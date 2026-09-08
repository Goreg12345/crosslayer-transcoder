"""Token-specific, error-preserving MOLT gate interventions on Qwen3."""

import hashlib
import json
import math
import threading
from contextlib import contextmanager
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from crosslayer_transcoder.dashboard.gates import MoltGates
from crosslayer_transcoder.dashboard.logits import transform_contribution


def target_gate(natural, amount, mode):
    if mode == "set":
        return torch.full_like(natural, amount)
    if mode == "add":
        return (natural + amount).clamp_min(0)
    if mode == "multiply":
        return natural * amount
    raise ValueError("Unknown gate mode")


def requested_amount(position, prompt_length, assignments, generated):
    """A generated index denotes a consumed token, not the token predicted by it."""
    if position < prompt_length:
        return assignments.get(position)
    index = position - prompt_length
    if generated and generated["start"] <= index <= generated["end"]:
        return generated["value"]
    return None


def reference_scale(mode, reference, active_mean):
    if mode == "multiply":
        return 1.0
    if reference == "raw":
        return 1.0
    if reference == "mean_active":
        if not math.isfinite(active_mean) or active_mean <= 0:
            raise ValueError(
                "No active-gate mean was observed for this transform; choose raw gate units"
            )
        return active_mean
    raise ValueError("Strength reference must be raw or mean_active")


def finite_number(value, name, minimum, maximum):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def integer(value, name, minimum, maximum):
    finite_number(value, name, minimum, maximum)
    if not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


class SteeringEngine:
    def __init__(self, directory, device="cuda:0"):
        self.directory = Path(directory)
        self.meta = json.loads((self.directory / "metadata.json").read_text())
        self.feature_stats = json.loads(
            (self.directory / "feature_stats.json").read_text()
        )
        self.device = device
        self.lock = threading.Lock()
        path = Path(self.meta["checkpoint"])
        if path.stat().st_mtime != self.meta["checkpoint_mtime"]:
            raise ValueError("Checkpoint changed since dashboard collection")
        checkpoint = torch.load(path, mmap=True, map_location="cpu", weights_only=False)
        self.state = checkpoint["state_dict"]
        self.layer = self.meta["layer"]
        hp = checkpoint["datamodule_hyper_parameters"]
        if (
            hp["model_arch"] != "qwen3"
            or hp["activation_input_location"] != "pre_norm"
            or hp["activation_output_location"] != "raw"
        ):
            raise ValueError(
                "Steering supports this Qwen3 pre_norm-to-raw configuration only"
            )
        self.gates = MoltGates(self.state, self.layer).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.meta["model"])
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.meta["model"],
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            .to(device)
            .eval()
            .requires_grad_(False)
        )
        self.eos_ids = self.model.generation_config.eos_token_id
        if isinstance(self.eos_ids, int):
            self.eos_ids = [self.eos_ids]
        self.eos_ids = set(self.eos_ids or [])

    def tokenize(self, prompt, prompt_format):
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 20000:
            raise ValueError("Enter a nonempty prompt of at most 20,000 characters")
        if prompt_format == "chat":
            ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        elif prompt_format == "raw":
            ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        else:
            raise ValueError("Prompt format must be chat or raw")
        if not 1 <= len(ids) <= 2048:
            raise ValueError(
                f"Prompt has {len(ids)} tokens; supported range is 1–2,048 (no truncation)"
            )
        digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        special = set(self.tokenizer.all_special_ids)
        return {
            "tokenization_id": digest,
            "ids": ids,
            "tokens": [
                {
                    "position": i,
                    "id": t,
                    "text": self.tokenizer.decode([t]),
                    "special": t in special,
                }
                for i, t in enumerate(ids)
            ],
        }

    def validate(self, body):
        tokenized = self.tokenize(body.get("prompt"), body.get("format", "chat"))
        if body.get("tokenization_id") != tokenized["tokenization_id"]:
            raise ValueError(
                "Prompt tokenization changed. Inspect the prompt and select tokens again."
            )
        feature = integer(
            body.get("feature"), "Transform ID", 0, self.meta["n_features"] - 1
        )
        mode = body.get("mode", "set")
        if mode not in ("set", "add", "multiply"):
            raise ValueError("Gate mode must be set, add, or multiply")
        reference = body.get("strength_reference", "raw")
        scale = reference_scale(
            mode, reference, self.feature_stats[feature]["mean_active"]
        )
        minimum = -100 if mode == "add" else 0
        rows = body.get("assignments", [])
        if not isinstance(rows, list) or len(rows) > len(tokenized["ids"]):
            raise ValueError("Invalid prompt-token assignments")
        assignments = {}
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Invalid token assignment")
            pos = integer(
                row.get("position"), "Token position", 0, len(tokenized["ids"]) - 1
            )
            if pos in assignments:
                raise ValueError("A token may have only one assignment")
            assignments[pos] = (
                finite_number(row.get("value"), "Gate strength", minimum, 100) * scale
            )
        generated = body.get("generated")
        max_new = integer(body.get("max_new_tokens", 64), "New tokens", 1, 256)
        if generated is not None:
            if not isinstance(generated, dict):
                raise ValueError("Invalid generated-token range")
            start = integer(generated.get("start"), "Generated start", 0, max_new - 1)
            end = integer(generated.get("end"), "Generated end", start, max_new - 1)
            generated = {
                "start": start,
                "end": end,
                "value": finite_number(
                    generated.get("value"), "Generated strength", minimum, 100
                )
                * scale,
            }
        return dict(
            prompt=body["prompt"],
            format=body.get("format", "chat"),
            tokenization_id=tokenized["tokenization_id"],
            ids=tokenized["ids"],
            feature=feature,
            mode=mode,
            strength_reference="natural" if mode == "multiply" else reference,
            reference_scale=scale,
            assignments=assignments,
            generated=generated,
            max_new_tokens=max_new,
            temperature=finite_number(body.get("temperature", 0), "Temperature", 0, 2),
            top_p=finite_number(body.get("top_p", 0.95), "Top-p", 0.01, 1),
            seed=integer(body.get("seed", 42), "Seed", 0, 2**32 - 1),
        )

    def natural_gate(self, x, feature):
        with torch.autocast(
            device_type=torch.device(self.device).type, dtype=torch.bfloat16
        ):
            return self.gates(x)[..., feature].float()

    @torch.inference_mode()
    def inspect(self, prompt, prompt_format, feature):
        integer(feature, "Transform ID", 0, self.meta["n_features"] - 1)
        data = self.tokenize(prompt, prompt_format)
        captured = []

        class Captured(Exception):
            pass

        def capture(_module, args):
            captured.append(self.natural_gate(args[0][0], feature).cpu().tolist())
            raise Captured()

        handle = self.model.model.layers[
            self.layer
        ].post_attention_layernorm.register_forward_pre_hook(capture)
        try:
            ids = torch.tensor([data["ids"]], device=self.device)
            try:
                self.model.model(
                    input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False
                )
            except Captured:
                pass
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError("Failed to capture prompt gates")
        for token, gate in zip(data["tokens"], captured[0]):
            token["gate"] = gate
        return data

    def feature_weights(self, feature):
        group, offset = 0, 0
        while feature >= offset + self.state[f"model.Us.{group}"].shape[0]:
            offset += self.state[f"model.Us.{group}"].shape[0]
            group += 1
        return (
            self.state[f"model.Vs.{group}"][feature - offset].to(self.device).float(),
            self.state[f"model.Us.{group}"][feature - offset].to(self.device).float(),
            self.state["model.output_standardizer.std"][self.layer]
            .to(self.device)
            .float(),
        )

    @contextmanager
    def intervention(self, config, trace, cursor):
        feature = config["feature"]
        v, u, std = self.feature_weights(feature)
        captured = {}

        def capture(_module, args):
            captured["input"] = args[0][0]

        def modify(_module, _args, output):
            x = captured.pop("input")
            positions, amounts = [], []
            for local in range(x.shape[0]):
                amount = requested_amount(
                    cursor[0] + local,
                    len(config["ids"]),
                    config["assignments"],
                    config["generated"],
                )
                if amount is not None:
                    positions.append(local)
                    amounts.append(amount)
            if not positions:
                return output
            selected = x[positions]
            natural = self.natural_gate(selected, feature)
            target = torch.stack(
                [
                    target_gate(natural[i], amount, config["mode"])
                    for i, amount in enumerate(amounts)
                ]
            )
            difference = target - natural
            # Preserve an exact no-op, including BF16 rounding.
            if bool((difference == 0).all()):
                delta = torch.zeros_like(selected, dtype=torch.float32)
            else:
                normalized = self.gates.standardizer(selected, self.layer).float()
                delta = transform_contribution(normalized, difference, v, u, std)
            if not torch.isfinite(delta).all():
                raise ValueError("Non-finite intervention; reduce steering strength")
            for i, local in enumerate(positions):
                trace.append(
                    {
                        "position": cursor[0] + local,
                        "source": "prompt"
                        if cursor[0] + local < len(config["ids"])
                        else "generated",
                        "before": natural[i].item(),
                        "after": target[i].item(),
                        "requested": amounts[i],
                        "delta_l2": delta[i].norm().item(),
                    }
                )
            if not bool((difference != 0).any()):
                return output
            updated = output.clone()
            updated[0, positions] = (output[0, positions].float() + delta).to(
                output.dtype
            )
            return updated

        layer = self.model.model.layers[self.layer]
        handles = [
            layer.post_attention_layernorm.register_forward_pre_hook(capture),
            layer.mlp.register_forward_hook(modify),
        ]
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def choose_token(self, logits, config, generator):
        if config["temperature"] == 0:
            return logits.argmax().view(1, 1)
        probs = torch.softmax(logits / config["temperature"], -1)
        sorted_probs, indices = probs.sort(descending=True)
        remove = sorted_probs.cumsum(-1) - sorted_probs > config["top_p"]
        sorted_probs[remove] = 0
        sample = torch.multinomial(sorted_probs, 1, generator=generator)
        return indices[sample].view(1, 1)

    @torch.inference_mode()
    def generate(self, config, cancel, progress, branch):
        ids = torch.tensor([config["ids"]], device=self.device)
        generator = torch.Generator(device=self.device).manual_seed(config["seed"])
        cache, produced, first_logits = None, [], None
        cursor, trace = [0], []
        from contextlib import nullcontext

        scope = (
            self.intervention(config, trace, cursor)
            if branch == "steered"
            else nullcontext()
        )
        with scope:
            for step in range(config["max_new_tokens"]):
                if cancel.is_set():
                    break
                out = self.model.model(
                    input_ids=ids, past_key_values=cache, use_cache=True
                )
                logits = self.model.lm_head(out.last_hidden_state[:, -1]).float()[0]
                if not torch.isfinite(logits).all():
                    raise ValueError(
                        "Non-finite model logits; reduce steering strength"
                    )
                if first_logits is None:
                    first_logits = logits.cpu()
                next_token = self.choose_token(logits, config, generator)
                token_id = next_token.item()
                produced.append(token_id)
                progress(
                    branch,
                    self.tokenizer.decode(produced, skip_special_tokens=True),
                    step + 1,
                )
                if token_id in self.eos_ids:
                    break
                cursor[0] += ids.shape[1]
                ids, cache = next_token, out.past_key_values
        return {
            "text": self.tokenizer.decode(produced, skip_special_tokens=True),
            "token_ids": produced,
            "trace": trace,
        }, first_logits

    def compare(self, config, cancel, progress):
        baseline, baseline_logits = self.generate(config, cancel, progress, "baseline")
        if cancel.is_set():
            return {"baseline": baseline, "steered": None, "cancelled": True}
        steered, steered_logits = self.generate(config, cancel, progress, "steered")
        result = {
            "baseline": baseline,
            "steered": steered,
            "cancelled": cancel.is_set(),
        }
        if baseline_logits is not None and steered_logits is not None:
            delta = steered_logits - baseline_logits
            selected = sorted(
                set(
                    baseline_logits.topk(8).indices.tolist()
                    + steered_logits.topk(8).indices.tolist()
                )
            )
            result["first_prediction"] = {
                "max_abs_logit_change": delta.abs().max().item(),
                "tokens": [
                    {
                        "id": i,
                        "token": self.tokenizer.decode([i]),
                        "baseline": baseline_logits[i].item(),
                        "steered": steered_logits[i].item(),
                        "delta": delta[i].item(),
                    }
                    for i in selected
                ],
            }
        result["config"] = config
        result["checkpoint"] = {
            k: self.meta[k] for k in ("checkpoint", "global_step", "model", "layer")
        }
        return result
