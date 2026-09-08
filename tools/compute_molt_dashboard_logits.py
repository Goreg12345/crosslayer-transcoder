#!/usr/bin/env python3
"""Compute context-dependent direct logit projections for a collected MOLT dashboard."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from molt_dashboard_sampling import example_peak_indices
from transformers import AutoModelForCausalLM, AutoTokenizer

from crosslayer_transcoder.dashboard.gates import MoltGates
from crosslayer_transcoder.dashboard.logits import (
    frozen_rms_readout,
    signed_topk,
    transform_contribution,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "directory", nargs="?", type=Path, default=Path("results/molt-qwen-dashboard")
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--top-k", type=int, default=10)
    args = p.parse_args()
    if args.top_k < 1:
        p.error("top-k must be positive")
    root, device = args.directory, args.device
    torch.set_num_threads(4)
    # Highest FP32 precision for transform and logit arithmetic.
    torch.set_float32_matmul_precision("highest")
    meta = json.loads((root / "metadata.json").read_text())
    checkpoint_path = Path(meta["checkpoint"])
    if (
        checkpoint_path.stat().st_size != meta["checkpoint_bytes"]
        or checkpoint_path.stat().st_mtime != meta["checkpoint_mtime"]
    ):
        raise ValueError(
            "Checkpoint changed since activation collection; collect a new dashboard sample"
        )
    cache = np.load(root / "activations.npz")
    tokens, offsets = cache["tokens"], cache["offsets"]
    positions, feature_ids, values = (
        cache["positions"],
        cache["features"],
        cache["values"],
    )
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", mmap=True, weights_only=False
    )
    state = checkpoint["state_dict"]
    gates = MoltGates(state, meta["layer"]).to(device).eval()
    lm = (
        AutoModelForCausalLM.from_pretrained(
            meta["model"], torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .to(device)
        .eval()
    )
    lm.requires_grad_(False)
    normalized_inputs, denominators = [], []
    captured = {}
    rms_epsilon = lm.model.norm.variance_epsilon

    def input_hook(_module, inputs):
        captured["input"] = inputs[0][0].detach()

    def final_hook(_module, inputs):
        x = inputs[0][0].float()
        captured["rms"] = (x.square().mean(-1) + rms_epsilon).sqrt()

    handles = [
        lm.model.layers[
            meta["layer"]
        ].post_attention_layernorm.register_forward_pre_hook(input_hook),
        lm.model.norm.register_forward_pre_hook(final_hook),
    ]
    try:
        with torch.inference_mode():
            for i in range(len(offsets) - 1):
                lo, hi = int(offsets[i]), int(offsets[i + 1])
                ids = torch.tensor(tokens[lo:hi].astype(np.int64), device=device)[None]
                # Run all transformer layers, but skip the expensive full-sequence vocabulary head.
                lm.model(
                    input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False
                )
                x = captured.pop("input")
                norm = gates.standardizer(x, meta["layer"])
                with torch.autocast(
                    device_type=torch.device(device).type, dtype=torch.bfloat16
                ):
                    observed = gates(x)
                ix = (positions >= lo) & (positions < hi)
                expected = torch.zeros_like(observed)
                expected[
                    torch.as_tensor(positions[ix] - lo, device=device),
                    torch.as_tensor(feature_ids[ix], device=device),
                ] = torch.as_tensor(values[ix], device=device, dtype=expected.dtype)
                torch.testing.assert_close(observed, expected, rtol=0, atol=0)
                normalized_inputs.append(norm.cpu())
                denominators.append(captured.pop("rms").cpu())
                if (i + 1) % 8 == 0:
                    print(
                        f"Baseline {i + 1}/{len(offsets) - 1} conversations; gates match cached values",
                        flush=True,
                    )
    finally:
        for handle in handles:
            handle.remove()
    x_all = torch.cat(normalized_inputs)
    rms_all = torch.cat(denominators)
    norm_weight = lm.model.norm.weight.detach().float().clone()
    unembed = lm.get_output_embeddings().weight.detach().float().clone()
    del lm, gates, normalized_inputs, denominators
    torch.cuda.empty_cache()
    output_std = (
        state["model.output_standardizer.std"][meta["layer"]].to(device).float()
    )
    order = np.argsort(feature_ids, kind="stable")
    cuts = np.searchsorted(feature_ids[order], np.arange(meta["n_features"] + 1))
    output = {}
    group, group_start = 0, 0
    for f in range(meta["n_features"]):
        while f >= group_start + state[f"model.Us.{group}"].shape[0]:
            group_start += state[f"model.Us.{group}"].shape[0]
            group += 1
        ix = order[cuts[f] : cuts[f + 1]]
        if not len(ix):
            continue
        pos, val = positions[ix], values[ix]
        convs, groups = example_peak_indices(pos, val, offsets)
        selected = [(title, j) for title, indices in groups for j in indices]
        selected_pos = [int(pos[j]) for _, j in selected]
        u = state[f"model.Us.{group}"][f - group_start].to(device).float()
        v = state[f"model.Vs.{group}"][f - group_start].to(device).float()
        # Accumulate the readout over ALL active tokens, not only top examples.
        total = torch.zeros(x_all.shape[1], device=device)
        with torch.inference_mode():
            for start in range(0, len(pos), 2048):
                chunk = pos[start : start + 2048]
                contribution = transform_contribution(
                    x_all[chunk].to(device).float(),
                    torch.as_tensor(val[start : start + 2048], device=device),
                    v,
                    u,
                    output_std,
                )
                total += frozen_rms_readout(
                    contribution, rms_all[chunk].to(device), norm_weight
                ).sum(0)
            contrib = transform_contribution(
                x_all[selected_pos].to(device).float(),
                torch.tensor([float(val[j]) for _, j in selected], device=device),
                v,
                u,
                output_std,
            )
            readouts = frozen_rms_readout(
                contrib, rms_all[selected_pos].to(device), norm_weight
            )
            vectors = torch.cat([total[None] / len(pos), readouts])
            scores = vectors @ unembed.T
            if not torch.isfinite(scores).all():
                raise ValueError(f"Nonfinite logits for transform {f}")
            data = [signed_topk(row, args.top_k) for row in scores]
        examples = []
        for (title, j), entry in zip(selected, data[1:]):
            conv = int(convs[j])
            absolute = int(pos[j])
            relative = absolute - int(offsets[conv])
            entry.update(
                position=absolute,
                conversation=conv,
                token_position=relative,
                gate=float(val[j]),
                group=title,
            )
            examples.append(entry)
        output[str(f)] = dict(active_tokens=len(pos), mean=data[0], examples=examples)
        if (f + 1) % 100 == 0:
            print(f"Projected {f + 1}/{meta['n_features']} transforms", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(meta["model"])
    ids = set()
    for data in output.values():
        for entry in [data["mean"], *data["examples"]]:
            ids.update(entry["top_token_ids"])
            ids.update(entry["bottom_token_ids"])
    vocab = {str(i): tokenizer.decode([i]) for i in ids}
    result = dict(
        method="direct projection with baseline final RMS frozen",
        formula="W_unembed @ (final_norm_weight * output_std * gate * ((x_normalized @ V) @ U) / baseline_final_rms)",
        interpretation="Positive/negative raw logit contribution at the next-token prediction. Downstream layers and normalization changes are excluded; not a causal ablation.",
        arithmetic="FP32 transforms and unembedding applied to cached-equivalent BF16 standardized inputs and gates",
        aggregation="Arithmetic mean over all active sampled tokens; averages may cancel context-dependent effects",
        checkpoint=meta["checkpoint"],
        global_step=meta["global_step"],
        tokens=len(tokens),
        top_k=args.top_k,
        vocab=vocab,
        features=output,
    )
    tmp = root / "logit_projections.json.tmp"
    tmp.write_text(json.dumps(result))
    tmp.replace(root / "logit_projections.json")
    print(f"Saved projections for {len(output)} active transforms", flush=True)


if __name__ == "__main__":
    main()
