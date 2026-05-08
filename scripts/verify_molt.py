"""Sanity-check a base LM vs. its MoLT-spliced counterpart on prompts.

Two questions this script answers:

  1. Does the MoLT-spliced model behave like the vanilla base LM? (full
     transcoder substitution faithfulness — KL / top-K agreement on the
     next-token distribution)
  2. Can the model do the task at all? (greedy completion + top-K next-token
     for either or both versions)

What "spliced" means here
-------------------------
For Gemma 3, MoLT was trained to map `pre_feedforward_layernorm.input` →
`post_feedforward_layernorm.output` (see
`crosslayer_transcoder/data/activation_sources.py`). To splice we register
two hooks per block:

  * forward-pre-hook on `pre_feedforward_layernorm` to capture the residual
    that MoLT consumes.
  * forward-hook on `post_feedforward_layernorm` to overwrite its output
    with `MoLT(captured_residual)`.

For GPT-2, MoLT was trained on `ln_2.input` → `mlp.output`; the splice
hooks ln_2 (pre) and mlp (post) instead.

Examples
--------
Compare vanilla and MoLT for Gemma 3-4B-IT on a one-shot reasoning prompt::

    uv run python scripts/verify_molt.py \\
        --hf-folder molt-multilayer-gemma3-4b-it-N50-100M-2gpu-b200-fp32weights \\
        --run-name 'sanity_comparison_completions' \\
        --base-model google/gemma-3-4b-it \\
        --prompt "3 days after Tuesday is" \\
        --chat-template --max-new-tokens 8

Multiple prompts at once::

    uv run python scripts/verify_molt.py ... \\
        --prompt "3 days after Tuesday is" \\
        --prompt "The capital of France is" \\
        --prompt "1 + 2 ="

Vanilla-only quick check (no MoLT load — useful when debugging the prompt or
chat template)::

    uv run python scripts/verify_molt.py \\
        --base-model google/gemma-3-4b-it \\
        --prompt "3 days after Tuesday is" \\
        --mode vanilla --chat-template
"""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import einops
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from crosslayer_transcoder.feature_dash.multilayer import (
    download_multilayer_from_hf,
    find_latest_step,
    load_multilayer_molt,
)


_DTYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


# ---------------------------------------------------------------------------
# Architecture-specific block resolution.
# ---------------------------------------------------------------------------


def _resolve_blocks(model) -> tuple[str, list]:
    """Return ("gpt2"|"gemma3", per-block module list).

    Mirrors the activation-source detection in `data/activation_sources.py`
    so the splice points line up with what MoLT was trained on.
    """
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return "gpt2", list(model.transformer.h)
    layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        layers = model.language_model.layers
    elif hasattr(model, "model") and hasattr(model.model, "language_model"):
        layers = model.model.language_model.layers
    if layers is None:
        raise ValueError(
            f"Don't know how to find transformer blocks on {type(model).__name__}"
        )
    sample = layers[0]
    if not hasattr(sample, "pre_feedforward_layernorm"):
        raise ValueError(
            f"Block type {type(sample).__name__} has no pre_feedforward_layernorm; "
            "this script only handles GPT-2 and Gemma 3 layouts."
        )
    return "gemma3", list(layers)


def _splice_targets(arch: str, block):
    """Return (capture_module, splice_module) for one transformer block.

    capture_module: forward-pre-hook reads MoLT's input residual from its arg.
    splice_module:  forward-hook overwrites the module's output with MoLT's recon.
    """
    if arch == "gpt2":
        return block.ln_2, block.mlp
    return block.pre_feedforward_layernorm, block.post_feedforward_layernorm


# ---------------------------------------------------------------------------
# MoLT splice forward (matches Molt.forward, but exposes per-batch shapes).
# ---------------------------------------------------------------------------


def _molt_layer_forward(inner_molt, resid: torch.Tensor, layer: int) -> torch.Tensor:
    """Run one per-layer Molt and return its standardized reconstruction.

    `resid` may be (B, T, D) at prefill or (B, 1, D) during generation; both
    flow through input/output standardizers without special casing.
    """
    acts = inner_molt.input_standardizer(resid, layer)
    pre = inner_molt.e(acts)
    gate = inner_molt.nonlinearity(pre)  # (..., n_features)
    raw = []
    for U, V in zip(inner_molt.Us, inner_molt.Vs):
        latents = einops.einsum(acts, V, "... d, n d r -> ... n r")
        raw.append(einops.einsum(latents, U, "... n r, n r d -> ... n d"))
    raw = torch.cat(raw, dim=-2)  # (..., n_features, d_acts)
    weighted = gate.unsqueeze(-1) * raw
    recons = weighted.sum(dim=-2)
    return inner_molt.output_standardizer(recons, layer)


@contextmanager
def splice_hooks(model, molt, arch: str, layers: list[int]):
    """Splice MoLT into `model` at the listed block indices for the duration.

    For each layer L in `layers`, registers:
      * pre-hook on the capture module that stores its input residual
      * post-hook on the splice module that returns MoLT's reconstruction
        (recomputed at output time so we always see the latest captured input)
    """
    blocks = _resolve_blocks(model)[1]
    captured: dict[int, torch.Tensor] = {}
    handles = []
    molt_dtype = next(molt.parameters()).dtype

    for L in layers:
        capture_mod, splice_mod = _splice_targets(arch, blocks[L])

        def make_pre(L=L):
            def pre(_m, args):
                captured[L] = args[0]
            return pre

        def make_post(L=L):
            def post(_m, _args, out):
                resid = captured[L]
                resid_in = resid.to(molt_dtype)
                rec = _molt_layer_forward(molt.molts[L], resid_in, L)
                # Match the dtype/device the model expects on the residual stream.
                rec = rec.to(dtype=out.dtype, device=out.device)
                return rec
            return post

        handles.append(capture_mod.register_forward_pre_hook(make_pre()))
        handles.append(splice_mod.register_forward_hook(make_post()))

    try:
        yield
    finally:
        for h in handles:
            h.remove()


# ---------------------------------------------------------------------------
# Prompt prep + comparison primitives.
# ---------------------------------------------------------------------------


def maybe_apply_chat_template(
    tokenizer, user_text: str, system: str | None
) -> str:
    """Wrap `user_text` in the chat template, stripping a duplicate BOS."""
    msgs: list[dict] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user_text})
    s = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    bos = getattr(tokenizer, "bos_token", None)
    if bos and s.startswith(bos):
        s = s[len(bos):]
    return s


def topk_str(probs: torch.Tensor, tokenizer, k: int) -> list[tuple[str, float]]:
    vals, idxs = torch.topk(probs, k)
    return [(tokenizer.decode([int(i)]), float(p)) for p, i in zip(vals, idxs)]


def kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> float:
    return float((p * (p.add(eps).log() - q.add(eps).log())).sum().item())


@torch.no_grad()
def next_token_probs(model, input_ids: torch.Tensor) -> torch.Tensor:
    out = model(input_ids=input_ids)
    return out.logits[0, -1].float().softmax(-1).cpu()


@torch.no_grad()
def greedy_complete(model, tokenizer, input_ids: torch.Tensor, n: int) -> str:
    """Append `n` greedy tokens. Returns only the newly generated text."""
    if n <= 0:
        return ""
    eos = tokenizer.eos_token_id
    cur = input_ids
    new_ids: list[int] = []
    for _ in range(n):
        logits = model(input_ids=cur).logits[0, -1]
        nxt = int(logits.argmax().item())
        new_ids.append(nxt)
        if eos is not None and nxt == eos:
            break
        cur = torch.cat([cur, torch.tensor([[nxt]], device=cur.device)], dim=1)
    return tokenizer.decode(new_ids)


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def run_prompt(
    model,
    tokenizer,
    prompt_text: str,
    arch: str,
    molt,
    *,
    mode: str,
    splice_layers: list[int] | None,
    topk: int,
    max_new_tokens: int,
) -> dict:
    """Run one prompt under selected mode(s) and return a comparison dict.

    `mode` is one of {"vanilla", "molt", "both"}. When MoLT is requested but
    `molt` is None, that side is skipped with a clear note in the output.
    """
    enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True)
    input_ids = enc.input_ids.to(next(model.parameters()).device)

    out: dict = {
        "prompt": prompt_text,
        "n_tokens": int(input_ids.shape[1]),
        "tokens": [tokenizer.decode([int(t)]) for t in input_ids[0]],
    }

    do_vanilla = mode in ("vanilla", "both")
    do_molt = mode in ("molt", "both") and molt is not None
    if mode in ("molt", "both") and molt is None:
        print("  (skipping MoLT side — checkpoints not loaded)", file=sys.stderr)

    if do_vanilla:
        probs_v = next_token_probs(model, input_ids)
        out["vanilla"] = {
            "top": topk_str(probs_v, tokenizer, topk),
            "completion": greedy_complete(
                model, tokenizer, input_ids, max_new_tokens
            ),
        }

    if do_molt:
        layers = splice_layers if splice_layers else list(range(molt.n_layers))
        with splice_hooks(model, molt, arch, layers):
            probs_m = next_token_probs(model, input_ids)
            comp_m = greedy_complete(
                model, tokenizer, input_ids, max_new_tokens
            )
        out["molt"] = {
            "top": topk_str(probs_m, tokenizer, topk),
            "completion": comp_m,
            "splice_layers": layers,
        }

    if do_vanilla and do_molt:
        out["kl_vanilla_molt"] = kl(probs_v, probs_m)
        out["agree_top1"] = (
            out["vanilla"]["top"][0][0] == out["molt"]["top"][0][0]
        )

    return out


def print_result(r: dict) -> None:
    print(f"\n=== Prompt: {r['prompt']!r}")
    print(f"  tokens ({r['n_tokens']}): {r['tokens']}")
    if "vanilla" in r:
        print("  vanilla:")
        print(f"    top-{len(r['vanilla']['top'])}: " + ", ".join(
            f"{t!r}={p:.3f}" for t, p in r["vanilla"]["top"]
        ))
        if r["vanilla"]["completion"]:
            print(f"    completion: {r['vanilla']['completion']!r}")
    if "molt" in r:
        print(f"  molt   (splice layers: {r['molt']['splice_layers']}):")
        print(f"    top-{len(r['molt']['top'])}: " + ", ".join(
            f"{t!r}={p:.3f}" for t, p in r["molt"]["top"]
        ))
        if r["molt"]["completion"]:
            print(f"    completion: {r['molt']['completion']!r}")
    if "kl_vanilla_molt" in r:
        print(
            f"  KL(vanilla || molt) = {r['kl_vanilla_molt']:.4f}   "
            f"top-1 agree: {r['agree_top1']}"
        )


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--prompt", action="append", required=True,
        help="prompt to evaluate; repeat for multiple",
    )
    p.add_argument(
        "--mode", choices=["vanilla", "molt", "both"], default="both",
        help="which model(s) to run (default: both)",
    )
    p.add_argument("--base-model", default="google/gemma-3-4b-it")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--dtype", default="bfloat16", choices=list(_DTYPE),
        help="dtype for the base LM (MoLT loads in fp32 to match the checkpoint)",
    )
    p.add_argument(
        "--ckpt-dir", default=None,
        help="local dir holding per-layer .pt files; mutually exclusive with --hf-folder",
    )
    p.add_argument("--hf-repo", default="kylelovesllms/molt-sweeps")
    p.add_argument(
        "--hf-folder", default=None,
        help="folder inside --hf-repo containing per-layer .pt files",
    )
    p.add_argument(
        "--run-name", default=None,
        help="filename prefix shared by every per-layer .pt; required when MoLT is used",
    )
    p.add_argument(
        "--step", default=None,
        help="checkpoint suffix like 'step19600' (default: latest)",
    )
    p.add_argument(
        "--splice-layers", default=None,
        help="comma-separated layer indices to splice MoLT into (default: all)",
    )
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument(
        "--chat-template", action="store_true",
        help="wrap each prompt with the tokenizer's chat template + generation prompt",
    )
    p.add_argument("--system-prompt", default=None)
    args = p.parse_args(argv)

    if args.mode != "vanilla":
        if args.ckpt_dir is None and args.hf_folder is None:
            p.error("--mode requires MoLT but neither --ckpt-dir nor --hf-folder was given")
        if args.run_name is None:
            p.error("--run-name is required when running MoLT")
    return args


def load_molt_if_needed(args) -> tuple[object | None, str | None]:
    if args.mode == "vanilla":
        return None, None
    if args.hf_folder is not None:
        print(f"=> downloading MoLT from {args.hf_repo}/{args.hf_folder} …", file=sys.stderr)
        ckpt_dir, step = download_multilayer_from_hf(
            repo_id=args.hf_repo,
            folder=args.hf_folder,
            run_name=args.run_name,
            step=args.step,
        )
    else:
        ckpt_dir = Path(args.ckpt_dir)
        step = args.step or find_latest_step(ckpt_dir, args.run_name)

    print(f"   loading checkpoints @ {step!r} from {ckpt_dir}", file=sys.stderr)
    molt, meta = load_multilayer_molt(
        ckpt_dir=ckpt_dir,
        run_name=args.run_name,
        step=step,
        device=args.device,
    )
    print(
        f"   d_acts={meta.d_acts} n_features={meta.n_features} "
        f"n_layers={meta.n_layers} ranks={meta.ranks} N={meta.N}",
        file=sys.stderr,
    )
    return molt, step


def main(argv=None) -> int:
    args = parse_args(argv)
    dtype = _DTYPE[args.dtype]

    print(f"=> loading base LM {args.base_model} ({args.dtype}) on {args.device} …", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = (
        AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype)
        .to(args.device)
        .eval()
    )
    arch, blocks = _resolve_blocks(model)
    print(f"   detected arch={arch}  n_blocks={len(blocks)}", file=sys.stderr)

    molt, _step = load_molt_if_needed(args)
    if molt is not None and molt.n_layers != len(blocks):
        print(
            f"WARN: MoLT has {molt.n_layers} layers but base LM has {len(blocks)} blocks. "
            "Splicing only the first min(...) layers.",
            file=sys.stderr,
        )

    splice_layers: list[int] | None = None
    if args.splice_layers:
        splice_layers = [int(x) for x in args.splice_layers.split(",") if x.strip()]

    for raw_prompt in args.prompt:
        prompt_text = (
            maybe_apply_chat_template(tokenizer, raw_prompt, args.system_prompt)
            if args.chat_template
            else raw_prompt
        )
        if args.chat_template:
            print(f"\nchat-templated: {prompt_text!r}", file=sys.stderr)
        result = run_prompt(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            arch=arch,
            molt=molt,
            mode=args.mode,
            splice_layers=splice_layers,
            topk=args.topk,
            max_new_tokens=args.max_new_tokens,
        )
        print_result(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
