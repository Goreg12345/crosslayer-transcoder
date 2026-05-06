"""Run analyze_prompt and render the selected features as a portable HTML bundle.

Pipeline:

  1. Load all per-layer MoLT checkpoints (one `MultiLayerMolt`).
  2. Run GPT-2 + MoLT on the prompt to find the top-K activating features
     per layer per token (same logic as `scripts/analyze_prompt.py`).
  3. Stream a corpus through GPT-2 + MoLT once to fill a `GateCollector`
     per layer with max-activating examples for *every* feature.
  4. Splice out only the (layer, feature) pairs that fired on the prompt
     and write a single self-contained `bundle.html` that opens by
     double-click.

Example:
    uv run python scripts/analyze_prompt_dashboard.py \\
        --prompt "Translate to Spanish: cat ->" \\
        --topk 8 \\
        --n-sequences 256 --seq-len 128 --device cpu \\
        --out feature_dash/translate-spanish-cat
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from crosslayer_transcoder.feature_dash.collect import _decode_token
from crosslayer_transcoder.feature_dash.multilayer import (
    MultiLayerLMRunner,
    collect_multilayer_features,
    find_latest_step,
    load_multilayer_molt,
)
from crosslayer_transcoder.feature_dash.multilayer_bundle import (
    PromptTrace,
    make_multilayer_bundle,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--prompt", required=True)
    p.add_argument("--ckpt-dir", default="checkpoints/molt-multilayer-N10-100M")
    p.add_argument("--run-name", default="molt-multilayer-N10-100M")
    p.add_argument("--step", type=int, default=None,
                   help="checkpoint step (default: latest)")
    p.add_argument("--topk", type=int, default=8,
                   help="top-K features per (layer, token) to include in the bundle")
    p.add_argument("--base-model-name", default="openai-community/gpt2")
    p.add_argument("--dataset-name", default="Skylion007/openwebtext")
    p.add_argument("--dataset-split", default="train")
    p.add_argument("--n-sequences", type=int, default=512,
                   help="corpus sequences to stream for max-activating examples")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--top-k-examples", type=int, default=20,
                   help="max-activating examples kept per feature")
    p.add_argument("--window", type=int, default=32,
                   help="tokens of context on each side of the peak")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--out", required=True,
                   help="output directory; bundle.html written inside")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


_DTYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@torch.no_grad()
def _gates_for_prompt(
    molt,
    base_model_name: str,
    prompt: str,
    device: str,
    dtype: torch.dtype,
) -> tuple[list[str], torch.Tensor]:
    """Run GPT-2 on `prompt` and compute MoLT gates for every layer.

    Returns (token strings, gates tensor of shape (n_layers, seq_len, n_features)).
    """
    tokenizer = GPT2TokenizerFast.from_pretrained(base_model_name)
    runner = MultiLayerLMRunner(
        model_name=base_model_name,
        n_layers=molt.n_layers,
        device=device,
        dtype=dtype,
    )
    try:
        enc = tokenizer(prompt, return_tensors="pt")
        token_ids = enc.input_ids
        resid = runner.residuals(token_ids)  # (1, T, L, D)
    finally:
        runner.close()

    gates_per_layer: list[torch.Tensor] = []
    for layer in range(molt.n_layers):
        inner = molt.molts[layer]
        acts = inner.input_standardizer(resid[:, :, layer, :], layer)
        pre = inner.e(acts)
        g = inner.nonlinearity(pre)  # (1, T, F)
        gates_per_layer.append(g[0].float().cpu())

    gates = torch.stack(gates_per_layer, dim=0)  # (L, T, F)
    tokens = [_decode_token(tokenizer, int(t)) for t in token_ids[0]]
    return tokens, gates


def _select_topk_per_token(
    gates: torch.Tensor, topk: int
) -> tuple[set[tuple[int, int]], dict[tuple[int, int], tuple[int, float]]]:
    """For each (layer, token) pick the top-K features with positive activation.

    Returns:
      - the set of (layer, feature_id) pairs to include in the bundle,
      - a dict (layer, feature_id) -> (best_token_pos, best_activation), used
        only as a quick textual report.
    """
    L, T, F = gates.shape
    selected: set[tuple[int, int]] = set()
    best: dict[tuple[int, int], tuple[int, float]] = {}

    k = min(topk, F)
    for layer in range(L):
        for t in range(T):
            row = gates[layer, t]
            vals, idxs = torch.topk(row, k)
            for v, i in zip(vals.tolist(), idxs.tolist()):
                if v <= 0.0:
                    continue
                key = (layer, int(i))
                selected.add(key)
                prev = best.get(key)
                if prev is None or v > prev[1]:
                    best[key] = (t, float(v))
    return selected, best


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    ckpt_dir = Path(args.ckpt_dir)
    step = args.step if args.step is not None else find_latest_step(ckpt_dir, args.run_name)
    print(f"Loading MoLT checkpoints @ step {step} from {ckpt_dir}", file=sys.stderr)

    dtype = _DTYPE[args.dtype]
    molt, meta = load_multilayer_molt(
        ckpt_dir=ckpt_dir,
        run_name=args.run_name,
        step=step,
        device=args.device,
    )

    # Stage 1: prompt activations.
    print(f"Running prompt through GPT-2 + MoLT…", file=sys.stderr)
    tokens, prompt_gates = _gates_for_prompt(
        molt=molt,
        base_model_name=args.base_model_name,
        prompt=args.prompt,
        device=args.device,
        dtype=dtype,
    )
    print(f"  tokens ({len(tokens)}): {tokens}", file=sys.stderr)
    selected, best = _select_topk_per_token(prompt_gates, args.topk)
    print(
        f"Selected {len(selected)} (layer, feature) pairs "
        f"across {meta.n_layers} layers (top-{args.topk} per token).",
        file=sys.stderr,
    )

    # Stage 2: corpus pass for max-activating examples.
    print(
        f"Streaming {args.n_sequences} sequences from {args.dataset_name} "
        f"through GPT-2 + MoLT…",
        file=sys.stderr,
    )
    collectors = collect_multilayer_features(
        molt=molt,
        base_model_name=args.base_model_name,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        n_sequences=args.n_sequences,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        top_k=args.top_k_examples,
        device=args.device,
        dtype=dtype,
        log_every=8 if args.verbose else 0,
    )

    # Stage 3: bundle.
    tokenizer = GPT2TokenizerFast.from_pretrained(args.base_model_name)
    prompt_traces: dict[tuple[int, int], PromptTrace] = {}
    for (layer, feat) in selected:
        prompt_traces[(layer, feat)] = PromptTrace(
            tokens=tokens,
            activations=prompt_gates[layer, :, feat].tolist(),
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = make_multilayer_bundle(
        collectors=collectors,
        meta=meta,
        tokenizer=tokenizer,
        out_path=out_dir,
        selected=selected,
        prompt=args.prompt,
        prompt_traces=prompt_traces,
        dataset_name=args.dataset_name,
        window=args.window,
    )

    size_mb = bundle_path.stat().st_size / 1e6
    print(
        f"\nWrote {bundle_path} ({size_mb:.1f} MB, {len(selected)} entries) — "
        f"open by double-click.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
