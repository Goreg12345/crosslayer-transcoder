"""Run analyze_prompt and render the selected features as a portable HTML bundle.

Pipeline:

  1. Load all per-layer MoLT checkpoints (one `MultiLayerMolt`). Checkpoints
     can be loaded from a local directory or downloaded from a folder inside
     a Hugging Face dataset/model repo.
  2. Run the base LM (GPT-2, Gemma 3, …) + MoLT on the prompt to find the
     top-K activating features per layer per token.
  3. Stream a corpus through the base LM + MoLT once to fill a
     `GateCollector` per layer with max-activating examples for *every*
     feature.
  4. Splice out only the (layer, feature) pairs that fired on the prompt
     and write a single self-contained `bundle.html` that opens by
     double-click.

Examples:
    # GPT-2 (local checkpoints)
    uv run python scripts/analyze_prompt_dashboard.py \\
        --prompt "Translate to Spanish: cat ->" \\
        --topk 8 \\
        --n-sequences 256 --seq-len 128 --device cpu \\
        --out feature_dash/translate-spanish-cat

    # Gemma 3 1B-it (HF folder download)
    uv run python scripts/analyze_prompt_dashboard.py \\
        --prompt "1+3=" \\
        --hf-folder molt-multilayer-gemma3-1b-it-N50-100M-4gpu-b200-fp32weights \\
        --run-name molt-multilayer-gemma3-1b-it-N50-100M-ddp-4gpu-b200 \\
        --base-model-name google/gemma-3-1b-it \\
        --dataset-name HuggingFaceFW/fineweb-edu --dataset-config sample-10BT \\
        --dtype bfloat16 \\
        --out feature_dash/gemma3-1b-prompt-1+3=
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

from crosslayer_transcoder.feature_dash.collect import (
    GateCollector,
    _decode_token,
    _iter_token_batches,
)
from crosslayer_transcoder.feature_dash.multilayer import (
    MultiLayerLMRunner,
    collect_multilayer_features,
    download_multilayer_from_hf,
    find_latest_step,
    load_multilayer_molt,
)
from crosslayer_transcoder.feature_dash.multilayer_bundle import (
    PromptTrace,
    make_multilayer_bundle,
)


def apply_chat_template_to_prompt(
    tokenizer,
    user_text: str,
    *,
    system_prompt: str | None = None,
) -> str:
    """Wrap `user_text` in the model's chat template and strip leading BOS.

    Returns the templated prompt as a plain string with `add_generation_prompt=True`.
    The tokenizer's BOS is stripped from the front because downstream calls of the
    form `tokenizer(prompt, return_tensors="pt")` add `add_special_tokens=True` by
    default and would otherwise prepend a second BOS.

    Errors if the tokenizer has no chat template (`apply_chat_template` itself
    raises). Returns `user_text` unchanged if `system_prompt` is None and the
    tokenizer's chat template is empty — but in practice IT models all set one.
    """
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_text})
    s = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    bos = getattr(tokenizer, "bos_token", None)
    if bos and s.startswith(bos):
        s = s[len(bos):]
    return s


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--prompt", required=True)

    # Either local --ckpt-dir or remote --hf-folder.
    p.add_argument("--ckpt-dir", default=None,
                   help="local directory holding per-layer .pt files; mutually exclusive with --hf-folder")
    p.add_argument("--hf-repo", default="kylelovesllms/molt-sweeps",
                   help="HF repo to download the per-layer .pt files from (with --hf-folder)")
    p.add_argument("--hf-folder", default=None,
                   help="folder inside --hf-repo holding the per-layer .pt files")
    p.add_argument("--run-name", required=True,
                   help="filename prefix shared by every per-layer .pt (e.g. molt-multilayer-N10-100M)")
    p.add_argument("--step", default=None,
                   help="checkpoint suffix like 'step36000' or 'tokens0100M352K' (default: latest)")

    p.add_argument("--topk", type=int, default=8,
                   help="top-K features per (layer, token) to include in the bundle")
    p.add_argument("--base-model-name", default="openai-community/gpt2",
                   help="HF model name of the base LM whose residuals MoLT was trained on")
    p.add_argument("--dataset-name", default="Skylion007/openwebtext")
    p.add_argument("--dataset-config", default=None,
                   help="HF datasets config name (some datasets need this, e.g. fineweb-edu sample-10BT)")
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
                   choices=["float32", "float16", "bfloat16"],
                   help="dtype for the base LM forward pass (MoLT stays in float32)")
    p.add_argument("--out", required=True,
                   help="output directory; bundle.html written inside")
    p.add_argument("--chat-template", action="store_true",
                   help="wrap --prompt as the user message in the tokenizer's "
                        "chat template (with add_generation_prompt=True). "
                        "Required for instruction-tuned models like "
                        "google/gemma-3-4b-it where raw completions don't trigger "
                        "the right behavior.")
    p.add_argument("--system-prompt", default=None,
                   help="optional system message; only meaningful with --chat-template")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    if (args.ckpt_dir is None) == (args.hf_folder is None):
        p.error("exactly one of --ckpt-dir or --hf-folder must be provided")
    return args


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
    tokenizer,
) -> tuple[list[str], torch.Tensor]:
    """Run the base LM on `prompt` and compute MoLT gates for every layer.

    Returns (token strings, gates tensor of shape (n_layers, seq_len, n_features)).
    """
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


def _collect_with_optional_config(
    args: argparse.Namespace,
    molt,
    tokenizer,
    dtype: torch.dtype,
) -> list[GateCollector]:
    """Stream a corpus once and fill one `GateCollector` per layer.

    `collect_multilayer_features` uses `_iter_token_batches`, which doesn't
    accept a `name=` config; for datasets that need one (e.g. fineweb-edu
    `sample-10BT`) we replicate the loop inline.
    """
    if not args.dataset_config:
        return collect_multilayer_features(
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
            tokenizer=tokenizer,
        )

    from datasets import load_dataset

    ds = load_dataset(
        args.dataset_name,
        name=args.dataset_config,
        split=args.dataset_split,
        streaming=True,
    )

    runner = MultiLayerLMRunner(
        model_name=args.base_model_name,
        n_layers=molt.n_layers,
        device=args.device,
        dtype=dtype,
    )
    molt = molt.to(args.device)
    collectors = [
        GateCollector(
            n_features=molt.n_features,
            top_k=args.top_k_examples,
            seq_len=args.seq_len,
        )
        for _ in range(molt.n_layers)
    ]

    try:
        buf: list[torch.Tensor] = []
        yielded = 0
        batches = 0
        for example in ds:
            if yielded >= args.n_sequences:
                break
            text = example.get("text") or example.get("content") or ""
            if not text:
                continue
            enc = tokenizer(
                text,
                truncation=True,
                max_length=args.seq_len,
                return_tensors="pt",
                add_special_tokens=False,
            )
            ids = enc["input_ids"][0]
            if ids.numel() < args.seq_len:
                continue
            buf.append(ids[: args.seq_len])
            if len(buf) == args.batch_size:
                tok = torch.stack(buf, dim=0)
                buf = []
                yielded += args.batch_size
                resid = runner.residuals(tok)
                for layer in range(molt.n_layers):
                    inner = molt.molts[layer]
                    acts = inner.input_standardizer(resid[:, :, layer, :], layer)
                    pre = inner.e(acts)
                    gates = inner.nonlinearity(pre)
                    collectors[layer].update(tok.cpu(), gates.float().cpu())
                batches += 1
                if args.verbose and batches % 8 == 0:
                    rates = [c.activation_rate().mean().item() for c in collectors]
                    print(
                        f"  batch {batches}: {collectors[0].total_tokens} tokens, "
                        f"mean fire rate {sum(rates)/len(rates):.4f}",
                        file=sys.stderr,
                    )
    finally:
        runner.close()
    return collectors


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # 1. Resolve checkpoint location.
    if args.hf_folder is not None:
        print(f"=> downloading {args.hf_folder} from {args.hf_repo}…", file=sys.stderr)
        ckpt_dir, step = download_multilayer_from_hf(
            repo_id=args.hf_repo,
            folder=args.hf_folder,
            run_name=args.run_name,
            step=args.step,
        )
        print(f"   step={step!r}, cached to {ckpt_dir}", file=sys.stderr)
    else:
        ckpt_dir = Path(args.ckpt_dir)
        step = (
            args.step
            if args.step is not None
            else find_latest_step(ckpt_dir, args.run_name)
        )

    print(f"Loading MoLT checkpoints @ step {step} from {ckpt_dir}", file=sys.stderr)

    dtype = _DTYPE[args.dtype]
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

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)

    prompt_text = args.prompt
    if args.chat_template:
        prompt_text = apply_chat_template_to_prompt(
            tokenizer, args.prompt, system_prompt=args.system_prompt,
        )
        print(f"Chat-templated prompt:\n{prompt_text!r}", file=sys.stderr)

    # 2. Prompt activations.
    print(f"Running prompt through {args.base_model_name} + MoLT…", file=sys.stderr)
    tokens, prompt_gates = _gates_for_prompt(
        molt=molt,
        base_model_name=args.base_model_name,
        prompt=prompt_text,
        device=args.device,
        dtype=dtype,
        tokenizer=tokenizer,
    )
    print(f"  tokens ({len(tokens)}): {tokens}", file=sys.stderr)
    selected, best = _select_topk_per_token(prompt_gates, args.topk)
    print(
        f"Selected {len(selected)} (layer, feature) pairs "
        f"across {meta.n_layers} layers (top-{args.topk} per token).",
        file=sys.stderr,
    )

    # 3. Corpus pass for max-activating examples.
    config_str = f":{args.dataset_config}" if args.dataset_config else ""
    print(
        f"Streaming {args.n_sequences} sequences from {args.dataset_name}{config_str} "
        f"through {args.base_model_name} + MoLT…",
        file=sys.stderr,
    )
    collectors = _collect_with_optional_config(args, molt, tokenizer, dtype)

    # 4. Bundle.
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
        prompt=prompt_text,
        prompt_traces=prompt_traces,
        dataset_name=args.dataset_name,
        window=args.window,
        base_model_name=args.base_model_name,
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
