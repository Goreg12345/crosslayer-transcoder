"""Build a multi-layer MoLT feature dashboard from a Hugging Face checkpoint folder.

The HF repo `kylelovesllms/molt-sweeps` stores multi-layer MoLT runs as one
folder per run, with one `.pt` per (layer, step). This script:

  1. Downloads every per-layer `.pt` for the latest step in the folder.
  2. Loads them as a `MultiLayerMolt`.
  3. Streams a corpus through the base LM (GPT-2 / Gemma 3 / etc.) and
     fills one `GateCollector` per layer.
  4. Optionally calls Claude (Haiku by default) to label each live feature.
  5. Writes a single self-contained `bundle_<run>.html`.

Examples
--------
GPT-2:
    uv run python scripts/build_multilayer_dashboard.py \\
        --hf-folder molt-multilayer-gpt2-N50-100M-fp32weights-full \\
        --run-name molt-multilayer-N50-100M-ddp \\
        --base-model openai-community/gpt2 \\
        --out feature_dash/molt-multilayer-gpt2-N50

Gemma 3 1B:
    uv run python scripts/build_multilayer_dashboard.py \\
        --hf-folder molt-multilayer-gemma3-1b-it-N50-100M-4gpu-b200-fp32weights \\
        --run-name molt-multilayer-gemma3-1b-it-N50-100M-ddp-4gpu-b200 \\
        --base-model google/gemma-3-1b-it \\
        --dataset-name HuggingFaceFW/fineweb-edu \\
        --dataset-config sample-10BT \\
        --out feature_dash/molt-multilayer-gemma3-1b
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch

from crosslayer_transcoder.feature_dash.annotate import (
    annotate_features,
    select_features_for_annotation,
)
from crosslayer_transcoder.feature_dash.collect import window_feature_summary
from crosslayer_transcoder.feature_dash.multilayer import (
    collect_multilayer_features,
    download_multilayer_from_hf,
    load_multilayer_molt,
)
from crosslayer_transcoder.feature_dash.multilayer_bundle import (
    make_multilayer_bundle,
)


_DTYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--hf-repo", default="kylelovesllms/molt-sweeps")
    p.add_argument("--hf-folder", required=True,
                   help="folder inside the HF repo holding the per-layer .pt files")
    p.add_argument("--run-name", required=True,
                   help="filename prefix shared by every per-layer .pt")
    p.add_argument("--step", default=None,
                   help="suffix like 'step36000' or 'tokens0100M352K' (default: latest)")
    p.add_argument("--base-model", required=True,
                   help="HF model name of the base LM whose residuals MoLT was trained on")
    p.add_argument("--dataset-name", default="Skylion007/openwebtext")
    p.add_argument("--dataset-config", default=None,
                   help="HF datasets `name=` config (some datasets need this)")
    p.add_argument("--dataset-split", default="train")
    p.add_argument("--n-sequences", type=int, default=1024)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--top-k-examples", type=int, default=20)
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "float16", "bfloat16"],
                   help="dtype for the base LM forward pass (MoLT stays in float32)")
    p.add_argument("--out", required=True, help="output directory; bundle.html written inside")

    # Annotation knobs.
    p.add_argument("--annotate", action="store_true",
                   help="call Claude to label every live feature")
    p.add_argument("--annotate-model", default="anthropic/claude-haiku-4.5",
                   help="OpenRouter model id (e.g. anthropic/claude-haiku-4.5)")
    p.add_argument("--annotate-workers", type=int, default=20)
    p.add_argument("--annotate-max-per-layer", type=int, default=None,
                   help="if set, keep only the top-K (by max activation) per layer")
    p.add_argument("--annotate-min-max-activation", type=float, default=0.05,
                   help="drop features whose peak activation is below this")
    p.add_argument("--annotate-only-estimate", action="store_true",
                   help="report the number of features to annotate and exit")

    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def _stream_and_collect(args, molt, tokenizer):
    """Wrapper around `collect_multilayer_features` that respects `--dataset-config`."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    if args.dataset_config:
        # Some datasets require a config name; the helper in `multilayer` only
        # streams the default config. Build the iterator inline so we can pass
        # it. For the standard openwebtext default, we fall back to the simpler path.
        from crosslayer_transcoder.feature_dash.multilayer import (
            MultiLayerLMRunner,
        )
        from crosslayer_transcoder.feature_dash.collect import GateCollector

        ds = load_dataset(
            args.dataset_name,
            name=args.dataset_config,
            split=args.dataset_split,
            streaming=True,
        )

        runner = MultiLayerLMRunner(
            model_name=args.base_model,
            n_layers=molt.n_layers,
            device=args.device,
            dtype=_DTYPE[args.dtype],
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
        return collectors, tokenizer

    # No custom dataset config — use the helper.
    collectors = collect_multilayer_features(
        molt=molt,
        base_model_name=args.base_model,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        n_sequences=args.n_sequences,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        top_k=args.top_k_examples,
        device=args.device,
        dtype=_DTYPE[args.dtype],
        log_every=8 if args.verbose else 0,
        tokenizer=tokenizer,
    )
    return collectors, tokenizer


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    print(f"=> downloading {args.hf_folder} from {args.hf_repo}…", file=sys.stderr)
    local_dir, step = download_multilayer_from_hf(
        repo_id=args.hf_repo,
        folder=args.hf_folder,
        run_name=args.run_name,
        step=args.step,
    )
    print(f"   step={step!r}, cached to {local_dir}", file=sys.stderr)

    print("=> loading MultiLayerMolt…", file=sys.stderr)
    molt, meta = load_multilayer_molt(
        ckpt_dir=local_dir,
        run_name=args.run_name,
        step=step,
        device=args.device,
    )
    print(
        f"   d_acts={meta.d_acts} n_features={meta.n_features} "
        f"n_layers={meta.n_layers} ranks={meta.ranks} N={meta.N}",
        file=sys.stderr,
    )
    total_features = meta.n_features * meta.n_layers
    print(f"   total features across all layers: {total_features:,}", file=sys.stderr)

    print(
        f"=> streaming {args.n_sequences} sequences "
        f"({args.dataset_name}{':' + args.dataset_config if args.dataset_config else ''}) "
        f"through {args.base_model}…",
        file=sys.stderr,
    )
    t0 = time.monotonic()
    collectors, tokenizer = _stream_and_collect(args, molt, tokenizer=None)
    elapsed = time.monotonic() - t0
    print(
        f"   collected {collectors[0].total_tokens:,} tokens in {elapsed:.1f}s",
        file=sys.stderr,
    )

    print("=> building per-layer feature summaries…", file=sys.stderr)
    feature_summaries_per_layer: list[list[dict]] = []
    selected: list[tuple[int, int]] = []
    live_per_layer = []
    for layer, coll in enumerate(collectors):
        summaries = []
        live = 0
        for f_id in range(meta.n_features):
            summary = coll.feature_summary(f_id)
            body = window_feature_summary(summary, tokenizer, window=args.window)
            summaries.append(body)
            if body["examples"]:
                selected.append((layer, f_id))
                live += 1
        feature_summaries_per_layer.append(summaries)
        live_per_layer.append(live)
    total_live = sum(live_per_layer)
    print(
        f"   live features (>=1 example): {total_live:,} "
        f"of {total_features:,} ({100 * total_live / max(1, total_features):.1f}%)",
        file=sys.stderr,
    )

    descriptions: dict[tuple[int, int], str] = {}
    if args.annotate or args.annotate_only_estimate:
        to_annotate = select_features_for_annotation(
            collectors,
            feature_summaries_per_layer,
            min_max_activation=args.annotate_min_max_activation,
            max_per_layer=args.annotate_max_per_layer,
        )
        # Cost / time rough estimate. Numbers are conservative — each request
        # is a few hundred input tokens (cached on first hit) plus ~50 output tokens.
        n = len(to_annotate)
        est_tokens_in = n * 700  # ~700 input tokens per req incl. examples
        est_tokens_out = n * 60
        # Haiku 4.5 list price (~$1/Min, $5/Mout) — informational only.
        est_cost = est_tokens_in * 1.0 / 1e6 + est_tokens_out * 5.0 / 1e6
        # Wallclock @ 20 workers, ~2s/req latency => 0.1s/req amortised.
        est_seconds = max(30, n / max(1, args.annotate_workers) * 2.0)
        print(
            f"=> annotation plan: {n:,} features, "
            f"~${est_cost:.2f}, ~{est_seconds/60:.1f} min @ {args.annotate_workers} workers",
            file=sys.stderr,
        )
        if args.annotate_only_estimate:
            return 0

        t1 = time.monotonic()
        descriptions = annotate_features(
            to_annotate,
            model=args.annotate_model,
            max_workers=args.annotate_workers,
        )
        actual = time.monotonic() - t1
        print(
            f"   annotated {sum(1 for v in descriptions.values() if v)}/{n} features "
            f"in {actual:.1f}s ({actual/60:.1f} min)",
            file=sys.stderr,
        )

    print("=> writing bundle…", file=sys.stderr)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = make_multilayer_bundle(
        collectors=collectors,
        meta=meta,
        tokenizer=tokenizer,
        out_path=out_dir,
        selected=selected,
        prompt=None,
        prompt_traces=None,
        dataset_name=args.dataset_name,
        window=args.window,
        descriptions=descriptions or None,
        base_model_name=args.base_model,
    )
    size_mb = bundle_path.stat().st_size / 1e6
    print(
        f"\nWrote {bundle_path} ({size_mb:.1f} MB, {len(selected)} live entries) — "
        f"open by double-click.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
