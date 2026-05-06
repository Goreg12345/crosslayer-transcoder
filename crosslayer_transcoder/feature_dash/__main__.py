"""CLI entrypoint: build a feature dashboard from a MoLT checkpoint.

    python -m crosslayer_transcoder.feature_dash \\
        --hf-filename gpt2-molt-lam-0_00015-50M.ckpt \\
        --out feature_dash/lam_0_00015_50M

Use `--local-ckpt` instead of `--hf-filename` to load a `.ckpt` from disk.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

from crosslayer_transcoder.feature_dash.bundle import make_bundle
from crosslayer_transcoder.feature_dash.collect import collect_features
from crosslayer_transcoder.feature_dash.dump import dump_dashboard
from crosslayer_transcoder.feature_dash.load import (
    DEFAULT_HF_REPO,
    load_molt,
    load_molt_from_hf,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m crosslayer_transcoder.feature_dash",
        description="Build a per-transform feature dashboard for a MoLT checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--hf-filename",
        help=f"Filename inside the HF repo (default repo: {DEFAULT_HF_REPO}).",
    )
    src.add_argument(
        "--local-ckpt",
        help="Path to a local .ckpt file. Mutually exclusive with --hf-filename.",
    )
    p.add_argument(
        "--repo-id",
        default=DEFAULT_HF_REPO,
        help="HF repo id when using --hf-filename.",
    )
    p.add_argument(
        "--revision",
        default=None,
        help="Optional HF revision (commit/tag/branch).",
    )

    p.add_argument(
        "--out", required=True, help="Output directory; created if missing."
    )
    p.add_argument(
        "--layer",
        type=int,
        default=8,
        help="Layer to capture residual at (matches MoLT training layer).",
    )

    p.add_argument(
        "--base-model-name",
        default="openai-community/gpt2",
        help="HF model name for the base LM that produces residuals.",
    )
    p.add_argument(
        "--dataset-name",
        default="Skylion007/openwebtext",
        help="HF dataset name for the corpus pass.",
    )
    p.add_argument(
        "--dataset-split", default="train", help="Dataset split to stream."
    )

    p.add_argument(
        "--n-sequences",
        type=int,
        default=1024,
        help="Number of sequences to stream from the corpus.",
    )
    p.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="Tokens per sequence; sequences shorter than this are dropped.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Sequences per LM forward pass.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of max-activating examples kept per transform.",
    )
    p.add_argument(
        "--window",
        type=int,
        default=32,
        help="Tokens of context on each side of the peak in the rendered view.",
    )

    p.add_argument(
        "--device",
        default=None,
        help="Torch device. Defaults to cuda if available, else cpu.",
    )
    p.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Base LM dtype.",
    )

    p.add_argument(
        "--no-assets",
        action="store_true",
        help="Skip copying the HTML/CSS/JS templates (data only).",
    )
    p.add_argument(
        "--bundle",
        action="store_true",
        help="Also write a single-file `bundle.html` with all data inlined "
             "(opens by double-click, no server needed — share with teammates).",
    )
    p.add_argument(
        "--bundle-only",
        action="store_true",
        help="Write only `bundle.html` — skip the multi-file dashboard "
             "(implies --no-assets and removes data/ after).",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true", help="Log progress."
    )

    return p


_DTYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.hf_filename:
        molt, meta = load_molt_from_hf(
            filename=args.hf_filename,
            repo_id=args.repo_id,
            revision=args.revision,
            device=device,
        )
    else:
        molt, meta = load_molt(args.local_ckpt, device=device)

    print(
        f"loaded MoLT: {meta.n_features} transforms, ranks={meta.ranks}, "
        f"d_acts={meta.d_acts}, base LM={meta.base_model_name or args.base_model_name}",
        file=sys.stderr,
    )

    collector = collect_features(
        molt=molt,
        layer=args.layer,
        base_model_name=args.base_model_name,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        n_sequences=args.n_sequences,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        top_k=args.top_k,
        device=device,
        dtype=_DTYPE[args.dtype],
        log_every=8 if args.verbose else 0,
    )

    from transformers import GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained(args.base_model_name)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_multi_file = not args.bundle_only
    write_bundle = args.bundle or args.bundle_only

    if write_multi_file:
        dump_dashboard(
            collector=collector,
            meta=meta,
            tokenizer=tokenizer,
            out_dir=out_dir,
            layer=args.layer,
            dataset_name=args.dataset_name,
            window=args.window,
            copy_assets=not args.no_assets,
        )

    bundle_path = None
    if write_bundle:
        # Pass the directory so the bundle is named after the checkpoint
        # (e.g. `bundle_gpt2-molt-lam-0_00015-50M.html`).
        bundle_path = make_bundle(
            collector=collector,
            meta=meta,
            tokenizer=tokenizer,
            out_path=out_dir,
            layer=args.layer,
            dataset_name=args.dataset_name,
            window=args.window,
        )

    if args.bundle_only:
        # Drop the data/ dir we never wrote — but if the user re-ran into
        # the same dir, leave any prior dump alone.
        pass

    print(f"\nDashboard written to {out_dir.resolve()}", file=sys.stderr)
    if write_multi_file and not args.no_assets:
        print(
            f"  Multi-file: cd {out_dir} && python -m http.server 8050\n"
            f"              Then open http://localhost:8050/index.html",
            file=sys.stderr,
        )
    if bundle_path is not None:
        size_mb = bundle_path.stat().st_size / 1e6
        print(
            f"  Bundle:     {bundle_path} ({size_mb:.1f} MB) — "
            f"open by double-click; share with teammates",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
