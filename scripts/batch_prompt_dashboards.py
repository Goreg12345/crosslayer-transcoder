"""Run analyze_prompt_dashboard for *multiple* prompts against *one* MoLT.

For each (base LM, MoLT) pair, the corpus pass that builds the per-feature
max-activating examples dominates wall-time. This script does that pass
exactly once and then writes one bundle.html per prompt.

Each prompt is specified as a positional `--prompt name=text` pair (the
`name` is used as the output subdirectory). Example:

    uv run python scripts/batch_prompt_dashboards.py \\
        --hf-folder molt-multilayer-gemma3-1b-it-N50-100M-4gpu-b200-fp32weights \\
        --run-name molt-multilayer-gemma3-1b-it-N50-100M-ddp-4gpu-b200 \\
        --base-model-name google/gemma-3-1b-it \\
        --dataset-name HuggingFaceFW/fineweb-edu --dataset-config sample-10BT \\
        --dtype bfloat16 \\
        --out feature_dash/gemma3-1b \\
        --prompt 'add-1+3=' '1+3=' \\
        --prompt 'add-2+3=' '2+3=' \\
        --prompt 'translate-hot-dog' 'hot->calor,dog->'
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

from crosslayer_transcoder.feature_dash.multilayer import (
    download_multilayer_from_hf,
    find_latest_step,
    load_multilayer_molt,
)
from crosslayer_transcoder.feature_dash.multilayer_bundle import (
    PromptTrace,
    make_multilayer_bundle,
)

# Re-use helpers from analyze_prompt_dashboard rather than duplicate them.
sys.path.insert(0, str(Path(__file__).parent))
from analyze_prompt_dashboard import (  # noqa: E402
    _DTYPE,
    _collect_with_optional_config,
    _gates_for_prompt,
    _select_topk_per_token,
    apply_chat_template_to_prompt,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ckpt-dir", default=None)
    p.add_argument("--hf-repo", default="kylelovesllms/molt-sweeps")
    p.add_argument("--hf-folder", default=None)
    p.add_argument("--run-name", required=True)
    p.add_argument("--step", default=None)

    p.add_argument(
        "--prompt",
        nargs=2,
        action="append",
        metavar=("NAME", "TEXT"),
        required=True,
        help="Bundle subdir name and prompt text. Repeat per prompt.",
    )
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--base-model-name", default="openai-community/gpt2")
    p.add_argument("--dataset-name", default="Skylion007/openwebtext")
    p.add_argument("--dataset-config", default=None)
    p.add_argument("--dataset-split", default="train")
    p.add_argument("--n-sequences", type=int, default=512)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--top-k-examples", type=int, default=20)
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--out", required=True,
                   help="parent output directory; one subdir per prompt name")
    p.add_argument("--chat-template", action="store_true",
                   help="wrap each --prompt TEXT as the user message in the "
                        "tokenizer's chat template (with add_generation_prompt=True). "
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

    # 2. Per-prompt gates (tiny: one short forward each).
    prompt_payloads: list[tuple[str, str, list[str], torch.Tensor, set[tuple[int, int]]]] = []
    for name, text in args.prompt:
        if args.chat_template:
            text = apply_chat_template_to_prompt(
                tokenizer, text, system_prompt=args.system_prompt,
            )
            print(f"[{name}] chat-templated:\n{text!r}", file=sys.stderr)
        print(f"\n[{name}] running prompt {text!r} through {args.base_model_name} + MoLT…",
              file=sys.stderr)
        tokens, prompt_gates = _gates_for_prompt(
            molt=molt,
            base_model_name=args.base_model_name,
            prompt=text,
            device=args.device,
            dtype=dtype,
            tokenizer=tokenizer,
        )
        print(f"  tokens ({len(tokens)}): {tokens}", file=sys.stderr)
        selected, _best = _select_topk_per_token(prompt_gates, args.topk)
        print(
            f"  selected {len(selected)} (layer, feature) pairs "
            f"(top-{args.topk} per token).",
            file=sys.stderr,
        )
        prompt_payloads.append((name, text, tokens, prompt_gates, selected))

    # 3. ONE corpus pass shared across all prompts.
    config_str = f":{args.dataset_config}" if args.dataset_config else ""
    print(
        f"\nStreaming {args.n_sequences} sequences from {args.dataset_name}{config_str} "
        f"through {args.base_model_name} + MoLT (one pass for all prompts)…",
        file=sys.stderr,
    )
    collectors = _collect_with_optional_config(args, molt, tokenizer, dtype)

    # 4. One bundle per prompt.
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    for name, text, tokens, prompt_gates, selected in prompt_payloads:
        prompt_traces: dict[tuple[int, int], PromptTrace] = {}
        for (layer, feat) in selected:
            prompt_traces[(layer, feat)] = PromptTrace(
                tokens=tokens,
                activations=prompt_gates[layer, :, feat].tolist(),
            )

        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = make_multilayer_bundle(
            collectors=collectors,
            meta=meta,
            tokenizer=tokenizer,
            out_path=out_dir,
            selected=selected,
            prompt=text,
            prompt_traces=prompt_traces,
            dataset_name=args.dataset_name,
            window=args.window,
            base_model_name=args.base_model_name,
        )
        size_mb = bundle_path.stat().st_size / 1e6
        print(
            f"[{name}] wrote {bundle_path} ({size_mb:.1f} MB, {len(selected)} entries)",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
