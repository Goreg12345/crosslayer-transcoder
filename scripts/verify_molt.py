"""Sanity-check a base LM vs. its MoLT-spliced counterpart on prompts.

Two questions this script answers:

  1. Does the MoLT-spliced model behave like the vanilla base LM? (full
     transcoder substitution faithfulness — KL / top-K agreement on the
     next-token distribution)
  2. Can the model do the task at all? (greedy completion + top-K next-token
     for either or both versions)

The splice + chat-template primitives live in
``crosslayer_transcoder.utils.molt_splice`` (shared with the training-time
``MoltEvalPromptCallback``); see that module's docstring for what "spliced"
means for each architecture.

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
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from crosslayer_transcoder.feature_dash.multilayer import (
    download_multilayer_from_hf,
    find_latest_step,
    load_multilayer_molt,
)
from crosslayer_transcoder.utils.molt_splice import (
    _DTYPE,
    apply_chat_template,
    compare_prompt,
    resolve_blocks,
)


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
    arch, blocks = resolve_blocks(model)
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
            apply_chat_template(tokenizer, raw_prompt, args.system_prompt)
            if args.chat_template
            else raw_prompt
        )
        if args.chat_template:
            print(f"\nchat-templated: {prompt_text!r}", file=sys.stderr)
        result = compare_prompt(
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
        if result.get("molt") is None and args.mode in ("molt", "both") and molt is None:
            print("  (skipping MoLT side — checkpoints not loaded)", file=sys.stderr)
        print_result(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
