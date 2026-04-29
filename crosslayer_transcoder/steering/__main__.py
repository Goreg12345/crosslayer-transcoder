"""CLI: causal validation by steering a base model with a MoLT transform.

Examples
--------
Local checkpoint, single alpha:
    python -m crosslayer_transcoder.steering \\
        --ckpt checkpoints/lam_0_00015_50M/clt.ckpt \\
        --transform-index 137 --alpha 8 \\
        --prompt "The cat sat on the"

HF checkpoint, alpha sweep:
    python -m crosslayer_transcoder.steering \\
        --ckpt gpt2-molt-lam-0_00015-50M.ckpt \\
        --hf-repo kylelovesllms/molt-sweeps \\
        --transform-index 137 --alpha 0,2,4,8,16 \\
        --prompt "Once upon a time"
"""

from __future__ import annotations

import argparse
import sys

import torch

from crosslayer_transcoder.feature_dash.load import load_molt, load_molt_from_hf
from crosslayer_transcoder.steering.generate import (
    generate_with_steering,
    load_base_model,
)
from crosslayer_transcoder.steering.intervention import TransformIntervention


def _parse_alphas(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m crosslayer_transcoder.steering",
        description="Steer a base model using a MoLT transform's contribution.",
    )
    p.add_argument(
        "--ckpt",
        required=True,
        help="Path to a Lightning .ckpt file, or (with --hf-repo) the filename within an HF repo.",
    )
    p.add_argument(
        "--hf-repo",
        default=None,
        help="If set, treat --ckpt as a filename in this HF repo and download it.",
    )
    p.add_argument("--transform-index", type=int, required=True)
    p.add_argument(
        "--alpha",
        default="0,2,8",
        help="Comma-separated steering strengths. Each is run as a separate generation.",
    )
    p.add_argument("--prompt", required=True)
    p.add_argument(
        "--layer",
        type=int,
        default=8,
        help="Transformer block to inject at. MoLT in this repo trains on layer 8 by default.",
    )
    p.add_argument(
        "--base-model",
        default=None,
        help="HF model id for the base LM. Defaults to the checkpoint's recorded base_model_name.",
    )
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    if args.hf_repo:
        molt, meta = load_molt_from_hf(args.ckpt, repo_id=args.hf_repo, device=args.device)
    else:
        molt, meta = load_molt(args.ckpt, device=args.device)

    base_name = args.base_model or meta.base_model_name
    if base_name is None:
        print(
            "error: checkpoint has no base_model_name in metadata — pass --base-model.",
            file=sys.stderr,
        )
        return 2

    print(f"loading base model: {base_name} on {args.device}")
    model, tok = load_base_model(base_name, args.device)

    alphas = _parse_alphas(args.alpha)
    print(
        f"\nsteering: transform={args.transform_index} (tier varies), layer={args.layer}, "
        f"alphas={alphas}\nprompt: {args.prompt!r}\n"
    )

    for alpha in alphas:
        if alpha == 0.0:
            intervention = None
            tag = "alpha=0 (baseline)"
        else:
            intervention = TransformIntervention(
                molt=molt,
                transform_index=args.transform_index,
                layer=args.layer,
                alpha=alpha,
            ).to(args.device)
            tag = f"alpha={alpha} (tier={intervention.tier}, rank={intervention.rank})"

        text = generate_with_steering(
            model,
            tok,
            args.prompt,
            intervention,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
        )
        print(f"=== {tag} ===")
        print(text)
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
