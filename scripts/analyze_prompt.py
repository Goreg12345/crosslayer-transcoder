"""
Analyze which MoLT transforms activate on a given prompt.

Loads all 12 per-layer MoLT checkpoints from a multi-layer training run, runs
GPT-2 small on the prompt, captures the MLP-input activations at every layer,
and reports the top-K activating transforms per layer per token.

Example:
    uv run python scripts/analyze_prompt.py \\
        --prompt "Translate to Spanish: cat ->" \\
        --step 50000 --topk 8

Transform indexing
------------------
N_features per layer = N * sum(rank_multipliers) = N * (1+2+4+8+16) = 31*N.
For N=10 that is 310. The features are concatenated by rank group:

    indices [0,           1*N)       -> rank 512  (group 0, x1)
    indices [1*N,         3*N)       -> rank 256  (group 1, x2)
    indices [3*N,         7*N)       -> rank 128  (group 2, x4)
    indices [7*N,        15*N)       -> rank 64   (group 3, x8)
    indices [15*N,       31*N)       -> rank 32   (group 4, x16)
"""

import argparse
import re
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from crosslayer_transcoder.model.jumprelu import JumpReLU
from crosslayer_transcoder.model.molt import MultiLayerMolt
from crosslayer_transcoder.model.standardize import (
    DimensionwiseInputStandardizer,
    DimensionwiseOutputStandardizer,
)


N_LAYERS = 12
D_ACTS = 768
RANKS = [512, 256, 128, 64, 32]
RANK_MULTIPLIERS = [1, 2, 4, 8, 16]


def transform_index_to_group(idx: int, n: int) -> tuple[int, int]:
    cumulative = 0
    for g, mult in enumerate(RANK_MULTIPLIERS):
        size = n * mult
        if idx < cumulative + size:
            return g, idx - cumulative
        cumulative += size
    raise ValueError(f"index {idx} out of range for N={n}")


def build_molt(n: int, device: str) -> MultiLayerMolt:
    n_features = n * sum(RANK_MULTIPLIERS)
    nonlinearity = JumpReLU(theta=0.03, bandwidth=1.0, n_layers=1, d_features=n_features)
    in_std = DimensionwiseInputStandardizer(n_layers=N_LAYERS, activation_dim=D_ACTS)
    out_std = DimensionwiseOutputStandardizer(n_layers=N_LAYERS, activation_dim=D_ACTS)
    model = MultiLayerMolt(
        n_layers=N_LAYERS, d_acts=D_ACTS, N=n, ranks=RANKS,
        nonlinearity=nonlinearity, input_standardizer=in_std, output_standardizer=out_std,
    )
    return model.to(device).eval()


def find_latest_step(ckpt_dir: Path, run_name: str) -> int:
    pattern = re.compile(rf"^{re.escape(run_name)}_layer_0_step(\d+)\.pt$")
    steps = []
    for p in ckpt_dir.iterdir():
        m = pattern.match(p.name)
        if m:
            steps.append(int(m.group(1)))
    if not steps:
        raise FileNotFoundError(f"no checkpoints in {ckpt_dir}")
    return max(steps)


def load_per_layer_checkpoints(model: MultiLayerMolt, ckpt_dir: Path, run_name: str, step: int, device: str):
    for layer in range(N_LAYERS):
        path = ckpt_dir / f"{run_name}_layer_{layer}_step{step}.pt"
        sd = torch.load(path, map_location=device)
        sd = {k: v.float() if v.is_floating_point() else v for k, v in sd.items()}
        model.molts[layer].load_state_dict(sd)
    # Standardizer mean/std buffers are restored from the checkpoint state dict, but
    # `is_initialized` is a plain Python attribute that load_state_dict doesn't touch.
    model.input_standardizer.is_initialized = True
    model.output_standardizer.is_initialized = True


@torch.no_grad()
def capture_mlp_inputs(gpt2: GPT2LMHeadModel, prompt: str, tokenizer: GPT2TokenizerFast, device: str):
    captured: list[torch.Tensor | None] = [None] * N_LAYERS

    def make_hook(layer_idx: int):
        def hook(_module, inputs):
            captured[layer_idx] = inputs[0].detach()
        return hook

    handles = [gpt2.transformer.h[i].ln_2.register_forward_pre_hook(make_hook(i)) for i in range(N_LAYERS)]
    try:
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        gpt2(**enc)
    finally:
        for h in handles:
            h.remove()

    stacked = torch.stack([c[0] for c in captured], dim=1)  # (seq, n_layers, d_acts)
    return enc.input_ids[0], stacked.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--step", type=int, default=None, help="checkpoint step (default: latest)")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--n", type=int, default=10, help="N from training config")
    ap.add_argument("--ckpt-dir", default="checkpoints/molt-multilayer-N10-100M")
    ap.add_argument("--run-name", default="molt-multilayer-N10-100M")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    step = args.step if args.step is not None else find_latest_step(ckpt_dir, args.run_name)
    print(f"Loading MoLT checkpoints @ step {step} from {ckpt_dir}")

    tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
    gpt2 = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(args.device).eval()
    molt = build_molt(args.n, args.device)
    load_per_layer_checkpoints(molt, ckpt_dir, args.run_name, step, args.device)

    token_ids, mlp_in = capture_mlp_inputs(gpt2, args.prompt, tokenizer, args.device)

    with torch.no_grad():
        gates, _, _ = molt(mlp_in)  # (seq, n_layers, n_features)

    tokens = [tokenizer.decode([t]) for t in token_ids]
    print(f"\nPrompt: {args.prompt!r}")
    print(f"Tokens ({len(tokens)}): {tokens}\n")

    for layer in range(N_LAYERS):
        print(f"=== Layer {layer} ===")
        layer_gates = gates[:, layer, :]
        for t_idx, tok in enumerate(tokens):
            row = layer_gates[t_idx]
            top_vals, top_idxs = torch.topk(row, args.topk)
            entries = []
            for val, idx in zip(top_vals.tolist(), top_idxs.tolist()):
                if val == 0.0:
                    continue
                g, j = transform_index_to_group(idx, args.n)
                entries.append(f"r{RANKS[g]}#{j}({val:.2f})")
            print(f"  pos={t_idx:>2} tok={tok!r:<14} | " + (" ".join(entries) if entries else "(all zero)"))
        print()


if __name__ == "__main__":
    main()
