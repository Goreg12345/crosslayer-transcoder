"""Greater-than heatmap for the GPT-2 MOLT *replacement model*.

Loads the per-layer MOLT transcoders from
  kylelovesllms/molt-sweeps :: molt-multilayer-gpt2-N50-100M-fp32weights-full
splices them into GPT-2 (every MLP output is replaced by its MOLT
reconstruction, in a single forward pass so downstream layers see the
perturbed residual stream), and re-runs the greater-than experiment.

We then compare the MOLT replacement heatmap against vanilla GPT-2 to see
whether the "greater-than" structure survives the transcoder substitution.

Run with:  uv run --with matplotlib python scripts/greater_than_molt.py
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import nnsight
import numpy as np
import torch
from huggingface_hub import hf_hub_download

from crosslayer_transcoder.model.jumprelu import JumpReLU
from crosslayer_transcoder.model.molt import Molt
from crosslayer_transcoder.model.standardize import (
    DimensionwiseInputStandardizer,
    DimensionwiseOutputStandardizer,
)

REPO = "kylelovesllms/molt-sweeps"
SUBDIR = "molt-multilayer-gpt2-N50-100M-fp32weights-full"
N_LAYERS = 12
D_ACTS = 768
N = 50
D_FEATURES = 1550  # 50 + 100 + 200 + 400 + 800


def build_molt():
    """A single per-layer MOLT matching the checkpoint architecture."""
    nl = JumpReLU(theta=0.03, bandwidth=1.0, n_layers=1, d_features=D_FEATURES)
    ins = DimensionwiseInputStandardizer(N_LAYERS, D_ACTS)
    outs = DimensionwiseOutputStandardizer(N_LAYERS, D_ACTS)
    return Molt(
        d_acts=D_ACTS, N=N, nonlinearity=nl,
        input_standardizer=ins, output_standardizer=outs,
    )


def load_molts(step: int, device, dtype=torch.float32):
    molts = []
    for layer in range(N_LAYERS):
        fname = f"{SUBDIR}/molt-multilayer-N50-100M-ddp_layer_{layer}_step{step}.pt"
        path = hf_hub_download(REPO, fname)
        sd = torch.load(path, map_location="cpu", weights_only=True)
        m = build_molt()
        m.load_state_dict(sd)
        # Standardizer buffers loaded from the checkpoint, but `is_initialized`
        # is a plain attribute (not in the state_dict) -> set it manually.
        m.input_standardizer.is_initialized = True
        m.output_standardizer.is_initialized = True
        m = m.to(device=device, dtype=dtype).eval().requires_grad_(False)
        molts.append(m)
        print(f"  loaded layer {layer}")
    return molts


def build_candidate_token_ids(tok):
    """Bare two-digit output-year token ids (all single tokens in GPT-2)."""
    ids = torch.zeros(100, dtype=torch.long)
    for zz in range(100):
        enc = tok(f"{zz:02d}")["input_ids"]
        assert len(enc) == 1, f"{zz:02d} not single token"
        ids[zz] = enc[0]
    return ids


@torch.no_grad()
def get_year_probs(gpt2, molts, prompts, cand_token_ids, batch_size, mode):
    """Return [n_prompts, 100] year-restricted probabilities at the final token.

    mode='clean'        -> vanilla GPT-2
    mode='sequential'   -> replacement model: each layer's MLP output replaced
                           in one trace, so layer L reads the *perturbed* stream
                           (errors compound across layers).
    mode='teacherforced'-> replace every MLP output too, but compute each layer's
                           reconstruction from the *clean* (unperturbed) input, so
                           reconstruction errors do not compound.
    """
    tok = gpt2.tokenizer
    cand = cand_token_ids.to(gpt2.device)
    out, argmaxes, year_mass = [], [], []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        enc = tok(chunk, return_tensors="pt").to(gpt2.device)
        toks = enc["input_ids"]
        B, S = toks.shape

        if mode == "teacherforced":
            # pass 1: capture clean MLP inputs (ascending layer order = forward order)
            clean_ins = []
            with gpt2.trace(toks):
                for L in range(N_LAYERS):
                    clean_ins.append(gpt2.transformer.h[L].ln_2.input.save())
            recons = []
            for L, molt in enumerate(molts):
                _, _, r = molt(clean_ins[L].reshape(B * S, -1), layer=L)
                recons.append(r.reshape(B, S, -1))

        with gpt2.trace(toks):
            if mode == "sequential":
                for L, molt in enumerate(molts):
                    mlp_in = gpt2.transformer.h[L].ln_2.input  # (B, S, D)
                    flat = mlp_in.reshape(B * S, -1)
                    _, _, r = molt(flat, layer=L)
                    gpt2.transformer.h[L].mlp.output = r.reshape(B, S, -1)
            elif mode == "teacherforced":
                for L in range(N_LAYERS):
                    gpt2.transformer.h[L].mlp.output = recons[L]
            logits = gpt2.lm_head.output.save()
        final = logits[:, -1, :]  # (B, vocab)
        year_logits = final[:, cand]  # (B, 100)
        out.append(year_logits.softmax(dim=-1).float().cpu())
        argmaxes.append(final.argmax(dim=-1).cpu())
        full = final.softmax(dim=-1)  # full-vocab probs
        year_mass.append(full[:, cand].sum(dim=-1).float().cpu())
    return (
        torch.cat(out, dim=0).numpy(),
        torch.cat(argmaxes, dim=0).numpy(),
        torch.cat(year_mass, dim=0).numpy(),
    )


def prob_diff(probs):
    """P(ZZ>YY) - P(ZZ<=YY) per start year, year-restricted."""
    upper = np.triu(np.ones((100, 100)), k=1)
    p_gt = (probs * upper).sum(1)
    return 2 * p_gt - probs.sum(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--step", type=int, default=36000, help="checkpoint step")
    p.add_argument("--century", default="19")
    p.add_argument("--row", type=int, default=41, help="YY for the line plot")
    p.add_argument("--lo", type=int, default=2)
    p.add_argument("--hi", type=int, default=98)
    p.add_argument("--batch", type=int, default=5, help="prompts per forward pass")
    p.add_argument("--out", default="greater_than_molt.png")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading GPT-2 on {device} ...")
    gpt2 = nnsight.LanguageModel("openai-community/gpt2", device_map=device, dispatch=True)
    gpt2.requires_grad_(False)

    print(f"Loading {N_LAYERS} MOLT layers (step {args.step}) ...")
    molts = load_molts(args.step, device)

    tok = gpt2.tokenizer
    cand_token_ids = build_candidate_token_ids(tok)
    prompts = [
        f"The war started in {args.century}{yy:02d} and ended in {args.century}"
        for yy in range(100)
    ]

    print("Running vanilla GPT-2 ...")
    clean, am_clean, mass_clean = get_year_probs(gpt2, molts, prompts, cand_token_ids, args.batch, mode="clean")
    print("Running MOLT replacement (sequential) ...")
    molt, am_seq, mass_seq = get_year_probs(gpt2, molts, prompts, cand_token_ids, args.batch, mode="sequential")
    print("Running MOLT replacement (teacher-forced) ...")
    tf, am_tf, mass_tf = get_year_probs(gpt2, molts, prompts, cand_token_ids, args.batch, mode="teacherforced")

    # ---- quantitative comparison ----
    valid = slice(args.lo, args.hi + 1)
    print("\n=== greater-than prob-diff  P(>YY)-P(<=YY), mean over start years ===")
    print(f"  vanilla GPT-2            : {prob_diff(clean).mean():+.3f}")
    print(f"  MOLT sequential          : {prob_diff(molt).mean():+.3f}")
    print(f"  MOLT teacher-forced      : {prob_diff(tf).mean():+.3f}")
    print("=== per-cell heatmap correlation vs vanilla ===")
    print(f"  sequential   : {np.corrcoef(clean[valid].ravel(), molt[valid].ravel())[0,1]:.3f}")
    print(f"  teacher-forced: {np.corrcoef(clean[valid].ravel(), tf[valid].ravel())[0,1]:.3f}")
    print("=== replacement faithfulness (final-token prediction) ===")
    print(f"  top-1 next-token agreement  seq={ (am_clean==am_seq).mean():.3f}  tf={(am_clean==am_tf).mean():.3f}")
    print(f"  mass on year tokens  clean={mass_clean.mean():.3f}  seq={mass_seq.mean():.3f}  tf={mass_tf.mean():.3f}")

    # ---- plot: three heatmaps (clean / sequential / teacher-forced) + line ----
    cmap = plt.get_cmap("Blues")
    fig, axd = plt.subplot_mosaic(
        [["clean", "seq", "tf"], ["line", "line", "line"]], figsize=(20, 11)
    )
    panels = [
        ("clean", clean, "vanilla GPT-2"),
        ("seq", molt, "MOLT replacement (sequential)"),
        ("tf", tf, "MOLT replacement (teacher-forced)"),
    ]
    for key, data, title in panels:
        ax = axd[key]
        im = ax.imshow(
            data[args.lo : args.hi + 1], origin="upper", aspect="auto",
            cmap=cmap, vmin=0, vmax=0.25, extent=[0, 100, args.hi, args.lo],
        )
        ax.set_xlabel("predicted year")
        ax.set_ylabel("YY")
        ax.set_title(f"{title}\nProbability Heatmap")
        fig.colorbar(im, ax=ax, label="probability")

    axl = axd["line"]
    axl.plot(np.arange(100), clean[args.row], lw=1.4, label="vanilla GPT-2")
    axl.plot(np.arange(100), molt[args.row], lw=1.4, label="MOLT sequential")
    axl.plot(np.arange(100), tf[args.row], lw=1.4, label="MOLT teacher-forced")
    axl.set_xlabel("Predicted Year")
    axl.set_ylabel("probability")
    axl.set_title(f"Probabilities when YY={args.row}")
    axl.legend()

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nSaved figure -> {args.out}")


if __name__ == "__main__":
    main()
