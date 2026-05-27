"""Visualize GPT-2's "greater-than" behavior (Hanna et al. 2023).

Template:  "The war started in 19YY and ended in 19__"

We feed one prompt per start year YY (00-99), read the model's next-token
distribution restricted to the 100 two-digit year tokens, and plot a heatmap:

    y-axis = start year YY (00-99)
    x-axis = candidate end year (00-99)
    cell   = P(end year = candidate | start year = YY)

A correctly-behaving model puts its probability mass *above* the diagonal
(the end year must be greater than the start year).

Run with:  uv run python scripts/greater_than_heatmap.py
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


def build_candidate_token_ids(tok: GPT2TokenizerFast):
    """Return (cand_token_ids[100], valid_mask[100]) for the OUTPUT candidates.

    The prompt ends with the century token (" 19"), so the next token GPT-2
    would emit to spell a year is the *bare* two-digit string "ZZ" with no
    leading space (e.g. " 1945" == " 19" + "45"). For GPT-2 all of 00-99 are
    single tokens, but we keep a validity mask so the code is robust to other
    tokenizers.
    """
    cand_token_ids = torch.zeros(100, dtype=torch.long)
    valid_mask = torch.zeros(100, dtype=torch.bool)
    for zz in range(100):
        ids = tok(f"{zz:02d}", return_tensors="pt").input_ids[0]
        if len(ids) == 1:
            cand_token_ids[zz] = ids.item()
            valid_mask[zz] = True
    return cand_token_ids, valid_mask


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gpt2", help="HF model name")
    p.add_argument("--century", default="19", help="two-digit century prefix")
    p.add_argument(
        "--normalize",
        choices=["vocab", "years"],
        default="years",
        help="softmax over full vocab then slice ('vocab', true probabilities) "
        "or renormalize over just the 100 year tokens ('years', sharper).",
    )
    p.add_argument(
        "--row",
        type=int,
        default=41,
        help="start year YY whose full predicted-year distribution is shown "
        "in the right-hand line plot.",
    )
    p.add_argument(
        "--lo", type=int, default=2, help="lowest YY shown in the heatmap."
    )
    p.add_argument(
        "--hi", type=int, default=98, help="highest YY shown in the heatmap."
    )
    p.add_argument("--out", default="greater_than_heatmap.png")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device} ...")
    tok = GPT2TokenizerFast.from_pretrained(args.model)
    model = GPT2LMHeadModel.from_pretrained(args.model).to(device).eval()

    cand_token_ids, valid_mask = build_candidate_token_ids(tok)
    n_valid = int(valid_mask.sum())
    print(f"{n_valid}/100 candidate two-digit years are single tokens.")
    if n_valid < 100:
        bad = [f"{i:02d}" for i in range(100) if not valid_mask[i]]
        print(f"  (not single-token, will be blanked in the plot: {bad})")

    # One prompt per start year. GPT-2 tokenizes " 19YY" as a single token and
    # the trailing " 19" as one token, so all 100 prompts have identical length
    # -> no padding needed, and the final real token is simply position -1.
    prompts = [
        f"The war started in {args.century}{yy:02d} and ended in {args.century}"
        for yy in range(100)
    ]
    enc = tok(prompts, return_tensors="pt").to(device)
    lengths = enc.attention_mask.sum(1)
    assert (lengths == lengths[0]).all(), (
        "Prompts have differing token lengths; some years split into multiple "
        "tokens. Use --century with cleaner tokenization or add padded indexing."
    )

    with torch.no_grad():
        logits = model(**enc).logits  # [100, seq, vocab]
    final_logits = logits[:, -1, :]  # [100, vocab]

    cand_token_ids = cand_token_ids.to(device)
    if args.normalize == "vocab":
        probs = final_logits.softmax(dim=-1)
        heat = probs[:, cand_token_ids]  # [start, candidate]
    else:  # renormalize over just the 100 candidate year tokens
        year_logits = final_logits[:, cand_token_ids]
        heat = year_logits.softmax(dim=-1)

    heat = heat.float().cpu().numpy()

    # Columns are candidate years; blank any that aren't valid single tokens.
    valid = valid_mask.cpu().numpy()
    heat[:, ~valid] = np.nan

    # ---- plot: heatmap (left) + single-row distribution (right) ----
    lo, hi = args.lo, args.hi
    sub = heat[lo : hi + 1]  # restrict displayed start years to [lo, hi]

    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad(color="lightgray")  # NaN cells
    fig, (axh, axl) = plt.subplots(1, 2, figsize=(16, 7))

    # Left: heatmap with YY running top->bottom (origin upper), like the paper.
    im = axh.imshow(
        sub, origin="upper", aspect="auto", cmap=cmap, vmin=0, vmax=0.25,
        extent=[0, 100, hi, lo],  # x: predicted year, y: YY (top=lo, bottom=hi)
    )
    axh.set_xlabel("predicted year")
    axh.set_ylabel("YY")
    axh.set_title(f"{args.model} Probability Heatmap")
    fig.colorbar(im, ax=axh, label="probability")

    # Right: full predicted-year distribution for one chosen start year.
    row = args.row
    dist = heat[row]
    axl.plot(np.arange(100), dist, lw=1.2)
    axl.set_xlabel("Predicted Year")
    axl.set_ylabel("probability")
    axl.set_title(f"{args.model} Probabilities when YY={row}")

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    argmax = int(np.nanargmax(dist))
    print(f"Saved figure -> {args.out}")
    print(f"  YY={row}: peak at predicted year {argmax} (p={dist[argmax]:.3f})")


if __name__ == "__main__":
    main()
