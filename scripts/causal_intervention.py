"""Causal intervention experiment for MoLT features on a year-prediction prompt.

For prompt "The war lasted from the year 1732 to the year 17", we ablate
candidate (layer, feature) pairs and measure the shift in the next-token
distribution — specifically, mass on years strictly later vs. earlier than
1732.

Intervention surface
--------------------
MoLT is a transcoder targeting GPT-2's MLP output (see clt_lightning.py). To
intervene, we splice MoLT's reconstruction in for GPT-2's MLP output at the
target layer(s). To ablate feature f, we zero its gate before recomputing the
reconstruction. This is standard transcoder-based activation patching.

Conditions reported per prompt:
  - clean:                vanilla GPT-2.
  - splice_all:           MoLT replaces every MLP output (full-substitution
                          faithfulness check).
  - splice_only_L<L>:     MoLT replaces MLP output at layer L only (the
                          per-feature faithfulness baseline).
  - ablate_<feature>:     splice at the feature's home layer and zero its gate.
  - ablate_year_triple:   ablate L9F266 + L10F247 + L6F113 simultaneously.

Outputs
-------
Prints a table to stdout and writes a JSON record to the output path.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import einops
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from crosslayer_transcoder.feature_dash.multilayer import (
    find_latest_step,
    load_multilayer_molt,
)


PROMPT = "The war lasted from the year 1732 to the year 17"
CKPT_DIR = Path("checkpoints/molt-multilayer-N10-100M")
RUN_NAME = "molt-multilayer-N10-100M"

CANDIDATES: list[tuple[str, int, int, str]] = [
    ("L9F266",  9, 266, "year-numeral detector"),
    ("L10F247", 10, 247, "year detector"),
    ("L6F113",  6, 113, "century/year"),
    ("L2F249",  2, 249, "pure ' to'"),
    ("L3F143",  3, 143, "pure ' from'"),
    ("L6F291",  6, 291, "range-connector"),
    ("L1F6",    1, 6,   "always-on (control)"),
]
TRIPLE = {"L9F266", "L10F247", "L6F113"}


@dataclass(frozen=True)
class Cond:
    name: str
    desc: str
    splice_layers: tuple[int, ...]
    ablate: tuple[tuple[int, int], ...]

    def per_layer_ablate(self) -> dict[int, list[int]]:
        d: dict[int, list[int]] = {}
        for layer, fid in self.ablate:
            d.setdefault(layer, []).append(fid)
        return d


def molt_mlp_out(
    inner_molt,
    resid: torch.Tensor,
    layer: int,
    ablate_features: Optional[list[int]] = None,
) -> torch.Tensor:
    """Compute MoLT's MLP-output reconstruction with optional feature ablation.

    Reproduces `Molt.forward` but allows zeroing specific gate entries before
    the rank-blocks are summed.
    """
    acts = inner_molt.input_standardizer(resid, layer)
    pre = inner_molt.e(acts)
    gate = inner_molt.nonlinearity(pre)
    if ablate_features:
        gate = gate.clone()
        for f in ablate_features:
            gate[..., f] = 0.0
    raw_recons = []
    for U, V in zip(inner_molt.Us, inner_molt.Vs):
        latents = einops.einsum(acts, V, "... d, n d r -> ... n r")
        raw_recons.append(einops.einsum(latents, U, "... n r, n r d -> ... n d"))
    raw_recons = torch.cat(raw_recons, dim=-2)
    weighted = gate.unsqueeze(-1) * raw_recons
    recons_norm = weighted.sum(dim=-2)
    return inner_molt.output_standardizer(recons_norm, layer)


@torch.no_grad()
def run_intervention(
    gpt2: GPT2LMHeadModel,
    molt,
    enc,
    splice_layers: set[int],
    ablate_per_layer: dict[int, list[int]],
) -> torch.Tensor:
    """Forward GPT-2 with MoLT-spliced MLP outputs at `splice_layers`."""
    handles = []
    captured: dict[int, torch.Tensor] = {}

    for L in splice_layers:
        block = gpt2.transformer.h[L]

        def pre_hook(_m, args, L=L):
            captured[L] = args[0]
        handles.append(block.ln_2.register_forward_pre_hook(pre_hook))

        def post_hook(_m, _args, _out, L=L):
            return molt_mlp_out(
                molt.molts[L], captured[L], L, ablate_per_layer.get(L)
            )
        handles.append(block.mlp.register_forward_hook(post_hook))

    try:
        logits = gpt2(**enc).logits[0, -1]
    finally:
        for h in handles:
            h.remove()
    return logits


def bucket_mass(probs: torch.Tensor, tok: GPT2TokenizerFast) -> dict[str, float]:
    def msum(strs: list[str]) -> float:
        s = 0.0
        for v in strs:
            ids = tok(v, add_special_tokens=False).input_ids
            if len(ids) == 1:
                s += probs[ids[0]].item()
        return s

    return {
        "32": msum(["32"]),
        "33-39": msum([str(i) for i in range(33, 40)]),
        "40-99": msum([str(i) for i in range(40, 100)]),
        "00-31": msum([f"{i:02d}" for i in range(0, 32)]),
    }


def topk(probs: torch.Tensor, tok: GPT2TokenizerFast, k: int = 10) -> list[tuple[str, float]]:
    vals, idxs = torch.topk(probs, k)
    return [(tok.decode([int(i)]), float(p)) for p, i in zip(vals, idxs)]


def kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> float:
    return float((p * (p.add(eps).log() - q.add(eps).log())).sum().item())


def build_conditions() -> list[Cond]:
    conds: list[Cond] = [Cond("clean", "vanilla GPT-2", (), ())]
    conds.append(Cond("splice_all", "full transcoder substitution", tuple(range(12)), ()))
    target_layers = sorted({L for _, L, _, _ in CANDIDATES})
    for L in target_layers:
        conds.append(Cond(f"splice_only_L{L}", f"splice MoLT at layer {L} only", (L,), ()))
    for name, L, f, desc in CANDIDATES:
        conds.append(Cond(f"ablate_{name}", f"zero {name} ({desc})", (L,), ((L, f),)))
    triple_cands = [c for c in CANDIDATES if c[0] in TRIPLE]
    conds.append(Cond(
        "ablate_year_triple",
        "zero L9F266+L10F247+L6F113 simultaneously",
        tuple(sorted({L for _, L, _, _ in triple_cands})),
        tuple((L, f) for _, L, f, _ in triple_cands),
    ))
    return conds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="feature_dash/year_1732/causal_intervention.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--step", type=int, default=None)
    args = ap.parse_args()

    step = args.step or find_latest_step(CKPT_DIR, RUN_NAME)
    print(f"Loading MoLT step {step} from {CKPT_DIR}")
    molt, _meta = load_multilayer_molt(
        ckpt_dir=CKPT_DIR, run_name=RUN_NAME, step=step, device=args.device
    )
    tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
    gpt2 = (
        GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
        .to(args.device)
        .eval()
    )
    enc = tok(PROMPT, return_tensors="pt").to(args.device)

    conds = build_conditions()
    results: list[dict] = []
    clean_probs: Optional[torch.Tensor] = None

    for cond in conds:
        logits = run_intervention(
            gpt2, molt, enc, set(cond.splice_layers), cond.per_layer_ablate()
        )
        probs = logits.softmax(-1).cpu()
        if cond.name == "clean":
            clean_probs = probs
        b = bucket_mass(probs, tok)
        score = b["33-39"] + b["40-99"] - b["00-31"]
        kl_clean = (
            kl(probs, clean_probs)
            if clean_probs is not None and cond.name != "clean"
            else 0.0
        )
        results.append({
            "name": cond.name,
            "desc": cond.desc,
            "splice_layers": list(cond.splice_layers),
            "ablate": [list(p) for p in cond.ablate],
            "buckets": b,
            "score": score,
            "kl_vs_clean": kl_clean,
            "top10": topk(probs, tok, 10),
        })

    print(f"\nPrompt: {PROMPT!r}")
    print(f"Tokens: {tok.convert_ids_to_tokens(enc.input_ids[0].tolist())}\n")
    print(
        f"{'condition':<24s} {'P(=32)':>7s} {'P(33-39)':>9s} "
        f"{'P(40-99)':>9s} {'P(<32)':>7s} {'score':>8s} {'KL_clean':>8s}  top1"
    )
    print("-" * 100)
    for r in results:
        b = r["buckets"]
        top1_tok, top1_p = r["top10"][0]
        print(
            f"{r['name']:<24s} {b['32']*100:6.2f}%  {b['33-39']*100:7.2f}% "
            f" {b['40-99']*100:7.2f}%  {b['00-31']*100:6.2f}%  {r['score']*100:+7.2f}  "
            f"{r['kl_vs_clean']:7.3f}  {top1_tok!r} {top1_p*100:.2f}%"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "prompt": PROMPT,
        "step": step,
        "results": results,
    }, indent=2))
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
