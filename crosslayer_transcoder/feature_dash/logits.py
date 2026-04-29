"""Per-feature logit-lens projection for MoLT transforms.

Each MoLT feature is a low-rank linear map
``M[f] = U[f].T @ V[f].T : R^d_acts -> R^d_acts`` gated by a scalar. There's no
single decoder direction (unlike an SAE), so we approximate one with the top-1
left singular vector of `M[f]` scaled by its singular value. We then project
that direction back to vocab space the same way the LM head does:

    direction_resid = u1 * output_std[layer] * ln_f.weight
    logits[f]       = (W_U @ direction_resid) * s1

This is the standard "logit lens" approximation — it ignores the layers of
attention/MLP that follow layer L, so absolute magnitudes are not literal
logit deltas. It still surfaces *which* tokens the feature pushes toward, and
it's free (no extra forward passes).

For rank-1 transforms it's exact (since `M` is exactly rank-1). For higher
ranks, top-1 SVD captures the dominant output direction; we keep just that
one to match the SAE_vis logit-table format. Future iterations could surface
top-k singular triples per feature.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from crosslayer_transcoder.model.molt import Molt

logger = logging.getLogger(__name__)


@torch.no_grad()
def compute_feature_logits(
    molt: Molt,
    layer: int,
    W_U: torch.Tensor,
    ln_f_weight: torch.Tensor,
    tokenizer,
    top_k: int = 10,
    n_bins: int = 50,
) -> list[dict]:
    """Per-feature logit-lens projection of the dominant output direction.

    Parameters
    ----------
    molt
        Loaded MoLT model. Reads `Us`, `Vs`, and `output_standardizer.std`.
    layer
        Layer index used to pick `output_standardizer.std[layer]`. Must match
        the layer at which gates were collected.
    W_U
        LM unembedding, shape `(vocab_size, d_acts)` — typically
        `model.lm_head.weight`.
    ln_f_weight
        Final layernorm gain, shape `(d_acts,)` — typically
        `model.transformer.ln_f.weight`. The ln_f bias and the per-token
        mean-subtraction Jacobian are dropped (small effect on token ranking).
    tokenizer
        HF-style tokenizer. Only `decode([id])` is called.
    top_k
        Number of boosted/suppressed tokens to surface per feature.
    n_bins
        Histogram bins over the per-feature logit vector.

    Returns
    -------
    list of dicts, length `molt.n_features`. Each dict:
        {
          "top_pos": [{"token": str, "value": float}, ...],   # length top_k
          "top_neg": [{"token": str, "value": float}, ...],   # length top_k
          "histogram": {"bin_edges": [float], "counts": [int]},
        }
    Dead/degenerate features (singular value ≈ 0) yield empty arrays.
    """
    out_std = molt.output_standardizer.std[layer].detach().cpu().float()
    W_U = W_U.detach().cpu().float()
    ln_f_weight = ln_f_weight.detach().cpu().float()

    if W_U.shape[1] != out_std.shape[0]:
        raise ValueError(
            f"d_acts mismatch: W_U is (vocab={W_U.shape[0]}, d_acts={W_U.shape[1]}) "
            f"but output_standardizer std at layer {layer} is "
            f"({out_std.shape[0]},)"
        )
    if ln_f_weight.shape != out_std.shape:
        raise ValueError(
            f"ln_f_weight shape {tuple(ln_f_weight.shape)} != "
            f"out_std shape {tuple(out_std.shape)}"
        )

    # Diagonal scaling commutes with W_U: pre-bake it.
    # (vocab_size, d_acts) * (d_acts,) -> (vocab_size, d_acts)
    W_U_eff = W_U * (out_std * ln_f_weight)

    results: list[dict] = []
    for tier_idx, (U, V) in enumerate(zip(molt.Us, molt.Vs)):
        # U: (n_in_tier, rank, d_acts), V: (n_in_tier, d_acts, rank).
        n_in_tier, rank, d_acts = U.shape
        results.extend(
            _compute_tier_logits(
                U=U.detach().cpu().float(),
                V=V.detach().cpu().float(),
                W_U_eff=W_U_eff,
                tokenizer=tokenizer,
                top_k=top_k,
                n_bins=n_bins,
            )
        )
        logger.info(
            "tier %d: computed logits for %d features (rank %d)",
            tier_idx,
            n_in_tier,
            rank,
        )

    if len(results) != molt.n_features:
        raise RuntimeError(
            f"computed {len(results)} feature logit entries, expected "
            f"{molt.n_features}"
        )
    return results


def _compute_tier_logits(
    U: torch.Tensor,
    V: torch.Tensor,
    W_U_eff: torch.Tensor,
    tokenizer,
    top_k: int,
    n_bins: int,
) -> list[dict]:
    """Top-1 SVD + logit projection for one tier, batched across its features.

    The QR trick avoids materialising any (d_acts, d_acts) per-feature matrix:
    M = (U.T) @ (V.T) is rank-r, so its top singular triple comes from QR of
    each factor and SVD of the resulting (r, r) middle matrix.
    """
    n, rank, d_acts = U.shape
    # Per-feature factors: A = U.T -> (n, d_acts, r), B = V.T -> (n, r, d_acts).
    A = U.transpose(-1, -2).contiguous()
    B = V.transpose(-1, -2).contiguous()

    Qa, Ra = torch.linalg.qr(A, mode="reduced")              # Qa: (n, d, r)
    Qb, Rb = torch.linalg.qr(B.transpose(-1, -2), mode="reduced")  # Qb: (n, d, r)
    # M = Qa @ Ra @ Rb.T @ Qb.T. Top-1 SVD of M comes from SVD of (n, r, r).
    C = Ra @ Rb.transpose(-1, -2)                            # (n, r, r)
    U_C, S_C, _ = torch.linalg.svd(C, full_matrices=False)    # U_C: (n, r, r), S_C: (n, r)

    u1 = (Qa @ U_C[:, :, 0:1]).squeeze(-1)                    # (n, d_acts)
    s1 = S_C[:, 0]                                           # (n,)

    # SVD signs are ambiguous (M = (-u1)*(-v1.T)*s1 is the same factorisation),
    # which would cause the same MoLT to flip "top boost" / "top suppress"
    # tables between runs. Orient each u1 so its largest-magnitude component
    # is positive — arbitrary but deterministic. The MoLT contribution to
    # the residual on a given input still depends on `v1.T @ acts`, so the
    # logit-lens table is "tokens boosted along u1," not "tokens this feature
    # boosts on every input."
    abs_u1 = u1.abs()
    pivot = abs_u1.argmax(dim=-1, keepdim=True)              # (n, 1)
    pivot_sign = u1.gather(-1, pivot).sign()                 # (n, 1)
    pivot_sign = torch.where(pivot_sign == 0, torch.ones_like(pivot_sign), pivot_sign)
    u1 = u1 * pivot_sign

    # Project each feature's direction through W_U_eff. Shape (n, vocab).
    logits = (u1 @ W_U_eff.T) * s1.unsqueeze(-1)

    out: list[dict] = []
    eps = 1e-9
    for i in range(n):
        row = logits[i]
        if not torch.isfinite(row).all() or float(s1[i].abs()) < eps:
            out.append({
                "top_pos": [],
                "top_neg": [],
                "histogram": {"bin_edges": [], "counts": []},
            })
            continue

        # Effective top-k may be smaller than `top_k` if vocab somehow is —
        # not a real concern for GPT-2's 50k vocab, but cheap to be safe.
        k = min(top_k, row.numel())
        pos_vals, pos_idx = row.topk(k)
        neg_vals, neg_idx = row.topk(k, largest=False)

        top_pos = [
            {"token": _decode(tokenizer, int(idx)), "value": float(v)}
            for idx, v in zip(pos_idx.tolist(), pos_vals.tolist())
        ]
        top_neg = [
            {"token": _decode(tokenizer, int(idx)), "value": float(v)}
            for idx, v in zip(neg_idx.tolist(), neg_vals.tolist())
        ]

        lo = float(row.min().item())
        hi = float(row.max().item())
        if hi - lo < eps:
            bin_edges: list[float] = []
            counts: list[int] = []
        else:
            counts_t = torch.histc(row, bins=n_bins, min=lo, max=hi).long()
            bin_edges = torch.linspace(lo, hi, n_bins + 1).tolist()
            counts = counts_t.tolist()

        out.append({
            "top_pos": top_pos,
            "top_neg": top_neg,
            "histogram": {"bin_edges": bin_edges, "counts": counts},
        })
    return out


def _decode(tokenizer, token_id: int) -> str:
    return tokenizer.decode([int(token_id)])


@torch.no_grad()
def compute_feature_logits_for_hf_lm(
    molt: Molt,
    layer: int,
    base_model_name: str = "openai-community/gpt2",
    tokenizer: Optional[object] = None,
    top_k: int = 10,
    n_bins: int = 50,
) -> list[dict]:
    """Convenience wrapper: load a fresh HF causal LM, pull `lm_head` and
    `transformer.ln_f.weight`, hand off to `compute_feature_logits`.

    Loads the LM only to read the unembedding + final-LN gain, then drops it.
    Costs ~500 MB / a few seconds of wall time for GPT-2; acceptable as a
    one-off offline step.
    """
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    model = GPT2LMHeadModel.from_pretrained(base_model_name).eval()
    if tokenizer is None:
        tokenizer = GPT2TokenizerFast.from_pretrained(base_model_name)
    try:
        W_U = model.lm_head.weight.detach()
        ln_f_weight = model.transformer.ln_f.weight.detach()
    except AttributeError as e:
        raise ValueError(
            f"{base_model_name} doesn't expose `lm_head` + `transformer.ln_f` — "
            "`compute_feature_logits_for_hf_lm` only handles GPT-2-style models."
        ) from e
    return compute_feature_logits(
        molt=molt,
        layer=layer,
        W_U=W_U,
        ln_f_weight=ln_f_weight,
        tokenizer=tokenizer,
        top_k=top_k,
        n_bins=n_bins,
    )
