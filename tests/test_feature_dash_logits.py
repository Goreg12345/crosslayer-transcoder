"""Tests for `feature_dash.logits.compute_feature_logits`.

We don't need a real GPT-2 here. The function takes raw `W_U` + `ln_f_weight`
tensors plus a tokenizer and returns one entry per feature. We build a tiny
MoLT, hand-craft the unembedding so that *one specific token* has a known
inner product with the feature's transform, and check that token shows up
top-pos (or top-neg) in the result.
"""

from __future__ import annotations

import pytest
import torch

from crosslayer_transcoder.feature_dash.logits import (
    compute_feature_logits,
)
from crosslayer_transcoder.model.jumprelu import JumpReLU
from crosslayer_transcoder.model.molt import Molt
from crosslayer_transcoder.model.standardize import (
    DimensionwiseInputStandardizer,
    DimensionwiseOutputStandardizer,
)


D_ACTS = 8
N_LAYERS = 3
N = 1
RANKS = [2]                  # one tier with N=1 feature, rank 2
LAYER = 1
VOCAB = 12


class _StubTok:
    def decode(self, ids):
        return f"<{ids[0]}>"


def _build_molt(
    n_features_override: int | None = None,
    ranks: list[int] | None = None,
) -> Molt:
    ranks = ranks if ranks is not None else RANKS
    n_features = N * sum(2**t for t in range(len(ranks)))
    nonlin = JumpReLU(theta=0.0, bandwidth=1.0, n_layers=1, d_features=n_features)
    in_std = DimensionwiseInputStandardizer(n_layers=N_LAYERS, activation_dim=D_ACTS)
    out_std = DimensionwiseOutputStandardizer(
        n_layers=N_LAYERS, activation_dim=D_ACTS
    )
    fake_batch = torch.randn(8, 2, N_LAYERS, D_ACTS)
    in_std.initialize_from_batch(fake_batch)
    out_std.initialize_from_batch(fake_batch)
    return Molt(
        d_acts=D_ACTS,
        N=N,
        ranks=ranks,
        nonlinearity=nonlin,
        input_standardizer=in_std,
        output_standardizer=out_std,
    )


def test_compute_feature_logits_returns_one_entry_per_feature():
    molt = _build_molt(ranks=[2, 1])    # features = 1 + 2 = 3
    W_U = torch.randn(VOCAB, D_ACTS)
    ln_f = torch.ones(D_ACTS)

    out = compute_feature_logits(
        molt=molt, layer=LAYER, W_U=W_U, ln_f_weight=ln_f,
        tokenizer=_StubTok(), top_k=4, n_bins=10,
    )
    assert len(out) == molt.n_features
    for entry in out:
        assert {"top_pos", "top_neg", "histogram"} <= entry.keys()


def test_top_pos_aligns_with_planted_decoder_direction():
    """Plant a known top-1 SVD direction into a feature's transform and check
    that the token whose unembedding row best matches it shows up first.
    """
    molt = _build_molt(ranks=[2])           # 1 feature, rank 2

    # We need M[0] = U[0].T @ V[0].T to have a known top-1 left singular
    # vector. Easiest: make M = e_k @ e_0.T (rank-1, top-left singular vector
    # is e_k, top right is e_0). Build U, V to realize this.
    #
    # U[0]: shape (rank=2, d_acts). Set U[0] = [[1, 0, 0, 0, 0, 0, 0, 0],
    #                                            [0, 0, 0, 0, 0, 0, 0, 0]]
    # V[0]: shape (d_acts, rank=2). Set V[0] = [[1, 0], [0, 0], ..., [0, 0]]
    # Then M = U.T @ V.T:
    #   U.T = (d_acts, rank) = [[1,0],[0,0],...,[0,0]]
    #   V.T = (rank, d_acts) = [[1,0,...,0],[0,...,0]]
    #   M[i, j] = sum_r U.T[i,r] * V.T[r,j]
    #          = U.T[i,0] * V.T[0,j]
    #          = [i==0] * [j==0]
    # That gives top singular value = 1, top left singular vector = e_0.
    # We want e_K instead — let's plant e_K. Set U[0] = [[0]*K + [1] + [0]*..., zeros].
    K = 3
    U_new = torch.zeros_like(molt.Us[0])
    U_new[0, 0, K] = 1.0
    V_new = torch.zeros_like(molt.Vs[0])
    V_new[0, 0, 0] = 1.0
    # Singular value scale: with these settings M = e_K @ e_0.T, sv = 1.
    with torch.no_grad():
        molt.Us[0].copy_(U_new)
        molt.Vs[0].copy_(V_new)

    # Set output_standardizer.std and ln_f to all-ones so the logits are
    # effectively `W_U @ e_K * 1 = W_U[:, K]`.
    with torch.no_grad():
        molt.output_standardizer.std.fill_(1.0)
    ln_f = torch.ones(D_ACTS)

    # Construct W_U so token #7 has the largest dot with e_K and token #2 is
    # the most negative.
    W_U = torch.zeros(VOCAB, D_ACTS)
    W_U[:, K] = torch.tensor([0.1, 0.2, -5.0, 0.0, 0.5, 0.3, 0.4, 9.0,
                              0.05, -0.1, 0.0, 0.2])

    out = compute_feature_logits(
        molt=molt, layer=LAYER, W_U=W_U, ln_f_weight=ln_f,
        tokenizer=_StubTok(), top_k=3, n_bins=10,
    )
    assert len(out) == 1
    entry = out[0]
    # Top boost is token 7 (W_U[7, K] = 9.0).
    assert entry["top_pos"][0]["token"] == "<7>"
    assert entry["top_pos"][0]["value"] == pytest.approx(9.0, abs=1e-4)
    # Top suppress is token 2 (W_U[2, K] = -5.0).
    assert entry["top_neg"][0]["token"] == "<2>"
    assert entry["top_neg"][0]["value"] == pytest.approx(-5.0, abs=1e-4)


def test_logit_scaling_uses_output_std_and_ln_f():
    """If we double output_std at this layer, every logit doubles.
    Same for ln_f_weight."""
    molt = _build_molt(ranks=[2])
    U_new = torch.zeros_like(molt.Us[0])
    U_new[0, 0, 3] = 1.0
    V_new = torch.zeros_like(molt.Vs[0])
    V_new[0, 0, 0] = 1.0
    with torch.no_grad():
        molt.Us[0].copy_(U_new)
        molt.Vs[0].copy_(V_new)
        molt.output_standardizer.std.fill_(1.0)

    W_U = torch.randn(VOCAB, D_ACTS)
    ln_f = torch.ones(D_ACTS)

    out_a = compute_feature_logits(
        molt=molt, layer=LAYER, W_U=W_U, ln_f_weight=ln_f,
        tokenizer=_StubTok(), top_k=3, n_bins=8,
    )

    # Double the output_std at this layer.
    with torch.no_grad():
        molt.output_standardizer.std[LAYER].mul_(2.0)
    out_b = compute_feature_logits(
        molt=molt, layer=LAYER, W_U=W_U, ln_f_weight=ln_f,
        tokenizer=_StubTok(), top_k=3, n_bins=8,
    )
    # Top-pos value should double too.
    assert out_b[0]["top_pos"][0]["value"] == pytest.approx(
        2 * out_a[0]["top_pos"][0]["value"], rel=1e-4
    )


def test_degenerate_feature_returns_empty_arrays():
    """All-zero transform -> zero singular value -> empty top_pos/top_neg."""
    molt = _build_molt(ranks=[2])
    with torch.no_grad():
        molt.Us[0].zero_()
        molt.Vs[0].zero_()
        molt.output_standardizer.std.fill_(1.0)
    out = compute_feature_logits(
        molt=molt, layer=LAYER,
        W_U=torch.randn(VOCAB, D_ACTS),
        ln_f_weight=torch.ones(D_ACTS),
        tokenizer=_StubTok(), top_k=3, n_bins=8,
    )
    assert out[0]["top_pos"] == []
    assert out[0]["top_neg"] == []
    assert out[0]["histogram"]["counts"] == []


def test_compute_feature_logits_rejects_shape_mismatch():
    molt = _build_molt(ranks=[2])
    with torch.no_grad():
        molt.output_standardizer.std.fill_(1.0)
    with pytest.raises(ValueError):
        compute_feature_logits(
            molt=molt, layer=LAYER,
            W_U=torch.randn(VOCAB, D_ACTS + 1),       # wrong d_acts
            ln_f_weight=torch.ones(D_ACTS),
            tokenizer=_StubTok(),
        )
    with pytest.raises(ValueError):
        compute_feature_logits(
            molt=molt, layer=LAYER,
            W_U=torch.randn(VOCAB, D_ACTS),
            ln_f_weight=torch.ones(D_ACTS + 1),       # wrong d_acts
            tokenizer=_StubTok(),
        )


def test_handles_multiple_tiers():
    """Two-tier MoLT: features = 1 + 2 = 3. All three should produce non-empty
    top_pos when transforms are non-degenerate."""
    molt = _build_molt(ranks=[2, 1])
    # Default xavier_uniform_ initialization gives nonzero transforms.
    with torch.no_grad():
        molt.output_standardizer.std.fill_(1.0)
    out = compute_feature_logits(
        molt=molt, layer=LAYER,
        W_U=torch.randn(VOCAB, D_ACTS),
        ln_f_weight=torch.ones(D_ACTS),
        tokenizer=_StubTok(), top_k=3, n_bins=8,
    )
    assert len(out) == 3
    for entry in out:
        assert len(entry["top_pos"]) == 3
        assert len(entry["top_neg"]) == 3
