"""Tests for the MoLT steering module.

Covers index resolution, the intervention math (must agree with MoLT's own
forward when alpha is set to the gate value), and the hook mechanics on a
GPT-2-shaped stub model so we don't depend on downloading real weights.
"""

import pytest
import torch
import torch.nn as nn

from crosslayer_transcoder.model.jumprelu import JumpReLU
from crosslayer_transcoder.model.molt import Molt
from crosslayer_transcoder.model.standardize import (
    DimensionwiseInputStandardizer,
    DimensionwiseOutputStandardizer,
)
from crosslayer_transcoder.steering.hooks import steered
from crosslayer_transcoder.steering.intervention import (
    TransformIntervention,
    _resolve_index,
)


D_ACTS = 16
N_LAYERS = 4
N = 3
RANKS = [4, 2]  # tier 0: 3 transforms rank 4; tier 1: 6 transforms rank 2
LAYER = 1


def _build_molt(seed: int = 0) -> Molt:
    torch.manual_seed(seed)
    n_features = N + 2 * N
    nonlin = JumpReLU(theta=0.03, bandwidth=1.0, n_layers=1, d_features=n_features)
    in_std = DimensionwiseInputStandardizer(n_layers=N_LAYERS, activation_dim=D_ACTS)
    out_std = DimensionwiseOutputStandardizer(n_layers=N_LAYERS, activation_dim=D_ACTS)

    fake_batch = torch.randn(32, 2, N_LAYERS, D_ACTS)
    in_std.initialize_from_batch(fake_batch)
    out_std.initialize_from_batch(fake_batch)

    return Molt(
        d_acts=D_ACTS,
        N=N,
        ranks=RANKS,
        nonlinearity=nonlin,
        input_standardizer=in_std,
        output_standardizer=out_std,
    ).eval()


def test_resolve_index_tier0():
    molt = _build_molt()
    # First N=3 indices are tier 0
    for i in range(N):
        tier, local = _resolve_index(molt, i)
        assert tier == 0
        assert local == i


def test_resolve_index_tier1():
    molt = _build_molt()
    # Next 2*N=6 indices are tier 1
    for i in range(N, N + 2 * N):
        tier, local = _resolve_index(molt, i)
        assert tier == 1
        assert local == i - N


def test_resolve_index_out_of_range():
    molt = _build_molt()
    with pytest.raises(IndexError):
        _resolve_index(molt, N + 2 * N)


def test_intervention_matches_molt_forward():
    """The core consistency check: if we set alpha to the gate value MoLT
    computes for transform i on a given input, then intervention(mlp_in)
    must equal transform i's share of MoLT's unstandardized reconstruction.
    """
    molt = _build_molt(seed=42)
    torch.manual_seed(7)
    mlp_in = torch.randn(2, D_ACTS)

    # Pick transform 5 (in tier 1) to exercise the non-trivial tier path.
    idx = 5
    tier, local = _resolve_index(molt, idx)

    # Run MoLT and pull out gate[i] and the per-transform raw recons.
    acts_std = molt.input_standardizer(mlp_in, layer=LAYER)
    pre_actvs = molt.e(acts_std)
    gate = molt.nonlinearity(pre_actvs)  # (B, n_features)

    # Per-transform raw recon for the picked one.
    V_i = molt.Vs[tier][local]  # (d_acts, rank)
    U_i = molt.Us[tier][local]  # (rank, d_acts)
    expected_std = (acts_std @ V_i) @ U_i  # (B, d_acts)
    expected_unstd = expected_std * molt.output_standardizer.std[LAYER]

    gate_i = gate[:, idx].unsqueeze(-1)  # (B, 1)
    expected_contribution = gate_i * expected_unstd

    # Now build an intervention with alpha set per-row to gate[i] — the test
    # uses alpha=1.0 and scales by gate_i manually, since alpha is a scalar.
    intervention = TransformIntervention(
        molt=molt, transform_index=idx, layer=LAYER, alpha=1.0
    )
    delta_at_alpha1 = intervention(mlp_in)
    got_contribution = gate_i * delta_at_alpha1

    assert torch.allclose(got_contribution, expected_contribution, atol=1e-5), (
        f"max abs diff: {(got_contribution - expected_contribution).abs().max().item()}"
    )


def test_intervention_alpha_scaling():
    """delta is linear in alpha."""
    molt = _build_molt(seed=1)
    mlp_in = torch.randn(3, D_ACTS)
    iv1 = TransformIntervention(molt=molt, transform_index=2, layer=LAYER, alpha=1.0)
    iv5 = TransformIntervention(molt=molt, transform_index=2, layer=LAYER, alpha=5.0)
    d1 = iv1(mlp_in)
    d5 = iv5(mlp_in)
    assert torch.allclose(d5, 5.0 * d1, atol=1e-6)


def test_intervention_zero_acts_zero_delta():
    """A linear intervention with no constant term produces zero on standardized
    zero-input. We standardize input first, so mlp_in == in_mean -> acts_std==0
    -> delta == 0. This distinguishes the gate-open mechanism from a static
    direction injection.
    """
    molt = _build_molt(seed=2)
    iv = TransformIntervention(molt=molt, transform_index=4, layer=LAYER, alpha=10.0)
    # mlp_in equal to the input mean produces standardized zeros.
    mlp_in = molt.input_standardizer.mean[LAYER].unsqueeze(0).clone()
    delta = iv(mlp_in)
    assert torch.allclose(delta, torch.zeros_like(delta), atol=1e-6)


def test_intervention_rejects_uninitialized_standardizer():
    n_features = N + 2 * N
    nonlin = JumpReLU(theta=0.03, bandwidth=1.0, n_layers=1, d_features=n_features)
    in_std = DimensionwiseInputStandardizer(n_layers=N_LAYERS, activation_dim=D_ACTS)
    out_std = DimensionwiseOutputStandardizer(n_layers=N_LAYERS, activation_dim=D_ACTS)
    molt = Molt(
        d_acts=D_ACTS,
        N=N,
        ranks=RANKS,
        nonlinearity=nonlin,
        input_standardizer=in_std,
        output_standardizer=out_std,
    )
    with pytest.raises(ValueError, match="not initialized"):
        TransformIntervention(molt=molt, transform_index=0, layer=LAYER, alpha=1.0)


# ----- hook tests on a GPT-2-shaped stub --------------------------------------


class _StubBlock(nn.Module):
    """Minimal GPT-2 block: residual + attn + (residual + mlp(ln_2(x))).

    We only need the post-MLP residual to come out correctly, so attn is just
    identity. The important thing is that ln_2 sees the pre-MLP residual and
    mlp's output is added back.
    """

    def __init__(self, d: int):
        super().__init__()
        self.ln_2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        # Skip attention; just exercise the ln_2/mlp path the hooks target.
        residual = x
        return residual + self.mlp(self.ln_2(residual))


class _StubTransformer(nn.Module):
    def __init__(self, d: int, n_layers: int):
        super().__init__()
        self.h = nn.ModuleList([_StubBlock(d) for _ in range(n_layers)])

    def forward(self, x):
        for block in self.h:
            x = block(x)
        return x


class _StubModel(nn.Module):
    def __init__(self, d: int, n_layers: int):
        super().__init__()
        self.transformer = _StubTransformer(d, n_layers)

    def forward(self, x):
        return self.transformer(x)


def test_steered_hook_alpha_zero_is_identity():
    """alpha=0 → intervention returns zeros → hooked forward equals plain forward."""
    torch.manual_seed(0)
    molt = _build_molt(seed=3)
    model = _StubModel(D_ACTS, N_LAYERS).eval()

    iv = TransformIntervention(molt=molt, transform_index=1, layer=LAYER, alpha=0.0)
    x = torch.randn(2, 5, D_ACTS)
    with torch.no_grad():
        out_baseline = model(x)
        with steered(model, iv):
            out_steered = model(x)
    assert torch.allclose(out_baseline, out_steered, atol=1e-6)


def test_steered_hook_adds_exact_intervention_delta():
    """The hooked forward differs from baseline by exactly intervention(mlp_in)
    summed into the residual stream at the patched layer. We capture mlp_in
    via a probe and verify the delta downstream-of-the-block equals it.
    """
    torch.manual_seed(0)
    molt = _build_molt(seed=4)
    model = _StubModel(D_ACTS, N_LAYERS).eval()
    iv = TransformIntervention(molt=molt, transform_index=2, layer=LAYER, alpha=3.5)

    x = torch.randn(2, 4, D_ACTS)

    captured: dict = {}

    def capture_pre_ln2(module, args):
        captured["mlp_in"] = args[0].detach().clone()

    block = model.transformer.h[LAYER]
    h = block.ln_2.register_forward_pre_hook(capture_pre_ln2)
    try:
        with torch.no_grad():
            out_baseline = model(x)
            with steered(model, iv):
                out_steered = model(x)
    finally:
        h.remove()

    # Recompute expected delta from the captured pre-LN residual. Since the
    # hook adds delta to mlp.output, and the block then adds residual+mlp_out,
    # the delta propagates unchanged through the residual add of this block.
    # Subsequent blocks are pure identities of (residual + mlp(ln_2(x))) — they
    # don't preserve the delta linearly. So compare at the *output of the
    # patched block*, not the final model output.
    #
    # Easiest way: re-run model with a probe on the patched block's output,
    # both with and without steering.
    block_outputs: dict = {"baseline": None, "steered": None}

    def probe(name):
        def _probe(_module, _args, output):
            block_outputs[name] = output.detach().clone()
        return _probe

    h_b = block.register_forward_hook(probe("baseline"))
    with torch.no_grad():
        _ = model(x)
    h_b.remove()

    h_s = block.register_forward_hook(probe("steered"))
    try:
        with torch.no_grad():
            with steered(model, iv):
                _ = model(x)
    finally:
        h_s.remove()

    delta_observed = block_outputs["steered"] - block_outputs["baseline"]
    delta_expected = iv(captured["mlp_in"])

    assert torch.allclose(delta_observed, delta_expected, atol=1e-5), (
        f"max abs diff: {(delta_observed - delta_expected).abs().max().item()}"
    )

    # Also: the full-model output must differ when alpha != 0.
    assert not torch.allclose(out_baseline, out_steered, atol=1e-4)


def test_steered_hook_cleanup():
    """Hooks are removed on context exit, even after exceptions."""
    molt = _build_molt(seed=5)
    model = _StubModel(D_ACTS, N_LAYERS).eval()
    iv = TransformIntervention(molt=molt, transform_index=0, layer=LAYER, alpha=1.0)

    block = model.transformer.h[LAYER]

    def n_hooks(m):
        return len(m._forward_hooks) + len(m._forward_pre_hooks)

    before = (n_hooks(block.ln_2), n_hooks(block.mlp))

    with pytest.raises(RuntimeError):
        with steered(model, iv):
            raise RuntimeError("boom")

    after = (n_hooks(block.ln_2), n_hooks(block.mlp))
    assert before == after, f"hooks leaked: before={before} after={after}"
