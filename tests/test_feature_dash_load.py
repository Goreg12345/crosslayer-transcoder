"""Smoke test for the MoLT feature-dash checkpoint loader.

Builds a tiny MoLT, wraps it in a MoltModule, saves a Lightning checkpoint via
trainer.save_checkpoint, then loads it back with `load_molt` and checks:

  - architecture is recovered (d_acts, n_features, n_layers, ranks, N),
  - standardizer buffers round-trip,
  - JumpReLU theta round-trips,
  - forward output matches the source model bit-for-bit on a fixed input,
  - the per-feature tier/rank index lines up with the rank tier sizes.
"""

from __future__ import annotations

import lightning as L
import pytest
import torch

from crosslayer_transcoder.feature_dash.load import infer_molt_arch, load_molt
from crosslayer_transcoder.model.clt_lightning import MoltModule
from crosslayer_transcoder.model.jumprelu import JumpReLU
from crosslayer_transcoder.model.molt import Molt
from crosslayer_transcoder.model.standardize import (
    DimensionwiseInputStandardizer,
    DimensionwiseOutputStandardizer,
)


D_ACTS = 16
N_LAYERS = 4
N = 3
RANKS = [8, 4]  # tier 0: 3 transforms, tier 1: 6 transforms -> 9 features
LAYER = 2


def _build_molt() -> Molt:
    n_features = N * 1 + N * 2  # = 9
    nonlin = JumpReLU(theta=0.05, bandwidth=1.0, n_layers=1, d_features=n_features)
    in_std = DimensionwiseInputStandardizer(n_layers=N_LAYERS, activation_dim=D_ACTS)
    out_std = DimensionwiseOutputStandardizer(n_layers=N_LAYERS, activation_dim=D_ACTS)

    fake_batch = torch.randn(8, 2, N_LAYERS, D_ACTS)
    in_std.initialize_from_batch(fake_batch)
    out_std.initialize_from_batch(fake_batch)

    return Molt(
        d_acts=D_ACTS,
        N=N,
        ranks=RANKS,
        nonlinearity=nonlin,
        input_standardizer=in_std,
        output_standardizer=out_std,
    )


def _save_lightning_checkpoint(tmp_path, molt: Molt) -> str:
    module = MoltModule(model=molt)
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        max_steps=0,
    )
    # `strategy.connect` wires the module so save_checkpoint works without
    # actually running `fit`.
    trainer.strategy.connect(module)
    ckpt_path = tmp_path / "molt.ckpt"
    trainer.save_checkpoint(str(ckpt_path))
    return str(ckpt_path)


def test_infer_molt_arch_from_state_dict():
    molt = _build_molt()
    sd = {f"model.{k}": v for k, v in molt.state_dict().items()}
    # mimic what Lightning would add
    sd["last_active"] = torch.zeros((molt.n_features,), dtype=torch.long)

    arch = infer_molt_arch(sd)
    assert arch == {
        "d_acts": D_ACTS,
        "n_features": N * 1 + N * 2,
        "n_layers": N_LAYERS,
        "ranks": RANKS,
        "N": N,
    }


def test_load_molt_roundtrip(tmp_path):
    torch.manual_seed(0)
    src = _build_molt()
    ckpt_path = _save_lightning_checkpoint(tmp_path, src)

    loaded, meta = load_molt(ckpt_path, device="cpu")

    assert meta.d_acts == D_ACTS
    assert meta.n_features == src.n_features
    assert meta.ranks == RANKS
    assert meta.N == N
    assert meta.n_layers == N_LAYERS

    # Tier index has the right tier counts.
    assert meta.feature_tier.count(0) == N
    assert meta.feature_tier.count(1) == 2 * N
    assert meta.feature_rank[0] == RANKS[0]
    assert meta.feature_rank[-1] == RANKS[1]
    assert len(meta.feature_tier) == src.n_features

    # Forward output matches bit-for-bit on a fixed input.
    src.eval()
    loaded.eval()
    x = torch.randn(5, D_ACTS)
    with torch.no_grad():
        g_src, rn_src, r_src = src(x, layer=LAYER)
        g_loaded, rn_loaded, r_loaded = loaded(x, layer=LAYER)
    assert torch.allclose(g_src, g_loaded)
    assert torch.allclose(rn_src, rn_loaded)
    assert torch.allclose(r_src, r_loaded)

    # Loaded model must be in eval and have grads disabled.
    assert not loaded.training
    assert all(not p.requires_grad for p in loaded.parameters())
