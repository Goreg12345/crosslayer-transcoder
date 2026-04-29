"""Smoke tests for the MoLT (Mixture of Low-rank Transcoders) port.

CPU test exercises the model wiring (low-rank transforms + JumpReLU + standardizers).
GPU test runs a fp32 and a 16-mixed forward+backward+optimizer step on synthetic
activations and checks that all parameters and the loss remain finite — mirrors
what the full-config CLI smoke covers, without the data generator or wandb.
"""

import pytest
import torch

from crosslayer_transcoder.model.jumprelu import JumpReLU
from crosslayer_transcoder.model.molt import Molt
from crosslayer_transcoder.model.standardize import (
    DimensionwiseInputStandardizer,
    DimensionwiseOutputStandardizer,
)


D_ACTS = 64
N_LAYERS = 12
N = 4
RANKS = [8, 4]
B = 4


def _build_molt(device: torch.device) -> Molt:
    n_features = N + 2 * N
    nonlin = JumpReLU(theta=0.03, bandwidth=1.0, n_layers=1, d_features=n_features)
    in_std = DimensionwiseInputStandardizer(n_layers=N_LAYERS, activation_dim=D_ACTS)
    out_std = DimensionwiseOutputStandardizer(n_layers=N_LAYERS, activation_dim=D_ACTS)

    fake_batch = torch.randn(B, 2, N_LAYERS, D_ACTS)
    in_std.initialize_from_batch(fake_batch)
    out_std.initialize_from_batch(fake_batch)

    return Molt(
        d_acts=D_ACTS,
        N=N,
        ranks=RANKS,
        nonlinearity=nonlin,
        input_standardizer=in_std,
        output_standardizer=out_std,
    ).to(device)


def test_molt_cpu_forward():
    torch.manual_seed(0)
    m = _build_molt(torch.device("cpu"))
    acts = torch.randn(B, D_ACTS)

    gate, recons_norm, recons = m(acts, layer=8)

    assert gate.shape == (B, m.n_features)
    assert recons_norm.shape == (B, D_ACTS)
    assert recons.shape == (B, D_ACTS)
    assert torch.isfinite(gate).all()
    assert torch.isfinite(recons).all()


def _run_train_step(device, autocast_dtype):
    """Forward + backward + optimizer step. Mirrors MoltModule.training_step
    without depending on the Lightning Trainer."""
    torch.manual_seed(0)
    m = _build_molt(device)
    optim = torch.optim.Adam(m.parameters(), lr=2e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=autocast_dtype is torch.float16)

    resid = torch.randn(B, D_ACTS, device=device)
    mlp_out = torch.randn(B, D_ACTS, device=device)

    optim.zero_grad(set_to_none=True)
    if autocast_dtype is None:
        gate, recons_norm, _ = m(resid, layer=8)
        target = m.output_standardizer.standardize(mlp_out, 8)
        mse = ((recons_norm - target) ** 2).mean()
        norms = m.transform_norm()
        sparsity = torch.tanh(norms * gate * 100.0).sum(dim=-1).mean() * 1.5e-4
        loss = mse + sparsity
        loss.backward()
        optim.step()
    else:
        with torch.amp.autocast("cuda", dtype=autocast_dtype):
            gate, recons_norm, _ = m(resid, layer=8)
            target = m.output_standardizer.standardize(mlp_out, 8)
            mse = ((recons_norm - target) ** 2).mean()
            norms = m.transform_norm()
            sparsity = torch.tanh(norms * gate * 100.0).sum(dim=-1).mean() * 1.5e-4
            loss = mse + sparsity
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

    return loss, m


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_molt_gpu_fp32_train_step():
    loss, m = _run_train_step(torch.device("cuda"), autocast_dtype=None)
    assert torch.isfinite(loss), f"non-finite loss: {loss.item()}"
    for name, p in m.named_parameters():
        assert torch.isfinite(p).all(), f"non-finite param after step: {name}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_molt_gpu_amp_train_step():
    loss, m = _run_train_step(torch.device("cuda"), autocast_dtype=torch.float16)
    assert torch.isfinite(loss), f"non-finite loss: {loss.item()}"
    for name, p in m.named_parameters():
        assert torch.isfinite(p).all(), f"non-finite param after AMP step: {name}"
