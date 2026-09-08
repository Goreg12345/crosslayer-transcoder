import torch

from crosslayer_transcoder.dashboard.logits import (
    frozen_rms_readout,
    signed_topk,
    transform_contribution,
)


def test_contribution_is_raw_reconstruction_on_minus_off():
    torch.manual_seed(0)
    x = torch.randn(7, 5)
    v, u = torch.randn(5, 3), torch.randn(3, 5)
    gate = torch.rand(7)
    std, mean, others = torch.rand(5), torch.randn(5), torch.randn(7, 5)
    on = (others + gate[:, None] * ((x @ v) @ u)) * std + mean
    off = others * std + mean
    torch.testing.assert_close(transform_contribution(x, gate, v, u, std), on - off)
    assert torch.equal(
        transform_contribution(x, gate * 0, v, u, std), torch.zeros_like(x)
    )


def test_frozen_rms_projection_equals_frozen_normalization_difference():
    torch.manual_seed(1)
    baseline, delta = torch.randn(7, 5), torch.randn(7, 5)
    norm_weight, unembed = torch.randn(5), torch.randn(11, 5)
    rms = (baseline.square().mean(-1) + 1e-6).sqrt()
    on = (baseline * norm_weight / rms[:, None]) @ unembed.T
    off = ((baseline - delta) * norm_weight / rms[:, None]) @ unembed.T
    torch.testing.assert_close(
        frozen_rms_readout(delta, rms, norm_weight) @ unembed.T, on - off
    )


def test_signed_topk_never_labels_zero_or_wrong_sign():
    result = signed_topk(torch.tensor([0.0, 2.0, -3.0, 1.0]), 10)
    assert result["top_token_ids"] == [1, 3]
    assert result["bottom_token_ids"] == [2]
