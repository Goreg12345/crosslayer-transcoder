import torch

from crosslayer_transcoder.dashboard.gates import MoltGates
from crosslayer_transcoder.model.jumprelu import JumpReLU
from crosslayer_transcoder.model.molt import Molt
from crosslayer_transcoder.model.standardize import (
    DimensionwiseInputStandardizer,
    DimensionwiseOutputStandardizer,
)


def test_gate_only_matches_full_molt_with_saved_normalization():
    torch.manual_seed(42)
    model = Molt(
        6,
        2,
        JumpReLU(theta=0.15, n_layers=1, d_features=6),
        DimensionwiseInputStandardizer(2, 6),
        DimensionwiseOutputStandardizer(2, 6),
        ranks=[3, 2],
    )
    model.initialize_standardizers(torch.randn(20, 2, 2, 6) * 3 + 2)
    state = {"model." + k: v for k, v in model.state_dict().items()}
    gate_only = MoltGates(state, layer=1)
    x = torch.randn(12, 6)
    expected = model(x, layer=1)[0]
    torch.testing.assert_close(gate_only(x), expected, rtol=0, atol=0)
    assert not any("Us" in k or "Vs" in k for k in gate_only.state_dict())


def test_threshold_is_strict_and_negative_pre_activation_is_inactive():
    state = {
        "model.e.weight": torch.eye(3),
        "model.e.bias": torch.zeros(3),
        "model.input_standardizer.mean": torch.zeros(1, 3),
        "model.input_standardizer.std": torch.ones(1, 3),
        "model.nonlinearity.theta": torch.tensor([[0.5, 0.5, -2.0]]),
        "model.nonlinearity.bandwidth": torch.tensor(1.0),
    }
    gate_only = MoltGates(state, layer=0)
    torch.testing.assert_close(
        gate_only(torch.tensor([[0.5, 0.6, -1.0]])), torch.tensor([[0.0, 0.6, 0.0]])
    )
