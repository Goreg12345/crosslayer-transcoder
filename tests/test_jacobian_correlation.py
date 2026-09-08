import torch

from crosslayer_transcoder.metrics.jacobian_correlation import (
    JacobianCorrelation,
    flattened_jacobian_cosine,
    molt_jacobian,
)
from tests.test_molt import build_molt


def test_flattened_jacobian_cosine_is_per_datapoint():
    identity = torch.eye(3).repeat(2, 1, 1)
    comparison = torch.stack([torch.eye(3), -torch.eye(3)])
    torch.testing.assert_close(
        flattened_jacobian_cosine(identity, comparison), torch.tensor([1.0, -1.0])
    )


def test_molt_analytic_jacobian_matches_actual_forward():
    torch.manual_seed(10)
    model = build_molt().double()
    # Keep all gates away from their discontinuity and active, so the ordinary
    # derivative of the represented JumpReLU function is unambiguous.
    model.e.bias.data.fill_(2.0)
    model.e.weight.data.mul_(0.01)
    inputs = torch.randn(2, model.d_acts, dtype=torch.double)

    expected = torch.stack(
        [
            torch.autograd.functional.jacobian(
                lambda x: model(x[None], layer=1)[2].squeeze(0), point
            )
            for point in inputs
        ]
    )
    actual = molt_jacobian(model, inputs, layer=1)
    torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=2e-5)


def test_jacobian_correlation_averages_datapoints():
    metric = JacobianCorrelation()
    target = torch.eye(2).repeat(2, 1, 1)
    replacement = torch.stack([torch.eye(2), torch.zeros(2, 2)])
    metric.update(replacement, target)
    torch.testing.assert_close(metric.compute(), torch.tensor(0.5))
