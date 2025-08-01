import torch
import torch.nn as nn


def rectangle(x):
    return heavyside_step(x + 0.5) - heavyside_step(x - 0.5)


def heavyside_step(x):
    return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))


class _JumpReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, theta, bandwidth):
        ctx.save_for_backward(input, theta)
        ctx.bandwidth = bandwidth
        feature_mask = torch.logical_and(input > theta, input > 0.0)
        features = feature_mask * input
        return features

    @staticmethod
    def backward(ctx, grad_output):
        input, theta = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0

        theta_grad = -(theta / bandwidth) * rectangle((input - theta) / bandwidth) * grad_output
        return grad_input, theta_grad, None


#  @torch.compile --> potentially causes segmentation faults
class JumpReLU(torch.nn.Module):
    def __init__(self, theta=0.0, bandwidth=1.0, n_layers=12, d_features=768 * 8):
        super().__init__()
        shape = (1, n_layers, d_features) if n_layers > 1 else (1, d_features)
        self.theta = nn.Parameter(torch.full(shape, theta))
        self.register_buffer("bandwidth", torch.tensor(bandwidth))

    def forward(self, input):
        return _JumpReLUFunction.apply(input, self.theta, self.bandwidth)


class HeavysideStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, theta, bandwidth):
        ctx.save_for_backward(input, theta)
        ctx.bandwidth = bandwidth
        return torch.where(input - theta > 0, torch.ones_like(input), torch.zeros_like(input))

    @staticmethod
    def backward(ctx, grad_output):
        input, theta = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        grad_input = grad_output.clone()
        grad_input = grad_output * 0.0

        theta_grad = -(1.0 / bandwidth) * rectangle((input - theta) / bandwidth) * grad_output
        return grad_input, theta_grad, None
