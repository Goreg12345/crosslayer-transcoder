from typing import Any, Dict

import torch
import torch.nn as nn

from crosslayer_transcoder.model.serializable_module import SerializableModule


def rectangle(x):
    return heavyside_step(x + 0.5) - heavyside_step(x - 0.5)


def heavyside_step(x):
    return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))


class ReLU(SerializableModule):
    def __init__(self):
        super().__init__()

    def forward(self, input, layer=None):
        return torch.relu(input)

    def to_config(self) -> Dict[str, Any]:
        return {
            "class_path": self.__class__.__module__ + "." + self.__class__.__name__,
            "init_args": {},
        }


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
class JumpReLU(SerializableModule):
    def __init__(self, theta=0.0, bandwidth=1.0, n_layers=12, d_features=768 * 8):
        super().__init__()
        self.theta = nn.Parameter(torch.full((1, n_layers, d_features), theta))
        self.register_buffer("bandwidth", torch.tensor(bandwidth))
        self._init_theta = theta
        self.n_layers = n_layers
        self.d_features = d_features

    def forward(self, input):
        return _JumpReLUFunction.apply(input, self.theta, self.bandwidth)

    def to_config(self) -> Dict[str, Any]:
        return {
            "class_path": self.__class__.__module__ + "." + self.__class__.__name__,
            "init_args": {
                "theta": self._init_theta,
                "bandwidth": self.bandwidth.item(),
                "n_layers": self.n_layers,
                "d_features": self.d_features,
            },
        }


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
