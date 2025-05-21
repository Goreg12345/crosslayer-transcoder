import torch


def rectangle(x):
        return heavyside_step(x + .5) - heavyside_step(x - .5)

def heavyside_step(x):
    return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

class JumpReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, theta, bandwidth):
        ctx.save_for_backward(input, theta)
        ctx.bandwidth = bandwidth
        feature_mask = (input > theta) & (input > 0.)
        features = feature_mask * input
        return features

    @staticmethod
    def backward(ctx, grad_output):
        input, theta = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        
        theta_grad = (
            -(theta / bandwidth)
            * rectangle((input - theta) / bandwidth)
            * grad_output
        )
        return grad_input, theta_grad, None
    

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
        grad_input = grad_output * 0.

        theta_grad = (
            -(1.0 / bandwidth) * rectangle((input - theta) / bandwidth) * grad_output
        )
        return grad_input, theta_grad, None