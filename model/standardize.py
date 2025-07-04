import torch
import torch.nn as nn
from jaxtyping import Float


class Standardizer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def initialize_from_batch(
        self, batch: Float[torch.Tensor, "batch_size io n_layers actv_dim"]
    ):
        pass

    def forward(self, batch: Float[torch.Tensor, "batch_size io n_layers actv_dim"]):
        return batch

    def standardize(
        self, batch: Float[torch.Tensor, "batch_size io n_layers actv_dim"]
    ):
        return batch


class DimensionwiseInputStandardizer(Standardizer):
    def __init__(self, n_layers, activation_dim):
        super().__init__()
        self.register_buffer("mean", torch.empty(n_layers, activation_dim))
        self.register_buffer("std", torch.empty(n_layers, activation_dim))
        self.is_initialized = False

    def initialize_from_batch(
        self, batch: Float[torch.Tensor, "batch_size io n_layers actv_dim"]
    ):
        self.mean.data = batch[:, 0].mean(dim=0)
        self.std.data = batch[:, 0].std(dim=0)
        self.std.data.clamp_(min=1e-8)
        self.is_initialized = True

    def forward(
        self,
        batch: Float[torch.Tensor, "batch_size io n_layers actv_dim"],
        layer="all",
    ):
        if not self.is_initialized:
            raise ValueError("Standardizer not initialized")
        if layer == "all":
            return (batch - self.mean) / self.std
        else:
            return (batch - self.mean[layer]) / self.std[layer]


class DimensionwiseOutputStandardizer(Standardizer):
    def __init__(self, n_layers, activation_dim):
        super().__init__()
        self.register_buffer("mean", torch.empty(n_layers, activation_dim))
        self.register_buffer("std", torch.empty(n_layers, activation_dim))
        self.is_initialized = False

    def initialize_from_batch(
        self, batch: Float[torch.Tensor, "batch_size io n_layers actv_dim"]
    ):
        self.mean.data = batch[:, 1].mean(dim=0)
        self.std.data = batch[:, 1].std(dim=0)
        self.std.data.clamp_(min=1e-8)
        self.is_initialized = True

    def forward(
        self,
        batch: Float[torch.Tensor, "batch_size io n_layers actv_dim"],
        layer="all",
    ):
        if not self.is_initialized:
            raise ValueError("Standardizer not initialized")
        if layer == "all":
            return (batch * self.std) + self.mean
        else:
            return (batch * self.std[layer]) + self.mean[layer]

    def standardize(
        self,
        mlp_out: Float[torch.Tensor, "batch_size n_layers actv_dim"],
        layer="all",
    ):
        if layer == "all":
            return (mlp_out - self.mean) / self.std
        else:
            return (mlp_out - self.mean[layer]) / self.std[layer]
