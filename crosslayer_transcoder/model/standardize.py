from einops import einsum
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

    def forward(
        self, batch: Float[torch.Tensor, "batch_size n_layers actv_dim"], layer="all"
    ):
        return batch

    def standardize(
        self, batch: Float[torch.Tensor, "batch_size n_layers actv_dim"], layer="all"
    ):
        return batch


class DimensionwiseInputStandardizer(Standardizer):
    def __init__(self, n_layers, activation_dim):
        super().__init__()
        self.register_buffer("mean", torch.empty(n_layers, activation_dim))
        self.register_buffer("std", torch.empty(n_layers, activation_dim))
        self.is_initialized = False

    @torch.no_grad()
    def initialize_from_batch(
        self, batch: Float[torch.Tensor, "batch_size io n_layers actv_dim"]
    ):
        batch = batch[:, 0]
        self.mean.data = batch.mean(dim=0)
        self.std.data = batch.std(dim=0)
        self.std.data.clamp_(min=1e-8)
        self.is_initialized = True

    def forward(
        self,
        batch: Float[torch.Tensor, "batch_size n_layers actv_dim"],
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


class SamplewiseInputStandardizer(Standardizer):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def initialize_from_batch(
        self, batch: Float[torch.Tensor, "batch_size io n_layers actv_dim"]
    ):
        pass

    def forward(
        self,
        batch: Float[torch.Tensor, "batch_size n_layers actv_dim"],
        layer="all",
    ):
        means = batch.mean(dim=-1, keepdim=True)
        stds = batch.std(dim=-1, keepdim=True)
        stds.clamp_(min=1e-8)
        return (batch - means) / stds


class LayerwiseInputStandardizer(Standardizer):
    def __init__(self, n_layers, n_exclude: int = 0):
        super().__init__()
        self.register_buffer("mean", torch.empty(n_layers))
        self.register_buffer("std", torch.empty(n_layers))
        self.n_exclude = n_exclude
        self.is_initialized = False

    @torch.no_grad()
    def initialize_from_batch(
        self, batch: Float[torch.Tensor, "batch_size io n_layers actv_dim"]
    ):
        batch = batch[:, 0]
        # exclude the n_exclude largest and smallest values
        topk = batch.topk(self.n_exclude, dim=-1, sorted=False)
        mink = batch.topk(self.n_exclude, dim=-1, sorted=False, largest=False)
        batch.scatter_(dim=-1, index=topk.indices, src=torch.zeros_like(batch))
        batch.scatter_(dim=-1, index=mink.indices, src=torch.zeros_like(batch))

        self.mean.data = batch.mean(dim=(0, 2))
        self.std.data = batch.std(dim=(0, 2))
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
            return (batch - self.mean[None, :, None]) / self.std[None, :, None]
        else:
            return (batch - self.mean[None, layer, None]) / self.std[None, layer, None]


class LayerwiseOutputStandardizer(Standardizer):
    def __init__(self, n_layers, n_exclude: int = 0):
        super().__init__()
        self.register_buffer("mean", torch.empty(n_layers))
        self.register_buffer("std", torch.empty(n_layers))
        self.n_exclude = n_exclude
        self.is_initialized = False

    def initialize_from_batch(
        self, batch: Float[torch.Tensor, "batch_size io n_layers actv_dim"]
    ):
        batch = batch[:, 1]
        # exclude the n_exclude largest and smallest values
        topk = batch.topk(self.n_exclude, dim=-1, sorted=False)
        mink = batch.topk(self.n_exclude, dim=-1, sorted=False, largest=False)
        batch.scatter_(dim=-1, index=topk.indices, src=torch.zeros_like(batch))
        batch.scatter_(dim=-1, index=mink.indices, src=torch.zeros_like(batch))

        self.mean.data = batch.mean(dim=(0, 2))
        self.std.data = batch.std(dim=(0, 2))
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
            return (batch * self.std[None, :, None]) + self.mean[None, :, None]
        else:
            return (batch * self.std[None, layer, None]) + self.mean[None, layer, None]

    def standardize(
        self,
        mlp_out: Float[torch.Tensor, "batch_size n_layers actv_dim"],
        layer="all",
    ):
        if layer == "all":
            return (mlp_out - self.mean[None, :, None]) / self.std[None, :, None]
        else:
            return (mlp_out - self.mean[None, layer, None]) / self.std[
                None, layer, None
            ]
