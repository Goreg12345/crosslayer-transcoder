from abc import ABC, abstractmethod
from typing import Union

import einops
import torch
from jaxtyping import Float


class TopK(ABC, torch.nn.Module):
    def __init__(self, k: int, e=None, n_layers=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k
        self.e = e
        self.n_layers = n_layers
        self.relu = torch.nn.ReLU()

    def forward(
        self,
        features: Float[torch.Tensor, "batch_size n_layers d_features"],
        layer: Union[int, str] = "all",
    ) -> Float[torch.Tensor, "batch_size n_layers d_features"]:
        if layer == "all":
            return self._training_forward(features)
        else:
            return self._inference_forward(features, layer)

    @abstractmethod
    def _training_forward(self, features: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def _inference_forward(self, features: torch.Tensor, layer: int) -> torch.Tensor:
        pass


class PerLayerTopK(TopK):
    @torch.no_grad()
    def _inference_forward(self, features: torch.Tensor, layer: int) -> torch.Tensor:
        return self._training_forward(features)

    def _training_forward(self, features: torch.Tensor) -> torch.Tensor:
        # b, l, d = features.shape does not work with nnsight because tuple extraction isn't implemented yet
        batch_size, n_layers, d_features = features.shape
        assert self.k <= d_features

        topk_features = torch.zeros_like(features)
        topk_vals, topk_idxs = torch.topk(features, self.k, dim=-1, sorted=False)
        topk_vals = self.relu(topk_vals)  # make sure that features are always positive
        topk_features.scatter_(dim=-1, index=topk_idxs, src=topk_vals)

        return topk_features


class BatchTopK(TopK):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("threshold", torch.zeros(1))

    @torch.no_grad()
    def _inference_forward(self, features: torch.Tensor, layer: int) -> torch.Tensor:
        threshold = max(0, self.threshold.item())
        return torch.where(features > threshold, features, torch.zeros_like(features))

    def _training_forward(
        self, features: Float[torch.Tensor, "batch_size n_layers d_features"]
    ) -> Float[torch.Tensor, "batch_size n_layers d_features"]:
        batch_size, n_layers, d_features = features.shape
        assert self.k <= d_features

        batch_k = self.k * batch_size * n_layers

        features = features.flatten()
        topk_features = torch.zeros_like(features)
        if batch_k == 0:
            return topk_features.reshape(batch_size, n_layers, d_features)
        topk_vals, topk_idxs = torch.topk(features, batch_k, dim=-1, sorted=False)
        topk_vals = self.relu(topk_vals)  # make sure that features are always positive
        topk_features.scatter_(dim=-1, index=topk_idxs, src=topk_vals)
        topk_features = topk_features.reshape(batch_size, n_layers, d_features)

        # update thresholds
        min_k = topk_vals.min().detach()
        self.threshold = (1 - self.e) * self.threshold + self.e * min_k
        return topk_features


class PerLayerBatchTopK(TopK):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("threshold", torch.zeros(self.n_layers))

    @torch.no_grad()
    def _inference_forward(self, features: torch.Tensor, layer: int) -> torch.Tensor:
        threshold = max(0, self.threshold[layer].item())
        return torch.where(features > threshold, features, torch.zeros_like(features))

    def _training_forward(
        self, features: Float[torch.Tensor, "batch_size n_layers d_features"]
    ) -> Float[torch.Tensor, "batch_size n_layers d_features"]:
        batch_size, n_layers, d_features = features.shape
        assert self.k <= d_features

        batch_k = self.k * batch_size

        features = einops.rearrange(features, "batch layer feature -> layer (batch feature)")
        topk_features = torch.zeros_like(features)
        if batch_k == 0:
            return einops.rearrange(
                topk_features, "layer (batch feature) -> batch layer feature", batch=batch_size
            )
        topk_vals, topk_idxs = torch.topk(features, batch_k, dim=-1, sorted=False)
        topk_vals = self.relu(topk_vals)  # make sure that features are always positive
        topk_features.scatter_(dim=-1, index=topk_idxs, src=topk_vals)
        topk_features = einops.rearrange(
            topk_features, "layer (batch feature) -> batch layer feature", batch=batch_size
        )

        # update thresholds
        min_k = topk_vals.detach().min(dim=-1).values
        self.threshold = (1 - self.e) * self.threshold + self.e * min_k
        return topk_features
