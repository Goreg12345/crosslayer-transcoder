import torch
from jaxtyping import Float


class PerLayerTopK(torch.nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.relu = torch.nn.ReLU()

    def forward(
        self,
        features: Float[torch.Tensor, "batch_size n_layers d_features"],
        training: bool = True,
    ) -> Float[torch.Tensor, "batch_size n_layers d_features"]:
        # b, l, d = features.shape does not work with nnsight because tuple extraction isn't implemented yet
        batch_size, n_layers, d_features = features.shape
        assert torch.tensor(self.k) <= d_features

        topk_features = torch.zeros_like(features)
        topk_vals, topk_idxs = torch.topk(features, self.k, dim=-1, sorted=False)
        topk_features.scatter_(dim=-1, index=topk_idxs, src=topk_vals)

        # make sure that features are always positive
        topk_features = self.relu(topk_features)
        return topk_features


class PerSampleTopK(torch.nn.Module):
    def __init__(self, k: int, e: float = 0.1):
        super().__init__()
        self.k = k
        self.e = e
        self.register_buffer("threshold", torch.zeros(1))
        self.relu = torch.nn.ReLU()

    def forward(
        self,
        features: Float[torch.Tensor, "batch_size n_layers d_features"],
        training: bool = True,
    ) -> Float[torch.Tensor, "batch_size n_layers d_features"]:
        if training:
            return self._training_forward(features)
        else:
            return self._inference_forward(features)

    @torch.no_grad()
    def _inference_forward(self, features: torch.Tensor) -> torch.Tensor:
        threshold = max(0, self.threshold.item())
        return torch.where(features > threshold, features, torch.zeros_like(features))

    def _training_forward(
        self,
        features: Float[torch.Tensor, "batch_size n_layers d_features"],
    ) -> Float[torch.Tensor, "batch_size n_layers d_features"]:
        batch_size, n_layers, d_features = features.shape
        assert self.k <= d_features * n_layers

        features = features.reshape(-1, d_features * n_layers)
        topk_features = torch.zeros_like(features)
        if self.k == 0:
            return topk_features.reshape(batch_size, n_layers, d_features)
        topk_vals, topk_idxs = torch.topk(features, self.k, dim=-1, sorted=False)
        topk_features.scatter_(dim=-1, index=topk_idxs, src=topk_vals)
        topk_features = topk_features.reshape(batch_size, n_layers, d_features)

        # make sure that features are always positive
        topk_features = self.relu(topk_features)

        # update thresholds
        min_k = topk_vals.min().detach()
        self.threshold = (1 - self.e) * self.threshold + self.e * min_k
        return topk_features


class BatchTopK(torch.nn.Module):
    def __init__(self, k: int, e: float = 0.1):
        super().__init__()
        self.k = k
        self.e = e
        self.register_buffer("threshold", torch.zeros(1))
        self.relu = torch.nn.ReLU()
        self.step = 0

    def forward(
        self,
        features: Float[torch.Tensor, "batch_size n_layers d_features"],
        training: bool = True,
    ) -> Float[torch.Tensor, "batch_size n_layers d_features"]:
        if training:
            return self._training_forward(features)
        else:
            return self._inference_forward(features)

    @torch.no_grad()
    def _inference_forward(self, features: torch.Tensor) -> torch.Tensor:
        threshold = max(0, self.threshold.item())
        return torch.where(features > threshold, features, torch.zeros_like(features))

    def update_k(self):
        return
        self.step += 1
        batch_size = 4000
        n_layers = 12
        if self.step < 500:
            self.k = 500
        elif self.step < 1000:
            self.k = 400
        elif self.step < 1500:
            self.k = 300
        elif self.step < 2000:
            self.k = 200
        elif self.step < 2500:
            self.k = 150
        elif self.step < 3000:
            self.k = 100
        elif self.step < 3500:
            self.k = 75
        elif self.step < 4000:
            self.k = 50
        elif self.step < 4500:
            self.k = 35
        elif self.step < 5000:
            self.k = 25
        elif self.step < 5500:
            self.k = 15
        elif self.step < 6000:
            self.k = 10
        else:
            self.k = 8
        self.k *= batch_size * n_layers

    def _training_forward(
        self, features: Float[torch.Tensor, "batch_size n_layers d_features"]
    ) -> Float[torch.Tensor, "batch_size n_layers d_features"]:
        self.update_k()
        batch_size, n_layers, d_features = features.shape
        assert self.k <= d_features * n_layers * batch_size

        features = features.flatten()
        topk_features = torch.zeros_like(features)
        if self.k == 0:
            return topk_features.reshape(batch_size, n_layers, d_features)
        topk_vals, topk_idxs = torch.topk(features, self.k, dim=-1, sorted=False)
        topk_features.scatter_(dim=-1, index=topk_idxs, src=topk_vals)
        topk_features = topk_features.reshape(batch_size, n_layers, d_features)
        # make sure that features are always positive
        topk_features = self.relu(topk_features)

        # update thresholds
        min_k = topk_vals.min().detach()
        self.threshold = (1 - self.e) * self.threshold + self.e * min_k
        return topk_features


if __name__ == "__main__":
    topk = PerSampleTopK(k=3, e=0.1)
    features = torch.randn(10, 12, 768 * 8)
    topk_features = topk(features)
    print(topk_features.shape)
