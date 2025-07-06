import torch
from jaxtyping import Float


class PerLayerTopK(torch.nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(
        self, features: Float[torch.Tensor, "batch_size n_layers d_features"]
    ) -> Float[torch.Tensor, "batch_size n_layers d_features"]:
        batch_size, n_layers, d_features = features.shape
        assert self.k <= d_features

        topk_features = torch.zeros_like(features)
        topk_vals, topk_idxs = torch.topk(features, self.k, dim=-1, sorted=False)
        topk_features.scatter_(dim=-1, index=topk_idxs, src=topk_vals)
        return topk_features


class PerSampleTopK(torch.nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(
        self, features: Float[torch.Tensor, "batch_size n_layers d_features"]
    ) -> Float[torch.Tensor, "batch_size n_layers d_features"]:
        batch_size, n_layers, d_features = features.shape
        assert self.k <= d_features * n_layers

        features = features.reshape(-1, d_features * n_layers)
        topk_features = torch.zeros_like(features)
        topk_vals, topk_idxs = torch.topk(features, self.k, dim=-1, sorted=False)
        topk_features.scatter_(dim=-1, index=topk_idxs, src=topk_vals)
        topk_features = topk_features.reshape(batch_size, n_layers, d_features)
        return topk_features


class BatchTopK(torch.nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(
        self, features: Float[torch.Tensor, "batch_size n_layers d_features"]
    ) -> Float[torch.Tensor, "batch_size n_layers d_features"]:
        batch_size, n_layers, d_features = features.shape
        assert self.k <= d_features * n_layers * batch_size

        features = features.flatten()
        topk_features = torch.zeros_like(features)
        topk_vals, topk_idxs = torch.topk(features, self.k, dim=-1, sorted=False)
        topk_features.scatter_(dim=-1, index=topk_idxs, src=topk_vals)
        topk_features = topk_features.reshape(batch_size, n_layers, d_features)
        return topk_features


if __name__ == "__main__":
    topk = BatchTopK(k=3)
    features = torch.randn(10, 12, 768 * 8)
    topk_features = topk(features)
    print(topk_features.shape)

    topk = PerSampleTopK(k=3)
    features = torch.randn(10, 12, 768 * 8)
    topk_features = topk(features)
    print(topk_features.shape)

    topk = PerLayerTopK(k=3)
    features = torch.randn(10, 12, 768 * 8)
    topk_features = topk(features)
    print(topk_features.shape)
