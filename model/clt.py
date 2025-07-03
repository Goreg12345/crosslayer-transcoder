from typing import Tuple

import torch
import torch.nn as nn
from einops import einsum
from jaxtyping import Float


class CrossLayerTranscoder(nn.Module):
    def __init__(
        self,
        nonlinearity: nn.Module,
        d_acts: int = 768,
        d_features: int = 6144,
        n_layers: int = 12,
    ):
        super().__init__()

        self.d_acts = d_acts
        self.d_features = d_features
        self.n_layers = n_layers
        self.nonlinearity = nonlinearity

        self.W_enc = nn.Parameter(torch.empty(n_layers, d_acts, d_features))
        self.W_dec = nn.Parameter(torch.empty(n_layers, n_layers, d_features, d_acts))
        self.register_buffer("mask", torch.triu(torch.ones(n_layers, n_layers)))

        self.reset_parameters()

    def reset_parameters(self):
        enc_uniform_thresh = 1 / (self.d_features**0.5)
        self.W_enc.data.uniform_(-enc_uniform_thresh, enc_uniform_thresh)

        dec_uniform_thresh = 1 / ((self.d_acts * self.n_layers) ** 0.5)
        self.W_dec.data.uniform_(-dec_uniform_thresh, dec_uniform_thresh)

    def forward(
        self, acts_norm: Float[torch.Tensor, "batch_size n_layers d_acts"]
    ) -> Tuple[
        Float[torch.Tensor, "batch_size n_layers d_features"],
        Float[torch.Tensor, "batch_size n_layers d_acts"],
    ]:
        features = einsum(
            acts_norm,
            self.W_enc,
            "batch_size n_layers d_acts, n_layers d_acts d_features -> batch_size n_layers d_features",
        )

        features = self.nonlinearity(features)
        recons = einsum(
            features,
            self.W_dec,
            self.mask,
            "batch_size from_layer d_features, from_layer to_layer d_features d_acts, "
            "from_layer to_layer -> batch_size to_layer d_acts",
        )

        return features, recons
