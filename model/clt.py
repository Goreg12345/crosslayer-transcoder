from typing import Tuple, Union

import torch
import torch.nn as nn
from einops import einsum
from jaxtyping import Float

from model.standardize import Standardizer


class SimpleCrossLayerTranscoder(nn.Module):
    def __init__(
        self,
        nonlinearity: nn.Module,
        input_standardizer: Standardizer,
        output_standardizer: Standardizer,
        d_acts: int = 768,
        d_features: int = 6144,
        n_layers: int = 12,
        enc_init_scaler: float = 1.0,
        plt: bool = False,
        tied_init: bool = False,
    ):
        super().__init__()

        self.d_acts = d_acts
        self.d_features = d_features
        self.n_layers = n_layers
        self.nonlinearity = nonlinearity
        self.input_standardizer = input_standardizer
        self.output_standardizer = output_standardizer
        self.enc_init_scaler = enc_init_scaler
        self.tied_init = tied_init

        self.W_enc = nn.Parameter(torch.empty(n_layers, d_acts, d_features))
        self.W_dec = nn.Parameter(torch.empty(n_layers, n_layers, d_features, d_acts))
        if not plt:
            self.register_buffer("mask", torch.triu(torch.ones(n_layers, n_layers)))
        else:
            self.register_buffer("mask", torch.eye(n_layers, n_layers))

        self.reset_parameters()

    def reset_parameters(self):
        enc_uniform_thresh = 1 / (self.enc_init_scaler * self.d_features**0.5)
        self.W_enc.data.uniform_(-enc_uniform_thresh, enc_uniform_thresh)

        # rescale to have same norm
        # norm = self.W_enc.norm(p=2, dim=1)
        # self.W_enc.data = self.W_enc.data / norm.unsqueeze(1)

        dec_uniform_thresh = 1 / ((self.d_acts * self.n_layers) ** 0.5)
        self.W_dec.data.uniform_(-dec_uniform_thresh, dec_uniform_thresh)
        mask = self.mask.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.d_features, self.d_acts)
        self.W_dec.data = torch.where(mask.bool(), self.W_dec.data, 0.0)

        if self.tied_init:
            for layer1 in range(self.n_layers):
                for layer2 in range(self.n_layers):
                    self.W_dec.data[layer1, layer2, :, :] = self.W_enc[layer1].data.T

        # rescale to have same norm
        # norm = self.W_dec.norm(p=2, dim=-1)
        # self.W_dec.data = self.W_dec.data / norm.unsqueeze(-1)

    def initialize_standardizers(self, batch: Float[torch.Tensor, "batch_size io n_layers d_acts"]):
        self.input_standardizer.initialize_from_batch(batch)
        self.output_standardizer.initialize_from_batch(batch)

    def decode(
        self, features: Float[torch.Tensor, "batch_size n_layers d_features"]
    ) -> Float[torch.Tensor, "batch_size n_layers d_acts"]:
        return einsum(
            features,
            self.W_dec,
            self.mask,
            "batch_size from_layer d_features, from_layer to_layer d_features d_acts, "
            "from_layer to_layer -> batch_size to_layer d_acts",
        )

    def forward(self, acts: Float[torch.Tensor, "batch_size n_layers d_acts"]) -> Tuple[
        Float[torch.Tensor, "batch_size n_layers d_features"],
        Float[torch.Tensor, "batch_size n_layers d_features"],
        Float[torch.Tensor, "batch_size n_layers d_acts"],
        Float[torch.Tensor, "batch_size n_layers d_acts"],
    ]:
        acts = self.input_standardizer(acts)

        pre_actvs = einsum(
            acts,
            self.W_enc,
            "batch_size n_layers d_acts, n_layers d_acts d_features -> batch_size n_layers d_features",
        )

        features = self.nonlinearity(pre_actvs)
        recons_norm = self.decode(features)

        recons = self.output_standardizer(recons_norm)

        return pre_actvs, features, recons_norm, recons


class Encoder(nn.Module):
    def __init__(self, d_acts: int, d_features: int, n_layers: int):
        super().__init__()
        self.d_acts = d_acts
        self.d_features = d_features
        self.n_layers = n_layers
        self.W = nn.Parameter(torch.empty((n_layers, d_acts, d_features)))
        self.reset_parameters()

    def reset_parameters(self):
        enc_uniform_thresh = 1 / (self.d_features**0.5)
        self.W.data.uniform_(-enc_uniform_thresh, enc_uniform_thresh)

    @torch.no_grad()
    def forward_layer(
        self, acts_norm: Float[torch.Tensor, "batch_size seq d_acts"], layer: int
    ) -> Float[torch.Tensor, "batch_size seq d_features"]:
        return einsum(
            acts_norm,
            self.W[layer],
            "batch_size seq d_acts, d_acts d_features -> batch_size seq d_features",
        )

    def forward(
        self, acts_norm: Float[torch.Tensor, "batch_size n_layers d_acts"], layer: str = "all"
    ) -> Float[torch.Tensor, "batch_size n_layers d_features"]:
        # for inference
        if layer != "all":
            return self.forward_layer(acts_norm, layer)

        # for training
        return einsum(
            acts_norm,
            self.W,
            "batch_size n_layers d_acts, n_layers d_acts d_features -> batch_size n_layers d_features",
        )


class Decoder(nn.Module):
    def __init__(self, d_acts: int, d_features: int, n_layers: int):
        super().__init__()
        self.d_acts = d_acts
        self.d_features = d_features
        self.n_layers = n_layers
        for i in range(n_layers):
            self.register_parameter(f"W_{i}", nn.Parameter(torch.empty((i + 1, d_features, d_acts))))
        self.reset_parameters()

    def reset_parameters(self):
        dec_uniform_thresh = 1 / ((self.d_acts * self.n_layers) ** 0.5)
        for i in range(self.n_layers):
            self.get_parameter(f"W_{i}").data.uniform_(-dec_uniform_thresh, dec_uniform_thresh)

    @torch.no_grad()
    def forward_layer(
        self, features: Float[torch.Tensor, "batch_size seq from_layer d_features"], layer: int
    ) -> Float[torch.Tensor, "batch_size seq d_acts"]:
        return einsum(
            features,
            self.get_parameter(f"W_{layer}"),
            "batch_size seq from_layer d_features, from_layer d_features d_acts -> batch_size seq d_acts",
        )

    def forward(
        self, features: Float[torch.Tensor, "batch_size n_layers d_features"], layer: str = "all"
    ) -> Float[torch.Tensor, "batch_size n_layers d_acts"]:
        if layer != "all":
            return self.forward_layer(features, layer)

        recons = []
        for l in range(self.n_layers):
            W = self.get_parameter(f"W_{l}")
            selected_features = features[:, : l + 1]
            l_recons = einsum(
                selected_features,
                W,
                "batch_size n_layers d_features, n_layers d_features d_acts -> batch_size n_layers d_acts",
            )
            l_recons = l_recons.sum(dim=1)
            recons.append(l_recons)
        recons = torch.stack(recons, dim=1)
        return recons


class CrossLayerTranscoder(nn.Module):
    def __init__(
        self,
        nonlinearity: nn.Module,
        input_standardizer: Standardizer,
        output_standardizer: Standardizer,
        encoder: Encoder,
        decoder: Decoder,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.nonlinearity = nonlinearity
        self.input_standardizer = input_standardizer
        self.output_standardizer = output_standardizer

        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()

    def initialize_standardizers(self, batch: Float[torch.Tensor, "batch_size io n_layers d_acts"]):
        self.input_standardizer.initialize_from_batch(batch)
        self.output_standardizer.initialize_from_batch(batch)

    def forward(self, acts: Float[torch.Tensor, "batch_size n_layers d_acts"]) -> Tuple[
        Float[torch.Tensor, "batch_size n_layers d_features"],  # pre_actvs
        Float[torch.Tensor, "batch_size n_layers d_features"],  # features
        Float[torch.Tensor, "batch_size n_layers d_acts"],  # recons_norm
        Float[torch.Tensor, "batch_size n_layers d_acts"],  # recons
    ]:
        acts = self.input_standardizer(acts)

        pre_actvs = self.encoder(acts)

        features = self.nonlinearity(pre_actvs)

        recons_norm = self.decoder(features)

        recons = self.output_standardizer(recons_norm)

        return pre_actvs, features, recons_norm, recons
