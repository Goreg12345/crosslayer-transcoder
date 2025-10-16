from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import yaml
from einops import einsum
from jaxtyping import Float
from safetensors.torch import save_file

from crosslayer_transcoder.model.serializable_module import SerializableModule
from crosslayer_transcoder.model.standardize import Standardizer


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
        mask = (
            self.mask.unsqueeze(-1)
            .unsqueeze(-1)
            .repeat(1, 1, self.d_features, self.d_acts)
        )
        self.W_dec.data = torch.where(mask.bool(), self.W_dec.data, 0.0)

        if self.tied_init:
            for layer1 in range(self.n_layers):
                for layer2 in range(self.n_layers):
                    self.W_dec.data[layer1, layer2, :, :] = self.W_enc[layer1].data.T

        # rescale to have same norm
        # norm = self.W_dec.norm(p=2, dim=-1)
        # self.W_dec.data = self.W_dec.data / norm.unsqueeze(-1)

    def initialize_standardizers(
        self, batch: Float[torch.Tensor, "batch_size io n_layers d_acts"]
    ):
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

    def forward(
        self, acts: Float[torch.Tensor, "batch_size n_layers d_acts"]
    ) -> Tuple[
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


class Encoder(SerializableModule):
    def __init__(self, d_acts: int, d_features: int, n_layers: int):
        super().__init__()
        self.d_acts = d_acts
        self.d_features = d_features
        self.n_layers = n_layers
        self.W = nn.Parameter(torch.empty((n_layers, d_acts, d_features)))
        self.bias = True
        self._is_folded = False
        if self.bias:
            self.b = nn.Parameter(torch.empty((n_layers, d_features)))
        self.reset_parameters()

    def reset_parameters(self):
        enc_uniform_thresh = 1 / (self.d_features**0.5)
        self.W.data.uniform_(-enc_uniform_thresh, enc_uniform_thresh)
        if self.bias:
            self.b.data.zero_()

    @torch.no_grad()
    def forward_layer(
        self, acts_norm: Float[torch.Tensor, "batch_size seq d_acts"], layer: int
    ) -> Float[torch.Tensor, "batch_size seq d_features"]:
        pre_actvs = einsum(
            acts_norm,
            self.W[layer],
            "batch_size seq d_acts, d_acts d_features -> batch_size seq d_features",
        )
        if self.bias:
            pre_actvs = pre_actvs + self.b[layer]
        return pre_actvs

    def forward(
        self,
        acts_norm: Float[torch.Tensor, "batch_size n_layers d_acts"],
        layer: str = "all",
    ) -> Float[torch.Tensor, "batch_size n_layers d_features"]:
        # for inference
        if layer != "all":
            return self.forward_layer(acts_norm, layer)

        # for training
        pre_actvs = einsum(
            acts_norm,
            self.W,
            "batch_size n_layers d_acts, n_layers d_acts d_features -> batch_size n_layers d_features",
        )
        pre_actvs = pre_actvs.contiguous()

        if self.bias:
            pre_actvs = pre_actvs + self.b.to(acts_norm.dtype)

        return pre_actvs

    def fold(self, input_standardizer: Standardizer):
        """In-place folding of the encoder weights. If the encoder is already folded, return self."""
        if self._is_folded:
            return self

        self.W.data = self.W / input_standardizer.std.unsqueeze(-1)

        self.b.data = self.b - (
            einsum(
                input_standardizer.mean,
                self.W,
                "n_layers d_acts, n_layers d_acts d_features -> n_layers d_features",
            )
        )

        self._is_folded = True
        return self

    def to_config(self) -> Dict[str, Any]:
        return {
            "class_path": self.__class__.__module__ + "." + self.__class__.__name__,
            "init_args": {
                "d_acts": self.d_acts,
                "d_features": self.d_features,
                "n_layers": self.n_layers,
            },
        }


class Decoder(SerializableModule):
    def __init__(self, d_acts: int, d_features: int, n_layers: int):
        super().__init__()
        self.d_acts = d_acts
        self.d_features = d_features
        self.n_layers = n_layers
        self.register_parameter(
            f"W", nn.Parameter(torch.empty((n_layers, d_features, d_acts)))
        )
        self.register_parameter(f"b", nn.Parameter(torch.empty((n_layers, d_acts))))
        self._is_folded = False
        self.reset_parameters()

    def reset_parameters(self):
        dec_uniform_thresh = 1 / ((self.d_acts * self.n_layers) ** 0.5)
        self.get_parameter(f"W").data.uniform_(-dec_uniform_thresh, dec_uniform_thresh)
        self.get_parameter(f"b").data.zero_()

    @torch.no_grad()
    def forward_layer(
        self,
        features: Float[torch.Tensor, "batch_size seq from_layer d_features"],
        layer: int,
    ) -> Float[torch.Tensor, "batch_size seq d_acts"]:
        if features.ndim == 4:  # (batch, seq, layer, d_features)
            features = features[:, :, layer, :]
        return (
            einsum(
                features,
                self.get_parameter(f"W")[layer],
                "batch_size seq d_features, d_features d_acts -> batch_size seq d_acts",
            )
            + self.b[layer]
        )

    def forward(
        self,
        features: Float[torch.Tensor, "batch_size n_layers d_features"],
        layer: str = "all",
    ) -> Float[torch.Tensor, "batch_size n_layers d_acts"]:
        if layer != "all":
            return self.forward_layer(features, layer)

        recons = (
            einsum(
                features,
                self.W,
                "batch_size n_layers d_features, n_layers d_features d_acts -> batch_size n_layers d_acts",
            )
            + self.b
        )
        return recons

    def fold(self, output_standardizer: Standardizer):
        """In-place folding of the decoder weights. If the decoder is already folded, return self."""
        if self._is_folded:
            return self

        self.W.data = einsum(
            self.W,
            output_standardizer.std,
            "n_layers d_features d_acts, n_layers d_acts -> n_layers d_features d_acts",
        )

        self.b.data = self.b * output_standardizer.std + output_standardizer.mean

        self._is_folded = True

        return self

    def to_config(self) -> Dict[str, Any]:
        return {
            "class_path": self.__class__.__module__ + "." + self.__class__.__name__,
            "init_args": {
                "d_acts": self.d_acts,
                "d_features": self.d_features,
                "n_layers": self.n_layers,
            },
        }


class CrosslayerDecoder(SerializableModule):
    def __init__(self, d_acts: int, d_features: int, n_layers: int):
        super().__init__()
        self.d_acts = d_acts
        self.d_features = d_features
        self.n_layers = n_layers
        for i in range(n_layers):
            self.register_parameter(
                f"W_{i}", nn.Parameter(torch.empty((i + 1, d_features, d_acts)))
            )
        self._is_folded = False
        self.register_parameter(f"b", nn.Parameter(torch.empty((n_layers, d_acts))))
        self.reset_parameters()

    def reset_parameters(self):
        dec_uniform_thresh = 1 / ((self.d_acts * self.n_layers) ** 0.5)
        for i in range(self.n_layers):
            self.get_parameter(f"W_{i}").data.uniform_(
                -dec_uniform_thresh, dec_uniform_thresh
            )
            # for l in range(i):
            #    self.get_parameter(f"W_{i}").data[l, :, :] = self.get_parameter(f"W_{l}").data[l, :, :] * 0.0

        self.b.data.zero_()

    @torch.no_grad()
    def forward_layer(
        self,
        features: Float[torch.Tensor, "batch_size seq from_layer d_features"],
        layer: int,
    ) -> Float[torch.Tensor, "batch_size seq d_acts"]:
        return (
            einsum(
                features,
                self.get_parameter(f"W_{layer}"),
                "batch_size seq from_layer d_features, from_layer d_features d_acts -> batch_size seq d_acts",
            )
            + self.b[layer]
        )

    def forward(
        self,
        features: Float[torch.Tensor, "batch_size n_layers d_features"],
        layer: str = "all",
    ) -> Float[torch.Tensor, "batch_size n_layers d_acts"]:
        if layer != "all":
            return self.forward_layer(features, layer)

        recons = torch.empty(
            features.shape[0],
            self.n_layers,
            self.d_acts,
            device=features.device,
            dtype=features.dtype,
        )
        for l in range(self.n_layers):
            W = self.get_parameter(f"W_{l}")
            selected_features = features[:, : l + 1]
            l_recons = einsum(
                selected_features,
                W,
                "batch_size n_layers d_features, n_layers d_features d_acts -> batch_size d_acts",
            )
            recons[:, l, :] = l_recons
        recons = recons + self.b.to(features.dtype)
        return recons

    def fold(self, output_standardizer: Standardizer):
        """In-place folding of the decoder weights. If the decoder is already folded, return self."""
        if self._is_folded:
            return self

        self.b.data = self.b * output_standardizer.std + output_standardizer.mean
        for layer in range(self.n_layers):
            std = output_standardizer.std[layer]
            self.get_parameter(f"W_{layer}").data = einsum(
                self.get_parameter(f"W_{layer}"),
                std,
                "n_layers d_features d_acts, d_acts -> n_layers d_features d_acts",
            )
        self._is_folded = True
        return self

    def to_config(self) -> Dict[str, Any]:
        return {
            "class_path": self.__class__.__module__ + "." + self.__class__.__name__,
            "init_args": {
                "d_acts": self.d_acts,
                "d_features": self.d_features,
                "n_layers": self.n_layers,
            },
        }


class MatryoshkaCrosslayerDecoder(SerializableModule):
    def __init__(self, d_acts: int, d_features: int, n_layers: int, group_indices: List[int]):
        super().__init__()
        self.d_acts = d_acts
        self.d_features = d_features
        self.n_layers = n_layers

        assert group_indices[0] == 0
        assert group_indices[-1] == d_features
        self.group_indices = group_indices
        self._is_folded = False

        for i in range(n_layers):
            self.register_parameter(f"W_{i}", nn.Parameter(torch.empty((i + 1, d_features, d_acts))))
        self.register_parameter(f"b", nn.Parameter(torch.empty((n_layers, d_acts))))
        self.reset_parameters()

    def reset_parameters(self):
        dec_uniform_thresh = 1 / ((self.d_acts * self.n_layers) ** 0.5)
        for i in range(self.n_layers):
            self.get_parameter(f"W_{i}").data.uniform_(-dec_uniform_thresh, dec_uniform_thresh)
            # for l in range(i):
            #    self.get_parameter(f"W_{i}").data[l, :, :] = self.get_parameter(f"W_{l}").data[l, :, :] * 0.0

        self.b.data.zero_()

    @torch.no_grad()
    def forward_layer(
        self, features: Float[torch.Tensor, "batch_size seq from_layer d_features"], layer: int
    ) -> Float[torch.Tensor, "batch_size seq d_acts"]:
        return (
            einsum(
                features,
                self.get_parameter(f"W_{layer}"),
                "batch_size seq from_layer d_features, from_layer d_features d_acts -> batch_size seq d_acts",
            )
            + self.b[layer]
        )

    def forward(
        self, features: Float[torch.Tensor, "batch_size n_layers d_features"], layer: str = "all"
    ) -> Float[torch.Tensor, "batch_size n_layers d_acts"]:
        if layer != "all":
            return self.forward_layer(features, layer)

        # (batch, n_groups, n_layers, d_acts)
        recons = torch.zeros(
            features.shape[0],
            len(self.group_indices) - 1,
            self.n_layers,
            self.d_acts,
            device=features.device,
            dtype=features.dtype,
        )
        # initialize first group with decoder bias
        recons[:, 0, :, :] = self.b.unsqueeze(0)
        for l in range(self.n_layers):
            W = self.get_parameter(f"W_{l}")
            for group_idx, (start_idx, end_idx) in enumerate(
                zip(self.group_indices[:-1], self.group_indices[1:])
            ):
                group_features = features[..., : l + 1, start_idx:end_idx]
                group_decoder_W = W[:, start_idx:end_idx, :]
                group_recons = einsum(
                    group_features,
                    group_decoder_W,
                    "batch layer d_group_features, layer d_group_features d_acts -> batch d_acts",
                )
                if group_idx == 0:
                    recons[:, group_idx, l, :] = group_recons  # + recons[:, group_idx, l, :]
                else:
                    recons[:, group_idx, l, :] = recons[:, group_idx - 1, l, :] + group_recons
        return recons

    def fold(self, output_standardizer: Standardizer):
        """In-place folding of the decoder weights. If the decoder is already folded, return self."""
        if self._is_folded:
            return self

        self.b.data = self.b * output_standardizer.std + output_standardizer.mean
        for layer in range(self.n_layers):
            std = output_standardizer.std[layer]
            self.get_parameter(f"W_{layer}").data = einsum(
                self.get_parameter(f"W_{layer}"),
                std,
                "n_layers d_features d_acts, d_acts -> n_layers d_features d_acts",
            )
        self._is_folded = True
        return self

    def to_config(self) -> Dict[str, Any]:
        return {
            "class_path": self.__class__.__module__ + "." + self.__class__.__name__,
            "init_args": {
                "d_acts": self.d_acts,
                "d_features": self.d_features,
                "n_layers": self.n_layers,
                "group_indices": self.group_indices,
            },
        }


class CrossLayerTranscoder(SerializableModule):
    def __init__(
        self,
        nonlinearity: SerializableModule,
        encoder: Encoder,
        decoder: Union[Decoder, CrosslayerDecoder, MatryoshkaCrosslayerDecoder],
        input_standardizer: Optional[Standardizer] = None,
        output_standardizer: Optional[Standardizer] = None,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.nonlinearity = nonlinearity
        self.input_standardizer = input_standardizer
        self.output_standardizer = output_standardizer
        self._is_folded = False

        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()

    def initialize_standardizers(
        self, batch: Float[torch.Tensor, "batch_size io n_layers d_acts"]
    ):
        self.input_standardizer.initialize_from_batch(batch)
        self.output_standardizer.initialize_from_batch(batch)

    def forward(
        self, acts: Float[torch.Tensor, "batch_size n_layers d_acts"]
    ) -> Tuple[
        Float[torch.Tensor, "batch_size n_layers d_features"],  # pre_actvs
        Float[torch.Tensor, "batch_size n_layers d_features"],  # features
        Float[torch.Tensor, "batch_size n_layers d_acts"],  # recons_norm
        Float[torch.Tensor, "batch_size n_layers d_acts"],  # recons
    ]:
        if self.input_standardizer is not None:
            acts = self.input_standardizer(acts)

        pre_actvs = self.encoder(acts)

        features = self.nonlinearity(pre_actvs)

        recons_norm = self.decoder(features)

        if self.output_standardizer is not None:
            recons = self.output_standardizer(recons_norm)
        else:
            recons = recons_norm

        return pre_actvs, features, recons_norm, recons

    def to_config(self) -> Dict[str, Any]:
        return {
            "class_path": self.__class__.__module__ + "." + self.__class__.__name__,
            "init_args": {
                "nonlinearity": self.nonlinearity.to_config(),
                "encoder": self.encoder.to_config(),
                "decoder": self.decoder.to_config(),
                "input_standardizer": self.input_standardizer.to_config()
                if self.input_standardizer is not None
                else None,
                "output_standardizer": self.output_standardizer.to_config()
                if self.output_standardizer is not None
                else None,
            },
        }

    def fold(self):
        if self._is_folded:
            return

        if self.input_standardizer is not None:
            self.encoder.fold(self.input_standardizer)
        if self.output_standardizer is not None:
            self.decoder.fold(self.output_standardizer)

        self._is_folded = True
        self.input_standardizer = None
        self.output_standardizer = None

    def save_pretrained(self, directory: Path, fold_standardizers: bool = True):
        directory.mkdir(parents=True, exist_ok=True)

        if fold_standardizers:
            self.fold()

        config = self.to_config()
        config["is_folded"] = self._is_folded
        with open(directory / "config.yaml", "w") as f:
            yaml.dump({"model": config}, f)

        save_file(self.state_dict(), directory / "checkpoint.safetensors")
