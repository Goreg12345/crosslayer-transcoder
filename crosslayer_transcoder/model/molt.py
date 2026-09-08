from typing import Any, Dict, Sequence

import einops
import torch
from jaxtyping import Float

from crosslayer_transcoder.model.serializable_module import SerializableModule
from crosslayer_transcoder.model.standardize import Standardizer


class Molt(SerializableModule):
    def __init__(
        self,
        d_acts: int,
        N: int,
        nonlinearity: SerializableModule,
        input_standardizer: Standardizer,
        output_standardizer: Standardizer,
        ranks: Sequence[int] = (512, 256, 128, 64, 32),
    ):
        super().__init__()

        self.d_acts = d_acts
        self.N = N
        self.ranks = list(ranks)
        self.nonlinearity = nonlinearity
        self.input_standardizer = input_standardizer
        self.output_standardizer = output_standardizer
        Us = []
        Vs = []
        rank_multiplier = 1
        n_features = 0
        d_latents = 0
        for rank in self.ranks:
            Us.append(torch.nn.Parameter(torch.empty(N * rank_multiplier, rank, d_acts)))
            Vs.append(torch.nn.Parameter(torch.empty(N * rank_multiplier, d_acts, rank)))
            n_features += N * rank_multiplier
            d_latents += N * rank_multiplier * rank
            rank_multiplier *= 2
        self.n_features = n_features
        self.e = torch.nn.Linear(d_acts, n_features)
        self.Us = torch.nn.ParameterList(Us)
        self.Vs = torch.nn.ParameterList(Vs)

        self.d_latents = d_latents

        self.reset_parameters()

    def reset_parameters(self):
        for U in self.Us:
            torch.nn.init.xavier_uniform_(U)
        for V in self.Vs:
            torch.nn.init.xavier_uniform_(V)

    def transform_norm(self):
        norms = []
        for U, V in zip(self.Us, self.Vs):
            # ||VU||_F^2 = tr((V^T V)(U U^T)). Computing the rank x rank
            # Gram matrices is exactly equivalent to materializing every
            # d_acts x d_acts transform, but uses dramatically less memory.
            vtv = einops.einsum(
                V,
                V,
                "n d_acts rank_left, n d_acts rank_right -> n rank_left rank_right",
            )
            uut = einops.einsum(
                U,
                U,
                "n rank_left d_acts, n rank_right d_acts -> n rank_left rank_right",
            )
            squared_norm = (vtv * uut).sum(dim=(1, 2))
            norms.append(squared_norm.clamp_min(0).sqrt())
        return torch.cat(norms, dim=0)

    def forward(
        self, acts: Float[torch.Tensor, "batch_size d_acts"], layer: int
    ) -> Float[torch.Tensor, "batch_size d_acts"]:
        _, gate, recons_norm, recons = self.forward_with_pre_activations(acts, layer)
        return gate, recons_norm, recons

    def forward_with_pre_activations(
        self, acts: Float[torch.Tensor, "batch_size d_acts"], layer: int
    ):
        """Run MOLT while retaining encoder pre-activations for auxiliary losses."""
        acts = self.input_standardizer(acts, layer)
        pre_actvs = self.e(acts)
        gate = self.nonlinearity(pre_actvs)  # (batch, n_transforms)

        raw_recons = []
        for U, V in zip(self.Us, self.Vs):
            latents = einops.einsum(
                acts,
                V,
                "batch d_acts, n_transforms d_acts d_transform -> batch n_transforms d_transform",
            )
            raw_recons.append(
                einops.einsum(
                    latents,
                    U,
                    "batch n_transforms d_transform, n_transforms d_transform d_acts -> batch n_transforms d_acts",
                )
            )

        raw_recons = torch.cat(raw_recons, dim=1)

        weighted_recons = gate.unsqueeze(-1) * raw_recons
        recons_norm = weighted_recons.sum(dim=1)

        recons = self.output_standardizer(recons_norm, layer)
        return pre_actvs, gate, recons_norm, recons

    def initialize_standardizers(
        self, batch: Float[torch.Tensor, "batch_size io n_layers d_acts"]
    ):
        self.input_standardizer.initialize_from_batch(batch)
        self.output_standardizer.initialize_from_batch(batch)

    def to_config(self) -> Dict[str, Any]:
        return {
            "class_path": self.__class__.__module__ + "." + self.__class__.__name__,
            "init_args": {
                "d_acts": self.d_acts,
                "N": self.N,
                "ranks": self.ranks,
                "nonlinearity": self.nonlinearity.to_config(),
                "input_standardizer": self.input_standardizer.to_config(),
                "output_standardizer": self.output_standardizer.to_config(),
            },
        }
