import copy

import einops
import torch
import torch.nn as nn
from jaxtyping import Float


class Molt(nn.Module):
    def __init__(
        self,
        d_acts: int,
        N: int,
        nonlinearity: nn.Module,
        input_standardizer: nn.Module,
        output_standardizer: nn.Module,
        ranks: list[int] = [512, 256, 128, 64, 32],
    ):
        super().__init__()

        self.d_acts = d_acts
        self.nonlinearity = nonlinearity
        self.input_standardizer = input_standardizer
        self.output_standardizer = output_standardizer
        Us = []
        Vs = []
        rank_multiplier = 1
        n_features = 0
        d_latents = 0
        for rank in ranks:
            Us.append(nn.Parameter(torch.empty(N * rank_multiplier, rank, d_acts)))
            Vs.append(nn.Parameter(torch.empty(N * rank_multiplier, d_acts, rank)))
            n_features += N * rank_multiplier
            d_latents += N * rank_multiplier * rank
            rank_multiplier *= 2
        self.n_features = n_features
        self.e = nn.Linear(d_acts, n_features)
        self.Us = nn.ParameterList(Us)
        self.Vs = nn.ParameterList(Vs)

        print(f"d_latents (transcoder equivalent): {d_latents}")
        self.d_latents = d_latents

        self.reset_parameters()

    def reset_parameters(self):
        for U in self.Us:
            nn.init.xavier_uniform_(U)
        for V in self.Vs:
            nn.init.xavier_uniform_(V)

    def transform_norm(self):
        norms = []
        for U, V in zip(self.Us, self.Vs):
            uv = einops.einsum(
                U,
                V,
                "n_transforms d_transform d_acts_out, n_transforms d_acts_in d_transform -> n_transforms d_acts_in d_acts_out",
            )
            norms.append(torch.norm(uv, dim=(1, 2)))
        return torch.cat(norms, dim=0)

    def forward(
        self, acts: Float[torch.Tensor, "batch_size d_acts"], layer: int
    ) -> Float[torch.Tensor, "batch_size d_acts"]:
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
        return gate, recons_norm, recons

    def initialize_standardizers(self, batch: Float[torch.Tensor, "batch_size io n_layers d_acts"]):
        self.input_standardizer.initialize_from_batch(batch)
        self.output_standardizer.initialize_from_batch(batch)


class MultiLayerMolt(nn.Module):
    """A stack of `n_layers` single-layer `Molt` models with shared standardizers.

    Each layer owns its encoder, low-rank transforms (Us, Vs), and nonlinearity
    parameters. The `nonlinearity` argument is a template that is deep-copied
    once per layer, so each layer trains independent JumpReLU thetas. The
    standardizers are shared across layers — they already carry per-layer
    mean/std for all `n_layers`, so duplicating them per inner Molt would be
    wasteful and inconsistent.
    """

    def __init__(
        self,
        n_layers: int,
        d_acts: int,
        N: int,
        nonlinearity: nn.Module,
        input_standardizer: nn.Module,
        output_standardizer: nn.Module,
        ranks: list[int] = [512, 256, 128, 64, 32],
    ):
        super().__init__()
        self.n_layers = n_layers
        self.input_standardizer = input_standardizer
        self.output_standardizer = output_standardizer
        self.molts = nn.ModuleList(
            [
                Molt(
                    d_acts=d_acts,
                    N=N,
                    nonlinearity=copy.deepcopy(nonlinearity),
                    input_standardizer=input_standardizer,
                    output_standardizer=output_standardizer,
                    ranks=ranks,
                )
                for _ in range(n_layers)
            ]
        )
        self.n_features = self.molts[0].n_features
        self.d_latents = self.molts[0].d_latents

    def initialize_standardizers(
        self, batch: Float[torch.Tensor, "batch_size io n_layers d_acts"]
    ):
        self.input_standardizer.initialize_from_batch(batch)
        self.output_standardizer.initialize_from_batch(batch)

    def transform_norm(self, layer: int) -> Float[torch.Tensor, "n_features"]:
        return self.molts[layer].transform_norm()

    def forward(
        self, acts: Float[torch.Tensor, "batch n_layers d_acts"]
    ) -> tuple[
        Float[torch.Tensor, "batch n_layers n_features"],
        Float[torch.Tensor, "batch n_layers d_acts"],
        Float[torch.Tensor, "batch n_layers d_acts"],
    ]:
        gates, recons_norms, reconss = [], [], []
        for layer in range(self.n_layers):
            gate, recons_norm, recons = self.molts[layer](acts[:, layer], layer=layer)
            gates.append(gate)
            recons_norms.append(recons_norm)
            reconss.append(recons)
        return (
            torch.stack(gates, dim=1),
            torch.stack(recons_norms, dim=1),
            torch.stack(reconss, dim=1),
        )
