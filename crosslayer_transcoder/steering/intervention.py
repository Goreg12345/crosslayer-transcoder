"""TransformIntervention — compute a single transform's contribution at a layer.

Pulls the U_i, V_i factors and the layer's input/output standardizer slices off
of a loaded `Molt`, freezes them as buffers, and exposes a forward that maps a
batch of pre-LN-MLP residuals to the additive perturbation that "transform i
firing at gate=alpha" would produce in real (unstandardized) MLP-out space.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Float

from crosslayer_transcoder.model.molt import Molt


def _resolve_index(molt: Molt, transform_index: int) -> tuple[int, int]:
    """Map a global transform index to (tier, local_index_within_tier)."""
    cumulative = 0
    for t, U in enumerate(molt.Us):
        n_in_tier = U.shape[0]
        if transform_index < cumulative + n_in_tier:
            return t, transform_index - cumulative
        cumulative += n_in_tier
    raise IndexError(
        f"transform_index {transform_index} out of range "
        f"(MoLT has {cumulative} transforms total)"
    )


class TransformIntervention(nn.Module):
    """A frozen, per-layer steering operator for a single MoLT transform.

    Forward takes pre-LN-MLP residuals and returns the perturbation to add to
    the MLP block's output. The mapping is

        delta = alpha * (acts_std @ V_i @ U_i) * out_std[L]
        acts_std = (mlp_in - in_mean[L]) / in_std[L]

    which is the contribution transform `i` would make to MLP-out if its gate
    were forced to `alpha` and every other transform were silenced. The mean
    offset of the output standardizer is intentionally dropped — this is an
    additive perturbation, not a replacement.
    """

    def __init__(
        self,
        molt: Molt,
        transform_index: int,
        layer: int,
        alpha: float,
    ):
        super().__init__()
        if not molt.input_standardizer.is_initialized:
            raise ValueError(
                "MoLT input_standardizer is not initialized — load from a "
                "trained checkpoint, or initialize with a real activation batch."
            )
        if not molt.output_standardizer.is_initialized:
            raise ValueError("MoLT output_standardizer is not initialized.")

        n_layers = molt.input_standardizer.mean.shape[0]
        if not (0 <= layer < n_layers):
            raise IndexError(
                f"layer={layer} out of range for MoLT with {n_layers} standardizer slices"
            )

        tier, local = _resolve_index(molt, transform_index)
        # Us[t]: (N*2^t, rank, d_acts)  ;  Vs[t]: (N*2^t, d_acts, rank)
        U_i = molt.Us[tier][local].detach().clone()  # (rank, d_acts)
        V_i = molt.Vs[tier][local].detach().clone()  # (d_acts, rank)
        self.register_buffer("U", U_i)
        self.register_buffer("V", V_i)

        self.register_buffer(
            "in_mean", molt.input_standardizer.mean[layer].detach().clone()
        )
        self.register_buffer(
            "in_std", molt.input_standardizer.std[layer].detach().clone()
        )
        self.register_buffer(
            "out_std", molt.output_standardizer.std[layer].detach().clone()
        )

        self.alpha = float(alpha)
        self.transform_index = transform_index
        self.layer = layer
        self.tier = tier
        self.local_index = local
        self.rank = U_i.shape[0]

    def forward(
        self,
        mlp_in: Float[torch.Tensor, "... d_acts"],
    ) -> Float[torch.Tensor, "... d_acts"]:
        # Cast factors to mlp_in's dtype so the intervention plays nicely with
        # bf16/fp16 generation. Buffers stay fp32 on disk.
        dtype = mlp_in.dtype
        acts_std = (mlp_in - self.in_mean.to(dtype)) / self.in_std.to(dtype)
        latent = acts_std @ self.V.to(dtype)  # (..., rank)
        raw = latent @ self.U.to(dtype)  # (..., d_acts)
        delta = (self.alpha * raw) * self.out_std.to(dtype)
        return delta
