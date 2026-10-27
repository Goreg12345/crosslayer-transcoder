"""Jacobian-based mechanistic faithfulness metrics.

The correlation used here is the cosine similarity between flattened Jacobian
matrices, computed separately for each datapoint and then averaged.
"""

from collections.abc import Callable

import torch
from torchmetrics import Metric

from crosslayer_transcoder.model.molt import Molt


def flattened_jacobian_cosine(
    replacement_jacobian: torch.Tensor,
    target_jacobian: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return cosine similarity along the final two (Jacobian) dimensions."""
    if replacement_jacobian.shape != target_jacobian.shape:
        raise ValueError(
            "Jacobian shapes differ: "
            f"{tuple(replacement_jacobian.shape)} != {tuple(target_jacobian.shape)}"
        )
    replacement = replacement_jacobian.float().flatten(start_dim=-2)
    target = target_jacobian.float().flatten(start_dim=-2)
    numerator = (replacement * target).sum(dim=-1)
    denominator = replacement.norm(dim=-1) * target.norm(dim=-1)
    return numerator / denominator.clamp_min(eps)


@torch.no_grad()
def molt_jacobian(molt: Molt, inputs: torch.Tensor, layer: int) -> torch.Tensor:
    """Compute the exact, almost-everywhere Jacobian of a MOLT.

    JumpReLU gates are differentiated as the represented function: inactive
    gates have derivative zero and active gates have derivative one. This
    deliberately does not use JumpReLU's straight-through training gradient.

    Args:
        molt: MOLT to differentiate.
        inputs: Tensor of shape ``[batch, d_acts]`` in the original activation
            coordinates.
        layer: Standardizer layer index.

    Returns:
        Tensor of shape ``[batch, d_acts, d_acts]``, with dimensions ordered as
        ``[batch, output, input]``.
    """
    if inputs.ndim != 2 or inputs.shape[-1] != molt.d_acts:
        raise ValueError(
            f"Expected inputs of shape [batch, {molt.d_acts}], got {tuple(inputs.shape)}"
        )

    compute_dtype = torch.float32
    x = molt.input_standardizer(inputs, layer).to(compute_dtype)
    encoder_weight = molt.e.weight.to(compute_dtype)
    pre_activations = x @ encoder_weight.T + molt.e.bias.to(compute_dtype)
    gates = molt.nonlinearity(pre_activations).to(compute_dtype)
    active = gates != 0

    batch_size, d_acts = inputs.shape
    jacobian = torch.zeros(
        batch_size, d_acts, d_acts, device=inputs.device, dtype=compute_dtype
    )
    feature_offset = 0
    for U_group, V_group in zip(molt.Us, molt.Vs):
        n_transforms = U_group.shape[0]
        group_active = active[:, feature_offset : feature_offset + n_transforms]
        group_gates = gates[:, feature_offset : feature_offset + n_transforms]
        group_encoder = encoder_weight[
            feature_offset : feature_offset + n_transforms
        ]
        U_group = U_group.to(compute_dtype)
        V_group = V_group.to(compute_dtype)

        # Sparse evaluation matters for trained MOLTs: work scales with L0, not
        # the total number of transforms.
        for batch_index in range(batch_size):
            indices = group_active[batch_index].nonzero(as_tuple=True)[0]
            if indices.numel() == 0:
                continue
            U = U_group[indices]
            V = V_group[indices]
            transforms = torch.bmm(V, U)  # [active, input, output]
            raw_reconstructions = torch.einsum(
                "d,ndo->no", x[batch_index], transforms
            )
            # d(g_i A_i x)/dx = (A_i x) e_i^T + g_i A_i^T.
            jacobian[batch_index] += torch.einsum(
                "no,ni->oi", raw_reconstructions, group_encoder[indices]
            )
            jacobian[batch_index] += torch.einsum(
                "n,nio->oi", group_gates[batch_index, indices], transforms
            )
        feature_offset += n_transforms

    input_std = molt.input_standardizer.std[layer].float()
    output_std = molt.output_standardizer.std[layer].float()
    return jacobian * output_std[None, :, None] / input_std[None, None, :]


def module_jacobian(
    function: Callable[[torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
    chunk_size: int | None = 64,
    output_device: torch.device | str | None = None,
) -> torch.Tensor:
    """Compute per-example Jacobians for a function mapping ``[d]`` to ``[d]``."""
    jacobian_fn = torch.func.jacrev(function, chunk_size=chunk_size)
    jacobians = []
    for point in inputs:
        jacobian = jacobian_fn(point).detach()
        if output_device is not None:
            jacobian = jacobian.to(output_device)
        jacobians.append(jacobian)
    return torch.stack(jacobians)


class JacobianCorrelation(Metric):
    """Mean per-datapoint cosine similarity of flattened Jacobians."""

    full_state_update = False

    def __init__(self) -> None:
        super().__init__()
        self.add_state(
            "correlation_sum", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("num_datapoints", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self, replacement_jacobian: torch.Tensor, target_jacobian: torch.Tensor
    ) -> None:
        correlations = flattened_jacobian_cosine(
            replacement_jacobian, target_jacobian
        )
        self.correlation_sum += correlations.sum().to(self.correlation_sum.device)
        self.num_datapoints += correlations.numel()

    def compute(self) -> torch.Tensor:
        if self.num_datapoints == 0:
            return torch.tensor(float("nan"), device=self.correlation_sum.device)
        return self.correlation_sum / self.num_datapoints
