"""Load a trained MoLT model from a Lightning checkpoint.

Architecture (d_acts, n_features, ranks, N, n_layers) is inferred from the
state_dict tensor shapes, so we don't need the original training YAML. The
nonlinearity is assumed to be `JumpReLU` and the standardizers are assumed to
be `Dimensionwise{Input,Output}Standardizer` — that's what every MoLT config
in this repo uses today. If a future MoLT variant swaps either of those, this
loader needs to grow a switch on the saved hparams.

Usage:
    # From the Hugging Face hub (default repo: kylelovesllms/molt-sweeps):
    molt, meta = load_molt_from_hf("gpt2-molt-lam-0_00015-50M.ckpt")

    # From a local file:
    molt, meta = load_molt("checkpoints/lam_0_00015_50M/clt.ckpt")

    # molt is in eval() with grads disabled. meta carries n_features, ranks,
    # tier->index mapping, base_model_name, training dataset, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
from huggingface_hub import hf_hub_download

from crosslayer_transcoder.model.jumprelu import JumpReLU
from crosslayer_transcoder.model.molt import Molt
from crosslayer_transcoder.model.standardize import (
    DimensionwiseInputStandardizer,
    DimensionwiseOutputStandardizer,
)

DEFAULT_HF_REPO = "kylelovesllms/molt-sweeps"


@dataclass
class MoltCheckpointMetadata:
    """Everything the dashboard needs that isn't the model weights themselves."""

    ckpt_path: str
    d_acts: int
    n_features: int
    n_layers: int
    ranks: list[int]
    N: int
    # Maps each feature index -> (tier_idx, rank). Tier 0 is the lowest-rank-multiplier
    # tier (N transforms), tier 1 has 2N transforms, etc.
    feature_tier: list[int] = field(default_factory=list)
    feature_rank: list[int] = field(default_factory=list)
    # Pulled from datamodule_hyper_parameters in the checkpoint, when present.
    base_model_name: Optional[str] = None
    training_dataset: Optional[str] = None
    # Lightning bookkeeping
    global_step: Optional[int] = None
    epoch: Optional[int] = None
    # If the checkpoint was pulled from HF, where it came from.
    hf_repo_id: Optional[str] = None
    hf_filename: Optional[str] = None
    hf_revision: Optional[str] = None


def _shape(sd: dict[str, torch.Tensor], key: str) -> tuple[int, ...]:
    if key not in sd:
        raise KeyError(
            f"Expected key '{key}' in checkpoint state_dict — is this a MoLT checkpoint?"
        )
    return tuple(sd[key].shape)


def infer_molt_arch(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Pull MoLT architecture out of state_dict tensor shapes."""
    n_features, d_acts = _shape(state_dict, "model.e.weight")

    n_layers = _shape(state_dict, "model.input_standardizer.mean")[0]

    # Walk Us.0, Us.1, ... while the keys exist. ranks come from Us.t.shape[1].
    ranks: list[int] = []
    tier = 0
    while f"model.Us.{tier}" in state_dict:
        u_shape = _shape(state_dict, f"model.Us.{tier}")
        # Us.t: (N * 2^t, ranks[t], d_acts)
        ranks.append(u_shape[1])
        tier += 1
    if not ranks:
        raise ValueError(
            "No Us.* keys in checkpoint — this doesn't look like a MoLT model."
        )

    # N = (Us.0 first dim) / 2^0 = Us.0.shape[0]
    N = _shape(state_dict, "model.Us.0")[0]

    # Sanity: features add up.
    expected_features = sum(N * (2**t) for t in range(len(ranks)))
    if expected_features != n_features:
        raise ValueError(
            f"Inferred N={N}, ranks={ranks} implies n_features={expected_features}, "
            f"but model.e.weight has {n_features}. Checkpoint may be from an "
            "incompatible MoLT variant."
        )

    return {
        "d_acts": d_acts,
        "n_features": n_features,
        "n_layers": n_layers,
        "ranks": ranks,
        "N": N,
    }


def _build_tier_index(N: int, ranks: list[int]) -> tuple[list[int], list[int]]:
    """Per-feature tier index and rank, in feature-id order."""
    feature_tier: list[int] = []
    feature_rank: list[int] = []
    for t, r in enumerate(ranks):
        n_in_tier = N * (2**t)
        feature_tier.extend([t] * n_in_tier)
        feature_rank.extend([r] * n_in_tier)
    return feature_tier, feature_rank


def load_molt(
    ckpt_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[Molt, MoltCheckpointMetadata]:
    """Reconstruct a `Molt` from a Lightning checkpoint and load its weights.

    The returned model is in eval mode with `requires_grad_(False)`. The
    standardizers are marked initialized (their mean/std buffers were saved
    with the checkpoint), so the model is ready for forward passes.
    """
    ckpt_path = str(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "state_dict" not in ckpt:
        raise ValueError(f"{ckpt_path} is not a Lightning checkpoint (no state_dict).")
    sd = ckpt["state_dict"]

    arch = infer_molt_arch(sd)
    d_acts = arch["d_acts"]
    n_features = arch["n_features"]
    n_layers = arch["n_layers"]
    ranks = arch["ranks"]
    N = arch["N"]

    nonlin = JumpReLU(theta=0.0, bandwidth=1.0, n_layers=1, d_features=n_features)
    in_std = DimensionwiseInputStandardizer(n_layers=n_layers, activation_dim=d_acts)
    out_std = DimensionwiseOutputStandardizer(n_layers=n_layers, activation_dim=d_acts)

    molt = Molt(
        d_acts=d_acts,
        N=N,
        nonlinearity=nonlin,
        input_standardizer=in_std,
        output_standardizer=out_std,
        ranks=ranks,
    )

    # Strip the "model." prefix that the LightningModule wrapping adds, and
    # drop anything that doesn't belong to the inner Molt (e.g. `last_active`,
    # which lives on the LightningModule, or any replacement_model state).
    molt_sd = {}
    for k, v in sd.items():
        if k.startswith("model."):
            molt_sd[k[len("model.") :]] = v

    missing, unexpected = molt.load_state_dict(molt_sd, strict=False)
    # Strict=False because the LightningModule may have stored extra buffers
    # that aren't part of Molt. Surface anything genuinely missing.
    if missing:
        raise RuntimeError(
            f"Missing keys when loading MoLT weights: {missing}. "
            "The checkpoint and inferred architecture disagree."
        )
    # `unexpected` is OK here — those are LightningModule-only tensors that
    # already got filtered by the prefix strip. If anything slips through it's
    # informational, not fatal.

    # Mark standardizers initialized — buffers were loaded from the ckpt.
    in_std.is_initialized = True
    out_std.is_initialized = True

    molt.eval()
    molt.requires_grad_(False)
    molt.to(device)

    feature_tier, feature_rank = _build_tier_index(N, ranks)

    dm_hp = ckpt.get("datamodule_hyper_parameters", {}) or {}
    meta = MoltCheckpointMetadata(
        ckpt_path=ckpt_path,
        d_acts=d_acts,
        n_features=n_features,
        n_layers=n_layers,
        ranks=ranks,
        N=N,
        feature_tier=feature_tier,
        feature_rank=feature_rank,
        base_model_name=dm_hp.get("model_name"),
        training_dataset=dm_hp.get("dataset_name"),
        global_step=ckpt.get("global_step"),
        epoch=ckpt.get("epoch"),
    )

    return molt, meta


def load_molt_from_hf(
    filename: str,
    repo_id: str = DEFAULT_HF_REPO,
    revision: Optional[str] = None,
    device: str | torch.device = "cpu",
    cache_dir: Optional[str] = None,
) -> tuple[Molt, MoltCheckpointMetadata]:
    """Download a MoLT checkpoint from the Hugging Face Hub and load it.

    Defaults `repo_id` to `kylelovesllms/molt-sweeps`. The hub cache handles
    re-use across calls; we never copy the file locally.

    Example:
        molt, meta = load_molt_from_hf("gpt2-molt-lam-0_00015-50M.ckpt")
    """
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        cache_dir=cache_dir,
    )
    molt, meta = load_molt(local_path, device=device)
    meta.hf_repo_id = repo_id
    meta.hf_filename = filename
    meta.hf_revision = revision
    return molt, meta
