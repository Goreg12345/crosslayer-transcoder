"""Multilayer extension of the feature dashboard.

The single-layer flow in `feature_dash.collect` / `feature_dash.bundle` is
designed for one MoLT trained on one residual stream. Multi-layer MoLT runs
train one Molt per transformer block and save them as separate `.pt` files
via `MoltPerLayerCheckpointCallback`. This module wires those together:

  * `load_multilayer_molt` — build a `MultiLayerMolt` and load all per-layer
    `.pt` files saved by the callback. Architecture is inferred from layer 0
    so a calling script doesn't have to repeat the YAML knobs.

  * `MultiLayerLMRunner` — register a forward pre-hook on every block's
    `ln_2` so one base-LM forward gives us all `n_layers` residual streams.

  * `collect_multilayer_features` — single corpus pass that updates one
    `GateCollector` per layer.

The `GateCollector` itself is unchanged — multilayer just maintains a list
of them, one per layer.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import torch

from crosslayer_transcoder.feature_dash.collect import (
    GateCollector,
    _iter_token_batches,
)
from crosslayer_transcoder.feature_dash.load import (
    MoltCheckpointMetadata,
    _build_tier_index,
)
from crosslayer_transcoder.model.jumprelu import JumpReLU
from crosslayer_transcoder.model.molt import MultiLayerMolt
from crosslayer_transcoder.model.standardize import (
    DimensionwiseInputStandardizer,
    DimensionwiseOutputStandardizer,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading per-layer .pt checkpoints saved by MoltPerLayerCheckpointCallback.
# ---------------------------------------------------------------------------


def find_latest_step(ckpt_dir: Path, run_name: str) -> int:
    """Pick the largest step among `<run_name>_layer_0_step<N>.pt` files."""
    pattern = re.compile(rf"^{re.escape(run_name)}_layer_0_step(\d+)\.pt$")
    steps = [
        int(m.group(1))
        for p in ckpt_dir.iterdir()
        if (m := pattern.match(p.name))
    ]
    if not steps:
        raise FileNotFoundError(f"no checkpoints matching {run_name} in {ckpt_dir}")
    return max(steps)


def _infer_arch_from_layer0(layer0_sd: dict[str, torch.Tensor]) -> dict[str, int | list[int]]:
    """Pull (d_acts, n_features, ranks, N) from a per-layer Molt state dict.

    Per-layer .pt files have *no* `model.` prefix (they're saved straight from
    `MultiLayerMolt.molts[layer].state_dict()`), so the keys here differ from
    the Lightning-checkpoint loader in `feature_dash.load`.
    """
    if "e.weight" not in layer0_sd:
        raise KeyError(
            "expected key 'e.weight' in per-layer Molt state dict — "
            "is this a per-layer MoLT checkpoint?"
        )
    n_features, d_acts = layer0_sd["e.weight"].shape

    ranks: list[int] = []
    tier = 0
    while f"Us.{tier}" in layer0_sd:
        ranks.append(layer0_sd[f"Us.{tier}"].shape[1])
        tier += 1
    if not ranks:
        raise ValueError("no Us.* keys in checkpoint — not a Molt state dict")

    N = layer0_sd["Us.0"].shape[0]
    expected = sum(N * (2**t) for t in range(len(ranks)))
    if expected != n_features:
        raise ValueError(
            f"inferred N={N}, ranks={ranks} -> {expected} features, "
            f"but e.weight has {n_features}"
        )
    n_layers = layer0_sd["input_standardizer.mean"].shape[0]

    return {
        "d_acts": int(d_acts),
        "n_features": int(n_features),
        "n_layers": int(n_layers),
        "ranks": ranks,
        "N": int(N),
    }


@dataclass
class MultiLayerCheckpointMetadata:
    """Architecture + bookkeeping for a multi-layer per-layer checkpoint set."""

    ckpt_dir: str
    run_name: str
    step: int
    d_acts: int
    n_features: int   # per layer
    n_layers: int
    ranks: list[int]
    N: int
    feature_tier: list[int] = field(default_factory=list)
    feature_rank: list[int] = field(default_factory=list)

    def per_layer_meta(self, layer: int) -> MoltCheckpointMetadata:
        """Adapter so single-layer dashboard helpers can consume one layer."""
        layer_path = (
            Path(self.ckpt_dir)
            / f"{self.run_name}_layer_{layer}_step{self.step}.pt"
        )
        return MoltCheckpointMetadata(
            ckpt_path=str(layer_path),
            d_acts=self.d_acts,
            n_features=self.n_features,
            n_layers=self.n_layers,
            ranks=list(self.ranks),
            N=self.N,
            feature_tier=list(self.feature_tier),
            feature_rank=list(self.feature_rank),
            global_step=self.step,
        )


def load_multilayer_molt(
    ckpt_dir: str | Path,
    run_name: str,
    step: int | None = None,
    device: str | torch.device = "cpu",
    jumprelu_theta: float = 0.03,
    jumprelu_bandwidth: float = 1.0,
) -> tuple[MultiLayerMolt, MultiLayerCheckpointMetadata]:
    """Build a `MultiLayerMolt` and load all per-layer `.pt` files.

    Architecture is inferred from layer 0's state dict. The standardizer
    `is_initialized` flag is a plain Python attr (not a buffer) so we set it
    explicitly after `load_state_dict` populates the saved mean/std.
    """
    ckpt_dir = Path(ckpt_dir)
    if step is None:
        step = find_latest_step(ckpt_dir, run_name)

    layer0_path = ckpt_dir / f"{run_name}_layer_0_step{step}.pt"
    layer0_sd = torch.load(layer0_path, map_location="cpu")
    arch = _infer_arch_from_layer0(layer0_sd)
    n_layers = arch["n_layers"]
    n_features = arch["n_features"]
    d_acts = arch["d_acts"]
    ranks = arch["ranks"]
    N = arch["N"]

    nonlin = JumpReLU(
        theta=jumprelu_theta,
        bandwidth=jumprelu_bandwidth,
        n_layers=1,
        d_features=n_features,
    )
    in_std = DimensionwiseInputStandardizer(n_layers=n_layers, activation_dim=d_acts)
    out_std = DimensionwiseOutputStandardizer(n_layers=n_layers, activation_dim=d_acts)

    molt = MultiLayerMolt(
        n_layers=n_layers,
        d_acts=d_acts,
        N=N,
        nonlinearity=nonlin,
        input_standardizer=in_std,
        output_standardizer=out_std,
        ranks=ranks,
    )

    for layer in range(n_layers):
        path = ckpt_dir / f"{run_name}_layer_{layer}_step{step}.pt"
        sd = torch.load(path, map_location="cpu")
        sd = {k: v.float() if v.is_floating_point() else v for k, v in sd.items()}
        molt.molts[layer].load_state_dict(sd)

    in_std.is_initialized = True
    out_std.is_initialized = True

    molt.eval()
    molt.requires_grad_(False)
    molt.to(device)

    feature_tier, feature_rank = _build_tier_index(N, ranks)
    meta = MultiLayerCheckpointMetadata(
        ckpt_dir=str(ckpt_dir),
        run_name=run_name,
        step=step,
        d_acts=d_acts,
        n_features=n_features,
        n_layers=n_layers,
        ranks=ranks,
        N=N,
        feature_tier=feature_tier,
        feature_rank=feature_rank,
    )
    return molt, meta


# ---------------------------------------------------------------------------
# Multi-layer base-LM runner: hook every block's ln_2 in one forward.
# ---------------------------------------------------------------------------


class MultiLayerLMRunner:
    """Capture the residual at every block's `ln_2` input in a single forward.

    Returns a stacked (B, T, n_layers, d_acts) tensor — index `[:, :, l, :]`
    is what `MultiLayerMolt.molts[l]` was trained on.
    """

    def __init__(
        self,
        model_name: str,
        n_layers: int,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        from transformers import GPT2LMHeadModel

        self.model = (
            GPT2LMHeadModel.from_pretrained(model_name).to(device).to(dtype).eval()
        )
        self.n_layers = n_layers
        self.device = torch.device(device)
        self.dtype = dtype
        self._captured: list[torch.Tensor | None] = [None] * n_layers
        self._handles = [
            self.model.transformer.h[i].ln_2.register_forward_pre_hook(
                self._make_hook(i)
            )
            for i in range(n_layers)
        ]

    def _make_hook(self, layer_idx: int):
        def hook(_module, args):
            self._captured[layer_idx] = args[0]
        return hook

    @torch.no_grad()
    def residuals(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Run the LM and stack captured residuals to (B, T, n_layers, d_acts)."""
        self._captured = [None] * self.n_layers
        token_ids = token_ids.to(self.device)
        self.model(token_ids)
        if any(c is None for c in self._captured):
            missing = [i for i, c in enumerate(self._captured) if c is None]
            raise RuntimeError(f"hooks didn't fire for layers {missing}")
        return torch.stack(self._captured, dim=2)  # (B, T, n_layers, d_acts)

    def close(self):
        for h in self._handles:
            h.remove()


# ---------------------------------------------------------------------------
# Top-level driver: stream tokens once, fan out to per-layer collectors.
# ---------------------------------------------------------------------------


def collect_multilayer_features(
    molt: MultiLayerMolt,
    base_model_name: str = "openai-community/gpt2",
    dataset_name: str = "Skylion007/openwebtext",
    dataset_split: str = "train",
    n_sequences: int = 1024,
    seq_len: int = 128,
    batch_size: int = 16,
    top_k: int = 20,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
    log_every: int = 16,
) -> list[GateCollector]:
    """Run the base LM once per batch, update one `GateCollector` per layer.

    Returns a list of `n_layers` collectors. Memory per collector is governed
    by `n_features * top_k * seq_len`; for N=10, top_k=20, seq_len=128 each
    collector is ~10 MB so the full 12-layer set is ~120 MB on CPU.
    """
    from transformers import GPT2TokenizerFast

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = GPT2TokenizerFast.from_pretrained(base_model_name)
    n_layers = molt.n_layers
    n_features = molt.n_features

    runner = MultiLayerLMRunner(
        model_name=base_model_name,
        n_layers=n_layers,
        device=device,
        dtype=dtype,
    )
    molt = molt.to(device)

    collectors = [
        GateCollector(n_features=n_features, top_k=top_k, seq_len=seq_len)
        for _ in range(n_layers)
    ]

    try:
        for batch_idx, tok in enumerate(
            _iter_token_batches(
                dataset_name=dataset_name,
                dataset_split=dataset_split,
                tokenizer=tokenizer,
                seq_len=seq_len,
                batch_size=batch_size,
                n_sequences=n_sequences,
            )
        ):
            resid = runner.residuals(tok)  # (B, T, n_layers, d_acts)
            B, T, L, D = resid.shape
            # Run each per-layer Molt on its own residual slice. We bypass the
            # MultiLayerMolt.forward batch wrapper because we want gates only
            # and per-layer access.
            for layer in range(n_layers):
                inner = molt.molts[layer]
                acts = inner.input_standardizer(resid[:, :, layer, :], layer)
                pre = inner.e(acts)
                gates = inner.nonlinearity(pre)  # (B, T, n_features)
                collectors[layer].update(tok.cpu(), gates.float().cpu())

            if log_every and batch_idx % log_every == 0:
                rates = [c.activation_rate().mean().item() for c in collectors]
                logger.info(
                    "batch %d: %d tokens collected, mean fire rate per layer: %s",
                    batch_idx,
                    collectors[0].total_tokens,
                    [f"{r:.4f}" for r in rates],
                )
    finally:
        runner.close()

    return collectors
