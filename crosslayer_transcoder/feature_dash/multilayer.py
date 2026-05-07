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


def find_latest_step(ckpt_dir: Path, run_name: str) -> str:
    """Pick the latest checkpoint suffix among `<run_name>_layer_0_<suffix>.pt`.

    Two suffix flavours are observed in the wild:
      * `step<N>.pt`   — used by `MoltPerLayerCheckpointCallback` (default)
      * `tokens<...>.pt` — used by some multi-GPU runs (e.g. the Gemma 3-1B
        checkpoints, where the suffix encodes seen tokens like `0100M352K`)

    We return the suffix string (excluding the `.pt`) so the caller can pass
    it back to `load_multilayer_molt` regardless of flavour.
    """
    prefix = f"{run_name}_layer_0_"
    suffixes: list[str] = []
    for p in ckpt_dir.iterdir():
        name = p.name
        if not (name.startswith(prefix) and name.endswith(".pt")):
            continue
        suffixes.append(name[len(prefix) : -len(".pt")])
    if not suffixes:
        raise FileNotFoundError(f"no checkpoints matching {run_name} in {ckpt_dir}")

    # Sort `step<N>` numerically, otherwise lexicographically (works for the
    # `tokens0100M352K` pattern because the zero-padded token counts sort).
    def _key(s: str) -> tuple[int, str]:
        m = re.match(r"^step(\d+)$", s)
        if m:
            return (int(m.group(1)), "")
        return (-1, s)

    return max(suffixes, key=_key)


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
    step: str  # full file suffix, e.g. "step36000" or "tokens0100M352K"
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
            / f"{self.run_name}_layer_{layer}_{self.step}.pt"
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
            global_step=None,
        )


def download_multilayer_from_hf(
    repo_id: str,
    folder: str,
    run_name: str,
    step: str | int | None = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> tuple[Path, str]:
    """Pull every per-layer `.pt` for a multi-layer MoLT run from the HF Hub.

    Returns `(local_dir, step_suffix)`. The local dir is whatever folder HF
    placed the files in (a snapshot under the HF cache); we don't copy.
    """
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    files = api.list_repo_files(repo_id, revision=revision)
    prefix = f"{folder}/{run_name}_layer_"
    candidates = [f for f in files if f.startswith(prefix) and f.endswith(".pt")]
    if not candidates:
        raise FileNotFoundError(
            f"no `.pt` files matching {prefix}*.pt in repo {repo_id}"
        )

    # Discover unique suffixes (e.g. "step36000", "tokens0100M352K") and pick the latest.
    re_suffix = re.compile(rf"^{re.escape(prefix)}(\d+)_(.+)\.pt$")
    suffix_to_layers: dict[str, set[int]] = {}
    for f in candidates:
        m = re_suffix.match(f)
        if not m:
            continue
        layer_idx = int(m.group(1))
        suffix = m.group(2)
        suffix_to_layers.setdefault(suffix, set()).add(layer_idx)
    if not suffix_to_layers:
        raise RuntimeError(f"could not parse any candidate filename in {repo_id}")

    if step is None:
        def _key(s: str) -> tuple[int, str]:
            m = re.match(r"^step(\d+)$", s)
            return (int(m.group(1)), "") if m else (-1, s)
        step = max(suffix_to_layers, key=_key)
    elif isinstance(step, int):
        step = f"step{step}"

    if step not in suffix_to_layers:
        raise FileNotFoundError(
            f"step suffix {step!r} not present in {repo_id}; available: "
            f"{sorted(suffix_to_layers)[-5:]}"
        )

    layers = sorted(suffix_to_layers[step])
    n_layers = max(layers) + 1
    if set(layers) != set(range(n_layers)):
        raise RuntimeError(
            f"per-layer files for {step} are non-contiguous: {layers}"
        )

    local_dir: Optional[Path] = None
    for layer in layers:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{folder}/{run_name}_layer_{layer}_{step}.pt",
            revision=revision,
            cache_dir=cache_dir,
        )
        if local_dir is None:
            local_dir = Path(path).parent
    assert local_dir is not None
    return local_dir, step


def load_multilayer_molt(
    ckpt_dir: str | Path,
    run_name: str,
    step: str | int | None = None,
    device: str | torch.device = "cpu",
    jumprelu_theta: float = 0.03,
    jumprelu_bandwidth: float = 1.0,
) -> tuple[MultiLayerMolt, MultiLayerCheckpointMetadata]:
    """Build a `MultiLayerMolt` and load all per-layer `.pt` files.

    Architecture is inferred from layer 0's state dict. The standardizer
    `is_initialized` flag is a plain Python attr (not a buffer) so we set it
    explicitly after `load_state_dict` populates the saved mean/std.

    `step` may be an int (legacy `step<N>` suffix), a full suffix string like
    `"step36000"` or `"tokens0100M352K"`, or `None` to pick the latest.
    """
    ckpt_dir = Path(ckpt_dir)
    if step is None:
        step = find_latest_step(ckpt_dir, run_name)
    elif isinstance(step, int):
        step = f"step{step}"

    layer0_path = ckpt_dir / f"{run_name}_layer_0_{step}.pt"
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
        path = ckpt_dir / f"{run_name}_layer_{layer}_{step}.pt"
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


def _resolve_mlp_input_layers(model) -> list[torch.nn.Module]:
    """Return the per-block module whose *input* is the MLP-input residual.

    MoLT was trained on the input to the second LayerNorm in each transformer
    block (the one that precedes the MLP). The module path differs between
    architectures:

      * GPT-2 / GPT-Neo style: `model.transformer.h[i].ln_2`
      * Gemma 3 / LLaMA style: `model.model.layers[i].pre_feedforward_layernorm`

    We try each in turn and raise if none match.
    """
    # GPT-2 family.
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return [block.ln_2 for block in model.transformer.h]
    # Gemma 3 / LLaMA family — Gemma 3 uses `pre_feedforward_layernorm`,
    # LLaMA-style models use `post_attention_layernorm` (which is the layernorm
    # immediately preceding the MLP, equivalent to GPT-2's `ln_2`).
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        sample = layers[0]
        if hasattr(sample, "pre_feedforward_layernorm"):
            return [block.pre_feedforward_layernorm for block in layers]
        if hasattr(sample, "post_attention_layernorm"):
            return [block.post_attention_layernorm for block in layers]
    raise ValueError(
        f"Don't know how to locate MLP-input layernorms in model of type "
        f"{type(model).__name__}; add a case to _resolve_mlp_input_layers."
    )


def _load_base_lm(model_name: str, device, dtype: torch.dtype):
    """Load a HF causal LM, picking the right loader for the architecture."""
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()


class MultiLayerLMRunner:
    """Capture the per-layer MLP-input residual in one base-LM forward.

    Returns a stacked (B, T, n_layers, d_acts) tensor — index `[:, :, l, :]`
    is what `MultiLayerMolt.molts[l]` was trained on.

    The pre-hook is registered on the per-block module whose *input* is the
    residual fed into the MLP path. For GPT-2 that's `ln_2`; for Gemma 3 it's
    `pre_feedforward_layernorm`. See `_resolve_mlp_input_layers`.
    """

    def __init__(
        self,
        model_name: str,
        n_layers: int,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.model = _load_base_lm(model_name, device, dtype)
        self.n_layers = n_layers
        self.device = torch.device(device)
        self.dtype = dtype
        self._captured: list[torch.Tensor | None] = [None] * n_layers

        target_modules = _resolve_mlp_input_layers(self.model)
        if len(target_modules) < n_layers:
            raise ValueError(
                f"base LM has {len(target_modules)} blocks but MoLT expects {n_layers}"
            )
        self._handles = [
            target_modules[i].register_forward_pre_hook(self._make_hook(i))
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
    tokenizer=None,
) -> list[GateCollector]:
    """Run the base LM once per batch, update one `GateCollector` per layer.

    Returns a list of `n_layers` collectors. Memory per collector is governed
    by `n_features * top_k * seq_len`; for N=10, top_k=20, seq_len=128 each
    collector is ~10 MB so the full 12-layer set is ~120 MB on CPU.

    Pass `tokenizer` to reuse a pre-loaded tokenizer; otherwise an `AutoTokenizer`
    is created from `base_model_name` (so non-GPT2 architectures like Gemma 3
    work without code changes).
    """
    from transformers import AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
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
