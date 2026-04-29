"""Streaming gate collection over OpenWebText.

For each MoLT transform, accumulate:
  1. firing rate (fraction of tokens with gate > 0),
  2. max activation,
  3. top-K sequences by per-sequence peak activation, with the per-token gate
     trace for that one transform (used by the renderer for token highlighting).

The corpus is `Skylion007/openwebtext` by default — this matches the training
distribution of the HF MoLT checkpoints.

The collector itself doesn't touch HuggingFace at all; it consumes batches of
`(token_ids, gates)`. `BaseLMRunner` and `collect_features` are the wiring
that produces those batches from a real LM and dataset.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator, Optional

import torch

from crosslayer_transcoder.model.molt import Molt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure data-side accumulator (no LM, no dataset). Unit-testable on CPU.
# ---------------------------------------------------------------------------


class GateCollector:
    """Per-feature firing-rate, max activation, and top-K sequence buffer.

    All state lives on CPU. The collector is pushed (token_ids, gates) batches
    by whatever drives it. Memory is bounded by `n_features * top_k * seq_len`
    on each of token_ids (int64) and activations (float32). For
    `n_features=1550, top_k=20, seq_len=128` that's ~47 MB persistent — fine.

    Also accumulates a log-spaced activation histogram per feature. Bins are
    fixed at construction (10^-3 .. 10^3, 60 bins) so different features —
    which can have very different scales — share the same axis. Values
    outside that range are dropped (gate values < 1e-3 are visually
    indistinguishable from noise; > 1e3 doesn't happen in trained MoLTs).
    """

    HIST_LOG_LO = -3.0
    HIST_LOG_HI = 3.0
    HIST_N_BINS = 60

    def __init__(self, n_features: int, top_k: int, seq_len: int):
        self.n_features = n_features
        self.K = top_k
        self.T = seq_len

        self.firing_count = torch.zeros(n_features, dtype=torch.long)
        self.total_tokens = 0
        self.max_activation = torch.full((n_features,), -float("inf"))

        # Top-K is keyed by per-sequence peak. -inf in unfilled slots.
        self.top_peaks = torch.full((n_features, top_k), -float("inf"))
        self.top_token_ids = torch.zeros(
            (n_features, top_k, seq_len), dtype=torch.long
        )
        self.top_activations = torch.zeros(
            (n_features, top_k, seq_len), dtype=torch.float32
        )

        # Activation histogram. `_hist_edges` has HIST_N_BINS+1 entries; bin i
        # covers [_hist_edges[i], _hist_edges[i+1]). Counts are per-feature.
        self._hist_edges = torch.logspace(
            self.HIST_LOG_LO, self.HIST_LOG_HI, self.HIST_N_BINS + 1
        )
        self.act_histogram = torch.zeros(
            (n_features, self.HIST_N_BINS), dtype=torch.long
        )

    @torch.no_grad()
    def update(self, batch_token_ids: torch.Tensor, gates: torch.Tensor) -> None:
        """Fold a batch into the running stats.

        batch_token_ids: (B, T) long, on CPU
        gates:           (B, T, F) float, on CPU
        """
        if batch_token_ids.device.type != "cpu" or gates.device.type != "cpu":
            raise ValueError("update expects CPU tensors")

        B, T, F = gates.shape
        if F != self.n_features:
            raise ValueError(f"gates has {F} features, expected {self.n_features}")
        if T != self.T or batch_token_ids.shape != (B, T):
            raise ValueError(
                f"shape mismatch: token_ids={tuple(batch_token_ids.shape)}, "
                f"gates={tuple(gates.shape)}, expected seq_len={self.T}"
            )

        # Firing count + max
        active = gates > 0
        self.firing_count += active.sum(dim=(0, 1)).long()
        self.total_tokens += B * T
        self.max_activation = torch.maximum(self.max_activation, gates.amax(dim=(0, 1)))

        # Top-K merge. Build a (F, K+B) candidate pool of peaks and pick top-K.
        peaks = gates.amax(dim=1)  # (B, F)

        all_peaks = torch.cat([self.top_peaks, peaks.T], dim=1)  # (F, K+B)
        # Token ids: old (F, K, T) and batch (1, B, T) broadcast then concat.
        batch_tok_expanded = batch_token_ids.unsqueeze(0).expand(F, -1, -1)
        all_token_ids = torch.cat(
            [self.top_token_ids, batch_tok_expanded], dim=1
        )  # (F, K+B, T)
        # Activations: old (F, K, T) and (B, T, F) -> permute -> (F, B, T) then concat.
        all_activations = torch.cat(
            [self.top_activations, gates.permute(2, 0, 1)], dim=1
        )  # (F, K+B, T)

        new_top_vals, new_top_idx = all_peaks.topk(self.K, dim=1)  # (F, K)

        F_idx = torch.arange(F).unsqueeze(1).expand(-1, self.K)
        self.top_peaks = new_top_vals
        self.top_token_ids = all_token_ids[F_idx, new_top_idx]
        self.top_activations = all_activations[F_idx, new_top_idx]

        # Histogram update.
        self._update_histogram(gates)

    def _update_histogram(self, gates: torch.Tensor) -> None:
        """Bin every nonzero gate value into its feature's log-spaced histogram.

        Values outside [10^HIST_LOG_LO, 10^HIST_LOG_HI) are silently dropped.
        Vectorised: we compute (B*T, F) bin indices, mask in-range entries, and
        scatter-add ones into a flat (F * HIST_N_BINS) view.
        """
        flat = gates.reshape(-1, self.n_features)  # (BT, F)
        # right=True gives the left-closed convention: bin i covers
        # [edges[i], edges[i+1]). After -1, in-range entries fall into
        # [0, HIST_N_BINS-1]; underflow becomes -1, overflow becomes
        # HIST_N_BINS, both of which we mask out below.
        bin_idx = torch.bucketize(flat, self._hist_edges, right=True) - 1
        in_range = (
            (flat > 0)
            & (bin_idx >= 0)
            & (bin_idx < self.HIST_N_BINS)
        )
        if not in_range.any():
            return
        pos_idx, feat_idx = torch.where(in_range)
        bins = bin_idx[pos_idx, feat_idx]
        flat_idx = feat_idx * self.HIST_N_BINS + bins
        self.act_histogram.view(-1).scatter_add_(
            0, flat_idx, torch.ones_like(flat_idx, dtype=torch.long)
        )

    def hist_edges(self) -> list[float]:
        """Bin edges shared by every feature's activation histogram."""
        return self._hist_edges.tolist()

    def activation_rate(self) -> torch.Tensor:
        """Per-feature fraction of tokens with gate > 0."""
        if self.total_tokens == 0:
            return torch.zeros(self.n_features)
        return self.firing_count.float() / float(self.total_tokens)

    def feature_summary(self, feature_id: int) -> "FeatureSummary":
        """Dump one feature's collected data into a small dataclass.

        Sequences whose per-sequence peak is <= 0 are dropped — those are
        sequences where the feature didn't activate at all, so there's nothing
        to highlight. A fully-dead feature returns an empty examples list.
        """
        peaks = self.top_peaks[feature_id]
        token_ids = self.top_token_ids[feature_id]
        activations = self.top_activations[feature_id]

        valid = peaks > 0
        peaks = peaks[valid]
        token_ids = token_ids[valid]
        activations = activations[valid]
        order = torch.argsort(peaks, descending=True)

        return FeatureSummary(
            feature_id=feature_id,
            activation_rate=self.activation_rate()[feature_id].item(),
            max_activation=float(self.max_activation[feature_id].item())
            if torch.isfinite(self.max_activation[feature_id])
            else 0.0,
            top_peaks=peaks[order].tolist(),
            top_token_ids=token_ids[order].tolist(),
            top_activations=activations[order].tolist(),
            act_histogram=self.act_histogram[feature_id].tolist(),
        )


@dataclass
class FeatureSummary:
    """Everything one feature contributes to the dashboard JSON."""

    feature_id: int
    activation_rate: float
    max_activation: float
    top_peaks: list[float]               # length <= K
    top_token_ids: list[list[int]]       # shape (n_examples, T)
    top_activations: list[list[float]]   # shape (n_examples, T)
    act_histogram: list[int]             # length HIST_N_BINS


# ---------------------------------------------------------------------------
# Step 3: window each example around its peak and decode token ids.
# ---------------------------------------------------------------------------


def _decode_token(tokenizer, token_id: int) -> str:
    """Decode a single token id to a display string.

    Uses single-id decode so byte-level BPE markers (e.g. GPT-2's `Ġ` for a
    leading space) come out as actual whitespace — that's what we want for
    rendering each token as its own `<span>`.
    """
    return tokenizer.decode([int(token_id)])


def window_example(
    token_ids: list[int],
    activations: list[float],
    tokenizer,
    window: int = 32,
) -> dict:
    """Trim one example to ±`window` tokens around its peak and decode ids.

    Returns the per-example dict that the dashboard JSON expects:
        {peak_activation, peak_token_pos, tokens, activations}
    where `peak_token_pos` is the index inside the *windowed* arrays.
    """
    if len(token_ids) != len(activations):
        raise ValueError(
            f"length mismatch: {len(token_ids)} ids vs {len(activations)} acts"
        )
    if len(token_ids) == 0:
        raise ValueError("empty example")

    peak_pos_full = max(range(len(activations)), key=lambda i: activations[i])
    start = max(0, peak_pos_full - window)
    end = min(len(token_ids), peak_pos_full + window + 1)

    windowed_ids = token_ids[start:end]
    windowed_acts = activations[start:end]
    windowed_tokens = [_decode_token(tokenizer, i) for i in windowed_ids]

    return {
        "peak_activation": float(activations[peak_pos_full]),
        "peak_token_pos": peak_pos_full - start,
        "tokens": windowed_tokens,
        "activations": [float(a) for a in windowed_acts],
    }


def window_feature_summary(
    summary: FeatureSummary,
    tokenizer,
    window: int = 32,
) -> dict:
    """Build the per-feature dashboard payload from a FeatureSummary.

    Schema (matches FEATURE-DASH.md §2 — minus tier/rank, which the dump step
    fills in from MoltCheckpointMetadata):
        {feature_id, activation_rate, max_activation, examples, act_histogram}
    """
    examples = [
        window_example(ids, acts, tokenizer, window=window)
        for ids, acts in zip(summary.top_token_ids, summary.top_activations)
    ]
    return {
        "feature_id": summary.feature_id,
        "activation_rate": summary.activation_rate,
        "max_activation": summary.max_activation,
        "examples": examples,
        "act_histogram": summary.act_histogram,
    }


# ---------------------------------------------------------------------------
# Base-LM forward-hook runner (captures residual at h[L].ln_2 input).
# ---------------------------------------------------------------------------


class BaseLMRunner:
    """Wrap a HF causal LM with a forward pre-hook on `transformer.h[L].ln_2`.

    That tensor — the input to the second LayerNorm — is what MoLT was trained
    on (cf. `crosslayer_transcoder/data/activation_sources.py`, which uses
    nnsight's `model.transformer.h[i].ln_2.input`). Capturing it via a
    PyTorch forward pre-hook is the same thing without the nnsight dep.
    """

    def __init__(
        self,
        model_name: str,
        layer: int,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        from transformers import GPT2LMHeadModel  # local import: heavy

        self.model = (
            GPT2LMHeadModel.from_pretrained(model_name).to(device).to(dtype).eval()
        )
        self.layer = layer
        self.device = torch.device(device)
        self.dtype = dtype
        self._captured: Optional[torch.Tensor] = None
        self._hook = self.model.transformer.h[layer].ln_2.register_forward_pre_hook(
            self._capture
        )

    def _capture(self, module, args):
        # forward_pre_hook: args is the tuple of positional args. ln_2 takes
        # `hidden_states` as its only positional arg.
        self._captured = args[0]

    @torch.no_grad()
    def residual_at_layer(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Run the LM and return the captured (B, T, d_acts) residual."""
        self._captured = None
        token_ids = token_ids.to(self.device)
        self.model(token_ids)
        if self._captured is None:
            raise RuntimeError(
                f"forward pre-hook on layer {self.layer} ln_2 didn't fire"
            )
        out = self._captured
        self._captured = None
        return out

    def close(self):
        self._hook.remove()


# ---------------------------------------------------------------------------
# Top-level driver: stream tokens, run LM, push gates into collector.
# ---------------------------------------------------------------------------


def _iter_token_batches(
    dataset_name: str,
    dataset_split: str,
    tokenizer,
    seq_len: int,
    batch_size: int,
    n_sequences: int,
) -> Iterator[torch.Tensor]:
    """Yield (B, T) int64 token-id batches from a streaming HF dataset.

    Skips sequences shorter than `seq_len` to keep all examples the same
    length — simpler downstream than masking.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=dataset_split, streaming=True)

    buf: list[torch.Tensor] = []
    yielded = 0
    for example in ds:
        if yielded >= n_sequences:
            return
        text = example.get("text") or ""
        if not text:
            continue
        enc = tokenizer(text, truncation=True, max_length=seq_len, return_tensors="pt")
        ids = enc["input_ids"][0]
        if ids.numel() < seq_len:
            continue
        buf.append(ids[:seq_len])
        if len(buf) == batch_size:
            yield torch.stack(buf, dim=0)
            yielded += batch_size
            buf = []
    if buf:
        yield torch.stack(buf, dim=0)


def collect_features(
    molt: Molt,
    layer: int,
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
) -> GateCollector:
    """Run the base LM over a corpus and collect MoLT gate stats per feature.

    Defaults match the HF MoLT checkpoints (GPT-2, OpenWebText). Returns a
    populated `GateCollector` ready to be serialised by Step 4.
    """
    from transformers import GPT2TokenizerFast

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = GPT2TokenizerFast.from_pretrained(base_model_name)
    runner = BaseLMRunner(base_model_name, layer, device, dtype)
    molt = molt.to(device)

    collector = GateCollector(
        n_features=molt.n_features, top_k=top_k, seq_len=seq_len
    )

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
            resid = runner.residual_at_layer(tok)  # (B, T, d_acts)
            resid_std = molt.input_standardizer(resid, layer)
            pre = molt.e(resid_std)
            gates = molt.nonlinearity(pre)  # (B, T, n_features)

            collector.update(tok.cpu(), gates.float().cpu())

            if log_every and batch_idx % log_every == 0:
                logger.info(
                    "batch %d: %d tokens collected, mean fire rate %.4f",
                    batch_idx,
                    collector.total_tokens,
                    collector.activation_rate().mean().item(),
                )
    finally:
        runner.close()

    return collector
