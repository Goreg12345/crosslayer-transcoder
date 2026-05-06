"""Build a single self-contained multilayer bundle.html.

Compared to `feature_dash.bundle.make_bundle`, this version:

  * Carries a flat list of (layer, feature_id) entries — useful when you
    only want to visualize a *subset* of features across multiple layers
    (e.g. those that activated on a specific prompt).
  * Optionally inlines the prompt-token activation trace alongside the
    corpus top-K examples, so the rendered page shows both why the feature
    was selected (its firing on the prompt) and what it usually fires on.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from crosslayer_transcoder.feature_dash.bundle import _sanitize_for_script_tag
from crosslayer_transcoder.feature_dash.collect import (
    GateCollector,
    window_feature_summary,
)
from crosslayer_transcoder.feature_dash.multilayer import MultiLayerCheckpointMetadata
from crosslayer_transcoder.feature_dash.render import TEMPLATE_DIR

logger = logging.getLogger(__name__)


@dataclass
class PromptTrace:
    """Per-feature activation trace on the analysis prompt."""

    tokens: list[str]
    activations: list[float]

    @property
    def peak_pos(self) -> int:
        return max(range(len(self.activations)), key=lambda i: self.activations[i])

    @property
    def peak(self) -> float:
        return float(self.activations[self.peak_pos])


def _ckpt_stem_multilayer(meta: MultiLayerCheckpointMetadata) -> str:
    raw = f"{meta.run_name}_step{meta.step}"
    return "".join(c if (c.isalnum() or c in "-_.") else "-" for c in raw).strip("-")


def make_multilayer_bundle(
    collectors: list[GateCollector],
    meta: MultiLayerCheckpointMetadata,
    tokenizer,
    out_path: str | Path,
    selected: Iterable[tuple[int, int]],
    prompt: Optional[str] = None,
    prompt_traces: Optional[dict[tuple[int, int], PromptTrace]] = None,
    dataset_name: str = "Skylion007/openwebtext",
    window: int = 32,
) -> Path:
    """Write a portable bundle.html covering only `selected` (layer, feature) pairs.

    `collectors` must be a list of length `meta.n_layers` (one per layer);
    each is consulted only for the feature ids present in `selected`.

    `prompt_traces`, when given, attaches the prompt-token activation trace
    to each entry so the rendered page shows both prompt highlighting and
    corpus top-K examples.
    """
    out_path = Path(out_path)
    if len(collectors) != meta.n_layers:
        raise ValueError(
            f"got {len(collectors)} collectors, expected {meta.n_layers}"
        )

    # De-dup, keep stable layer-then-feature order for the index.
    selected_sorted = sorted(set(selected))

    if out_path.is_dir() or (not out_path.exists() and out_path.suffix == ""):
        out_path = out_path / f"bundle_{_ckpt_stem_multilayer(meta)}.html"

    entries: list[dict] = []
    features: dict[str, dict] = {}

    total_tokens = collectors[0].total_tokens if collectors else 0

    for layer, feature_id in selected_sorted:
        if not (0 <= layer < meta.n_layers):
            raise ValueError(f"layer {layer} out of range")
        if not (0 <= feature_id < meta.n_features):
            raise ValueError(f"feature {feature_id} out of range")

        coll = collectors[layer]
        summary = coll.feature_summary(feature_id)
        body = window_feature_summary(summary, tokenizer, window=window)
        body["layer"] = layer
        body["tier"] = meta.feature_tier[feature_id]
        body["rank"] = meta.feature_rank[feature_id]

        prompt_peak = None
        prompt_peak_pos = None
        if prompt_traces is not None and (layer, feature_id) in prompt_traces:
            trace = prompt_traces[(layer, feature_id)]
            body["prompt_tokens"] = trace.tokens
            body["prompt_activations"] = trace.activations
            body["prompt_peak"] = trace.peak
            body["prompt_peak_pos"] = trace.peak_pos
            prompt_peak = trace.peak
            prompt_peak_pos = trace.peak_pos

        key = f"L{layer}F{feature_id}"
        features[key] = body

        entries.append({
            "key": key,
            "layer": layer,
            "feature_id": feature_id,
            "tier": meta.feature_tier[feature_id],
            "rank": meta.feature_rank[feature_id],
            "activation_rate": float(body["activation_rate"]),
            "max_activation": float(body["max_activation"]),
            "prompt_peak": prompt_peak,
            "prompt_peak_pos": prompt_peak_pos,
        })

    metadata = {
        "schema_version": 1,
        "ckpt_dir": meta.ckpt_dir,
        "run_name": meta.run_name,
        "step": meta.step,
        "ckpt_stem": _ckpt_stem_multilayer(meta),
        "n_layers": meta.n_layers,
        "n_features_per_layer": meta.n_features,
        "ranks": list(meta.ranks),
        "N": meta.N,
        "n_tokens_collected": total_tokens,
        "dashboard_dataset": dataset_name,
        "seq_len": collectors[0].T if collectors else None,
        "top_k": collectors[0].K if collectors else None,
        "window": window,
        "prompt": prompt,
        "entries": entries,
    }

    template = (TEMPLATE_DIR / "bundle_multilayer.html").read_text()
    css = (TEMPLATE_DIR / "dashboard.css").read_text()

    metadata_json = _sanitize_for_script_tag(json.dumps(metadata, separators=(",", ":")))
    features_json = _sanitize_for_script_tag(json.dumps(features, separators=(",", ":")))

    html = (
        template
        .replace("__BUNDLE_CSS__", css)
        .replace("__METADATA_JSON__", metadata_json)
        .replace("__FEATURES_JSON__", features_json)
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    logger.info(
        "wrote %s (%.1f MB, %d entries across %d layers)",
        out_path,
        out_path.stat().st_size / 1e6,
        len(entries),
        meta.n_layers,
    )
    return out_path
