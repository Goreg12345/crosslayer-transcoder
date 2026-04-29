"""Serialise a populated GateCollector + checkpoint metadata to disk.

Layout produced (matches FEATURE-DASH.md §2):

    <out_dir>/
      data/
        metadata.json
        features/
          0000.json
          0001.json
          ...

The renderer in Step 5 will consume these files; nothing here knows about HTML.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from crosslayer_transcoder.feature_dash.collect import (
    GateCollector,
    window_feature_summary,
)
from crosslayer_transcoder.feature_dash.load import MoltCheckpointMetadata
from crosslayer_transcoder.feature_dash.render import copy_render_assets

logger = logging.getLogger(__name__)


SCHEMA_VERSION = 1


def _metadata_payload(
    meta: MoltCheckpointMetadata,
    collector: GateCollector,
    layer: int,
    dataset_name: str,
    seq_len: int,
    top_k: int,
    window: int,
) -> dict[str, Any]:
    payload = asdict(meta)
    rates = collector.activation_rate().tolist()
    # max_activation may be -inf for features that never saw a batch (e.g. an
    # empty collector); clamp to 0.0 for JSON / display.
    max_acts = [
        float(v) if v != float("-inf") else 0.0
        for v in collector.max_activation.tolist()
    ]
    payload.update(
        {
            "schema_version": SCHEMA_VERSION,
            "layer": layer,
            "n_tokens_collected": collector.total_tokens,
            "dashboard_dataset": dataset_name,
            "seq_len": seq_len,
            "top_k": top_k,
            "window": window,
            # Per-feature aggregates so the index page renders from one fetch.
            "feature_activation_rate": rates,
            "feature_max_activation": max_acts,
            # Bin edges shared by every feature's activation histogram.
            "act_histogram_edges": collector.hist_edges(),
        }
    )
    return payload


def _feature_payload(
    collector: GateCollector,
    feature_id: int,
    meta: MoltCheckpointMetadata,
    tokenizer,
    window: int,
    feature_logits: Optional[list[dict]] = None,
) -> dict[str, Any]:
    summary = collector.feature_summary(feature_id)
    body = window_feature_summary(summary, tokenizer, window=window)
    body["tier"] = meta.feature_tier[feature_id]
    body["rank"] = meta.feature_rank[feature_id]
    if feature_logits is not None:
        body["logits"] = feature_logits[feature_id]
    return body


def dump_dashboard(
    collector: GateCollector,
    meta: MoltCheckpointMetadata,
    tokenizer,
    out_dir: str | Path,
    layer: int,
    dataset_name: str = "Skylion007/openwebtext",
    seq_len: Optional[int] = None,
    window: int = 32,
    copy_assets: bool = True,
    feature_logits: Optional[list[dict]] = None,
) -> Path:
    """Write `metadata.json` + `features/<id>.json` per transform, then copy
    the static HTML/CSS/JS into `out_dir` (set `copy_assets=False` to skip).

    `seq_len` defaults to the collector's seq_len. Returns the data root.
    `feature_logits`, if provided, is the result of
    `crosslayer_transcoder.feature_dash.logits.compute_feature_logits` — a
    list of length `meta.n_features` with per-feature top-pos/top-neg/histogram
    entries. When omitted, no logit panel data is written.
    """
    out_dir = Path(out_dir)
    data_dir = out_dir / "data"
    feat_dir = data_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    seq_len = seq_len if seq_len is not None else collector.T
    top_k = collector.K

    if feature_logits is not None and len(feature_logits) != meta.n_features:
        raise ValueError(
            f"feature_logits has length {len(feature_logits)}, "
            f"expected {meta.n_features}"
        )

    md = _metadata_payload(
        meta=meta,
        collector=collector,
        layer=layer,
        dataset_name=dataset_name,
        seq_len=seq_len,
        top_k=top_k,
        window=window,
    )
    md["has_logits"] = feature_logits is not None
    (data_dir / "metadata.json").write_text(json.dumps(md, indent=2))

    width = _feature_id_width(meta.n_features)
    for f_id in range(meta.n_features):
        payload = _feature_payload(
            collector, f_id, meta, tokenizer, window, feature_logits
        )
        (feat_dir / f"{f_id:0{width}d}.json").write_text(
            json.dumps(payload, separators=(",", ":"))
        )
        if f_id and f_id % 200 == 0:
            logger.info("dumped feature %d/%d", f_id, meta.n_features)

    logger.info("wrote %d features + metadata to %s", meta.n_features, data_dir)

    if copy_assets:
        copy_render_assets(out_dir)
        logger.info(
            "wrote dashboard assets — open %s/index.html via "
            "`python -m http.server` to view",
            out_dir,
        )

    return data_dir


def _feature_id_width(n_features: int) -> int:
    """Zero-padding width for feature filenames; matches the JS in index.js."""
    return max(4, len(str(n_features - 1)))
