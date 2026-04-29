"""Build a single self-contained `bundle.html` from a dumped dashboard.

The bundle has metadata + every feature's JSON inlined as `<script
type="application/json">` blocks, plus the dashboard CSS inlined in `<style>`.
It opens by double-click — `fetch()` is not used, so `file://` works.

Routing is hash-based:
    bundle.html#index           -> sortable feature table
    bundle.html#feature=<id>    -> per-feature view

Use after `dump_dashboard(...)`:

    from crosslayer_transcoder.feature_dash import dump_dashboard, make_bundle
    dump_dashboard(collector, meta, tokenizer, out_dir, layer=8)
    bundle_path = make_bundle(out_dir)   # -> <out_dir>/bundle.html

Or skip the disk dump entirely with `make_bundle_from_state`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from crosslayer_transcoder.feature_dash.collect import (
    GateCollector,
    window_feature_summary,
)
from crosslayer_transcoder.feature_dash.dump import _feature_id_width, _metadata_payload
from crosslayer_transcoder.feature_dash.load import MoltCheckpointMetadata
from crosslayer_transcoder.feature_dash.render import TEMPLATE_DIR

logger = logging.getLogger(__name__)


def _ckpt_stem(meta: MoltCheckpointMetadata | None, fallback_metadata: dict | None = None) -> str:
    """Derive a filesystem-safe stem from the checkpoint name.

    Prefers `hf_filename` (typical when loaded from the Hub), falls back to
    the basename of `ckpt_path`. The `.ckpt` (or other) suffix is stripped.
    """
    if meta is not None:
        source = meta.hf_filename or meta.ckpt_path or ""
    else:
        assert fallback_metadata is not None
        source = (
            fallback_metadata.get("hf_filename")
            or fallback_metadata.get("ckpt_path")
            or ""
        )
    stem = Path(source).stem if source else ""
    # Replace any path separators or whitespace with `-`; drop control chars.
    cleaned = "".join(
        c if (c.isalnum() or c in "-_.") else "-" for c in stem
    ).strip("-")
    return cleaned or "checkpoint"


def default_bundle_filename(meta: MoltCheckpointMetadata) -> str:
    """`bundle_<ckpt-stem>.html`, e.g. `bundle_gpt2-molt-lam-0_00015-50M.html`."""
    return f"bundle_{_ckpt_stem(meta)}.html"


def _sanitize_for_script_tag(s: str) -> str:
    """Escape `</` so the JSON blob can't accidentally close the <script> tag.

    `<script type="application/json">` content is opaque to the HTML parser
    *except* for the literal sequence `</script>` (and a few related variants).
    Replacing `<` with `\\u003c` is the standard hardening.
    """
    return s.replace("<", "\\u003c")


def make_bundle_from_disk(out_dir: str | Path) -> Path:
    """Read `<out_dir>/data/` and write `<out_dir>/bundle_<ckpt-stem>.html`."""
    out_dir = Path(out_dir)
    data_dir = out_dir / "data"
    if not (data_dir / "metadata.json").is_file():
        raise FileNotFoundError(
            f"{data_dir/'metadata.json'} not found — run dump_dashboard first"
        )

    metadata = json.loads((data_dir / "metadata.json").read_text())
    n_features = metadata["n_features"]
    width = _feature_id_width(n_features)

    features: dict[str, dict] = {}
    for f_id in range(n_features):
        path = data_dir / "features" / f"{f_id:0{width}d}.json"
        features[str(f_id)] = json.loads(path.read_text())

    bundle_name = f"bundle_{_ckpt_stem(None, fallback_metadata=metadata)}.html"
    return _write_bundle(out_dir / bundle_name, metadata, features)


def make_bundle(
    collector: GateCollector,
    meta: MoltCheckpointMetadata,
    tokenizer,
    out_path: str | Path,
    layer: int,
    dataset_name: str = "Skylion007/openwebtext",
    seq_len: Optional[int] = None,
    window: int = 32,
    feature_logits: Optional[list[dict]] = None,
) -> Path:
    """Build a bundle directly from in-memory state — no disk dump needed.

    If `out_path` is a directory (or doesn't exist and ends with a separator),
    the bundle is written as `bundle_<ckpt-stem>.html` inside it. Otherwise
    `out_path` is treated as the explicit file to write.

    `feature_logits` mirrors `dump_dashboard`'s parameter: when provided, each
    per-feature payload includes `"logits"`.
    """
    out_path = Path(out_path)
    treat_as_dir = out_path.is_dir() or (
        not out_path.exists() and out_path.suffix == ""
    )
    if treat_as_dir:
        out_path = out_path / default_bundle_filename(meta)

    if feature_logits is not None and len(feature_logits) != meta.n_features:
        raise ValueError(
            f"feature_logits has length {len(feature_logits)}, "
            f"expected {meta.n_features}"
        )

    metadata = _metadata_payload(
        meta=meta,
        collector=collector,
        layer=layer,
        dataset_name=dataset_name,
        seq_len=seq_len if seq_len is not None else collector.T,
        top_k=collector.K,
        window=window,
    )
    metadata["has_logits"] = feature_logits is not None

    features: dict[str, dict] = {}
    for f_id in range(meta.n_features):
        summary = collector.feature_summary(f_id)
        body = window_feature_summary(summary, tokenizer, window=window)
        body["tier"] = meta.feature_tier[f_id]
        body["rank"] = meta.feature_rank[f_id]
        if feature_logits is not None:
            body["logits"] = feature_logits[f_id]
        features[str(f_id)] = body

    return _write_bundle(out_path, metadata, features)


def _write_bundle(out_path: Path, metadata: dict, features: dict) -> Path:
    template = (TEMPLATE_DIR / "bundle.html").read_text()
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
        "wrote %s (%.1f MB, %d features)",
        out_path,
        out_path.stat().st_size / 1e6,
        len(features),
    )
    return out_path
