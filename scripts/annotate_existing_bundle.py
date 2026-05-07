"""Add LLM descriptions to an already-built multilayer bundle.

Takes a `bundle_*.html` written by `build_multilayer_dashboard.py`, runs the
selected features through the OpenRouter annotator, and writes a new bundle
with the descriptions inlined. The original is left untouched (a `.bak` copy
is made next to it for safety).

This is the path you want when:
  * you built the dashboards without `--annotate` to get something to look at
    fast, and now want to fill in descriptions; or
  * you want to re-annotate with a different model without re-running the GPU
    forward pass.

Examples
--------
    OPEN_ROUTER_API_KEY=... uv run python scripts/annotate_existing_bundle.py \\
        --bundle feature_dash/molt-multilayer-gpt2-N50/bundle_*.html \\
        --model anthropic/claude-haiku-4.5 \\
        --workers 20
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
import time
from pathlib import Path

from crosslayer_transcoder.feature_dash.annotate import annotate_features


_METADATA_RE = re.compile(
    r'<script id="metadata" type="application/json">(.*?)</script>',
    re.DOTALL,
)
_FEATURES_RE = re.compile(
    r'<script id="features" type="application/json">(.*?)</script>',
    re.DOTALL,
)


def _unsanitize(s: str) -> str:
    """Reverse `_sanitize_for_script_tag` so the JSON parses."""
    return s.replace("\\u003c", "<")


def _sanitize(s: str) -> str:
    return s.replace("<", "\\u003c")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", required=True, help="path to the bundle .html")
    p.add_argument("--model", default="anthropic/claude-haiku-4.5",
                   help="OpenRouter model id (default: anthropic/claude-haiku-4.5)")
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--max-tokens", type=int, default=80)
    p.add_argument("--max-per-layer", type=int, default=None,
                   help="cap features per layer (top-K by max_activation)")
    p.add_argument("--min-max-activation", type=float, default=0.0,
                   help="skip features whose peak activation is below this")
    p.add_argument("--out", default=None,
                   help="output path; defaults to overwriting --bundle "
                        "(a .bak copy is created either way)")
    p.add_argument("--dry-run", action="store_true",
                   help="report what would be annotated and exit")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    bundle_path = Path(args.bundle)
    if not bundle_path.is_file():
        print(f"bundle not found: {bundle_path}", file=sys.stderr)
        return 1

    print(f"=> reading {bundle_path} ({bundle_path.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)
    html = bundle_path.read_text()
    md_match = _METADATA_RE.search(html)
    feat_match = _FEATURES_RE.search(html)
    if not md_match or not feat_match:
        print("bundle is missing the metadata/features JSON blocks", file=sys.stderr)
        return 2
    metadata = json.loads(_unsanitize(md_match.group(1)))
    features = json.loads(_unsanitize(feat_match.group(1)))

    # Build the (layer, feature_id, examples) list. Drop entries that already
    # have a description so we don't pay to re-annotate.
    items: list[tuple[int, int, list[dict]]] = []
    skipped_have_desc = 0
    skipped_low_act = 0
    skipped_no_examples = 0
    by_layer: dict[int, list[tuple[int, dict, str]]] = {}

    for key, body in features.items():
        layer = body["layer"]
        feature_id = body["feature_id"]
        examples = body.get("examples") or []
        existing = (body.get("description") or "").strip()
        if existing:
            skipped_have_desc += 1
            continue
        if not examples:
            skipped_no_examples += 1
            continue
        if body.get("max_activation", 0.0) < args.min_max_activation:
            skipped_low_act += 1
            continue
        by_layer.setdefault(layer, []).append((feature_id, body, key))

    # Per-layer cap.
    for layer, lst in by_layer.items():
        if args.max_per_layer is not None and len(lst) > args.max_per_layer:
            lst.sort(key=lambda t: t[1].get("max_activation", 0.0), reverse=True)
            by_layer[layer] = lst[: args.max_per_layer]

    for layer, lst in by_layer.items():
        for feature_id, body, _ in lst:
            items.append((layer, feature_id, body["examples"]))

    print(
        f"   {len(items):,} features to annotate "
        f"(skipped: {skipped_have_desc:,} already-described, "
        f"{skipped_low_act:,} low max_act, {skipped_no_examples:,} no examples)",
        file=sys.stderr,
    )

    # Cost / time rough estimate.
    n = len(items)
    est_in = n * 700
    est_out = n * 60
    # Haiku 4.5 list price ~$1/Min, $5/Mout (informational).
    est_cost = est_in / 1e6 * 1.0 + est_out / 1e6 * 5.0
    est_seconds = max(30, n / max(1, args.workers) * 2.0)
    print(
        f"   est ~${est_cost:.2f}, ~{est_seconds/60:.1f} min @ {args.workers} workers",
        file=sys.stderr,
    )
    if args.dry_run or n == 0:
        return 0

    t0 = time.monotonic()
    descriptions = annotate_features(
        items,
        model=args.model,
        max_workers=args.workers,
        max_tokens=args.max_tokens,
    )
    actual = time.monotonic() - t0
    n_filled = sum(1 for v in descriptions.values() if v)
    print(
        f"   annotated {n_filled}/{n} features in {actual:.1f}s ({actual/60:.1f} min)",
        file=sys.stderr,
    )

    # Splice descriptions back in: both the per-feature body and the
    # corresponding entry in metadata.entries.
    for (layer, feature_id), desc in descriptions.items():
        if not desc:
            continue
        key = f"L{layer}F{feature_id}"
        if key in features:
            features[key]["description"] = desc
    for entry in metadata.get("entries", []):
        layer = entry["layer"]
        feature_id = entry["feature_id"]
        if (layer, feature_id) in descriptions and descriptions[(layer, feature_id)]:
            entry["description"] = descriptions[(layer, feature_id)]

    metadata_json = _sanitize(json.dumps(metadata, separators=(",", ":")))
    features_json = _sanitize(json.dumps(features, separators=(",", ":")))

    new_html = (
        html[: md_match.start(1)]
        + metadata_json
        + html[md_match.end(1) : feat_match.start(1)]
        + features_json
        + html[feat_match.end(1) :]
    )

    out_path = Path(args.out) if args.out else bundle_path
    if out_path == bundle_path:
        backup = bundle_path.with_suffix(bundle_path.suffix + ".bak")
        if not backup.exists():
            shutil.copy2(bundle_path, backup)
            print(f"   backup written to {backup}", file=sys.stderr)

    out_path.write_text(new_html)
    print(
        f"=> wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB) — "
        f"open by double-click",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
