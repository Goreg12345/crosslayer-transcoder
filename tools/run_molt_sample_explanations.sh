#!/usr/bin/env bash
# Complete a large sample's gate dashboard and labels after its collector exits.
set -euo pipefail
if [[ "${4:-}" != "--approved-full-run" ]]; then
    echo 'Full expansion is paused for review of the 32-transform pilot. Run tools/run_molt_pilot.py for the pilot; pass --approved-full-run only after explicit user approval.' >&2
    exit 2
fi
cd "$(dirname "${BASH_SOURCE[0]}")/.."
SAMPLE_DIRECTORY="${1:?Pass the collected sample directory}"
COLLECTOR_UNIT="${2:?Pass the collector systemd user unit}"
PUBLISH_PATH="${3:?Pass the new dashboard symlink path}"
export CUDA_VISIBLE_DEVICES=
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1

.venv/bin/python - "$SAMPLE_DIRECTORY" "$COLLECTOR_UNIT" <<'PY'
import json
from pathlib import Path
import subprocess
import sys
import time

root, unit = Path(sys.argv[1]), sys.argv[2]
while True:
    progress = root / "collection-progress.json"
    data = json.loads(progress.read_text()) if progress.exists() else {}
    if data.get("complete"):
        if not data.get("target_reached"):
            raise RuntimeError("Collection finished below its token target")
        metadata = json.loads((root / "metadata.json").read_text())
        if metadata["tokens"] != data["tokens"]:
            raise RuntimeError("Collection metadata disagrees with progress")
        print(f"Collection ready: {data['tokens']:,} tokens", flush=True)
        break
    status = subprocess.run(
        ["systemctl", "--user", "is-active", unit], capture_output=True, text=True
    ).stdout.strip()
    if status not in {"active", "activating", "reloading"}:
        raise RuntimeError(f"Collector stopped before completing ({status})")
    print(f"Waiting for collection: {data.get('tokens', 0):,} tokens", flush=True)
    time.sleep(30)
PY

.venv-dashboard/bin/python tools/render_molt_dashboard.py "$SAMPLE_DIRECTORY"
# Add a separate route on the existing MOLT server; keep its original sample intact.
.venv/bin/python - "$SAMPLE_DIRECTORY" "$PUBLISH_PATH" <<'PY'
from pathlib import Path
import sys
root, link = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
if link.is_symlink() and link.resolve() == root:
    pass
elif link.exists() or link.is_symlink():
    raise RuntimeError(f"Publish path already exists: {link}")
else:
    link.symlink_to(root, target_is_directory=True)
print(f"Gate dashboard published at {link}", flush=True)
PY

local/autointerp-venv/bin/python tools/explain_molt_dashboard.py "$SAMPLE_DIRECTORY" \
    --base-url https://geor.taild80f17.ts.net/v1 \
    --model qwen3.8-27b-nvfp4 --api-key-env vLLM_API_KEY \
    --limit 0 --concurrency 32 --context-limit 10000
.venv-dashboard/bin/python tools/render_molt_dashboard.py "$SAMPLE_DIRECTORY"

.venv/bin/python - "$SAMPLE_DIRECTORY" <<'PY'
from collections import Counter
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
summary = json.loads((root / "autointerp" / "summary.json").read_text())
counts = Counter(row["status"] for row in summary["features"].values())
print(f"Published explanation statuses: {dict(counts)}", flush=True)
if counts["error"] or counts["pending"]:
    raise RuntimeError("Some explanations failed or remain pending; rerun to resume")
PY
