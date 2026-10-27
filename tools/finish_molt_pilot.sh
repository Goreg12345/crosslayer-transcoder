#!/usr/bin/env bash
# Resume incomplete cases after the active pilot exits, then publish its pages.
set -euo pipefail
cd "$(dirname "$0")/.."
sample=${1:-results/molt-qwen-dashboard-campaign-layer22-10m}
unit=${2:-molt-pilot32-layer22.service}
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=

while systemctl --user is-active --quiet "$unit"; do
    sleep 10
done

complete() {
    .venv/bin/python - "$sample/pilot32/summary.json" <<'PY'
import json
import sys
from pathlib import Path
p = Path(sys.argv[1])
sys.exit(0 if p.exists() and json.loads(p.read_text())["run"]["state"] == "complete" else 1)
PY
}

# Only invalid/missing responses are retried. Completed scores are retained.
for attempt in 1 2; do
    if complete; then break; fi
    echo "Resuming incomplete pilot methods (attempt $attempt/2)"
    local/autointerp-venv/bin/python tools/run_molt_pilot.py "$sample" || true
done

.venv/bin/python tools/render_molt_pilot.py "$sample"
.venv-dashboard/bin/python tools/render_molt_dashboard.py "$sample"
if ! complete; then
    echo "Pilot pages published with incomplete cases; inspect summary.json and audit logs."
    exit 1
fi
echo "All 32 pilot transforms scored and published. Expansion remains paused."
