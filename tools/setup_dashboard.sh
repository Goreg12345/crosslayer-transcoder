#!/usr/bin/env bash
# Run from the repository root after setting up the project's .venv.
set -euo pipefail
uv venv --system-site-packages --python .venv/bin/python .venv-dashboard
.venv/bin/python - <<'PY'
from pathlib import Path
base = Path('.venv/lib/python3.12/site-packages').resolve()
site = next(Path('.venv-dashboard/lib').glob('python*/site-packages'))
(site/'training_environment.pth').write_text(str(base)+'\n')
PY
uv pip install --python .venv-dashboard/bin/python --no-deps -r tools/dashboard-requirements.txt
.venv-dashboard/bin/python -c 'import sae_vis; print("SAE-Vis ready")'
