#!/usr/bin/env bash
# An API-only Delphi overlay; the training environment is never installed into.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
revision=4fea06e6e8b68eeaf302474325fca13df95c5d6f
mkdir -p local
if [[ ! -d local/delphi/.git ]]; then
    git clone --no-checkout https://github.com/EleutherAI/delphi.git local/delphi
    git -C local/delphi checkout "$revision"
fi
[[ "$(git -C local/delphi rev-parse HEAD)" == "$revision" ]] || {
    echo 'Unexpected Delphi revision; refusing to change the existing checkout.' >&2
    exit 1
}
if [[ ! -x local/autointerp-venv/bin/python ]]; then
    uv venv --system-site-packages --python .venv/bin/python local/autointerp-venv
fi
.venv/bin/python - <<'PY'
import ast
from pathlib import Path
import subprocess
import sysconfig

site = next(Path('local/autointerp-venv/lib').glob('python*/site-packages'))
(site / 'training_environment.pth').write_text(sysconfig.get_path('purelib') + '\n')
(site / 'delphi_source.pth').write_text(str(Path('local/delphi').resolve()) + '\n')
# Upstream eagerly imports vLLM, FAISS, and embedding models for API-only use.
# Make only package re-exports lazy; explainer, prompt, and scorer code is intact.
for name in ['clients', 'latents', 'scorers']:
    rel = f'delphi/{name}/__init__.py'
    original = subprocess.check_output(
        ['git', '-C', 'local/delphi', 'show', f'HEAD:{rel}'], text=True)
    imports = {alias.asname or alias.name: ('.' * node.level + node.module, alias.name)
               for node in ast.parse(original).body if isinstance(node, ast.ImportFrom)
               for alias in node.names}
    (Path('local/delphi') / rel).write_text(
        '"""Lazy optional backend imports for remote-only use."""\n'
        'from importlib import import_module\n'
        f'_IMPORTS = {imports!r}\n__all__ = list(_IMPORTS)\n'
        'def __getattr__(name):\n'
        '    if name not in _IMPORTS:\n        raise AttributeError(name)\n'
        '    module, attribute = _IMPORTS[name]\n'
        '    value = getattr(import_module(module, __name__), attribute)\n'
        '    globals()[name] = value\n    return value\n')
PY
uv pip install --python local/autointerp-venv/bin/python \
    orjson==3.12.0 blobfile==3.3.0 aiofiles==25.1.0 \
    beartype==0.22.9 tiktoken==0.14.0 simple-parsing==0.1.9
local/autointerp-venv/bin/python -c \
    'from delphi.explainers import DefaultExplainer; from delphi.scorers import DetectionScorer; print("Delphi remote explainers ready")'
