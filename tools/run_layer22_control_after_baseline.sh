#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/georg/Code/crosslayer-transcoder"
baseline_session="molt-gemma3"
legacy_checkpoint="$repo_dir/checkpoints/molt-gemma3-4b-it-rtx6000/clt.ckpt"
baseline_dir="$repo_dir/checkpoints/molt-gemma3-4b-it-rtx6000/layer-22/baseline"
baseline_checkpoint="$baseline_dir/training.ckpt"

cd "$repo_dir"
while tmux list-sessions -F '#{session_name}' 2>/dev/null | grep -Fxq "$baseline_session"; do
    sleep 30
done

if [[ ! -f "$legacy_checkpoint" ]]; then
    echo "Baseline ended without its expected checkpoint: $legacy_checkpoint" >&2
    exit 1
fi

global_step="$({ uv run python - "$legacy_checkpoint" <<'PY'
import sys
import torch

checkpoint = torch.load(
    sys.argv[1], map_location="cpu", mmap=True, weights_only=False
)
print(checkpoint.get("global_step", -1))
PY
} | tail -n 1)"

if (( global_step < 100000 )); then
    echo "Baseline checkpoint is incomplete (global_step=$global_step); control not launched." >&2
    exit 1
fi

mkdir -p "$baseline_dir"
if [[ -e "$baseline_checkpoint" ]]; then
    echo "Refusing to overwrite existing baseline artifact: $baseline_checkpoint" >&2
    exit 1
fi
mv "$legacy_checkpoint" "$baseline_checkpoint"

set -a
source .env
set +a
export HF_TOKEN="$HUGGINGFACE_API_KEY"

exec uv run clt fit \
    --config config/molt-gemma3-4b-it-5090-extractor-rtx6000.yaml \
    --config config/molt-gemma3-4b-it-preact-control.yaml
