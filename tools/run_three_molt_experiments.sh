#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/georg/Code/crosslayer-transcoder"
base_config="config/molt-gemma3-4b-it-5090-extractor-rtx6000.yaml"

cd "$repo_dir"
set -a
source .env
set +a
export HF_TOKEN="$HUGGINGFACE_API_KEY"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

run_experiment() {
    local label="$1"
    local checkpoint="$2"
    shift 2
    local log_file="logs/${label}.log"
    local config_args=(--config "$base_config")
    local override
    for override in "$@"; do
        config_args+=(--config "$override")
    done

    if [[ -e "$checkpoint" ]]; then
        echo "Refusing to overwrite existing checkpoint: $checkpoint" >&2
        return 1
    fi

    echo "Starting $label"
    uv run clt fit "${config_args[@]}" 2>&1 | tee "$log_file"

    if [[ ! -s "$checkpoint" ]]; then
        echo "$label exited without producing $checkpoint; stopping chain." >&2
        return 1
    fi
    echo "Completed $label"
}

mkdir -p logs
run_experiment \
    "molt-gemma3-4b-it-zero443" \
    "checkpoints/molt-gemma3-4b-it-rtx6000/layer-22/zero443/training.ckpt" \
    "config/molt-gemma3-4b-it-zero443.yaml"
run_experiment \
    "molt-qwen3-4b-chat-ultrachat" \
    "checkpoints/molt-qwen3-4b/layer-22/chat-ultrachat/training.ckpt" \
    "config/molt-qwen3-4b-chat.yaml"
run_experiment \
    "molt-qwen3-4b-base-openwebtext" \
    "checkpoints/molt-qwen3-4b/layer-22/base-openwebtext/training.ckpt" \
    "config/molt-qwen3-4b-chat.yaml" \
    "config/molt-qwen3-4b-base-openwebtext.yaml"
