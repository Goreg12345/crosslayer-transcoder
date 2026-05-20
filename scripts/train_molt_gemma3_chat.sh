#!/usr/bin/env bash
# Train a multi-layer MoLT on Gemma3-4B-IT with chat-formatted data.
#
# Model dims (residual width + block count) are DERIVED from the model config at
# launch (scripts/model_dims.py) and injected as CLI overrides, so the example
# config (config/molt-multilayer-gemma3-4b-chat.yaml) never hardcodes them.
#
# Usage:
#   ./scripts/train_molt_gemma3_chat.sh                         # uses default model
#   MODEL=google/gemma-3-4b-it ./scripts/train_molt_gemma3_chat.sh
#   ./scripts/train_molt_gemma3_chat.sh --trainer.max_steps=1000   # extra overrides
#
# Requires HF access to the (gated) Gemma 3 weights — `huggingface-cli login`.
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="${MODEL:-google/gemma-3-4b-it}"
CONFIG="${CONFIG:-config/molt-multilayer-gemma3-4b-chat.yaml}"

echo "=> deriving dims for $MODEL …" >&2
read -r D L < <(uv run python scripts/model_dims.py "$MODEL" --shell)
echo "   d_acts=$D  n_layers=$L" >&2

uv run clt fit \
  --config "$CONFIG" \
  --data.init_args.model_name "$MODEL" \
  --data.init_args.n_layers "$L" \
  --data.init_args.activation_dim "$D" \
  --model.init_args.model.init_args.n_layers "$L" \
  --model.init_args.model.init_args.d_acts "$D" \
  --model.init_args.model.init_args.input_standardizer.init_args.n_layers "$L" \
  --model.init_args.model.init_args.input_standardizer.init_args.activation_dim "$D" \
  --model.init_args.model.init_args.output_standardizer.init_args.n_layers "$L" \
  --model.init_args.model.init_args.output_standardizer.init_args.activation_dim "$D" \
  "$@"
