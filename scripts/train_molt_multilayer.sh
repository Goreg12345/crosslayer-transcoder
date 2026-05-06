#!/usr/bin/env bash
# Train a 12-layer MoLT on GPT2-small at N=50 with the original transform
# distribution (ranks=[512, 256, 128, 64, 32]) for 100M tokens.
#
# Per-layer transforms are saved as `{wandb_run_name}_layer_{l}.pt` under
# `checkpoints/molt-multilayer-N50-100M/` at end of training.
#
# Memory: requires an 80 GB-class GPU (A100/H100) — the Adam optimizer state
# alone is ~28 GB. See config header for tuning notes if running on smaller
# hardware.
#
# Usage:
#   ./scripts/train_molt_multilayer.sh                  # uses default config
#   ./scripts/train_molt_multilayer.sh --trainer.max_steps=1000   # override
set -euo pipefail

cd "$(dirname "$0")/.."

uv run clt fit \
  --config config/molt-multilayer.yaml \
  "$@"
