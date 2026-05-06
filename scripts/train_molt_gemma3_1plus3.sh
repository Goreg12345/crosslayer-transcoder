#!/usr/bin/env bash
# Launch the 1-producer + 3-DDP-consumer Gemma3-4B-it MoLT training run.
#
# GPU 0 runs a standalone producer (Gemma3 forward + activation buffer).
# GPUs 1,2,3 run Lightning DDP, each rank attaching to that producer's buffer.
#
# The producer is launched in the background, then we wait for it to print
# "buffer ready" before starting the trainer. On exit (Ctrl+C, trainer crash,
# trainer success) we tear the producer down too.
#
# Usage:
#   ./scripts/train_molt_gemma3_1plus3.sh                        # default config
#   ./scripts/train_molt_gemma3_1plus3.sh --trainer.max_steps=1000   # CLI override
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-config/molt-multilayer-gemma3-4b-it-1_plus_3ddp.yaml}"
PRODUCER_LOG="${PRODUCER_LOG:-/tmp/molt-gemma3-producer.log}"

echo "[launcher] producer log -> $PRODUCER_LOG"
: > "$PRODUCER_LOG"

uv run python -m crosslayer_transcoder.data.standalone_producer --config "$CONFIG" \
  > "$PRODUCER_LOG" 2>&1 &
PRODUCER_PID=$!
echo "[launcher] standalone_producer pid=$PRODUCER_PID"

cleanup() {
  if kill -0 "$PRODUCER_PID" 2>/dev/null; then
    echo "[launcher] sending SIGINT to producer pid=$PRODUCER_PID"
    kill -INT "$PRODUCER_PID" 2>/dev/null || true
    # give it ~15s to drain wandb / unlink shm
    for _ in $(seq 1 15); do
      if ! kill -0 "$PRODUCER_PID" 2>/dev/null; then break; fi
      sleep 1
    done
    if kill -0 "$PRODUCER_PID" 2>/dev/null; then
      echo "[launcher] producer still alive, SIGKILL"
      kill -9 "$PRODUCER_PID" 2>/dev/null || true
    fi
  fi
}
trap cleanup EXIT INT TERM

echo "[launcher] waiting for producer to publish 'buffer ready' ..."
for _ in $(seq 1 600); do
  if grep -q "buffer ready" "$PRODUCER_LOG"; then
    echo "[launcher] producer reports buffer ready"
    break
  fi
  if ! kill -0 "$PRODUCER_PID" 2>/dev/null; then
    echo "[launcher] producer exited before buffer was ready; tail of $PRODUCER_LOG:"
    tail -n 50 "$PRODUCER_LOG"
    exit 1
  fi
  sleep 1
done

if ! grep -q "buffer ready" "$PRODUCER_LOG"; then
  echo "[launcher] timed out waiting for producer; tail of $PRODUCER_LOG:"
  tail -n 50 "$PRODUCER_LOG"
  exit 1
fi

echo "[launcher] starting trainer (Lightning DDP on devices [1,2,3])"
uv run clt fit --config "$CONFIG" "$@"
