#!/usr/bin/env bash
# Ctrl-C stops training. Run the same command again to resume the last checkpoint.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
layers=(22 0 1 10 30 2 3 4 5 6 7 8 9 11 12 13 14 15 16 17 18 19 20 21 23 24 25 26 27 28 29 31 32 33 34 35)
dry_run=false
for arg in "$@"; do
    case "$arg" in
        --first-five) layers=("${layers[@]:0:5}") ;;
        --dry-run) dry_run=true ;;
        *) echo "Usage: $0 [--first-five] [--dry-run]" >&2; exit 2 ;;
    esac
done
mkdir -p logs/qwen3-all-mlps
exec 9>logs/qwen3-all-mlps/campaign.lock
flock -n 9 || { echo 'Campaign already running' >&2; exit 1; }
if ! "$dry_run"; then
    set -a
    source .env
    set +a
fi
training_pid=''
cleanup() {
    if [[ -n "$training_pid" ]]; then
        kill -TERM -- "-$training_pid" 2>/dev/null || true
        sleep 2
        kill -KILL -- "-$training_pid" 2>/dev/null || true
        wait "$training_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for layer in "${layers[@]}"; do
    checkpoint_dir="checkpoints/molt-qwen3-4b/layer-$layer/N80-rank512to32-lr4e-5-sp7.5e-5-native-ramp80"
    checkpoint="$checkpoint_dir/training.ckpt"
    resume=()
    if [[ -f "$checkpoint" ]]; then
        step=$(.venv/bin/python tools/molt_checkpoint_step.py "$checkpoint")
        if (( step >= 100000 )); then
            echo "Layer $layer complete; keeping $checkpoint"
            continue
        fi
        resume=(--ckpt_path "$checkpoint")
        echo "Resuming layer $layer at step $step"
    fi
    command=(.venv/bin/clt fit --config config/molt-qwen3-all-mlps.yaml
        --model.init_args.layer "$layer" --data.init_args.activation_layer "$layer"
        --trainer.logger.init_args.name "qwen3-4b-layer${layer}-N80-r512-32-lr4e-5-sp7.5e-5"
        --data.init_args.wandb_logging.run_name "qwen3-4b-layer${layer}-activations"
        --trainer.callbacks "[{class_path: crosslayer_transcoder.utils.callbacks.TimedCheckpointCallback, init_args: {checkpoint_dir: $checkpoint_dir, interval_minutes: 90, filename: training.ckpt}}]"
        "${resume[@]}")
    if "$dry_run"; then printf '%q ' "${command[@]}"; printf '\n'; continue; fi
    if pgrep -f '[/]clt fit([[:space:]]|$)' >/dev/null; then
        echo 'Another training run is active' >&2; exit 1
    fi
    log="logs/qwen3-all-mlps/layer-${layer}-$(date +%s%N).log"
    echo "Training layer $layer; $log"
    setsid "${command[@]}" 9>&- > >(tee "$log") 2>&1 &
    training_pid=$!
    while kill -0 "$training_pid" 2>/dev/null; do
        if tail -c 65536 "$log" | grep -F 'Traceback (most recent call last)' >/dev/null; then
            echo "Training failed; see $log" >&2; exit 1
        fi
        sleep 10 & wait $! || true
    done
    if wait "$training_pid"; then
        training_status=0
    else
        training_status=$?
    fi
    training_pid=''
    if (( training_status != 0 )); then
        echo "Training layer $layer exited with status $training_status; see $log" >&2
        exit "$training_status"
    fi
    step=$(.venv/bin/python tools/molt_checkpoint_step.py "$checkpoint")
    if (( step < 100000 )); then echo "Stopped at step $step; run this script again to resume"; exit 1; fi
    echo "Layer $layer complete at step $step; continuing to next layer"
done
echo 'Campaign complete; all requested layers reached step 100000'
