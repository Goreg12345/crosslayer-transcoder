#!/bin/bash

# CrossLayer Transcoder Batch Encoder Init Scaler Experiment Runner
# This script creates tmux sessions for each GPU and runs experiments with different enc_init_scaler values
# Each GPU runs a different subset of experiments to avoid duplication

set -e

# Configuration
BASE_CONFIG="config/batch_topk_gpu_3_tests.yaml"
CLI_COMMAND="python cli.py fit --config"

# Encoder init scaler values to test
ENC_INIT_SCALER_VALUES=(0.001 0.01 0.05 0.1 0.5 0.8 1.0 1.2 1.5 2 10 50)

# Generate all combinations upfront (in this case, just the enc_init_scaler values)
generate_all_combinations() {
    local combinations=()
    
    for enc_init_scaler in "${ENC_INIT_SCALER_VALUES[@]}"; do
        combinations+=("${enc_init_scaler}")
    done
    
    echo "${combinations[@]}"
}

# Function to start a tmux session and run experiments
run_experiments() {
    local gpu_id="$1"
    shift
    local experiments=("$@")
    
    local session_name="gpu${gpu_id}_batch_enc_init_scaler"
    
    echo "Starting batch Encoder Init Scaler experiments for GPU $gpu_id"
    echo "  → Running ${#experiments[@]} experiments"
    
    # Kill existing session if it exists
    tmux kill-session -t "$session_name" 2>/dev/null || true
    
    # Create new tmux session
    tmux new-session -d -s "$session_name"
    
    # Build a single chained command to run experiments sequentially
    local chained_cmd=""
    
    # Generate commands for assigned experiments
    for experiment in "${experiments[@]}"; do
        local enc_init_scaler="$experiment"
        
        local cmd="timeout 1h $CLI_COMMAND $BASE_CONFIG"
        
        # Add GPU-specific overrides
        cmd="$cmd --trainer.devices [${gpu_id}]"
        cmd="$cmd --model.init_args.replacement_model.init_args.device_map cuda:${gpu_id}"
        cmd="$cmd --data.init_args.device_map cuda:1"  # Keep data generation on GPU 0
        
        # Add encoder init scaler override
        cmd="$cmd --model.init_args.model.init_args.enc_init_scaler $enc_init_scaler"
        
        echo "  Adding to sequence: GPU $gpu_id, enc_init_scaler=$enc_init_scaler"
        
        if [ -z "$chained_cmd" ]; then
            chained_cmd="$cmd"
        else
            chained_cmd="$chained_cmd; echo 'Previous experiment finished (success/failure/timeout). Starting next experiment...'; $cmd"
        fi
    done
    
    # Send the complete chained command to tmux
    echo "  Sending chained command to tmux session '$session_name'"
    tmux send-keys -t "$session_name" "$chained_cmd" Enter
    
    # Detach the session
    tmux detach-client -s "$session_name" 2>/dev/null || true
    
    echo "✓ Started tmux session '$session_name' for GPU $gpu_id"
    echo "  → Running ${#experiments[@]} experiments (2h timeout each)"
}

# Distribute experiments across GPUs
distribute_experiments() {
    local all_combinations=($(generate_all_combinations))
    local total_experiments=${#all_combinations[@]}
    local gpus=(0 1)
    local num_gpus=${#gpus[@]}
    
    echo "Total experiments to distribute: $total_experiments"
    echo "GPUs available: ${gpus[@]}"
    
    # Calculate experiments per GPU
    local base_experiments_per_gpu=$((total_experiments / num_gpus))
    local extra_experiments=$((total_experiments % num_gpus))
    
    echo "Base experiments per GPU: $base_experiments_per_gpu"
    echo "Extra experiments for first $extra_experiments GPUs: 1 each"
    
    local start_idx=0
    
    for i in "${!gpus[@]}"; do
        local gpu_id=${gpus[$i]}
        local experiments_for_this_gpu=$base_experiments_per_gpu
        
        # Add one extra experiment to first few GPUs if there's a remainder
        if [ $i -lt $extra_experiments ]; then
            experiments_for_this_gpu=$((experiments_for_this_gpu + 1))
        fi
        
        # Extract experiments for this GPU
        local gpu_experiments=("${all_combinations[@]:$start_idx:$experiments_for_this_gpu}")
        
        echo ""
        echo "GPU $gpu_id will run experiments $((start_idx + 1)) to $((start_idx + experiments_for_this_gpu))"
        
        # Start experiments for this GPU
        run_experiments "$gpu_id" "${gpu_experiments[@]}"
        
        start_idx=$((start_idx + experiments_for_this_gpu))
    done
}

# Main execution
echo "=== CrossLayer Transcoder Batch Encoder Init Scaler Experiment Runner ==="
echo "Base config: $BASE_CONFIG"
echo "Testing hyperparameters:"
echo "  - enc_init_scaler: [${ENC_INIT_SCALER_VALUES[*]}]"
echo ""
echo "Starting tmux sessions with distributed experiments..."

# Distribute and run experiments
distribute_experiments

echo ""
echo "=== All experiments started ==="
echo "Active tmux sessions:"
tmux list-sessions | grep batch_enc_init_scaler || echo "No batch Encoder Init Scaler sessions found"

echo ""
echo "To monitor a specific GPU session:"
echo "  tmux attach-session -t gpu0_batch_enc_init_scaler" 
echo "  tmux attach-session -t gpu1_batch_enc_init_scaler"

echo ""
echo "To list all sessions: tmux list-sessions"
echo "To kill a session: tmux kill-session -t <session_name>"

# Calculate total experiments across all GPUs
total_combinations=${#ENC_INIT_SCALER_VALUES[@]}
echo ""
echo "Total unique experiments: $total_combinations"
echo "Estimated total runtime: ~$((total_combinations * 2)) hours (assuming 2h per experiment)"
echo "With parallel execution across 2 GPUs: ~$((total_combinations * 2 / 2)) hours" 