#!/bin/bash

# CrossLayer Transcoder Batch TopK Experiment Runner
# This script creates tmux sessions for each GPU and runs experiments with different hyperparameter combinations
# Each GPU runs a different subset of experiments to avoid duplication

set -e

# Configuration
BASE_CONFIG="config/batch_topk_gpu_3_tests.yaml"
CLI_COMMAND="python cli.py fit --config"

# Hyperparameters to test
TOPK_AUX_K_VALUES=(0 100_000 1_000_000 5_000_000)
TOKENS_TILL_DEAD_VALUES=(100_000 1_000_000)
AUX_LOSS_SCALE_VALUES=(0.03 0.15)

# Generate all combinations upfront
generate_all_combinations() {
    local combinations=()
    
    for topk_aux_k in "${TOPK_AUX_K_VALUES[@]}"; do
        for tokens_till_dead in "${TOKENS_TILL_DEAD_VALUES[@]}"; do
            for aux_loss_scale in "${AUX_LOSS_SCALE_VALUES[@]}"; do
                combinations+=("${topk_aux_k},${tokens_till_dead},${aux_loss_scale}")
            done
        done
    done
    
    echo "${combinations[@]}"
}

# Function to start a tmux session and run experiments
run_experiments() {
    local gpu_id="$1"
    shift
    local experiments=("$@")
    
    local session_name="gpu${gpu_id}_batch_topk"
    
    echo "Starting batch TopK experiments for GPU $gpu_id"
    echo "  → Running ${#experiments[@]} experiments"
    
    # Kill existing session if it exists
    tmux kill-session -t "$session_name" 2>/dev/null || true
    
    # Create new tmux session
    tmux new-session -d -s "$session_name"
    
    # Build a single chained command to run experiments sequentially
    local chained_cmd=""
    
    # Generate commands for assigned experiments
    for experiment in "${experiments[@]}"; do
        IFS=',' read -r topk_aux_k tokens_till_dead aux_loss_scale <<< "$experiment"
        
        local cmd="timeout 2h $CLI_COMMAND $BASE_CONFIG"
        
        # Add GPU-specific overrides
        cmd="$cmd --trainer.devices [${gpu_id}]"
        cmd="$cmd --model.init_args.replacement_model.init_args.device_map cuda:${gpu_id}"
        cmd="$cmd --data.init_args.device_map cuda:0"  # Keep data generation on GPU 0
        
        # Add hyperparameter overrides
        cmd="$cmd --model.init_args.topk_aux.init_args.k $topk_aux_k"
        cmd="$cmd --model.init_args.tokens_till_dead $tokens_till_dead"
        cmd="$cmd --model.init_args.aux_loss_scale $aux_loss_scale"
        
        echo "  Adding to sequence: GPU $gpu_id, topk_aux_k=$topk_aux_k, tokens_till_dead=$tokens_till_dead, aux_loss_scale=$aux_loss_scale"
        
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
    local gpus=(1 2 3)
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
echo "=== CrossLayer Transcoder Batch TopK Experiment Runner ==="
echo "Base config: $BASE_CONFIG"
echo "Testing hyperparameters:"
echo "  - topk_aux.k: [${TOPK_AUX_K_VALUES[*]}]"
echo "  - tokens_till_dead: [${TOKENS_TILL_DEAD_VALUES[*]}]"
echo "  - aux_loss_scale: [${AUX_LOSS_SCALE_VALUES[*]}]"
echo ""
echo "Starting tmux sessions with distributed experiments..."

# Distribute and run experiments
distribute_experiments

echo ""
echo "=== All experiments started ==="
echo "Active tmux sessions:"
tmux list-sessions | grep batch_topk || echo "No batch TopK sessions found"

echo ""
echo "To monitor a specific GPU session:"
echo "  tmux attach-session -t gpu1_batch_topk" 
echo "  tmux attach-session -t gpu2_batch_topk"
echo "  tmux attach-session -t gpu3_batch_topk"

echo ""
echo "To list all sessions: tmux list-sessions"
echo "To kill a session: tmux kill-session -t <session_name>"

# Calculate total experiments across all GPUs
total_combinations=$((${#TOPK_AUX_K_VALUES[@]} * ${#TOKENS_TILL_DEAD_VALUES[@]} * ${#AUX_LOSS_SCALE_VALUES[@]}))
echo ""
echo "Total unique experiments: $total_combinations"
echo "Estimated total runtime: ~$((total_combinations * 2)) hours (assuming 2h per experiment)"
echo "With parallel execution across 3 GPUs: ~$((total_combinations * 2 / 3)) hours" 