#!/bin/bash

# CrossLayer Transcoder Experiment Runner
# This script creates tmux sessions for each GPU and runs experiments with different parameter combinations

set -e

# Configuration
BASE_CONFIG_DIR="config"
CLI_COMMAND="python cli.py fit --config"



# Function to start a tmux session and run experiments
run_experiments() {
    local gpu_id="$1"
    local base_config="$2"
    local nonlinearity_type="$3"
    
    local session_name="gpu${gpu_id}_experiments"
    
    echo "Starting experiments for GPU $gpu_id with config $base_config"
    
    # Kill existing session if it exists
    tmux kill-session -t "$session_name" 2>/dev/null || true
    
    # Create new tmux session
    tmux new-session -d -s "$session_name"
    
    # Feature counts to test
    local feature_counts=(7000 10_000)
    
    # Build a single chained command to run experiments sequentially
    local chained_cmd=""
    
    if [ "$gpu_id" == "0" ]; then
        # GPU 0: Only run with original config (JumpReLU)
        for features in "${feature_counts[@]}"; do
            local cmd="timeout 2h $CLI_COMMAND $base_config --model.init_args.model.init_args.d_features $features --model.init_args.dead_features.init_args.n_features $features"
            echo "Adding to sequence: $cmd"
            
            if [ -z "$chained_cmd" ]; then
                chained_cmd="$cmd"
            else
                chained_cmd="$chained_cmd; echo 'Previous experiment finished (success/failure/timeout). Starting next experiment...'; $cmd"
            fi
        done
    else
        # GPUs 1, 2, 3: Run with different k values
        local base_k_values=(5 8 10 20 50)
        
        for features in "${feature_counts[@]}"; do
            for base_k in "${base_k_values[@]}"; do
                local k_value="$base_k"
                
                # Apply multipliers based on nonlinearity type
                if [ "$nonlinearity_type" == "per_layer" ]; then
                    # Per Layer TopK: use base k value directly
                    k_value="$base_k"
                elif [ "$nonlinearity_type" == "per_sample" ]; then
                    # Per Sample TopK: multiply base k by 12 (number of layers)
                    k_value=$((base_k * 12))
                elif [ "$nonlinearity_type" == "batch" ]; then
                    # Batch TopK: multiply base k by 12 * 4000 (layers * batch_size)
                    k_value=$((base_k * 12 * 4000))
                fi
                
                local cmd="timeout 2h $CLI_COMMAND $base_config --model.init_args.model.init_args.d_features $features --model.init_args.dead_features.init_args.n_features $features --model.init_args.model.init_args.nonlinearity.init_args.k $k_value"
                
                echo "Adding to sequence: $cmd (k=$k_value, timeout=2h)"
                
                if [ -z "$chained_cmd" ]; then
                    chained_cmd="$cmd"
                else
                    chained_cmd="$chained_cmd; echo 'Previous experiment finished (success/failure/timeout). Starting next experiment...'; $cmd"
                fi
            done
        done
    fi
    
    # Send the complete chained command to tmux
    echo "Sending chained command to tmux session '$session_name'"
    tmux send-keys -t "$session_name" "$chained_cmd" Enter
    
    # Detach the session
    tmux detach-client -s "$session_name" 2>/dev/null || true
    
    echo "✓ Started tmux session '$session_name' for GPU $gpu_id"
}

# Main execution
echo "=== CrossLayer Transcoder Experiment Runner ==="
echo "Starting tmux sessions with parameter overrides..."

# GPU 0: Default config (JumpReLU) - DISABLED
# run_experiments 0 "$BASE_CONFIG_DIR/default.yaml" "jumprelu"

# GPU 1: Per Layer TopK
run_experiments 1 "$BASE_CONFIG_DIR/per_layer_topk_gpu_1.yaml" "per_layer"

# GPU 2: Per Sample TopK  
run_experiments 2 "$BASE_CONFIG_DIR/per_sample_topk_gpu_2.yaml" "per_sample"

# GPU 3: Batch TopK
run_experiments 3 "$BASE_CONFIG_DIR/batch_topk_gpu_3.yaml" "batch"

echo ""
echo "=== All experiments started ==="
echo "Active tmux sessions:"
tmux list-sessions | grep gpu || echo "No GPU sessions found"

echo ""
echo "To monitor a specific GPU session:"
echo "  tmux attach-session -t gpu1_experiments" 
echo "  tmux attach-session -t gpu2_experiments"
echo "  tmux attach-session -t gpu3_experiments"

echo ""
echo "To list all sessions: tmux list-sessions"
echo "To kill a session: tmux kill-session -t <session_name>" 