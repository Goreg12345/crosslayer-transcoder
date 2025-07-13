#!/bin/bash

# CrossLayer Transcoder Batch Optimizer Experiment Runner
# This script creates a tmux session on GPU 0 and runs experiments with different optimizer configurations
# Sweeps over beta1, beta2, optimizer type, tied_init, learning rate, and warmup steps

set -e

# Configuration
BASE_CONFIG="config/batch_topk_optims.yaml"
CLI_COMMAND="python cli.py fit --config"

# Hyperparameter values to sweep
BETA1_VALUES=(0 0.9)
BETA2_VALUES=(0 0.9 0.98 0.999)
OPTIMIZER_VALUES=("adam" "adamw")
TIED_INIT_VALUES=(True False)
LR_VALUES=(1e-4 1e-3)
WARMUP_STEPS_VALUES=(0 500)

# Generate all combinations upfront
generate_all_combinations() {
    local combinations=()
    
    for beta1 in "${BETA1_VALUES[@]}"; do
        for beta2 in "${BETA2_VALUES[@]}"; do
            for optimizer in "${OPTIMIZER_VALUES[@]}"; do
                for tied_init in "${TIED_INIT_VALUES[@]}"; do
                    for lr in "${LR_VALUES[@]}"; do
                        for warmup_steps in "${WARMUP_STEPS_VALUES[@]}"; do
                            combinations+=("${beta1},${beta2},${optimizer},${tied_init},${lr},${warmup_steps}")
                        done
                    done
                done
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
    
    local session_name="gpu${gpu_id}_batch_optimizer"
    
    echo "Starting batch Optimizer experiments for GPU $gpu_id"
    echo "  → Running ${#experiments[@]} experiments"
    
    # Kill existing session if it exists
    tmux kill-session -t "$session_name" 2>/dev/null || true
    
    # Create new tmux session
    tmux new-session -d -s "$session_name"
    
    # Create a script that will run all experiments sequentially
    local script_file="/tmp/batch_optimizer_experiments_${session_name}.sh"
    echo "#!/bin/bash" > "$script_file"
    echo "set -e" >> "$script_file"
    echo "" >> "$script_file"
    
    # Generate commands for assigned experiments
    for experiment in "${experiments[@]}"; do
        # Parse experiment parameters
        IFS=',' read -r beta1 beta2 optimizer tied_init lr warmup_steps <<< "$experiment"
        
        local cmd="timeout 2h $CLI_COMMAND $BASE_CONFIG"
        
        # Add GPU-specific overrides
        cmd="$cmd --trainer.devices [${gpu_id}]"
        cmd="$cmd --model.init_args.replacement_model.init_args.device_map cuda:${gpu_id}"
        cmd="$cmd --data.init_args.device_map cuda:0"  # Keep data generation on GPU 0
        
        # Add optimizer parameter overrides
        cmd="$cmd --model.init_args.learning_rate $lr"
        cmd="$cmd --model.init_args.optimizer $optimizer"
        cmd="$cmd --model.init_args.beta1 $beta1"
        cmd="$cmd --model.init_args.beta2 $beta2"
        cmd="$cmd --model.init_args.warmup_steps $warmup_steps"
        cmd="$cmd --model.init_args.model.init_args.tied_init $tied_init"
        
        # Set wandb run name with parameter combination
        local wandb_run_name="opt-${optimizer}_b1-${beta1}_b2-${beta2}_lr-${lr}_warmup-${warmup_steps}_tied-${tied_init}"
        cmd="$cmd --trainer.logger.init_args.name $wandb_run_name"
        
        echo "  Adding to sequence: GPU $gpu_id, beta1=$beta1, beta2=$beta2, optimizer=$optimizer, tied_init=$tied_init, lr=$lr, warmup_steps=$warmup_steps"
        
        # Add to script file
        echo "echo 'Starting experiment: $wandb_run_name'" >> "$script_file"
        echo "$cmd" >> "$script_file"
        echo "echo 'Experiment finished (success/failure/timeout): $wandb_run_name'" >> "$script_file"
        echo "" >> "$script_file"
    done
    
    # Make script executable
    chmod +x "$script_file"
    
    # Send the script execution command to tmux
    echo "  Sending script execution command to tmux session '$session_name'"
    tmux send-keys -t "$session_name" "$script_file" Enter
    
    # Detach the session
    tmux detach-client -s "$session_name" 2>/dev/null || true
    
    echo "✓ Started tmux session '$session_name' for GPU $gpu_id"
    echo "  → Running ${#experiments[@]} experiments (2h timeout each)"
}

# Distribute experiments across GPUs
distribute_experiments() {
    local all_combinations=($(generate_all_combinations))
    local total_experiments=${#all_combinations[@]}
    local gpus=(0 2)
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
echo "=== CrossLayer Transcoder Batch Optimizer Experiment Runner ==="
echo "Base config: $BASE_CONFIG"
echo "Testing hyperparameters:"
echo "  - beta1: [${BETA1_VALUES[*]}]"
echo "  - beta2: [${BETA2_VALUES[*]}]"
echo "  - optimizer: [${OPTIMIZER_VALUES[*]}]"
echo "  - tied_init: [${TIED_INIT_VALUES[*]}]"
echo "  - learning_rate: [${LR_VALUES[*]}]"
echo "  - warmup_steps: [${WARMUP_STEPS_VALUES[*]}]"
echo ""
echo "Starting tmux sessions with distributed experiments..."

# Distribute and run experiments
distribute_experiments

echo ""
echo "=== All experiments started ==="
echo "Active tmux sessions:"
tmux list-sessions | grep batch_optimizer || echo "No batch Optimizer sessions found"

echo ""
echo "To monitor a specific GPU session:"
echo "  tmux attach-session -t gpu0_batch_optimizer" 
echo "  tmux attach-session -t gpu2_batch_optimizer"

echo ""
echo "To list all sessions: tmux list-sessions"
echo "To kill a session: tmux kill-session -t <session_name>"

# Calculate total experiments across all GPUs
total_combinations=${#BETA1_VALUES[@]}
total_combinations=$((total_combinations * ${#BETA2_VALUES[@]}))
total_combinations=$((total_combinations * ${#OPTIMIZER_VALUES[@]}))
total_combinations=$((total_combinations * ${#TIED_INIT_VALUES[@]}))
total_combinations=$((total_combinations * ${#LR_VALUES[@]}))
total_combinations=$((total_combinations * ${#WARMUP_STEPS_VALUES[@]}))
echo ""
echo "Total unique experiments: $total_combinations"
echo "Estimated total runtime: ~$((total_combinations * 2)) hours (assuming 2h per experiment)"
echo "With parallel execution across 2 GPUs: ~$((total_combinations * 2 / 2)) hours" 