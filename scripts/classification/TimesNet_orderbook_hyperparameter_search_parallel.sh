#!/bin/bash

model_name=TimesNet

# Configuration
MAX_PARALLEL_JOBS=8  # Adjust based on your GPU memory and system resources
DELAY_BETWEEN_JOBS=2  # Seconds to wait between starting jobs

# Define hyperparameter arrays
e_layers_options=(3 6)
d_model_options=(128 256 512 1024 2048)
d_ff_options=(256 512 1024 2048)
learning_rate_options=(0.0001)

# Create log directory
mkdir -p logs/hyperparameter_search

# Function to run a single experiment
run_experiment() {
    local e_layers=$1
    local d_model=$2
    local d_ff=$3
    local learning_rate=$4
    local experiment_num=$5
    local total_experiments=$6
    
    # Create unique model ID for this configuration
    model_id="Orderbook_e${e_layers}_d${d_model}_ff${d_ff}_lr${learning_rate}"
    
    echo "[$experiment_num/$total_experiments] Starting experiment: $model_id (PID: $$)"
    echo "Parameters: e_layers=$e_layers, d_model=$d_model, d_ff=$d_ff, learning_rate=$learning_rate"
    
    # Log file for this experiment
    log_file="logs/hyperparameter_search/${model_id}.log"
    
    # Run the experiment
    python -u run.py \
        --task_name classification \
        --is_training 1 \
        --root_path ./dataset/Orderbook/ \
        --model_id "$model_id" \
        --model $model_name \
        --data Orderbook \
        --e_layers $e_layers \
        --batch_size 32 \
        --d_model $d_model \
        --d_ff $d_ff \
        --top_k 3 \
        --des "HyperSearch_e${e_layers}_d${d_model}_ff${d_ff}_lr${learning_rate}" \
        --itr 1 \
        --num_kernels 6 \
        --learning_rate $learning_rate \
        --train_epochs 1 \
        --patience 10 \
        --class_loss weighted_ce \
        --class_weights "1.0,100.0" \
        2>&1 | tee "$log_file"
    
    echo "[$experiment_num/$total_experiments] Completed experiment: $model_id"
}

# Function to wait for job slots to become available
wait_for_jobs() {
    while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL_JOBS ]; do
        sleep 1
    done
}

# Counter for experiment tracking
experiment_count=0
total_experiments=$((${#e_layers_options[@]} * ${#d_model_options[@]} * ${#d_ff_options[@]} * ${#learning_rate_options[@]}))

echo "Starting parallel hyperparameter search with $total_experiments total experiments..."
echo "Max parallel jobs: $MAX_PARALLEL_JOBS"
echo "=========================================="

# Track start time
start_time=$(date +%s)

# Loop through all combinations
for e_layers in "${e_layers_options[@]}"; do
    for d_model in "${d_model_options[@]}"; do
        for d_ff in "${d_ff_options[@]}"; do
            for learning_rate in "${learning_rate_options[@]}"; do
                experiment_count=$((experiment_count + 1))
                
                # Wait for available job slot
                wait_for_jobs
                
                # Run experiment in background
                run_experiment $e_layers $d_model $d_ff $learning_rate $experiment_count $total_experiments &
                
                # Small delay to prevent overwhelming the system
                sleep $DELAY_BETWEEN_JOBS
                
                echo "Jobs running: $(jobs -r | wc -l)/$MAX_PARALLEL_JOBS"
            done
        done
    done
done

# Wait for all remaining jobs to complete
echo "Waiting for all experiments to complete..."
wait

# Calculate total time
end_time=$(date +%s)
total_time=$((end_time - start_time))
hours=$((total_time / 3600))
minutes=$(((total_time % 3600) / 60))
seconds=$((total_time % 60))

echo "=========================================="
echo "Hyperparameter search completed!"
echo "Total experiments run: $experiment_count"
echo "Total time: ${hours}h ${minutes}m ${seconds}s"
echo "Check logs/hyperparameter_search/ for individual experiment logs."

# Optional: Create a summary of all experiments
echo "Creating experiment summary..."
summary_file="logs/hyperparameter_search/experiment_summary.txt"
echo "Hyperparameter Search Summary" > "$summary_file"
echo "Generated on: $(date)" >> "$summary_file"
echo "Total experiments: $experiment_count" >> "$summary_file"
echo "Total time: ${hours}h ${minutes}m ${seconds}s" >> "$summary_file"
echo "Max parallel jobs: $MAX_PARALLEL_JOBS" >> "$summary_file"
echo "" >> "$summary_file"
echo "Experiment configurations:" >> "$summary_file"

for log_file in logs/hyperparameter_search/Orderbook_*.log; do
    if [ -f "$log_file" ]; then
        basename=$(basename "$log_file" .log)
        echo "$basename" >> "$summary_file"
    fi
done

echo "Summary saved to: $summary_file" 