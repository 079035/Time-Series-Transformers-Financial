#!/bin/bash

model_name=TimesNet

# Define hyperparameter arrays
e_layers_options=(3 6)
d_model_options=(128 256 512)
d_ff_options=(256 512 1024)
learning_rate_options=(0.001 0.0001)

# Create log directory
mkdir -p logs/hyperparameter_search

# Counter for experiment tracking
experiment_count=0
total_experiments=$((${#e_layers_options[@]} * ${#d_model_options[@]} * ${#d_ff_options[@]} * ${#learning_rate_options[@]}))

echo "Starting hyperparameter search with $total_experiments total experiments..."

# Loop through all combinations
for e_layers in "${e_layers_options[@]}"; do
    for d_model in "${d_model_options[@]}"; do
        for d_ff in "${d_ff_options[@]}"; do
            for learning_rate in "${learning_rate_options[@]}"; do
                experiment_count=$((experiment_count + 1))
                
                # Create unique model ID for this configuration
                model_id="Orderbook_e${e_layers}_d${d_model}_ff${d_ff}_lr${learning_rate}"
                
                echo "[$experiment_count/$total_experiments] Running experiment: $model_id"
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
                    --train_epochs 100 \
                    --patience 10 \
                    --use_gpu True \
                    --class_loss weighted_ce \
                    --class_weights "1.0,100.0" \
                    2>&1 | tee "$log_file"
                
                echo "Completed experiment $experiment_count/$total_experiments"
                echo "----------------------------------------"
            done
        done
    done
done

echo "Hyperparameter search completed! Check logs/hyperparameter_search/ for individual experiment logs."
echo "Total experiments run: $experiment_count" 