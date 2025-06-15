#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

# Train iTransformer on Orderbook data for binary classification
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/Orderbook/ \
  --model_id Orderbook \
  --model $model_name \
  --data Orderbook \
  --e_layers 3 \
  --batch_size 32 \
  --d_model 256 \
  --d_ff 512 \
  --top_k 3 \
  --des 'Binary_Classification' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10 \
  --use_gpu \
  --seq_len 96 
