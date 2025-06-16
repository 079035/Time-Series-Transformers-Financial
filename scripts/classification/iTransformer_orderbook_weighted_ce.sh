#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

# Train iTransformer on Orderbook data with Weighted Cross Entropy Loss
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/Orderbook/ \
  --model_id Orderbook_weighted_ce \
  --model $model_name \
  --data Orderbook \
  --e_layers 3 \
  --batch_size 32 \
  --d_model 256 \
  --d_ff 512 \
  --top_k 3 \
  --des 'Weighted_CE' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10 \
  --use_gpu True \
  --seq_len 96 \
  --class_loss weighted_ce \
  --class_weights "1.0,100.0" 