#!/bin/bash

DIR=checkpoint_$(date +%F-%H%M.%S)
if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="output/$DIR"
fi

mkdir -p "${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 run.py \
  --hf_bert "bert-base-uncased" \
  --do_train \
  --output_dir "${OUTPUT_DIR}" \
  --eval_strategy steps \
  --eval_steps 200 \
  --logging_steps 200 \
  --save_steps 400 \
  --use_legacy_prediction_loop True \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --num_train_epochs 40 \
  --save_strategy steps \
  --load_best_model_at_end False \
  --fp16 \
  --seed 42 \
  --query_encoder_model_name_or_path "bert-base-uncased" \
  --document_encoder_model_name_or_path "bert-base-uncased" \
  # --query_encoder_model_name_or_path "/home/quanmai/workspace/IR/mulRetrievers/gpt/setBert/pretrain/output/checkpoint_2024-06-01-1345.33/best_model/" \
  # --document_encoder_model_name_or_path "/home/quanmai/workspace/IR/mulRetrievers/gpt/setBert/pretrain/output/checkpoint_2024-06-01-1345.33/best_model/" \