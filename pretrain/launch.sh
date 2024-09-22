#!/bin/bash

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="output/checkpoint_$(date +%F-%H%M.%S)"
fi

mkdir -p "${OUTPUT_DIR}"

# accelerate launch --num_processes 1 run.py --do_train --do_eval --output_dir "${OUTPUT_DIR}" --fp16 --use_legacy_prediction_loop True
# python run.py --do_train --do_eval --output_dir "${OUTPUT_DIR}" --fp16 --use_legacy_prediction_loop True
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 run.py \
  --model_name_or_path "bert-large-uncased" \
  --do_train \
  --output_dir "${OUTPUT_DIR}" \
  --fp16 \
  --eval_strategy steps \
  --eval_steps 400 \
  --logging_steps 400 \
  --use_legacy_prediction_loop True \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --num_train_epochs 10 \
  --save_steps 400 \
  --training_loss contrastive \
  # --save_strategy steps \
  # --per_device_train_batch_size 128 \
  # --per_device_eval_batch_size 256 \
  