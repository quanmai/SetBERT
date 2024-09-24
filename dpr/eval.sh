#!/bin/bash

DIR="checkpoint_2024-06-12-2210.11" # your checkpoint

python eval.py \
    --pretrained_model_path ./output/$DIR/query_encoder \
    --embedding_dir embedding/$DIR \
    --max_query_length 64 \
    --safe_tensor \
    # --topk 1000 \
