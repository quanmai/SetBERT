#!/bin/bash

# DIR="checkpoint_2024-05-27-1441.32" # 0.219 0.526
# DIR="checkpoint_2024-05-28-1351.45" # 0.182 0.452
# DIR="checkpoint_2024-05-28-2212.09" # 0.231 0.541
# DIR="checkpoint_2024-05-29-1313.25" # 0.182 0.462
# DIR="checkpoint_2024-05-29-2342.10" # 0.233 0.540
# DIR="checkpoint_2024-06-01-1603.30" # 0.246 0.558
# DIR="checkpoint_2024-06-02-1453.42"  # 0.288 0.61
# DIR="checkpoint_2024-06-06-1044.25"  # 0.263 0.575
DIR="checkpoint_2024-06-12-2210.11"
# DIR="checkpoint_2024-06-10-1620.25"

python eval.py \
    --pretrained_model_path /home/quanmai/workspace/IR/mulRetrievers/gpt/setBert/dpr/output/$DIR/query_encoder \
    --embedding_dir embedding/$DIR \
    --max_query_length 64 \
    --safe_tensor \
    # --topk 1000 \
