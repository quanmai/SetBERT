#!/bin/bash
DIR="checkpoint_2024-07-24-1637.32"
accelerate launch doc2emb.py \
    --document_path ../../../data/documents.jsonl \
    --sequence_len 256 \
    --safetensor \
    --pretrained_model_path /home/quanmai/workspace/IR/mulRetrievers/gpt/setBert/dpr/output/$DIR/doc_encoder \
    --output_dir embedding/$DIR \