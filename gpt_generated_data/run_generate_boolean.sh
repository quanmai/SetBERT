#!/bin/bash

python generate_boolean_query.py \
    --total-samples 50000 \
    --num-chunks 10 \
    --num-workers 20 \
    --num-per-call 10 \
    --max-tokens 4096 \
    --boolean-type or \