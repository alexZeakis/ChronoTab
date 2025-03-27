#!/bin/bash

datasets=("Gas" "Electricity" "Weather" "ILI")
args=("Llama3.1" "Qwen2.5" "TableGPT2")

for dataset in "${datasets[@]}"; do
    for arg in "${args[@]}"; do
        python ../python/main.py \
        --dataset "$dataset" \
        --no_predictions 1  \
        --context_window 10 \
        --serialization "table" \
        --model "$arg" \
        --num_repeats 5 \
        --timestamp "aware" \
        --domain "agnostic" \
        --data_dir "../data/clean/" \
        --out_file "../logs/model/${dataset}_logs.jsonl"
    done
done