#!/bin/bash

datasets=("Gas" "Electricity" "Weather")
args=("aware" "agnostic")

for dataset in "${datasets[@]}"; do
    for arg in "${args[@]}"; do
        python ../python/main.py \
        --dataset "$dataset" \
        --no_predictions 1  \
        --context_window 10 \
        --serialization "table" \
        --model "TableGPT2" \
        --num_repeats 5 \
        --timestamp "aware" \
        --domain "$arg" \
        --data_dir "../data/clean/" \
        --out_file "../logs/domain/${dataset}_logs.jsonl"
    done
done