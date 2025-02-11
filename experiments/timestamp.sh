#!/bin/bash

datasets=("Gas" "Electricity" "Weather")
args=("agnostic" "aware")

for dataset in "${datasets[@]}"; do
    for arg in "${args[@]}"; do
        python ../python/main.py \
        --dataset "$dataset" \
        --no_predictions 1  \
        --context_window 10 \
        --serialization "table" \
        --model "TableGPT2" \
        --num_repeats 5 \
        --timestamp "$arg" \
        --domain "agnostic" \
        --data_dir "../data/clean/" \
        --out_file "../logs/timestamp/${dataset}_logs.jsonl"
    done
done