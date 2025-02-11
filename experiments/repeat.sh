#!/bin/bash

datasets=("Gas" "Electricity" "Weather")
args=(1 5 10)

for dataset in "${datasets[@]}"; do
    for arg in "${args[@]}"; do
        python ../python/main.py \
        --dataset "$dataset" \
        --no_predictions 1  \
        --context_window 10 \
        --serialization "table" \
        --model "TableGPT2" \
        --num_repeats $arg \
        --timestamp "aware" \
        --domain "agnostic" \
        --data_dir "../data/clean/" \
        --out_file "../logs/repeat/${dataset}_logs.jsonl"
    done
done