#!/bin/bash

datasets=("Gas" "Electricity" "Weather" "ILI")
args=(5 10 20)

for dataset in "${datasets[@]}"; do
    for arg in "${args[@]}"; do
        python ../python/main.py \
        --dataset "$dataset" \
        --no_predictions 1  \
        --context_window $arg \
        --serialization "table" \
        --model "TableGPT2" \
        --num_repeats 5 \
        --timestamp "aware" \
        --domain "agnostic" \
        --data_dir "../data/clean/" \
        --out_file "../logs/context/${dataset}_logs.jsonl"
    done
done