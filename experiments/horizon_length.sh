#!/bin/bash

datasets=("Gas" "Electricity" "Weather" "ILI")
args=(1 3 5)

for dataset in "${datasets[@]}"; do
    for arg in "${args[@]}"; do
        python ../python/main.py \
        --dataset "$dataset" \
        --no_predictions $arg  \
        --context_window 10 \
        --serialization "table" \
        --model "TableGPT2" \
        --num_repeats 5 \
        --timestamp "aware" \
        --domain "agnostic" \
        --data_dir "../data/clean/" \
        --out_file "../logs/horizon/${dataset}_logs.jsonl"
        #break
    done
    #break
done