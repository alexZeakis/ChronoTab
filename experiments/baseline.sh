#!/bin/bash

datasets=("Gas" "Electricity" "Weather" "ILI")
#datasets=("ILI")

for dataset in "${datasets[@]}"; do
    python ../python/baselines/chronos/run_chronos.py \
    --dataset "$dataset" \
    --no_predictions 1  \
    --context_window 10 \
    --num_repeats 5 \
    --data_dir "../data/clean/" \
    --out_file "../logs/baselines/chronos/${dataset}_logs.jsonl"

    python ../python/baselines/MOIRAI/run_moirai.py \
    --dataset "$dataset" \
    --no_predictions 1  \
    --context_window 10 \
    --num_repeats 5 \
    --data_dir "../data/clean/" \
    --out_file "../logs/baselines/moirai/${dataset}_logs.jsonl"

    python ../python/baselines/ARIMA/run_arima.py \
    --dataset "$dataset" \
    --no_predictions 1  \
    --context_window 10 \
    --num_repeats 5 \
    --data_dir "../data/clean/" \
    --out_file "../logs/baselines/arima/${dataset}_logs.jsonl"
    
done