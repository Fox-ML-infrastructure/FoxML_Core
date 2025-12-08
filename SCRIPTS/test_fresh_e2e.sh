#!/bin/bash
# Fresh E2E test with new config system

python TRAINING/train.py \
    --data-dir "data/data_labeled/interval=5m" \
    --symbols AAPL MSFT \
    --output-dir "test_e2e_fresh" \
    --auto-targets \
    --top-n-targets 3 \
    --max-targets-to-evaluate 23 \
    --auto-features \
    --top-m-features 50 \
    --min-cs 3 \
    --max-rows-per-symbol 5000 \
    --max-rows-train 10000 \
    --families lightgbm random_forest
