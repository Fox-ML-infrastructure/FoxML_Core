#!/bin/bash
# End-to-end test command for target ranking with ~23 targets
# Tests the unified ranking and selection pipeline with multiple model families

python TRAINING/train.py \
    --data-dir "data/data_labeled/interval=5m" \
    --symbols AAPL MSFT \
    --output-dir "test_e2e_ranking_unified" \
    --auto-targets \
    --top-n-targets 23 \
    --max-targets-to-evaluate 23 \
    --auto-features \
    --top-m-features 50 \
    --min-cs 3 \
    --max-rows-per-symbol 5000 \
    --max-rows-train 10000 \
    --families lightgbm xgboost random_forest catboost neural_network lasso mutual_information univariate_selection

