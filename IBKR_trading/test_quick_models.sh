#!/bin/bash

# Quick Test Script for Core Models
# Tests a subset of models to verify the system works

set -e

echo "🧪 Quick Model Training Test"
echo "============================"

# Configuration
DATA_DIR="/home/Jennifer/secure/trader/5m_with_barrier_targets_full/interval=5m"
SYMBOLS="CBRE CERN CFG"  # Just 3 symbols for quick test
OUTPUT_DIR="/home/Jennifer/secure/trader/IBKR_trading/quick_test_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "📊 Quick Test Configuration:"
echo "  Data Directory: ${DATA_DIR}"
echo "  Symbols: ${SYMBOLS}"
echo "  Output Directory: ${OUTPUT_DIR}"
echo ""

# Run quick test with just a few models
cd /home/Jennifer/secure/trader/TRAINING

python train_with_strategies.py \
    --data-dir "${DATA_DIR}" \
    --symbols ${SYMBOLS} \
    --output-dir "${OUTPUT_DIR}/quick_${TIMESTAMP}" \
    --strategies single_task \
    --models LightGBM XGBoost MLP CNN1D LSTM \
    --targets fwd_ret_5m fwd_ret_15m fwd_ret_30m \
    --batch-size 3 \
    --batch-id 0 \
    --deterministic \
    --log-level INFO \
    --save-models \
    --save-predictions \
    --save-metrics

echo ""
echo "✅ Quick test completed!"
echo "  - Output: ${OUTPUT_DIR}/quick_${TIMESTAMP}"
echo "  - Models tested: LightGBM, XGBoost, MLP, CNN1D, LSTM"
echo "  - Strategies: single_task"
echo "  - Targets: fwd_ret_5m, fwd_ret_15m, fwd_ret_30m"
echo ""
