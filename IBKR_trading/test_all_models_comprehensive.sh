#!/bin/bash

# Comprehensive Test Script for All Models
# Tests 20 models across 13 symbols, all targets, all strategies

set -e

echo "🚀 Starting Comprehensive Model Training Test"
echo "=============================================="

# Configuration
DATA_DIR="/home/Jennifer/secure/trader/5m_with_barrier_targets_full/interval=5m"
SYMBOLS="CBRE CERN CFG CVX GCI HOG MLM MXIM NI OI PANW SIVB XOM"
OUTPUT_DIR="/home/Jennifer/secure/trader/TRAINING/test_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${OUTPUT_DIR}/comprehensive_test_${TIMESTAMP}.log"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "📊 Test Configuration:"
echo "  Data Directory: ${DATA_DIR}"
echo "  Symbols: ${SYMBOLS}"
echo "  Output Directory: ${OUTPUT_DIR}"
echo "  Log File: ${LOG_FILE}"
echo ""

# Check if data directory exists
if [ ! -d "${DATA_DIR}" ]; then
    echo "❌ Error: Data directory not found: ${DATA_DIR}"
    exit 1
fi

# Check if TRAINING directory exists
if [ ! -d "/home/Jennifer/secure/trader/TRAINING" ]; then
    echo "❌ Error: TRAINING directory not found"
    exit 1
fi

echo "✅ Data directory found"
echo "✅ TRAINING directory found"
echo ""

# Start comprehensive test
echo "🧪 Running Comprehensive Model Training Test..."
echo "This will test:"
echo "  - 20 model families (LightGBM, XGBoost, MLP, CNN1D, LSTM, Transformer, etc.)"
echo "  - All training strategies (single_task, multi_task, cascade)"
echo "  - All available targets (forward returns + barrier targets)"
echo "  - 13 symbols"
echo ""

# Run the comprehensive test
cd /home/Jennifer/secure/trader/TRAINING

python train_with_strategies.py \
    --data-dir "${DATA_DIR}" \
    --symbols ${SYMBOLS} \
    --output-dir "${OUTPUT_DIR}/comprehensive_${TIMESTAMP}" \
    --strategy all \
    --experimental \
    --log-level INFO \
    2>&1 | tee "${LOG_FILE}"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Comprehensive test completed successfully!"
    echo ""
    echo "📈 Results Summary:"
    echo "  - Log file: ${LOG_FILE}"
    echo "  - Output directory: ${OUTPUT_DIR}/comprehensive_${TIMESTAMP}"
    echo ""
    echo "🔍 To view results:"
    echo "  - Check log: cat ${LOG_FILE}"
    echo "  - View output: ls -la ${OUTPUT_DIR}/comprehensive_${TIMESTAMP}"
    echo ""
    echo "📊 Model Performance:"
    if [ -f "${OUTPUT_DIR}/comprehensive_${TIMESTAMP}/metrics.json" ]; then
        echo "  - Metrics saved to: ${OUTPUT_DIR}/comprehensive_${TIMESTAMP}/metrics.json"
    fi
    if [ -d "${OUTPUT_DIR}/comprehensive_${TIMESTAMP}/models" ]; then
        echo "  - Models saved to: ${OUTPUT_DIR}/comprehensive_${TIMESTAMP}/models"
        echo "  - Model count: $(find ${OUTPUT_DIR}/comprehensive_${TIMESTAMP}/models -name "*.pkl" | wc -l)"
    fi
else
    echo ""
    echo "❌ Comprehensive test failed!"
    echo "  - Check log file: ${LOG_FILE}"
    echo "  - Check for errors in the output above"
    exit 1
fi

echo ""
echo "🎯 Test completed at $(date)"
echo "=============================================="
