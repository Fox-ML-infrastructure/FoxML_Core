#!/bin/bash
# Test script for auto-rerun functionality
# Tests the intelligent training pipeline with auto-target ranking and auto-rerun

set -e

echo "🧪 Testing Auto-Rerun Functionality"
echo "===================================="
echo ""
echo "This test will:"
echo "  1. Auto-rank targets (top 3)"
echo "  2. Automatically detect and fix leakage"
echo "  3. Auto-rerun targets after fixes (up to 3 times)"
echo "  4. Train models on the selected targets"
echo ""

# Default values
DATA_DIR="${DATA_DIR:-data/data_labeled/interval=5m}"
SYMBOLS="${SYMBOLS:-AAPL MSFT}"
OUTPUT_DIR="${OUTPUT_DIR:-test_auto_rerun_output}"
TOP_N="${TOP_N:-3}"
MAX_ROWS="${MAX_ROWS:-5000}"

echo "Configuration:"
echo "  Data directory: $DATA_DIR"
echo "  Symbols: $SYMBOLS"
echo "  Output directory: $OUTPUT_DIR"
echo "  Top N targets: $TOP_N"
echo "  Max rows per symbol: $MAX_ROWS"
echo ""

# Run the training pipeline
python TRAINING/train.py \
    --data-dir "$DATA_DIR" \
    --symbols $SYMBOLS \
    --output-dir "$OUTPUT_DIR" \
    --auto-targets \
    --top-n-targets $TOP_N \
    --min-cs 3 \
    --max-rows-per-symbol $MAX_ROWS \
    --max-rows-train 10000 \
    --no-auto-features

echo ""
echo "✅ Test completed!"
echo "   Check $OUTPUT_DIR for results"
echo "   Check CONFIG/excluded_features.yaml and CONFIG/feature_registry.yaml for auto-fixes"

