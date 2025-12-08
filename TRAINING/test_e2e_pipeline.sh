#!/bin/bash
# Full End-to-End Test: Target Ranking → Feature Selection → Training Pipeline
# This tests the complete intelligent training workflow

set -e

cd "$(dirname "$0")/.."

echo "================================================================================"
echo "🧪 FULL END-TO-END TEST: Intelligent Training Pipeline"
echo "================================================================================"
echo ""
echo "This will test:"
echo "  1. Target Ranking (auto-select top targets)"
echo "  2. Feature Selection (auto-select features per target)"
echo "  3. Model Training (train all selected models)"
echo ""
echo "Expected runtime: ~15-30 minutes (depending on data size)"
echo ""

# Test parameters - reasonable defaults for full test
DATA_DIR="data/data_labeled/interval=5m"
SYMBOLS="AAPL MSFT GOOGL"  # 3 symbols for cross-sectional data
OUTPUT_DIR="test_e2e_output"
TOP_N_TARGETS=5
TOP_M_FEATURES=50
FAMILIES="LightGBM XGBoost MLP"  # Mix of CPU and GPU families
MIN_CS=10
MAX_ROWS_PER_SYMBOL=10000
MAX_ROWS_TRAIN=50000

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Data directory not found: $DATA_DIR"
    echo "   Please update DATA_DIR in this script"
    exit 1
fi

echo "📊 Test Configuration:"
echo "   Data dir: $DATA_DIR"
echo "   Symbols: $SYMBOLS"
echo "   Output: $OUTPUT_DIR"
echo "   Top N targets: $TOP_N_TARGETS"
echo "   Top M features per target: $TOP_M_FEATURES"
echo "   Model families: $FAMILIES"
echo "   Min cross-sectional samples: $MIN_CS"
echo "   Max rows per symbol: $MAX_ROWS_PER_SYMBOL"
echo "   Max training rows: $MAX_ROWS_TRAIN"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# Run full E2E test
echo ""
echo "🚀 Starting full end-to-end test..."
echo ""

python TRAINING/train.py \
    --data-dir "$DATA_DIR" \
    --symbols $SYMBOLS \
    --output-dir "$OUTPUT_DIR" \
    --auto-targets \
    --top-n-targets $TOP_N_TARGETS \
    --auto-features \
    --top-m-features $TOP_M_FEATURES \
    --families $FAMILIES \
    --strategy single_task \
    --min-cs $MIN_CS \
    --max-rows-per-symbol $MAX_ROWS_PER_SYMBOL \
    --max-rows-train $MAX_ROWS_TRAIN

EXIT_CODE=$?

echo ""
echo "================================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ E2E Test completed successfully!"
    echo "================================================================================"
    echo ""
    echo "📁 Check results:"
    echo "   - Target rankings: $OUTPUT_DIR/target_rankings/"
    echo "   - Feature selections: $OUTPUT_DIR/feature_selections/"
    echo "   - Training results: $OUTPUT_DIR/training_results/"
    echo "   - Cache: $OUTPUT_DIR/cache/"
    echo ""
    echo "📊 Summary:"
    echo "   - Targets ranked and selected"
    echo "   - Features selected per target"
    echo "   - Models trained for each target"
    echo ""
else
    echo "❌ E2E Test failed with exit code: $EXIT_CODE"
    echo "================================================================================"
    echo ""
    echo "Check logs above for errors"
    exit $EXIT_CODE
fi

