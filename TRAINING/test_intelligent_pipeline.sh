#!/bin/bash
# Fast test script for intelligent training pipeline
# Tests ranking → selection → training with minimal data

set -e

cd "$(dirname "$0")/.."

echo "================================================================================"
echo "🧪 FAST TEST: Intelligent Training Pipeline"
echo "================================================================================"
echo ""
echo "This will test:"
echo "  1. Target ranking (top 2 targets)"
echo "  2. Feature selection (top 20 features per target)"
echo "  3. Model training (LightGBM only, 1 target)"
echo ""
echo "Expected runtime: ~5-10 minutes"
echo ""

# Test parameters
DATA_DIR="data/data_labeled/interval=5m"
SYMBOLS="AAPL MSFT"  # Just 2 symbols for speed
OUTPUT_DIR="test_intelligent_output"

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
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# Run intelligent trainer with minimal settings
python -m TRAINING.orchestration.intelligent_trainer \
    --data-dir "$DATA_DIR" \
    --symbols $SYMBOLS \
    --output-dir "$OUTPUT_DIR" \
    --auto-targets \
    --top-n-targets 2 \
    --auto-features \
    --top-m-features 20 \
    --families LightGBM \
    --strategy single_task \
    --min-cs 3 \
    --max-rows-per-symbol 5000 \
    --max-rows-train 10000

echo ""
echo "================================================================================"
echo "✅ Test completed!"
echo "================================================================================"
echo ""
echo "Check output:"
echo "  - Target rankings: $OUTPUT_DIR/target_rankings/"
echo "  - Feature selections: $OUTPUT_DIR/feature_selections/"
echo "  - Training results: $OUTPUT_DIR/training_results/"
echo "  - Cache: $OUTPUT_DIR/cache/"
echo ""

