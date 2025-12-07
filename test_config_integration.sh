#!/bin/bash
# Quick test to verify centralized configuration integration
# Tests that configs are loading correctly from YAML files

set -e

echo "🧪 Testing Centralized Configuration Integration"
echo "=================================================="
echo ""

# Get data directory (adjust if needed)
DATA_DIR="${DATA_DIR:-data/data_labeled/interval=5m}"
OUTPUT_DIR="test_config_output_$(date +%Y%m%d_%H%M%S)"

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Data directory not found: $DATA_DIR"
    echo "   Please set DATA_DIR environment variable or update the script"
    exit 1
fi

# Find first available symbol
FIRST_SYMBOL=$(ls "$DATA_DIR" 2>/dev/null | head -1 | sed 's/symbol=//' | head -1)
if [ -z "$FIRST_SYMBOL" ]; then
    echo "❌ No symbols found in $DATA_DIR"
    exit 1
fi

echo "📊 Test Configuration:"
echo "   Data Directory: $DATA_DIR"
echo "   Symbol: $FIRST_SYMBOL"
echo "   Output: $OUTPUT_DIR"
echo "   Models: LightGBM, XGBoost, MLP, CNN1D (mix of CPU/GPU)"
echo ""

cd TRAINING

echo "🚀 Running test..."
python train_with_strategies.py \
    --data-dir "../$DATA_DIR" \
    --symbols "$FIRST_SYMBOL" \
    --targets fwd_ret_5m \
    --families LightGBM XGBoost MLP CNN1D \
    --strategy single_task \
    --output-dir "../$OUTPUT_DIR" \
    --max-samples-per-symbol 50 \
    --max-rows-train 1000 \
    --epochs 5 \
    --min-cs 1 \
    --threads 4 \
    --log-level INFO \
    --model-types cross-sectional

echo ""
echo "✅ Test completed!"
echo "   Check output directory: $OUTPUT_DIR"
echo "   Verify that configs were loaded from YAML files in logs"
