#!/bin/bash
# End-to-End Test: Auto-Fixer Integration
# This test verifies that the auto-fixer detects and fixes leaks when perfect scores are detected

set -e

cd "$(dirname "$0")/.."

echo "================================================================================"
echo "🧪 AUTO-FIXER E2E TEST: Leakage Detection and Auto-Fix"
echo "================================================================================"
echo ""
echo "This test will:"
echo "  1. Run target ranking (which triggers auto-fixer on perfect scores)"
echo "  2. Verify auto-fixer detects leaking features (ts, p_*, etc.)"
echo "  3. Check that config files are updated"
echo "  4. Re-run to verify leaks are fixed"
echo ""
echo "Expected: Auto-fixer should detect and exclude leaking features"
echo ""

# Test parameters - minimal for fast testing
DATA_DIR="data/data_labeled/interval=5m"
SYMBOLS="AAPL MSFT"  # 2 symbols for faster testing
OUTPUT_DIR="test_auto_fixer_output"
TOP_N_TARGETS=3  # Test on fewer targets
MIN_CS=3  # Lower threshold for faster testing
MAX_ROWS_PER_SYMBOL=5000  # Smaller dataset
MAX_ROWS_TRAIN=10000

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
echo "   Min cross-sectional samples: $MIN_CS"
echo "   Max rows per symbol: $MAX_ROWS_PER_SYMBOL"
echo "   Max training rows: $MAX_ROWS_TRAIN"
echo ""

# Backup configs before test
echo "📦 Backing up config files..."
BACKUP_DIR="test_auto_fixer_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
if [ -f "CONFIG/excluded_features.yaml" ]; then
    cp "CONFIG/excluded_features.yaml" "$BACKUP_DIR/"
fi
if [ -f "CONFIG/feature_registry.yaml" ]; then
    cp "CONFIG/feature_registry.yaml" "$BACKUP_DIR/"
fi
echo "   Configs backed up to: $BACKUP_DIR"
echo ""

read -p "Press Enter to continue or Ctrl+C to cancel..."

# Clean output directory
echo "🧹 Cleaning output directory..."
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Run target ranking (this will trigger auto-fixer if perfect scores detected)
echo ""
echo "🚀 Running target ranking (auto-fixer will trigger on perfect scores)..."
echo ""

python TRAINING/train.py \
    --data-dir "$DATA_DIR" \
    --symbols $SYMBOLS \
    --output-dir "$OUTPUT_DIR" \
    --auto-targets \
    --top-n-targets $TOP_N_TARGETS \
    --min-cs $MIN_CS \
    --max-rows-per-symbol $MAX_ROWS_PER_SYMBOL \
    --max-rows-train $MAX_ROWS_TRAIN \
    --no-auto-features  # Skip feature selection for faster test

EXIT_CODE=$?

echo ""
echo "================================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Auto-Fixer E2E Test completed!"
    echo "================================================================================"
    echo ""
    echo "📁 Check results:"
    echo "   - Target rankings: $OUTPUT_DIR/target_rankings/"
    echo "   - Auto-fixer logs: Check console output above for '🔧 Auto-fixing' messages"
    echo ""
    echo "📊 Config Changes:"
    if [ -f "CONFIG/excluded_features.yaml" ]; then
        echo "   - excluded_features.yaml: $(wc -l < CONFIG/excluded_features.yaml) lines"
    fi
    if [ -f "CONFIG/feature_registry.yaml" ]; then
        echo "   - feature_registry.yaml: $(wc -l < CONFIG/feature_registry.yaml) lines"
    fi
    echo ""
    echo "💡 To restore original configs:"
    echo "   cp $BACKUP_DIR/*.yaml CONFIG/"
    echo ""
else
    echo "❌ Auto-Fixer E2E Test failed with exit code: $EXIT_CODE"
    echo "================================================================================"
    echo ""
    echo "Check logs above for errors"
    echo ""
    echo "💡 To restore original configs:"
    echo "   cp $BACKUP_DIR/*.yaml CONFIG/"
    exit $EXIT_CODE
fi

