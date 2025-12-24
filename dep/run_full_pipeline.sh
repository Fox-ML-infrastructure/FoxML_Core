#!/bin/bash
#
# Run Full Pipeline
# Process all symbols and train models
#

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                        FULL PIPELINE EXECUTION                               ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Configuration
DATA_DIR="${DATA_DIR:-data/data_labeled/interval=5m}"
OUTPUT_DIR="${OUTPUT_DIR:-DATA_PROCESSING/data/labeled}"
MODELS="${MODELS:-lightgbm,xgboost,ensemble}"
VARIANT="${VARIANT:-conservative}"
PARALLEL_JOBS="${PARALLEL_JOBS:-4}"
LOG_DIR="logs"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="$LOG_DIR/pipeline_$TIMESTAMP.log"

# Create log directory
mkdir -p "$LOG_DIR"

echo "Configuration:"
echo "  Data Directory: $DATA_DIR"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Models: $MODELS"
echo "  Variant: $VARIANT"
echo "  Parallel Jobs: $PARALLEL_JOBS"
echo "  Log File: $LOG_FILE"
echo ""

# Get list of symbols
echo -e "${YELLOW}[Phase 1: Discovery]${NC} Finding symbols..."
SYMBOLS=$(ls "$DATA_DIR"/*.parquet 2>/dev/null | xargs -n 1 basename | sed 's/\.parquet$//' | sort)
SYMBOL_COUNT=$(echo "$SYMBOLS" | wc -w)

if [ $SYMBOL_COUNT -eq 0 ]; then
    echo -e "${RED}❌ No parquet files found in $DATA_DIR${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Found $SYMBOL_COUNT symbols${NC}"
echo ""

# Phase 2: Data Processing
echo -e "${YELLOW}[Phase 2: Data Processing]${NC} Processing symbols (parallel)..."
echo "$SYMBOLS" | tr ' ' '\n' > /tmp/symbols_to_process.txt

PROCESSED=0
FAILED=0

# Process in parallel
cat /tmp/symbols_to_process.txt | parallel -j $PARALLEL_JOBS --bar \
    "python SCRIPTS/process_single_symbol.py {} >> $LOG_FILE 2>&1 && echo '✅ {}' || echo '❌ {}'" \
    | tee /tmp/process_results.txt

# Count results
PROCESSED=$(grep -c "✅" /tmp/process_results.txt || echo 0)
FAILED=$(grep -c "❌" /tmp/process_results.txt || echo 0)

echo ""
echo "Data Processing Results:"
echo -e "  ${GREEN}✅ Processed: $PROCESSED${NC}"
if [ $FAILED -gt 0 ]; then
    echo -e "  ${RED}❌ Failed: $FAILED${NC}"
fi
echo ""

if [ $PROCESSED -eq 0 ]; then
    echo -e "${RED}❌ No symbols processed successfully. Check $LOG_FILE${NC}"
    exit 1
fi

# Phase 3: Model Training
echo -e "${YELLOW}[Phase 3: Model Training]${NC} Training models..."

# Convert comma-separated models to array
IFS=',' read -ra MODEL_ARRAY <<< "$MODELS"

TRAINED=0
TRAIN_FAILED=0

for symbol in $(grep "✅" /tmp/process_results.txt | cut -d' ' -f2); do
    for model in "${MODEL_ARRAY[@]}"; do
        echo "  Training $model on $symbol..."
        if python SCRIPTS/train_single_model.py "$symbol" "$model" --variant "$VARIANT" >> "$LOG_FILE" 2>&1; then
            echo -e "    ${GREEN}✅ $symbol + $model${NC}"
            ((TRAINED++))
        else
            echo -e "    ${RED}❌ $symbol + $model (see log)${NC}"
            ((TRAIN_FAILED++))
        fi
    done
done

echo ""
echo "Training Results:"
echo -e "  ${GREEN}✅ Trained: $TRAINED models${NC}"
if [ $TRAIN_FAILED -gt 0 ]; then
    echo -e "  ${RED}❌ Failed: $TRAIN_FAILED models${NC}"
fi
echo ""

# Summary
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                           PIPELINE COMPLETE                                  ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Summary:"
echo "  Symbols Found:      $SYMBOL_COUNT"
echo "  Symbols Processed:  $PROCESSED"
echo "  Models Trained:     $TRAINED"
echo ""
echo "Output Locations:"
echo "  Labeled Data:  $OUTPUT_DIR/"
echo "  Trained Models: models/"
echo "  Full Log:      $LOG_FILE"
echo ""

if [ $TRAINED -gt 0 ]; then
    echo -e "${GREEN}✅ Pipeline completed successfully!${NC}"
    echo ""
    echo "Next step: python SCRIPTS/view_results.py"
    exit 0
else
    echo -e "${RED}❌ Pipeline completed with errors. Check $LOG_FILE${NC}"
    exit 1
fi

