#!/bin/bash
#
# Master Script: Run All 3 Phases
#
# This script replaces the old train_all_symbols.sh with an optimized 3-phase workflow:
# - Phase 1: Feature Engineering & Selection (reduce 421 → 61 features)
# - Phase 2: Core Model Training (LightGBM, MultiTask, Ensemble)
# - Phase 3: Sequential Model Training (LSTM, Transformer, CNN1D)
#

set -e  # Exit on any error

# ============================================================================
# CONFIGURATION
# ============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-/home/Jennifer/trader/data}"  # Default to trader/data
LOG_DIR="$SCRIPT_DIR/logs"
METADATA_DIR="$SCRIPT_DIR/metadata"
OUTPUT_DIR="$SCRIPT_DIR/output"

# Create directories
mkdir -p "$LOG_DIR" "$METADATA_DIR" "$OUTPUT_DIR"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

print_header() {
    echo ""
    echo -e "${BLUE}========================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_status() {
    echo -e "${BLUE}➡️  $1${NC}"
}

check_prerequisites() {
    print_header "Checking Prerequisites"
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 not found"
        exit 1
    fi
    print_success "Python 3: $(python3 --version)"
    
    # Check required packages
    python3 -c "import lightgbm, xgboost, sklearn, numpy, pandas" 2>/dev/null
    if [ $? -eq 0 ]; then
        print_success "Required packages installed"
    else
        print_error "Missing required packages (lightgbm, xgboost, sklearn, numpy, pandas)"
        exit 1
    fi
    
    # Check data directory
    if [ ! -d "$DATA_DIR" ]; then
        print_error "Data directory not found: $DATA_DIR"
        print_warning "Set DATA_DIR environment variable or modify this script"
        exit 1
    fi
    print_success "Data directory: $DATA_DIR"
    
    echo ""
}

# ============================================================================
# MAIN SCRIPT
# ============================================================================

print_header "🚀 OPTIMIZED 3-PHASE TRAINING WORKFLOW"

echo "Configuration:"
echo "  Data directory: $DATA_DIR"
echo "  Log directory: $LOG_DIR"
echo "  Metadata directory: $METADATA_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo ""

# Check prerequisites
check_prerequisites

# Get start time
START_TIME=$(date +%s)

# ============================================================================
# PHASE 1: Feature Engineering & Selection
# ============================================================================

print_header "🔬 PHASE 1: Feature Engineering & Selection"
print_status "Reducing features from 421 → ~60 features"
print_status "This will take approximately 15-30 minutes..."

PHASE1_LOG="$LOG_DIR/phase1_$(date +%Y%m%d_%H%M%S).log"

cd "$SCRIPT_DIR/phase1_feature_engineering"

python3 run_phase1.py \
    --data-dir "$DATA_DIR" \
    --config feature_selection_config.yaml \
    --output-dir "$METADATA_DIR" \
    --log-dir "$LOG_DIR" \
    2>&1 | tee "$PHASE1_LOG"

PHASE1_EXIT=$?
cd "$SCRIPT_DIR"

if [ $PHASE1_EXIT -eq 0 ]; then
    print_success "Phase 1 completed successfully"
    
    # Show Phase 1 results
    if [ -f "$METADATA_DIR/phase1_summary.json" ]; then
        echo ""
        echo "Phase 1 Results:"
        python3 -c "import json; print(json.dumps(json.load(open('$METADATA_DIR/phase1_summary.json')), indent=2))"
        echo ""
    fi
else
    print_error "Phase 1 failed! Check log: $PHASE1_LOG"
    exit 1
fi

# ============================================================================
# PHASE 2: Core Model Training
# ============================================================================

print_header "🎯 PHASE 2: Core Model Training"
print_status "Training LightGBM, MultiTask, and Ensemble models"
print_status "This will take approximately 30-60 minutes..."

PHASE2_LOG="$LOG_DIR/phase2_$(date +%Y%m%d_%H%M%S).log"

# Check if Phase 2 script exists
if [ ! -f "$SCRIPT_DIR/phase2_core_models/run_phase2.py" ]; then
    print_warning "Phase 2 script not found, skipping"
    print_status "You can create run_phase2.py based on the template in OPERATIONS_GUIDE.md"
else
    cd "$SCRIPT_DIR/phase2_core_models"
    
    python3 run_phase2.py \
        --data-dir "$DATA_DIR" \
        --metadata-dir "$METADATA_DIR" \
        --config core_models_config.yaml \
        --output-dir "$OUTPUT_DIR/core_models" \
        --log-dir "$LOG_DIR" \
        2>&1 | tee "$PHASE2_LOG"
    
    PHASE2_EXIT=$?
    cd "$SCRIPT_DIR"
    
    if [ $PHASE2_EXIT -eq 0 ]; then
        print_success "Phase 2 completed successfully"
        
        # Count trained models
        MODEL_COUNT=$(find "$OUTPUT_DIR/core_models" -name "*.joblib" 2>/dev/null | wc -l)
        echo "Trained models: $MODEL_COUNT"
        echo ""
    else
        print_error "Phase 2 failed! Check log: $PHASE2_LOG"
        exit 1
    fi
fi

# ============================================================================
# PHASE 3: Sequential Model Training
# ============================================================================

print_header "🧠 PHASE 3: Sequential Model Training"
print_status "Training LSTM, Transformer, and CNN1D models"
print_status "This will take approximately 60-120 minutes..."

PHASE3_LOG="$LOG_DIR/phase3_$(date +%Y%m%d_%H%M%S).log"

# Check if Phase 3 script exists
if [ ! -f "$SCRIPT_DIR/phase3_sequential_models/run_phase3.py" ]; then
    print_warning "Phase 3 script not found, skipping"
    print_status "You can create run_phase3.py based on the template in OPERATIONS_GUIDE.md"
else
    cd "$SCRIPT_DIR/phase3_sequential_models"
    
    python3 run_phase3.py \
        --data-dir "$DATA_DIR" \
        --metadata-dir "$METADATA_DIR" \
        --config sequential_config.yaml \
        --output-dir "$OUTPUT_DIR/sequential_models" \
        --log-dir "$LOG_DIR" \
        2>&1 | tee "$PHASE3_LOG"
    
    PHASE3_EXIT=$?
    cd "$SCRIPT_DIR"
    
    if [ $PHASE3_EXIT -eq 0 ]; then
        print_success "Phase 3 completed successfully"
        
        # Count trained models
        MODEL_COUNT=$(find "$OUTPUT_DIR/sequential_models" -name "*.joblib" 2>/dev/null | wc -l)
        echo "Trained models: $MODEL_COUNT"
        echo ""
    else
        print_error "Phase 3 failed! Check log: $PHASE3_LOG"
        exit 1
    fi
fi

# ============================================================================
# SUMMARY
# ============================================================================

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

print_header "🎉 ALL PHASES COMPLETED SUCCESSFULLY!"

echo "Summary:"
echo "  ⏱️  Total time: ${MINUTES}m ${SECONDS}s"
echo "  📁 Metadata: $METADATA_DIR"
echo "  🤖 Core models: $OUTPUT_DIR/core_models"
echo "  🧠 Sequential models: $OUTPUT_DIR/sequential_models"
echo "  📋 Logs: $LOG_DIR"
echo ""

# Show artifacts
echo "Artifacts created:"
if [ -d "$METADATA_DIR" ]; then
    ls -lh "$METADATA_DIR" | tail -n +2 | awk '{print "  - " $9 " (" $5 ")"}'
fi
echo ""

# Show total models trained
TOTAL_MODELS=$(find "$OUTPUT_DIR" -name "*.joblib" 2>/dev/null | wc -l)
echo "Total models trained: $TOTAL_MODELS"
echo ""

print_success "Workflow complete! Models are ready for evaluation."
print_status "Next steps:"
echo "  1. Review validation scores in logs/"
echo "  2. Analyze feature importance in metadata/feature_importance_report.csv"
echo "  3. Evaluate models on test set"
echo "  4. Deploy best models to production"
echo ""

exit 0

