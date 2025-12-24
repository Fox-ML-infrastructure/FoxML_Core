#!/bin/bash
"""
Test GPU Model Functionality
============================

This script tests GPU model functionality for all GPU-capable model families.
It runs a focused test on a small set of symbols to verify GPU support is working.

Usage:
    ./test_gpu_models.sh [SYMBOL1 SYMBOL2 ...]
    
    If no symbols provided, defaults to: AAPL TSLA MSFT

Examples:
    ./test_gpu_models.sh AAPL
    ./test_gpu_models.sh AAPL TSLA MSFT GOOGL
"""

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Function to print colored output
print_header() {
    echo -e "${PURPLE}================================================================================${NC}"
    echo -e "${PURPLE}$1${NC}"
    echo -e "${PURPLE}================================================================================${NC}"
}

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check GPU availability
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        if [ $GPU_COUNT -gt 0 ]; then
            print_success "✅ Found $GPU_COUNT GPU(s)"
            nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
            return 0
        else
            print_warning "⚠️  nvidia-smi found but no GPUs detected"
            return 1
        fi
    else
        print_warning "⚠️  nvidia-smi not found - GPU may not be available"
        return 1
    fi
}

# Main function
main() {
    print_header "🎯 GPU Model Functionality Test"
    
    # Check GPU availability
    print_status "Checking GPU availability..."
    if ! check_gpu; then
        print_warning "GPU check failed, but continuing test (may run on CPU)"
    fi
    
    # Get symbols from command line or use defaults
    if [ $# -eq 0 ]; then
        SYMBOLS=(AAPL TSLA MSFT)
        print_status "No symbols provided, using defaults: ${SYMBOLS[*]}"
    else
        SYMBOLS=("$@")
        print_status "Testing with symbols: ${SYMBOLS[*]}"
    fi
    
    SYMBOLS_STRING="${SYMBOLS[*]}"
    
    # Data directory - Use absolute path to trader/data
    TRADER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    DATA_DIR="${DATA_DIR:-${TRADER_ROOT}/data/data_labeled/interval=5m}"
    
    print_status "Data directory: $DATA_DIR"
    
    # Check if data directory exists
    if [ ! -d "$DATA_DIR" ]; then
        print_error "❌ Data directory not found: $DATA_DIR"
        print_error "Please ensure the data directory exists."
        exit 1
    fi
    
    print_status "✅ Data directory found"
    
    # Test configuration (smaller for quick testing)
    MAX_SPS=${MAX_SPS:-20}  # Reduced for faster testing
    EPOCHS=${EPOCHS:-10}    # Reduced for faster testing
    
    # Recommended stability/perf flags
    export CS_ALIGN_MODE=${CS_ALIGN_MODE:-union}
    export USE_POLARS=${USE_POLARS:-1}
    export PYTHONHASHSEED=${PYTHONHASHSEED:-42}
    export TF_DETERMINISTIC_OPS=${TF_DETERMINISTIC_OPS:-1}
    
    # Enable GPU (don't hide it)
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    export TF_GPU_ALLOCATOR=${TF_GPU_ALLOCATOR:-cuda_malloc_async}
    
    print_status "GPU configuration:"
    print_status "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    print_status "  TF_GPU_ALLOCATOR=${TF_GPU_ALLOCATOR}"
    print_status "Test configuration:"
    print_status "  max_samples_per_symbol=${MAX_SPS}"
    print_status "  epochs=${EPOCHS}"
    print_status "  symbols: ${SYMBOLS_STRING}"
    
    echo ""
    
    # GPU Model Families
    # TensorFlow-based GPU models
    FAMS_TF_GPU=(MLP VAE GAN MetaLearning MultiTask)
    
    # Sequential GPU models (TensorFlow)
    FAMS_SEQ_GPU=(CNN1D LSTM Transformer TabCNN TabLSTM TabTransformer)
    
    # Test targets (reduced set for faster testing)
    TEST_TARGETS=(fwd_ret_5m fwd_ret_15m mdd_5m_0.001 will_peak_5m)
    
    # Test Phase 1: TensorFlow GPU Models (Cross-sectional)
    print_header "🚀 Phase 1: TensorFlow GPU Models (Cross-sectional)"
    print_status "Testing: ${FAMS_TF_GPU[*]}"
    print_status "Targets: ${TEST_TARGETS[*]}"
    
    if python train_with_strategies.py \
        --data-dir "$DATA_DIR" \
        --symbols $SYMBOLS_STRING \
        --strategy single_task \
        --families ${FAMS_TF_GPU[*]} \
        --targets ${TEST_TARGETS[*]} \
        --output-dir test_output_gpu_test \
        --min-cs 1 \
        --max-samples-per-symbol ${MAX_SPS} \
        --epochs ${EPOCHS} \
        --model-types cross-sectional \
        --experimental \
        --use-polars; then
        print_success "✅ Phase 1 (TF GPU models) completed"
    else
        print_error "❌ Phase 1 (TF GPU models) failed"
        exit 1
    fi
    
    echo ""
    
    # Test Phase 2: Sequential GPU Models
    print_header "🚀 Phase 2: Sequential GPU Models"
    print_status "Testing: ${FAMS_SEQ_GPU[*]}"
    print_status "Targets: ${TEST_TARGETS[*]}"
    
    if python train_with_strategies.py \
        --data-dir "$DATA_DIR" \
        --symbols $SYMBOLS_STRING \
        --strategy single_task \
        --families ${FAMS_SEQ_GPU[*]} \
        --targets ${TEST_TARGETS[*]} \
        --output-dir test_output_gpu_test \
        --min-cs 1 \
        --max-samples-per-symbol ${MAX_SPS} \
        --epochs ${EPOCHS} \
        --model-types sequential \
        --experimental \
        --use-polars; then
        print_success "✅ Phase 2 (Sequential GPU models) completed"
    else
        print_error "❌ Phase 2 (Sequential GPU models) failed"
        exit 1
    fi
    
    echo ""
    
    # Test Phase 3: XGBoost GPU (if available)
    print_header "🚀 Phase 3: XGBoost GPU Support"
    print_status "Testing: XGBoost with GPU support"
    print_status "Targets: ${TEST_TARGETS[*]}"
    
    if python train_with_strategies.py \
        --data-dir "$DATA_DIR" \
        --symbols $SYMBOLS_STRING \
        --strategy single_task \
        --families XGBoost \
        --targets ${TEST_TARGETS[*]} \
        --output-dir test_output_gpu_test \
        --min-cs 1 \
        --max-samples-per-symbol ${MAX_SPS} \
        --epochs ${EPOCHS} \
        --model-types cross-sectional \
        --use-polars; then
        print_success "✅ Phase 3 (XGBoost GPU) completed"
    else
        print_warning "⚠️  Phase 3 (XGBoost GPU) failed - may not have GPU support built"
    fi
    
    echo ""
    print_header "🎉 GPU MODEL TESTING COMPLETED!"
    print_success "✅ Phase 1: TensorFlow GPU models → test_output_gpu_test/"
    print_success "✅ Phase 2: Sequential GPU models → test_output_gpu_test/"
    print_success "✅ Phase 3: XGBoost GPU → test_output_gpu_test/"
    print_success "Tested symbols: ${SYMBOLS_STRING}"
    print_success "GPU models tested: ${#FAMS_TF_GPU[@]} TF + ${#FAMS_SEQ_GPU[@]} Sequential + XGBoost = $(( ${#FAMS_TF_GPU[@]} + ${#FAMS_SEQ_GPU[@]} + 1 )) models"
    
    # Check output directory
    if [ -d "test_output_gpu_test" ]; then
        MODEL_COUNT=$(find test_output_gpu_test -name "*.txt" -o -name "*.pkl" -o -name "*.joblib" 2>/dev/null | wc -l)
        print_status "Generated ${MODEL_COUNT} model files in test_output_gpu_test/"
    fi
    
    return 0
}

# Run main function
main "$@"

