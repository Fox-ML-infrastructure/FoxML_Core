#!/bin/bash
"""
Train All Available Symbols - Two-Phase Training
==============================================

This script automatically discovers all available symbols and trains them
using the two-phase approach: cross-sectional models first, then sequential models.

Usage:
    ./train_all_symbols.sh
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

# Main function
main() {
    print_header "🎯 TRAINING System - All Available Symbols"
    print_header "Two-Phase Training: Cross-sectional → Sequential Models"
    
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
    
    # Discover all available symbols
    print_status "🔍 Discovering all available symbols..."
    SYMBOLS=()
    for symbol_dir in "$DATA_DIR"/symbol=*; do
        if [ -d "$symbol_dir" ]; then
            symbol_name=$(basename "$symbol_dir" | sed 's/symbol=//')
            SYMBOLS+=("$symbol_name")
        fi
    done
    
    if [ ${#SYMBOLS[@]} -eq 0 ]; then
        print_error "❌ No symbols found in data directory"
        exit 1
    fi
    
    # Convert array to space-separated string
    SYMBOLS_STRING="${SYMBOLS[*]}"
    
    print_status "✅ Found ${#SYMBOLS[@]} symbols: ${SYMBOLS_STRING}"
    
    # Batching configuration (can be overridden via env BATCH_SIZE)
    BATCH_SIZE=${BATCH_SIZE:-15}
    SYMBOLS_COUNT=${#SYMBOLS[@]}
    TOTAL_BATCHES=$(( (SYMBOLS_COUNT + BATCH_SIZE - 1) / BATCH_SIZE ))
    
    # Production-grade training budgets (override via env):
    #   MAX_SPS  → max samples per timestamp for cross-sectional sampling
    #   EPOCHS   → training epochs per model
    MAX_SPS=${MAX_SPS:-50}
    EPOCHS=${EPOCHS:-180}

    # Recommended stability/perf flags (override via env)
    export CS_ALIGN_MODE=${CS_ALIGN_MODE:-union}
    export USE_POLARS=${USE_POLARS:-1}
    export PYTHONHASHSEED=${PYTHONHASHSEED:-42}
    export TF_DETERMINISTIC_OPS=${TF_DETERMINISTIC_OPS:-1}

    print_status "Running two-phase training in ${TOTAL_BATCHES} batch(es) of up to ${BATCH_SIZE} symbols each (total ${SYMBOLS_COUNT})..."
    print_status "Phase 1: Cross-sectional models (LightGBM, XGBoost, MLP, Ensemble, etc.)"
    print_status "Phase 2: Sequential models (CNN1D, LSTM, Transformer, TabCNN, etc.)"
    print_status "All strategies: single_task, multi_task, cascade"
    print_status "All targets: fwd_ret, mdd, mfe, will_peak, will_valley"
    print_status "Budgets → max_samples_per_symbol=${MAX_SPS}, epochs=${EPOCHS}"

    # Family splits to prevent GPU OOM: run TF-heavy on CPU with smaller sampling
    FAMS_NON_TF=(LightGBM XGBoost QuantileLightGBM RewardBased NGBoost GMMRegime ChangePoint FTRLProximal Ensemble MetaLearning MultiTask)
    FAMS_TF=(MLP VAE GAN)
    MAX_SPS_TF=${MAX_SPS_TF:-30}
    EPOCHS_TF=${EPOCHS_TF:-100}
    
    echo ""
    
    # Phase 1: Cross-sectional models only
    print_header "🚀 Phase 1: Cross-sectional Models (Fast Training)"
    print_status "Training: LightGBM, XGBoost, MLP, Ensemble, RewardBased, QuantileLightGBM, NGBoost, GMMRegime, ChangePoint, FTRLProximal, VAE, GAN, MetaLearning, MultiTask"
    print_status "Skipping: CNN1D, LSTM, Transformer, TabCNN, TabLSTM, TabTransformer"
    
    for (( batch=0; batch<${TOTAL_BATCHES}; batch++ )); do
        start_index=$(( batch * BATCH_SIZE ))
        remaining=$(( SYMBOLS_COUNT - start_index ))
        take=$(( remaining < BATCH_SIZE ? remaining : BATCH_SIZE ))
        # Slice current batch
        BATCH_SYMBOLS=("${SYMBOLS[@]:start_index:take}")
        BATCH_SYMBOLS_STRING="${BATCH_SYMBOLS[*]}"
        print_status "📦 Phase 1 - Batch $((batch+1))/${TOTAL_BATCHES}: ${take} symbols"
        
        # Phase 1A: Non-TF families on GPU (memory efficient)
        if python train_with_strategies.py \
            --data-dir "$DATA_DIR" \
            --symbols $BATCH_SYMBOLS_STRING \
            --strategy all \
            --families ${FAMS_NON_TF[*]} \
            --targets fwd_ret_5m fwd_ret_10m fwd_ret_15m fwd_ret_30m fwd_ret_60m \
                      mdd_5m_0.001 mdd_10m_0.001 mdd_15m_0.001 mdd_30m_0.001 \
                      mfe_5m_0.001 mfe_10m_0.001 mfe_15m_0.001 mfe_30m_0.001 \
                      will_peak_5m will_peak_10m will_peak_15m will_peak_30m \
                      will_valley_5m will_valley_10m will_valley_15m will_valley_30m \
            --output-dir test_output_all_symbols_cross_sectional \
            --min-cs 1 \
            --max-samples-per-symbol ${MAX_SPS} \
            --epochs ${EPOCHS} \
            --model-types cross-sectional \
            --experimental \
            --use-polars; then
            print_success "   ✅ Phase 1A (non-TF) batch $((batch+1)) completed"
        else
            print_error "   ❌ Phase 1A (non-TF) batch $((batch+1)) failed"
            return 1
        fi

        # Phase 1B: TF-heavy families on CPU to avoid 8 GB GPU OOM; smaller sampling
        if CUDA_VISIBLE_DEVICES="" TF_GPU_ALLOCATOR=${TF_GPU_ALLOCATOR:-cuda_malloc_async} \
           python train_with_strategies.py \
            --data-dir "$DATA_DIR" \
            --symbols $BATCH_SYMBOLS_STRING \
            --strategy all \
            --families ${FAMS_TF[*]} \
            --targets fwd_ret_5m fwd_ret_10m fwd_ret_15m fwd_ret_30m fwd_ret_60m \
                      mdd_5m_0.001 mdd_10m_0.001 mdd_15m_0.001 mdd_30m_0.001 \
                      mfe_5m_0.001 mfe_10m_0.001 mfe_15m_0.001 mfe_30m_0.001 \
                      will_peak_5m will_peak_10m will_peak_15m will_peak_30m \
                      will_valley_5m will_valley_10m will_valley_15m will_valley_30m \
            --output-dir test_output_all_symbols_cross_sectional \
            --min-cs 1 \
            --max-samples-per-symbol ${MAX_SPS_TF} \
            --epochs ${EPOCHS_TF} \
            --model-types cross-sectional \
            --experimental \
            --use-polars; then
            print_success "   ✅ Phase 1B (TF on CPU) batch $((batch+1)) completed"
        else
            print_error "   ❌ Phase 1B (TF on CPU) batch $((batch+1)) failed"
            return 1
        fi
    done
    
    echo ""
    
    # Phase 2: Sequential models only
    print_header "🚀 Phase 2: Sequential Models (Deep Learning)"
    print_status "Training: CNN1D, LSTM, Transformer, TabCNN, TabLSTM, TabTransformer"
    print_status "Using pre-trained cross-sectional models as features (if available)"
    
    for (( batch=0; batch<${TOTAL_BATCHES}; batch++ )); do
        start_index=$(( batch * BATCH_SIZE ))
        remaining=$(( SYMBOLS_COUNT - start_index ))
        take=$(( remaining < BATCH_SIZE ? remaining : BATCH_SIZE ))
        BATCH_SYMBOLS=("${SYMBOLS[@]:start_index:take}")
        BATCH_SYMBOLS_STRING="${BATCH_SYMBOLS[*]}"
        print_status "📦 Phase 2 - Batch $((batch+1))/${TOTAL_BATCHES}: ${take} symbols"
        
        # Phase 2 (sequential TF families) on CPU to avoid GPU OOM
        if CUDA_VISIBLE_DEVICES="" TF_GPU_ALLOCATOR=${TF_GPU_ALLOCATOR:-cuda_malloc_async} \
           python train_with_strategies.py \
            --data-dir "$DATA_DIR" \
            --symbols $BATCH_SYMBOLS_STRING \
            --strategy all \
            --families CNN1D LSTM Transformer TabCNN TabLSTM TabTransformer \
            --targets fwd_ret_5m fwd_ret_10m fwd_ret_15m fwd_ret_30m fwd_ret_60m \
                      mdd_5m_0.001 mdd_10m_0.001 mdd_15m_0.001 mdd_30m_0.001 \
                      mfe_5m_0.001 mfe_10m_0.001 mfe_15m_0.001 mfe_30m_0.001 \
                      will_peak_5m will_peak_10m will_peak_15m will_peak_30m \
                      will_valley_5m will_valley_10m will_valley_15m will_valley_30m \
            --output-dir test_output_all_symbols_sequential \
            --min-cs 1 \
            --max-samples-per-symbol ${MAX_SPS_TF} \
            --epochs ${EPOCHS_TF} \
            --model-types sequential \
            --experimental \
            --use-polars; then
            print_success "   ✅ Phase 2 batch $((batch+1)) completed"
        else
            print_error "   ❌ Phase 2 batch $((batch+1)) failed"
            return 1
        fi
    done
    
    echo ""
    print_header "🎉 TWO-PHASE TRAINING COMPLETED!"
    print_success "✅ Phase 1: Cross-sectional models → test_output_all_symbols_cross_sectional/"
    print_success "✅ Phase 2: Sequential models → test_output_all_symbols_sequential/"
    print_success "Total symbols: ${#SYMBOLS[@]}"
    print_success "Total combinations: 14 cross-sectional + 6 sequential = 20 models"
    print_success "All strategies: single_task, multi_task, cascade"
    print_success "All targets: 20 different targets across 5 timeframes"
    print_success "Total model combinations: ${#SYMBOLS[@]} symbols × 20 models × 3 strategies × 20 targets"
    return 0
}

# Run main function
main "$@"
