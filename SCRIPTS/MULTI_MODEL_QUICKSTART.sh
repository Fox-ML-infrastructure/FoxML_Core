#!/bin/bash
# Multi-Model Feature Selection Quick Start
# This script demonstrates the full workflow

set -e  # Exit on error

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=============================================================================="
echo "🚀 MULTI-MODEL FEATURE SELECTION QUICKSTART"
echo "=============================================================================="
echo ""
echo "This script will:"
echo "  1. Rank target predictability (find best targets)"
echo "  2. Run multi-model feature selection on top target"
echo "  3. Compare with single-model (LightGBM) features"
echo ""
read -p "Press Enter to continue..."

# ============================================================================
# STEP 1: Rank Target Predictability
# ============================================================================
echo ""
echo "=============================================================================="
echo "STEP 1: Ranking Target Predictability"
echo "=============================================================================="
echo "Testing on 5 representative symbols: AAPL, MSFT, GOOGL, TSLA, SPY"
echo "This will take ~5-10 minutes..."
echo ""

python SCRIPTS/rank_target_predictability.py \
  --symbols AAPL,MSFT,GOOGL,TSLA,SPY \
  --model-families lightgbm,random_forest,neural_network \
  --output-dir results/target_rankings

echo ""
echo "✅ Target ranking complete!"
echo "Results saved to: results/target_rankings/"
echo ""
echo "Top 5 targets:"
head -6 results/target_rankings/target_predictability_rankings.csv | column -t -s,

# ============================================================================
# STEP 2: Multi-Model Feature Selection
# ============================================================================
echo ""
echo "=============================================================================="
echo "STEP 2: Multi-Model Feature Selection"
echo "=============================================================================="
echo "Running on best target with 4 model families"
echo "Testing on 5 symbols first (faster)..."
echo ""

# Get top target from rankings
TOP_TARGET=$(tail -n +2 results/target_rankings/target_predictability_rankings.csv | head -1 | cut -d',' -f3)
echo "Selected target: $TOP_TARGET"

python SCRIPTS/multi_model_feature_selection.py \
  --symbols AAPL,MSFT,GOOGL,TSLA,SPY \
  --target-column "$TOP_TARGET" \
  --top-n 60 \
  --enable-families lightgbm,xgboost,random_forest,neural_network \
  --output-dir DATA_PROCESSING/data/features/multi_model

echo ""
echo "✅ Multi-model selection complete!"
echo ""
echo "Top 10 consensus features:"
head -11 DATA_PROCESSING/data/features/multi_model/feature_importance_multi_model.csv | column -t -s,

# ============================================================================
# STEP 3: Compare with Single-Model Features (if available)
# ============================================================================
echo ""
echo "=============================================================================="
echo "STEP 3: Comparison with Single-Model Selection"
echo "=============================================================================="

# Check if single-model features exist
SINGLE_MODEL_FEATURES="DATA_PROCESSING/data/features/selected_features.txt"

if [ -f "$SINGLE_MODEL_FEATURES" ]; then
    echo "Found existing LightGBM-only features. Comparing..."
    echo ""
    
    python SCRIPTS/compare_feature_sets.py \
      --set1 "$SINGLE_MODEL_FEATURES" \
      --set2 DATA_PROCESSING/data/features/multi_model/selected_features.txt \
      --name1 "LightGBM-only" \
      --name2 "Multi-model" \
      --output results/feature_comparison.csv
    
    echo ""
    echo "✅ Comparison saved to: results/feature_comparison.csv"
else
    echo "No existing single-model features found. Skipping comparison."
    echo "To compare, first run:"
    echo "  python SCRIPTS/select_features.py --target-column $TOP_TARGET"
fi

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "=============================================================================="
echo "✅ QUICKSTART COMPLETE!"
echo "=============================================================================="
echo ""
echo "📁 Output files:"
echo "  • Target rankings:        results/target_rankings/"
echo "  • Multi-model features:   DATA_PROCESSING/data/features/multi_model/"
echo "  • Feature comparison:     results/feature_comparison.csv"
echo ""
echo "🎯 Next steps:"
echo ""
echo "1. Review target rankings:"
echo "   cat results/target_rankings/target_predictability_rankings.yaml"
echo ""
echo "2. Use multi-model features in training:"
echo "   features=\$(cat DATA_PROCESSING/data/features/multi_model/selected_features.txt)"
echo ""
echo "3. Run on full universe (728 symbols):"
echo "   python SCRIPTS/multi_model_feature_selection.py --target-column $TOP_TARGET --top-n 60"
echo ""
echo "4. Read full documentation:"
echo "   cat INFORMATION/MULTI_MODEL_FEATURE_SELECTION.md"
echo ""
echo "=============================================================================="

