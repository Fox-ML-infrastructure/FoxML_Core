#!/bin/bash
# Show what output you already have and what to expect from new scripts

echo "================================================================================"
echo "YOUR EXISTING OUTPUT (Old Single-Model Scripts)"
echo "================================================================================"
echo ""

echo "📁 Feature selection output (LightGBM-only):"
echo ""
ls -lhd DATA_PROCESSING/data/features/*/

echo ""
echo "Example: peak_60m features"
echo "────────────────────────────────────────────────────────────────────────────────"
ls -lh DATA_PROCESSING/data/features/peak_60m/
echo ""

if [ -f "DATA_PROCESSING/data/features/peak_60m/selected_features.txt" ]; then
    echo "Top 10 features (LightGBM-only):"
    head -10 DATA_PROCESSING/data/features/peak_60m/selected_features.txt | nl
    echo ""
    echo "Total features: $(wc -l < DATA_PROCESSING/data/features/peak_60m/selected_features.txt)"
fi

echo ""
echo "================================================================================"
echo "NEW SCRIPTS OUTPUT (Multi-Model - Not Run Yet)"
echo "================================================================================"
echo ""

echo "🎯 Target Ranking Output (rank_target_predictability.py):"
echo "   Location: results/target_rankings/"
echo "   Files:"
echo "     • target_predictability_rankings.csv  - Full rankings with R² scores"
echo "     • target_predictability_rankings.yaml - Recommendations"
echo ""
echo "   Status: $([ -f 'results/target_rankings/target_predictability_rankings.csv' ] && echo '✅ EXISTS' || echo '❌ NOT CREATED YET')"
echo ""

echo "🤖 Multi-Model Feature Selection Output (multi_model_feature_selection.py):"
echo "   Location: DATA_PROCESSING/data/features/multi_model/"
echo "   Files:"
echo "     • selected_features.txt                    - Top N consensus features"
echo "     • feature_importance_multi_model.csv       - Detailed rankings"
echo "     • model_agreement_matrix.csv               - Which models agree"
echo "     • importance_lightgbm.csv                  - Per-family rankings"
echo "     • importance_xgboost.csv"
echo "     • importance_random_forest.csv"
echo "     • importance_neural_network.csv"
echo ""
echo "   Status: $([ -f 'DATA_PROCESSING/data/features/multi_model/selected_features.txt' ] && echo '✅ EXISTS' || echo '❌ NOT CREATED YET')"
echo ""

echo "================================================================================"
echo "TO RUN NEW SCRIPTS"
echo "================================================================================"
echo ""
echo "1️⃣  Target Ranking (10 minutes):"
echo "    python SCRIPTS/rank_target_predictability.py --symbols AAPL,MSFT,GOOGL,TSLA,JPM"
echo ""
echo "2️⃣  Multi-Model Feature Selection (2-10 hours depending on dataset):"
echo "    python SCRIPTS/multi_model_feature_selection.py \\"
echo "      --target-column y_will_peak_60m_0.8 \\"
echo "      --top-n 60"
echo ""
echo "3️⃣  Compare old vs new:"
echo "    python SCRIPTS/compare_feature_sets.py \\"
echo "      --set1 DATA_PROCESSING/data/features/peak_60m/selected_features.txt \\"
echo "      --set2 DATA_PROCESSING/data/features/multi_model/selected_features.txt"
echo ""

