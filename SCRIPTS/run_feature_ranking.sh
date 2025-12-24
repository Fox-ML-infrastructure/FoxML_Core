#!/bin/bash
# Quick script to run feature ranking by IC and predictive power

set -e

# Activate conda environment
echo "Activating trader_env..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate trader_env

cd /home/Jennifer/trader

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║     FEATURE RANKING BY IC AND PREDICTIVE POWER                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Run the feature ranking
python SCRIPTS/rank_features_by_ic_and_predictive.py \
  --symbols AAPL,MSFT,GOOGL,TSLA,JPM \
  --target-rankings results/final_clean/target_predictability_rankings.yaml \
  --top-n-targets 10 \
  --output-dir results/feature_selection

echo ""
echo "✅ Feature ranking complete!"
echo "📊 Results saved to: results/feature_selection/"
echo ""
echo "Next steps:"
echo "  1. Review: results/feature_selection/feature_rankings_ic_predictive.csv"
echo "  2. Select top 50-100 features with highest combined_score"
echo "  3. Use selected features in your models"
echo ""

