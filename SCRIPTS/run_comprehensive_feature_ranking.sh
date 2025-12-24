#!/bin/bash
# Comprehensive Feature Ranking - Quick Start Script
# Run this in your terminal with: bash SCRIPTS/run_comprehensive_feature_ranking.sh

set -e

# Activate conda environment
echo "Activating trader_env..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate trader_env

cd /home/Jennifer/trader

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║     COMPREHENSIVE FEATURE RANKING                                ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Option 1: Quality Audit (target-independent)
echo "Option 1: Quality Audit (no target needed)"
echo "  Ranks features by data quality, variance, and redundancy"
echo ""
echo "  Command:"
echo "  python SCRIPTS/rank_features_comprehensive.py \\"
echo "    --symbols AAPL,MSFT,GOOGL,TSLA,JPM \\"
echo "    --output-dir results/feature_quality_audit"
echo ""

# Option 2: Full Ranking with Target
echo "Option 2: Full Ranking with Target (predictive + quality)"
echo "  Ranks features by both predictive power and data quality"
echo ""
echo "  Command:"
echo "  python SCRIPTS/rank_features_comprehensive.py \\"
echo "    --symbols AAPL,MSFT,GOOGL,TSLA,JPM \\"
echo "    --target y_will_peak_60m_0.8 \\"
echo "    --output-dir results/peak_60m_feature_ranking"
echo ""

# Option 3: Custom target
echo "Option 3: Use your own target"
echo "  python SCRIPTS/rank_features_comprehensive.py \\"
echo "    --symbols AAPL,MSFT,GOOGL,TSLA,JPM \\"
echo "    --target <YOUR_TARGET_COLUMN> \\"
echo "    --output-dir results/custom_feature_ranking"
echo ""

echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "Choose an option above and run the command, or run:"
echo "  python SCRIPTS/rank_features_comprehensive.py --help"
echo "  for all options"
echo ""

