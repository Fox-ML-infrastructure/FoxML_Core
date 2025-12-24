#!/bin/bash
# Quick demo run of target ranking (shows output immediately)

set -e

cd "$(dirname "$0")/.."

echo "================================================================================"
echo "🎯 TARGET RANKING DEMO"
echo "================================================================================"
echo ""
echo "Testing on 1 symbol (AAPL), 1 target (peak_60m) to show output format..."
echo ""

# Create output directory
mkdir -p results/target_rankings_demo

# Run on just 1 target for speed
echo "Running..."
python SCRIPTS/rank_target_predictability.py \
  --symbols AAPL \
  --targets peak_60m \
  --output-dir results/target_rankings_demo

echo ""
echo "================================================================================"
echo "✅ OUTPUT CREATED"
echo "================================================================================"
echo ""

echo "📁 Output files:"
ls -lh results/target_rankings_demo/

echo ""
echo "📊 Rankings:"
cat results/target_rankings_demo/target_predictability_rankings.yaml

echo ""
echo "================================================================================"
echo "NEXT: Run on all targets"
echo "================================================================================"
echo ""
echo "python SCRIPTS/rank_target_predictability.py --symbols AAPL,MSFT,GOOGL,TSLA,JPM"
echo ""

