#!/bin/bash
# Week 1: Baseline Validation
# Establish clean baseline before adding any new features

set -e

cd "$(dirname "$0")/.."

echo "================================================================================"
echo "🎯 WEEK 1: BASELINE VALIDATION"
echo "================================================================================"
echo ""
echo "This will:"
echo "  1. Rank all targets by predictability (30 min)"
echo "  2. Run multi-model selection on top 3 targets (overnight)"
echo "  3. Document baseline metrics"
echo ""
echo "Purpose: Establish clean baseline to measure improvements against"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# Create baseline output directory
BASELINE_DIR="results/baseline_week1"
mkdir -p "$BASELINE_DIR"

echo ""
echo "================================================================================"
echo "STEP 1: Target Predictability Ranking (30 minutes)"
echo "================================================================================"
echo ""
echo "Testing on 5 representative symbols: AAPL, MSFT, GOOGL, TSLA, JPM"
echo "This evaluates all enabled targets across 3 model families"
echo ""

python scripts/rank_target_predictability.py \
  --symbols AAPL,MSFT,GOOGL,TSLA,JPM \
  --output-dir "$BASELINE_DIR/target_rankings" \
  | tee "$BASELINE_DIR/target_ranking.log"

echo ""
echo "✅ Target ranking complete!"
echo ""
echo "📊 Top 5 targets:"
head -6 "$BASELINE_DIR/target_rankings/target_predictability_rankings.csv" | column -t -s,

# Save top 3 targets for next step
TOP_3_TARGETS=$(tail -n +2 "$BASELINE_DIR/target_rankings/target_predictability_rankings.csv" | \
                head -3 | cut -d',' -f3 | tr '\n' ' ')

echo ""
echo "Top 3 targets for feature selection: $TOP_3_TARGETS"
echo "$TOP_3_TARGETS" > "$BASELINE_DIR/top_3_targets.txt"

echo ""
echo "================================================================================"
echo "STEP 2: Multi-Model Feature Selection (2-10 hours)"
echo "================================================================================"
echo ""
echo "Running on top 3 targets with 4 model families"
echo "This will take several hours - consider running overnight"
echo ""
read -p "Run now? (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    for target in $TOP_3_TARGETS; do
        echo ""
        echo "────────────────────────────────────────────────────────────────────────────"
        echo "🤖 Multi-model selection for: $target"
        echo "────────────────────────────────────────────────────────────────────────────"
        
        python scripts/multi_model_feature_selection.py \
          --target-column "$target" \
          --top-n 60 \
          --enable-families lightgbm,xgboost,random_forest,neural_network \
          --output-dir "$BASELINE_DIR/features_${target}" \
          | tee "$BASELINE_DIR/features_${target}.log"
        
        echo ""
        echo "✅ Completed: $target"
    done
else
    echo ""
    echo "Skipped. Run manually with:"
    for target in $TOP_3_TARGETS; do
        echo ""
        echo "python scripts/multi_model_feature_selection.py \\"
        echo "  --target-column $target \\"
        echo "  --top-n 60 \\"
        echo "  --output-dir $BASELINE_DIR/features_${target}"
    done
fi

echo ""
echo "================================================================================"
echo "STEP 3: Document Baseline Metrics"
echo "================================================================================"
echo ""

cat > "$BASELINE_DIR/baseline_metrics.md" << 'EOF'
# Baseline Metrics (Week 1)

## Date
$(date '+%Y-%m-%d %H:%M:%S')

## Target Rankings

Top 5 Targets:
$(head -6 results/baseline_week1/target_rankings/target_predictability_rankings.csv | column -t -s,)

## Multi-Model Feature Selection

### Target 1: [Fill in from top_3_targets.txt]
- Consensus Score Range: [Check CSV]
- Top 10 Features: [Check CSV]
- Model Agreement: [Check agreement matrix]

### Target 2: [Fill in]
- Similar metrics

### Target 3: [Fill in]
- Similar metrics

## Observations

- Which targets are most predictable?
- Which features appear across multiple targets?
- Do regime-related features rank high? (Should be NO for baseline)
- Model agreement: Do all 4 models agree on top features?

## Next Steps

Week 2: Add regime detection features and re-run to measure improvement
EOF

echo "✅ Created baseline metrics template: $BASELINE_DIR/baseline_metrics.md"
echo ""

echo "================================================================================"
echo "✅ BASELINE VALIDATION COMPLETE"
echo "================================================================================"
echo ""
echo "📁 Output location: $BASELINE_DIR/"
echo ""
echo "📋 Files created:"
ls -lh "$BASELINE_DIR/" | grep -v "^d"
echo ""
echo "📊 Review results:"
echo "  1. Target rankings:  cat $BASELINE_DIR/target_rankings/target_predictability_rankings.yaml"
echo "  2. Feature rankings: head -20 $BASELINE_DIR/features_*/feature_importance_multi_model.csv"
echo "  3. Baseline doc:     vim $BASELINE_DIR/baseline_metrics.md"
echo ""
echo "🎯 Next: Week 2 - Add regime features"
echo "   bash scripts/run_regime_enhancement.sh"
echo ""

