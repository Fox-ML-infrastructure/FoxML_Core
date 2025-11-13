#!/bin/bash
# Week 2: Regime Detection Enhancement
# Add regime features and measure improvement vs baseline

set -e

cd "$(dirname "$0")/.."

echo "================================================================================"
echo "🎯 WEEK 2: REGIME DETECTION ENHANCEMENT"
echo "================================================================================"
echo ""
echo "Prerequisites:"
echo "  ✅ Baseline validation completed (Week 1)"
echo "  ✅ Baseline metrics documented"
echo ""
echo "This will:"
echo "  1. Add regime detection features to your data"
echo "  2. Re-run multi-model selection on same targets"
echo "  3. Compare results to baseline"
echo "  4. Measure improvement"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# Check baseline exists
if [ ! -f "results/baseline_week1/top_3_targets.txt" ]; then
    echo "❌ Baseline not found. Run baseline validation first:"
    echo "   bash scripts/run_baseline_validation.sh"
    exit 1
fi

# Create regime output directory
REGIME_DIR="results/regime_week2"
mkdir -p "$REGIME_DIR"

# Get top 3 targets from baseline
TOP_3_TARGETS=$(cat results/baseline_week1/top_3_targets.txt)
echo "Testing on top 3 targets from baseline: $TOP_3_TARGETS"
echo ""

echo "================================================================================"
echo "STEP 1: Add Regime Features to Data Pipeline"
echo "================================================================================"
echo ""
echo "Creating script to add regime features..."

cat > scripts/add_regime_to_data.py << 'EOPY'
#!/usr/bin/env python
"""
Add regime features to labeled data.

This script:
1. Loads existing labeled data
2. Adds regime detection features
3. Saves to new location (preserves original)
"""

import sys
from pathlib import Path
import pandas as pd
import logging

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from DATA_PROCESSING.features.regime_features import add_all_regime_features

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path, 
                       default=_REPO_ROOT / 'data/data_labeled/interval=5m')
    parser.add_argument('--output-dir', type=Path,
                       default=_REPO_ROOT / 'data/data_labeled_with_regime/interval=5m')
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols (default: all)')
    parser.add_argument('--dry-run', action='store_true', help='Test on one symbol')
    args = parser.parse_args()
    
    # Find symbols
    symbol_dirs = sorted([d for d in args.input_dir.glob('symbol=*') if d.is_dir()])
    
    if args.symbols:
        requested = [s.strip().upper() for s in args.symbols.split(',')]
        symbol_dirs = [d for d in symbol_dirs if d.name.replace('symbol=', '') in requested]
    
    if args.dry_run:
        symbol_dirs = symbol_dirs[:1]
        logger.info("🧪 DRY RUN: Testing on 1 symbol")
    
    logger.info(f"Processing {len(symbol_dirs)} symbols...")
    logger.info("")
    
    for i, symbol_dir in enumerate(symbol_dirs, 1):
        symbol = symbol_dir.name.replace('symbol=', '')
        parquet_file = symbol_dir / f"{symbol}.parquet"
        
        if not parquet_file.exists():
            logger.warning(f"[{i}/{len(symbol_dirs)}] {symbol}: File not found, skipping")
            continue
        
        try:
            # Load
            df = pd.read_parquet(parquet_file)
            logger.info(f"[{i}/{len(symbol_dirs)}] {symbol}: Loaded {len(df)} rows")
            
            # Add regime features
            df_with_regime = add_all_regime_features(df, trend_lookback=50)
            
            # Save to new location
            output_symbol_dir = args.output_dir / f"symbol={symbol}"
            output_symbol_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_symbol_dir / f"{symbol}.parquet"
            
            df_with_regime.to_parquet(output_file, index=False)
            
            n_new = len(df_with_regime.columns) - len(df.columns)
            logger.info(f"              ✅ Added {n_new} regime features, saved to {output_file}")
            logger.info("")
            
        except Exception as e:
            logger.error(f"[{i}/{len(symbol_dirs)}] {symbol}: ERROR - {e}")
            continue
    
    logger.info("="*80)
    logger.info("✅ Regime features added!")
    logger.info(f"Output: {args.output_dir}")
    logger.info("")
    logger.info("Next: Run multi-model selection on data with regime features")

if __name__ == '__main__':
    main()
EOPY

chmod +x scripts/add_regime_to_data.py

echo "✅ Created scripts/add_regime_to_data.py"
echo ""
echo "Running on top 3 target symbols + test set (10 symbols total)..."
echo ""

# Add regime features (test on small set first)
python scripts/add_regime_to_data.py \
  --symbols AAPL,MSFT,GOOGL,TSLA,JPM \
  | tee "$REGIME_DIR/add_regime_features.log"

echo ""
echo "✅ Regime features added to test symbols"
echo ""

echo "================================================================================"
echo "STEP 2: Run Multi-Model Selection with Regime Features"
echo "================================================================================"
echo ""
echo "Re-running on same top 3 targets to measure improvement..."
echo ""

for target in $TOP_3_TARGETS; do
    echo ""
    echo "────────────────────────────────────────────────────────────────────────────"
    echo "🤖 Multi-model selection with regime features: $target"
    echo "────────────────────────────────────────────────────────────────────────────"
    
    python scripts/multi_model_feature_selection.py \
      --target-column "$target" \
      --top-n 60 \
      --data-dir data/data_labeled_with_regime/interval=5m \
      --enable-families lightgbm,xgboost,random_forest,neural_network \
      --output-dir "$REGIME_DIR/features_${target}" \
      | tee "$REGIME_DIR/features_${target}.log"
    
    echo ""
    echo "✅ Completed: $target with regime features"
done

echo ""
echo "================================================================================"
echo "STEP 3: Compare Baseline vs Regime Enhancement"
echo "================================================================================"
echo ""

cat > "$REGIME_DIR/comparison_report.md" << 'EOF'
# Regime Enhancement Results (Week 2)

## Date
$(date '+%Y-%m-%d %H:%M:%S')

## Comparison: Baseline vs With Regime Features

### Target 1: [First target from top_3]

**Baseline (Week 1):**
- Top feature: [From baseline CSV]
- Consensus score range: [From baseline CSV]

**With Regime (Week 2):**
- Top feature: [From regime CSV]
- Consensus score range: [From regime CSV]
- Regime features in top 20? [YES/NO - check CSV]

**Improvement:**
- R² change: [Calculate if metrics available]
- New features discovered: [List regime features that ranked high]

### Target 2: [Second target]
[Similar analysis]

### Target 3: [Third target]
[Similar analysis]

## Regime Feature Rankings

Check if regime features rank high:
$(grep -i "regime" results/regime_week2/features_*/feature_importance_multi_model.csv | head -20)

## Model Agreement on Regime Features

Do all 4 models agree regime features are important?
[Check model_agreement_matrix.csv]

## Key Findings

1. Do regime features rank in top 20? YES/NO
2. Which regime features are most important?
   - regime_trend, regime_chop, or regime_vol?
   - Regime-conditional returns (ret_*_in_trend)?
   - Regime transitions (entered_trend, etc)?

3. Estimated R² improvement: +X%
4. Worth keeping? YES/NO

## Decision

- [ ] Keep regime features (they improve performance)
- [ ] Remove regime features (no improvement)
- [ ] Need more testing on full dataset

## Next Steps

If regime features help:
- Week 3: Add VIX features
- Re-run comparison

If regime features don't help:
- Try different regime detection parameters
- Or skip to fractional differentiation (Week 4)
EOF

echo "✅ Created comparison template: $REGIME_DIR/comparison_report.md"
echo ""

# Auto-generate comparison snippet
echo "Comparing top 10 features..."
echo "" >> "$REGIME_DIR/comparison_report.md"
echo "## Auto-Generated Comparison" >> "$REGIME_DIR/comparison_report.md"
echo "" >> "$REGIME_DIR/comparison_report.md"

for target in $TOP_3_TARGETS; do
    echo "### $target" >> "$REGIME_DIR/comparison_report.md"
    echo "" >> "$REGIME_DIR/comparison_report.md"
    
    if [ -f "results/baseline_week1/features_${target}/feature_importance_multi_model.csv" ]; then
        echo "**Baseline Top 10:**" >> "$REGIME_DIR/comparison_report.md"
        echo "\`\`\`" >> "$REGIME_DIR/comparison_report.md"
        head -11 "results/baseline_week1/features_${target}/feature_importance_multi_model.csv" >> "$REGIME_DIR/comparison_report.md"
        echo "\`\`\`" >> "$REGIME_DIR/comparison_report.md"
        echo "" >> "$REGIME_DIR/comparison_report.md"
    fi
    
    if [ -f "$REGIME_DIR/features_${target}/feature_importance_multi_model.csv" ]; then
        echo "**With Regime Top 10:**" >> "$REGIME_DIR/comparison_report.md"
        echo "\`\`\`" >> "$REGIME_DIR/comparison_report.md"
        head -11 "$REGIME_DIR/features_${target}/feature_importance_multi_model.csv" >> "$REGIME_DIR/comparison_report.md"
        echo "\`\`\`" >> "$REGIME_DIR/comparison_report.md"
        echo "" >> "$REGIME_DIR/comparison_report.md"
        
        # Count regime features in top 20
        regime_count=$(head -21 "$REGIME_DIR/features_${target}/feature_importance_multi_model.csv" | grep -i "regime" | wc -l)
        echo "**Regime features in top 20:** $regime_count" >> "$REGIME_DIR/comparison_report.md"
        echo "" >> "$REGIME_DIR/comparison_report.md"
    fi
done

echo "================================================================================"
echo "✅ REGIME ENHANCEMENT COMPLETE"
echo "================================================================================"
echo ""
echo "📁 Output location: $REGIME_DIR/"
echo ""
echo "📊 Review results:"
echo "  1. Comparison report: cat $REGIME_DIR/comparison_report.md"
echo "  2. Feature rankings:  head -20 $REGIME_DIR/features_*/feature_importance_multi_model.csv"
echo "  3. Check regime rank: grep -i regime $REGIME_DIR/features_*/feature_importance_multi_model.csv"
echo ""
echo "🎯 Next Steps:"
echo "  1. Review comparison_report.md"
echo "  2. If regime features rank high → Keep them, proceed to Week 3 (VIX)"
echo "  3. If regime features rank low → Tweak parameters or skip to Week 4"
echo ""

