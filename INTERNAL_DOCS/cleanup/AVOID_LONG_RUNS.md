# How to Avoid 4-Day Runs

Based on the leakage.md findings, here's how to avoid running scripts for days:

## The Problem

Before the leakage fix, you had to:
- Run target ranking with all models
- Process all symbols
- Wait 4+ days for results
- Then discover leakage issues and start over

## The Solution: Fast Initial Ranking + Checkpointing

### Step 1: Fast Initial Target Ranking (30-45 minutes)

Use the fast configuration for initial screening:

```bash
conda activate trader_env
cd /home/Jennifer/trader

# Use fast config (3 models instead of 11, fewer samples)
python SCRIPTS/rank_target_predictability.py \
  --discover-all \
  --symbols AAPL,MSFT,GOOGL,TSLA,JPM \
  --output-dir results/target_rankings_fast \
  --multi-model-config CONFIG/fast_target_ranking.yaml
```

**Time:** ~30-45 minutes (vs 2-3 hours with all models)

**What you get:**
- Quick ranking of all targets
- Identifies top 5-10 targets worth deeper analysis
- Leakage detection flags suspicious targets
- Checkpoint saved after each target (can resume if interrupted)

### Step 2: Review Results

```bash
# View results
head -20 results/target_rankings_fast/target_predictability_rankings.csv

# Filter for good targets
python -c "
import pandas as pd
df = pd.read_csv('results/target_rankings_fast/target_predictability_rankings.csv')
good = df[(df['composite_score'] > 0.4) & (df['mean_r2'] > 0.05) & (df['leakage_flag'] == 'OK')]
print('Top targets for detailed analysis:')
print(good[['target_name', 'composite_score', 'mean_r2', 'n_models']].head(10).to_string())
"
```

**Action:** Select top 3-5 targets for detailed analysis.

### Step 3: Detailed Analysis on Top Targets Only (1-2 hours)

Only run full model suite on the top targets:

```bash
# Use default config (all 11 models) but only on top targets
python SCRIPTS/rank_target_predictability.py \
  --targets peak_60m_0.8,valley_60m_0.8,swing_high_15m_0.05 \
  --symbols AAPL,MSFT,GOOGL,TSLA,JPM \
  --output-dir results/target_rankings_detailed
```

**Time:** ~1-2 hours (only 3-5 targets vs all targets)

**Benefit:** Full model robustness on targets that matter

### Step 4: Feature Ranking (Use Checkpointing)

For feature ranking, use checkpointing to resume if interrupted:

```bash
# Start feature ranking (can take hours)
python SCRIPTS/rank_features_by_ic_and_predictive.py \
  --symbols AAPL,MSFT,GOOGL,TSLA,JPM \
  --target-rankings results/target_rankings_fast/target_predictability_rankings.yaml \
  --top-n-targets 5 \
  --output-dir results/feature_selection \
  --resume  # Resume from checkpoint if available
```

**If interrupted:**
```bash
# Just run again with --resume flag
python SCRIPTS/rank_features_by_ic_and_predictive.py \
  --symbols AAPL,MSFT,GOOGL,TSLA,JPM \
  --target-rankings results/target_rankings_fast/target_predictability_rankings.yaml \
  --top-n-targets 5 \
  --output-dir results/feature_selection \
  --resume  # Automatically resumes from checkpoint
```

## Time Comparison

| Approach | Time | Models | Use Case |
|----------|------|--------|----------|
| **Old (all models, all targets)** | 4+ days | 11 | Too slow |
| **Fast initial (3 models, all targets)** | 30-45 min | 3 | Initial screening |
| **Detailed (11 models, top 5 targets)** | 1-2 hours | 11 | Final validation |
| **Total optimized workflow** | ~2-3 hours | Mixed | Best approach |

## Key Strategies

### 1. Use Fast Config for Initial Ranking

The `CONFIG/fast_target_ranking.yaml` uses:
- Only 3 fastest models (LightGBM, Random Forest, Neural Network disabled)
- Fewer samples per symbol (10k vs 50k)
- Faster cross-validation (3 folds vs 5)
- Still includes leakage detection

**Result:** 10-20x faster, still reliable for ranking

### 2. Use Checkpointing

All scripts now support checkpointing:
- Saves progress after each target/symbol
- Resume with `--resume` flag
- No lost work if interrupted

### 3. Two-Stage Approach

1. **Stage 1 (Fast):** Screen all targets with fast config
2. **Stage 2 (Detailed):** Deep analysis on top 5-10 targets only

### 4. Monitor Progress Over SSH

```bash
# Watch progress in real-time
journalctl -f SYSLOG_IDENTIFIER=rank_target_predictability

# Check checkpoint status
python -c "
import json
with open('results/target_rankings_fast/checkpoint.json') as f:
    data = json.load(f)
    print(f'Completed: {len(data.get(\"completed_items\", {}))} targets')
    print(f'Failed: {len(data.get(\"failed_items\", []))} targets')
"
```

## Recommended Workflow

```bash
# Day 1: Fast initial ranking (30-45 min)
python SCRIPTS/rank_target_predictability.py \
  --discover-all \
  --symbols AAPL,MSFT,GOOGL,TSLA,JPM \
  --output-dir results/target_rankings_fast \
  --multi-model-config CONFIG/fast_target_ranking.yaml

# Review results, select top 5 targets

# Day 1: Detailed analysis on top targets (1-2 hours)
python SCRIPTS/rank_target_predictability.py \
  --targets <top_target_1>,<top_target_2>,<top_target_3>,<top_target_4>,<top_target_5> \
  --symbols AAPL,MSFT,GOOGL,TSLA,JPM \
  --output-dir results/target_rankings_detailed

# Day 1-2: Feature ranking (can run overnight with checkpointing)
nohup python SCRIPTS/rank_features_by_ic_and_predictive.py \
  --symbols AAPL,MSFT,GOOGL,TSLA,JPM \
  --target-rankings results/target_rankings_fast/target_predictability_rankings.yaml \
  --top-n-targets 5 \
  --output-dir results/feature_selection \
  --resume \
  > logs/feature_ranking.log 2>&1 &

# Monitor progress
journalctl -f SYSLOG_IDENTIFIER=rank_features_by_ic_and_predictive
```

**Total time:** ~2-3 hours active work + overnight feature ranking (with checkpointing)

## If You Need to Resume

All scripts support resuming:

```bash
# Target ranking
python SCRIPTS/rank_target_predictability.py \
  --discover-all \
  --symbols AAPL,MSFT,GOOGL,TSLA,JPM \
  --output-dir results/target_rankings_fast \
  --resume  # Automatically skips completed targets

# Feature ranking
python SCRIPTS/rank_features_by_ic_and_predictive.py \
  --symbols AAPL,MSFT,GOOGL,TSLA,JPM \
  --target-rankings results/target_rankings_fast/target_predictability_rankings.yaml \
  --top-n-targets 5 \
  --output-dir results/feature_selection \
  --resume  # Automatically skips completed targets
```

## Summary

**Before (4 days):**
- All models, all targets, no checkpointing
- Had to restart if interrupted
- Discovered leakage issues after completion

**After (2-3 hours + overnight):**
- Fast initial screening (30-45 min)
- Detailed analysis on top targets only (1-2 hours)
- Checkpointing prevents lost work
- Leakage detection flags issues immediately
- Can monitor progress over SSH

**Key files:**
- `CONFIG/fast_target_ranking.yaml` - Fast configuration
- `CONFIG/multi_model_feature_selection.yaml` - Full configuration (for detailed analysis)
- Checkpoint files in output directories - For resuming

