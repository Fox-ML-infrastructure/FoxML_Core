# Target to Feature Ranking Workflow

## Step 1: Run Target Predictability Ranking

Rank all targets to identify which are most predictable.

```bash
conda activate trader_env
cd /home/Jennifer/trader

python SCRIPTS/rank_target_predictability.py \
  --discover-all \
  --symbols AAPL,MSFT,GOOGL,TSLA,JPM \
  --output-dir results/target_rankings
```

**Output:**
- `results/target_rankings/target_predictability_rankings.csv`
- `results/target_rankings/target_predictability_rankings.yaml`
- `results/target_rankings/checkpoint.json` (for resuming)

**What to look for:**
- Targets with `composite_score > 0.5` and `mean_r2 > 0.1` (good targets)
- Targets with `leakage_flag != "OK"` (review these - may have issues)
- Targets with `n_models >= 6` (robust across models)

**Time:** ~2-3 hours for all targets

---

## Step 2: Review Target Rankings

```bash
# View top targets
head -20 results/target_rankings/target_predictability_rankings.csv

# Or filter for good targets
python -c "
import pandas as pd
df = pd.read_csv('results/target_rankings/target_predictability_rankings.csv')
good = df[(df['composite_score'] > 0.5) & (df['mean_r2'] > 0.1) & (df['leakage_flag'] == 'OK')]
print('Top targets to use:')
print(good[['target_name', 'composite_score', 'mean_r2', 'n_models']].head(10).to_string())
"
```

**Action:** Select top 3-5 targets for feature ranking.

---

## Step 3: Run Feature Ranking for Top Targets

Use the feature ranking script with your top targets.

### Option A: Auto-use top targets from rankings

```bash
python SCRIPTS/rank_features_by_ic_and_predictive.py \
  --symbols AAPL,MSFT,GOOGL,TSLA,JPM \
  --target-rankings results/target_rankings/target_predictability_rankings.yaml \
  --top-n-targets 5 \
  --output-dir results/feature_selection
```

### Option B: Specify targets manually

```bash
python SCRIPTS/rank_features_by_ic_and_predictive.py \
  --symbols AAPL,MSFT,GOOGL,TSLA,JPM \
  --targets y_will_peak_60m_0.8,y_will_valley_60m_0.8 \
  --output-dir results/feature_selection
```

**Output:**
- `results/feature_selection/feature_rankings.csv`
- `results/feature_selection/feature_rankings_by_model.csv`
- `results/feature_selection/checkpoint.json` (for resuming)

**What this does:**
- Computes IC (correlation) for each feature with each target
- Computes predictive power using all enabled models
- Combines IC + predictive power into final ranking
- Ranks features across all targets

**Time:** ~2-3 hours per symbol (or ~10-15 hours for 5 symbols)

---

## Step 4: Review Feature Rankings

```bash
# View top features
head -30 results/feature_selection/feature_rankings.csv

# View per-model breakdown
head -30 results/feature_selection/feature_rankings_by_model.csv
```

**What to look for:**
- Features with `combined_score > 0.7` (excellent)
- Features important in 8+ models (high consensus)
- Features that work across multiple targets

**Action:** Select top 50-100 features for production use.

---

## Step 5: (Optional) Run Multi-Model Feature Selection

For detailed per-symbol analysis on a specific target:

```bash
python SCRIPTS/multi_model_feature_selection.py \
  --symbols AAPL,MSFT,GOOGL,TSLA,JPM \
  --target-column y_will_peak_60m_0.8 \
  --top-n 60 \
  --output-dir results/multi_model_features
```

**Output:**
- `results/multi_model_features/selected_features.txt`
- `results/multi_model_features/feature_importance_multi_model.csv`
- `results/multi_model_features/model_agreement_matrix.csv`

**Use this when:**
- You want detailed per-symbol feature importance
- You want to see model agreement matrix
- You're selecting features for a specific target

---

## Summary Workflow

1. **Target Ranking** → Find best targets
2. **Review Targets** → Select top 3-5 targets
3. **Feature Ranking** → Find best features for those targets
4. **Review Features** → Select top 50-100 features
5. **Multi-Model Selection** (optional) → Detailed analysis for specific target

---

## Quick Reference Commands

```bash
# 1. Target ranking
python SCRIPTS/rank_target_predictability.py --discover-all --symbols AAPL,MSFT,GOOGL,TSLA,JPM --output-dir results/target_rankings

# 2. Feature ranking (auto-use top 5 targets)
python SCRIPTS/rank_features_by_ic_and_predictive.py --symbols AAPL,MSFT,GOOGL,TSLA,JPM --target-rankings results/target_rankings/target_predictability_rankings.yaml --top-n-targets 5 --output-dir results/feature_selection

# 3. Feature ranking (manual targets)
python SCRIPTS/rank_features_by_ic_and_predictive.py --symbols AAPL,MSFT,GOOGL,TSLA,JPM --targets y_will_peak_60m_0.8,y_will_valley_60m_0.8 --output-dir results/feature_selection

# Resume from checkpoint (if interrupted)
python SCRIPTS/rank_target_predictability.py --discover-all --symbols AAPL,MSFT,GOOGL,TSLA,JPM --output-dir results/target_rankings --resume
```

---

## Expected Results

### Good Targets:
- `composite_score > 0.5`
- `mean_r2 > 0.1`
- `leakage_flag == "OK"`
- `n_models >= 6`

### Good Features:
- `combined_score > 0.7`
- Important in 8+ models
- High IC (correlation) with targets
- Works across multiple targets

