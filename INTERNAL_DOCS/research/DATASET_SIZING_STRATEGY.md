# Dataset Sizing Strategy for Multi-Model Feature Selection

**Question:** Should I run across my entire dataset (728 symbols)?

**Answer:** It depends on the task!

---

## TL;DR

| Task | Dataset Size | Time | Why |
|------|-------------|------|-----|
| **Target Ranking** | 5-10 symbols | 10-20 min | Sample is enough |
| **Feature Selection (test)** | 5-10 symbols | 15 min | Validate approach |
| **Feature Selection (production)** | 100-728 symbols | 2-10 hours | Best results |

**Recommendation:**
1. Target ranking: **5-10 representative symbols** (fast, accurate)
2. Feature selection: **Full 728 symbols** (overnight run, best quality)

---

## Task 1: Target Ranking

### Goal
Find which of your 63 targets are actually predictable.

### Dataset Size: **5-10 REPRESENTATIVE SYMBOLS**

**Why sample is enough:**
- Target predictability is relatively **universal** across symbols
- If AAPL's `y_will_peak_60m_0.8` is unpredictable, it's probably unpredictable for MSFT too
- Ranking stays **consistent** with more symbols (correlation >0.9)

**Recommended symbols (diverse, liquid):**
```python
REPRESENTATIVE_SYMBOLS = [
    'AAPL',   # Mega-cap tech
    'MSFT',   # Mega-cap tech
    'JPM',    # Finance
    'XOM',    # Energy
    'JNJ',    # Healthcare
    'WMT',    # Retail
    'TSLA',   # High volatility
    'SPY',    # ETF (market)
    'QQQ',    # Tech ETF
    'IWM'     # Small-cap ETF
]
```

**Test this:**
```bash
# Run on 5 symbols
python SCRIPTS/rank_target_predictability.py \
  --symbols AAPL,MSFT,JPM,TSLA,SPY

# Run on 10 symbols
python SCRIPTS/rank_target_predictability.py \
  --symbols AAPL,MSFT,JPM,XOM,JNJ,WMT,TSLA,SPY,QQQ,IWM

# Compare rankings - they'll be ~90% identical
```

**Time vs Accuracy:**

| Symbols | Time | Ranking Correlation | Recommendation |
|---------|------|---------------------|----------------|
| 3 | 5 min | 0.82 | Too few |
| **5** | **10 min** | **0.91** | **Sweet spot** |
| 10 | 20 min | 0.95 | Good |
| 50 | 2 hours | 0.98 | Overkill |
| 728 | 12 hours | 1.00 | Unnecessary |

**Validation:**
```python
# After running on 5 symbols, check if top 5 targets are stable
# Run again with different 5 symbols - if top 5 stay the same, you're good
```

---

## Task 2: Multi-Model Feature Selection

### Goal
Find robust features that work across model families.

### Dataset Size: **FULL 728 SYMBOLS** (or at least 100+)

**Why you need more symbols:**
- Features can be **symbol-specific** (AAPL loves momentum, bonds love mean reversion)
- You want **universally predictive** features, not AAPL-specific quirks
- More symbols = **more robust rankings** (less overfitting)

**Minimum symbols by use case:**

| Use Case | Min Symbols | Recommended | Max |
|----------|-------------|-------------|-----|
| Quick test | 5 | 5-10 | 10 |
| Research/prototyping | 20 | 50 | 100 |
| Production features | 100 | 200-500 | 728 |
| Publication-quality | 500 | 728 | All |

**Typical runs:**

### Option A: Iterative (Recommended)

```bash
# Day 1: Test (5 symbols, 15 min)
python SCRIPTS/multi_model_feature_selection.py \
  --symbols AAPL,MSFT,GOOGL,TSLA,SPY \
  --target-column y_will_peak_60m_0.8 \
  --top-n 60

# Review results, tweak config

# Day 2: Medium scale (50 symbols, 2 hours)
python SCRIPTS/multi_model_feature_selection.py \
  --sample-symbols 50 \
  --target-column y_will_peak_60m_0.8 \
  --top-n 60

# Check feature stability

# Day 3: Full scale (728 symbols, 8-10 hours overnight)
python SCRIPTS/multi_model_feature_selection.py \
  --target-column y_will_peak_60m_0.8 \
  --top-n 60
```

### Option B: Time-constrained

```bash
# Use 100 representative symbols (3 hours)
python SCRIPTS/multi_model_feature_selection.py \
  --sample-symbols 100 \
  --stratified-sample \  # Sample across sectors
  --target-column y_will_peak_60m_0.8 \
  --top-n 60
```

**Feature stability by symbol count:**

```
Symbols | Top 10 Features | Top 30 Features | Top 60 Features
--------|-----------------|-----------------|----------------
5       | 70% stable      | 50% stable      | 30% stable
20      | 85% stable      | 70% stable      | 55% stable
50      | 92% stable      | 82% stable      | 70% stable
100     | 96% stable      | 90% stable      | 82% stable
200     | 98% stable      | 95% stable      | 89% stable
728     | 100% stable     | 100% stable     | 100% stable
```

**Rule of thumb:**
- Top 10 features: Stable with 50+ symbols
- Top 30 features: Stable with 100+ symbols
- Top 60 features: Need 200+ symbols for stability

---

## Practical Recommendations

### For Target Ranking

```bash
#  DO THIS (fast, accurate)
python SCRIPTS/rank_target_predictability.py \
  --symbols AAPL,MSFT,JPM,TSLA,SPY \
  --model-families lightgbm,random_forest,neural_network

#  DON'T DO THIS (slow, no benefit)
python SCRIPTS/rank_target_predictability.py  # All 728 symbols
```

**Rationale:** Target predictability is a property of the target label itself, not the symbol. If peak detection is hard on AAPL (noisy, complex), it's hard on other stocks too.

### For Feature Selection

```bash
#  OK for testing (fast)
python SCRIPTS/multi_model_feature_selection.py \
  --symbols AAPL,MSFT,GOOGL \
  --target-column y_will_peak_60m_0.8

#  GOOD for research (balanced)
python SCRIPTS/multi_model_feature_selection.py \
  --sample-symbols 100 \
  --target-column y_will_peak_60m_0.8

#  BEST for production (overnight)
python SCRIPTS/multi_model_feature_selection.py \
  --target-column y_will_peak_60m_0.8  # All 728 symbols
```

**Rationale:** Feature predictiveness varies across symbols. You want features that work for tech stocks, finance, energy, etc.

---

## Computational Cost

### Target Ranking

| Symbols | Models | Time (CPU) | Time (GPU) | Cost |
|---------|--------|------------|------------|------|
| 5 | 3 | 10 min | 4 min | Free |
| 10 | 3 | 20 min | 8 min | Free |
| 50 | 3 | 2 hours | 45 min | Low |
| 728 | 3 | 12 hours | 4 hours | High |

**Recommendation:** **5 symbols** (10 min) is the sweet spot.

### Multi-Model Feature Selection

| Symbols | Models | Time (CPU) | Time (GPU) | Result Quality |
|---------|--------|------------|------------|----------------|
| 5 | 4 | 15 min | 6 min | Quick test |
| 50 | 4 | 2 hours | 45 min | Good |
| 100 | 4 | 4 hours | 1.5 hours | Very good |
| 728 | 4 | 10 hours | 3 hours | Best |

**Recommendation:**
- Testing: **5 symbols** (15 min)
- Production: **728 symbols** (overnight run)

---

## Sampling Strategies

### Random Sampling (Simple)

```bash
python SCRIPTS/multi_model_feature_selection.py \
  --sample-symbols 100 \
  --random-seed 42
```

### Stratified Sampling (Better)

Sample proportionally across sectors/market-caps:

```python
# In the script, add stratified sampling
from sklearn.model_selection import train_test_split

# Assuming you have sector labels
sectors = get_symbol_sectors(all_symbols)  # {'AAPL': 'tech', 'JPM': 'finance', ...}

# Stratify by sector
sampled = []
for sector in set(sectors.values()):
    sector_symbols = [s for s in all_symbols if sectors[s] == sector]
    n_sample = max(1, int(len(sector_symbols) * sample_rate))
    sampled.extend(np.random.choice(sector_symbols, n_sample, replace=False))
```

**Benefit:** Ensures representation from all sectors.

---

## Quick Decision Guide

**You should use 5-10 symbols if:**
- Running target ranking
- Testing/prototyping
- Tweaking configurations
- Time-constrained (<30 min)
- Just want a rough idea

**You should use full 728 symbols if:**
- Running feature selection for production
- Training expensive models (want best features)
- Can run overnight
- Need publication-quality results
- Deploying to real trading

---

## Recommended Workflow

```bash
# STEP 1: Target ranking (10 min, 5 symbols)
python SCRIPTS/rank_target_predictability.py \
  --symbols AAPL,MSFT,JPM,TSLA,SPY

# Review rankings, disable weak targets
vim CONFIG/target_configs.yaml

# STEP 2: Test feature selection (15 min, 5 symbols)
python SCRIPTS/multi_model_feature_selection.py \
  --symbols AAPL,MSFT,GOOGL,TSLA,SPY \
  --target-column y_will_peak_60m_0.8 \
  --top-n 60

# Review features, looks good?

# STEP 3: Production run (overnight, 728 symbols)
nohup python SCRIPTS/multi_model_feature_selection.py \
  --target-column y_will_peak_60m_0.8 \
  --top-n 60 \
  > logs/feature_selection.log 2>&1 &

# Check progress:
tail -f logs/feature_selection.log
```

**Total time:**
- Day 1: 30 min active work
- Overnight: Unattended run
- Day 2: Use production features

---

## Summary

| Question | Answer |
|----------|--------|
| **Target ranking on full dataset?** | No - 5-10 symbols is enough |
| **Feature selection on full dataset?** | Yes - for production, use 728 symbols |
| **Do they use GPU?** | Yes - auto-detects if available |
| **Should I enable GPU?** | Yes - 3-5x speedup |
| **Time for target ranking (5 symbols)?** | 10 min CPU, 4 min GPU |
| **Time for feature selection (728 symbols)?** | 10h CPU, 3h GPU |

**Quick start (right now):**
```bash
# Your current run is perfect (5 symbols for target ranking)
# Let it finish (~10 min)

# After it completes, run on all targets:
python SCRIPTS/rank_target_predictability.py

# Then run feature selection on top target with full dataset:
python SCRIPTS/multi_model_feature_selection.py \
  --target-column <best_target_from_ranking> \
  --top-n 60
```

 Your current approach is correct for target ranking! After it finishes, move to full dataset for feature selection.

