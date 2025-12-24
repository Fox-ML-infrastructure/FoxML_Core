# Alpha Enhancement Roadmap

Strategic plan for improving predictive edge using your multi-model feature selection system.

---

## Current State (Your Baseline)

 **You have:**
- 728 symbols with 5-minute bars
- 63 targets (including triple-barrier-like targets already!)
- Multi-model feature selection system (LightGBM, XGBoost, RF, NN)
- Target predictability ranking system
- ~40-60 engineered features per target

 **You need baseline metrics:**
- Which targets are actually predictable? (R² scores)
- Which features have universal alpha?
- Current Sharpe ratio / hit rate

---

## Phase 1: Validate Current System (Week 1) **START HERE**

### Goal
Establish clean baseline before adding complexity.

### Actions

```bash
# 1. Rank all 63 targets by predictability (30 min)
python SCRIPTS/rank_target_predictability.py

# 2. Review rankings, disable bottom 50%
cat results/target_rankings/target_predictability_rankings.yaml
vim CONFIG/target_configs.yaml  # Set enabled: false for weak targets

# 3. Run multi-model selection on top 3 targets (overnight)
for target in y_will_peak_60m_0.8 y_first_touch_60m_0.8 y_will_valley_60m_0.8; do
    python SCRIPTS/multi_model_feature_selection.py \
      --target-column $target \
      --top-n 60
done

# 4. Document baseline
echo "Baseline R²: <fill in>" > docs/baseline_metrics.md
echo "Top features: <list>" >> docs/baseline_metrics.md
```

### Success Criteria
- Know which 10-15 targets are worth training
- Have consensus features for top 3 targets
- Baseline R² documented

**Time:** 1 day active work + 1 overnight run

---

## Phase 2: Quick Alpha Wins (Week 2-3)

### 2A. **Regime Detection Features** **HIGHEST ROI**

**Why first:**
- Multiplicative effect (features work differently in different regimes)
- Easy to implement
- Multi-model system will tell you if they help

**Implementation:**

```python
# Add to DATA_PROCESSING/features/regime_features.py

import numpy as np
import pandas as pd

def detect_regime(df: pd.DataFrame, lookback: int = 50) -> pd.Series:
    """
    Detect market regime: trending, choppy, or volatile

    Returns: regime column with values [0, 1, 2]
    """
    # Trend strength: correlation between price and time
    trend_strength = df['close'].rolling(lookback).apply(
        lambda x: np.corrcoef(x, np.arange(len(x)))[0, 1]
    )

    # Volatility: rolling std
    volatility = df['close'].pct_change().rolling(lookback).std()
    volatility_zscore = (volatility - volatility.rolling(200).mean()) / volatility.rolling(200).std()

    # Classify regime
    regime = pd.Series(0, index=df.index)  # 0 = choppy (default)
    regime[trend_strength.abs() > 0.7] = 1  # 1 = trending
    regime[volatility_zscore > 1.5] = 2     # 2 = high volatility

    return regime


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add regime-aware features"""

    # Detect regime
    df['regime'] = detect_regime(df, lookback=50)
    df['regime_trend'] = (df['regime'] == 1).astype(int)
    df['regime_chop'] = (df['regime'] == 0).astype(int)
    df['regime_vol'] = (df['regime'] == 2).astype(int)

    # Regime-conditional features
    for lookback in [5, 10, 20]:
        # Returns work differently in different regimes
        ret = df['close'].pct_change(lookback)
        df[f'ret_{lookback}m_in_trend'] = ret * df['regime_trend']
        df[f'ret_{lookback}m_in_chop'] = ret * df['regime_chop']
        df[f'ret_{lookback}m_in_vol'] = ret * df['regime_vol']

    # RSI works differently in different regimes
    rsi = compute_rsi(df['close'], 14)
    df['rsi_in_trend'] = rsi * df['regime_trend']
    df['rsi_in_chop'] = rsi * df['regime_chop']

    return df
```

**Test immediately:**

```bash
# Add regime features to your data pipeline
python DATA_PROCESSING/pipeline/run_pipeline.py --add-regime-features

# Run multi-model selection with regime features
python SCRIPTS/multi_model_feature_selection.py \
  --target-column y_will_peak_60m_0.8 \
  --top-n 60

# Check if regime features rank high
grep "regime" DATA_PROCESSING/data/features/multi_model/feature_importance_multi_model.csv
```

**Expected impact:** +5-15% improvement in R² if regimes matter for your data.

**Time:** 2 days

---

### 2B. **Triple-Barrier Targets** **YOU ALREADY HAVE THESE!**

Check your target columns:

```bash
python -c "
import pandas as pd
df = pd.read_parquet('data/data_labeled/interval=5m/symbol=AAPL/AAPL.parquet')
targets = [c for c in df.columns if c.startswith('y_')]
print('\n'.join(targets))
"
```

Look for:
- `y_will_peak_*` ← These ARE triple-barrier-like (take profit)
- `y_will_valley_*` ← Stop loss
- `y_first_touch_*` ← First barrier hit

**Action:** Your target ranking will tell you which are most predictable!

**Time:** Already done

---

### 2C. **VIX Features** (If Available)

**Only if you have VIX data:**

```python
# Add to DATA_PROCESSING/features/macro_features.py

def add_vix_features(stock_df: pd.DataFrame, vix_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add VIX (fear gauge) features to stock data

    Args:
        stock_df: Stock 5-minute bars
        vix_df: VIX 5-minute bars (from same time)
    """
    # Merge on timestamp
    stock_df = stock_df.merge(
        vix_df[['timestamp', 'close']],
        on='timestamp',
        how='left',
        suffixes=('', '_vix')
    )

    # VIX level (absolute fear)
    stock_df['vix_level'] = stock_df['close_vix']

    # VIX change (fear acceleration)
    stock_df['vix_change_5m'] = stock_df['close_vix'].pct_change()
    stock_df['vix_change_15m'] = stock_df['close_vix'].pct_change(3)

    # VIX term structure (if you have VIX futures)
    # Higher VIX futures = backwardation = market expects vol to fall

    # Stock sensitivity to VIX (beta to fear)
    stock_ret = stock_df['close'].pct_change()
    vix_ret = stock_df['close_vix'].pct_change()
    stock_df['vix_beta_50'] = stock_ret.rolling(50).cov(vix_ret) / vix_ret.rolling(50).var()

    return stock_df
```

**Test:**

```bash
python SCRIPTS/multi_model_feature_selection.py --target-column y_will_peak_60m_0.8

# Check VIX feature ranking
grep "vix" DATA_PROCESSING/data/features/multi_model/feature_importance_multi_model.csv
```

**Expected:** VIX features will rank high for risk-sensitive stocks (financials, small caps).

**Time:** 3 days (need to get VIX data first)

---

## Phase 3: Advanced Features (Week 4+)

### 3A. **Fractional Differentiation**

**When:** After validating regime features work.

**Why:** Makes price series stationary for deep learning models.

```python
# Add to DATA_PROCESSING/features/advanced_features.py

def fractional_diff(series: pd.Series, d: float, threshold: float = 0.01) -> pd.Series:
    """
    Fractional differentiation (Marcos López de Prado)

    Args:
        series: Price series
        d: Differentiation order (0.0 = raw, 1.0 = full diff)
        threshold: Weight cutoff for computational efficiency

    Returns:
        Stationary series with memory
    """
    import numpy as np
    from scipy.special import comb

    # Compute weights
    weights = [1.0]
    for k in range(1, len(series)):
        weight = -weights[-1] * (d - k + 1) / k
        if abs(weight) < threshold:
            break
        weights.append(weight)

    weights = np.array(weights[::-1])

    # Apply weights (convolution)
    diff_series = pd.Series(dtype=float, index=series.index)
    for i in range(len(weights), len(series)):
        diff_series.iloc[i] = np.dot(weights, series.iloc[i-len(weights)+1:i+1])

    return diff_series


# Test different d values
for d in [0.3, 0.5, 0.7]:
    df[f'price_fracdiff_{d:.1f}'] = fractional_diff(df['close'], d)
```

**Test:** Run multi-model selection, see if fractional diff features rank high.

**Time:** 3 days

---

### 3B. **Neural Network Meta-Features**

**When:** After Phase 2 complete.

**Why:** Let NN discover non-linear feature interactions.

```python
# Add to TRAINING/features/meta_features.py

from sklearn.neural_network import MLPRegressor
import numpy as np

def create_neural_meta_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    hidden_layers=(128, 64, 32)
) -> np.ndarray:
    """
    Train neural network, extract penultimate layer as meta-features

    Returns:
        meta_features: (N, hidden_layers[-1]) array of learned features
    """
    # Train NN
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        max_iter=300,
        early_stopping=True,
        random_state=42
    )
    model.fit(X, y)

    # Extract activations from second-to-last layer
    # This requires custom forward pass
    from sklearn.neural_network._multilayer_perceptron import MLPRegressor as MLP

    # Forward pass to penultimate layer
    X_scaled = model._scaler.transform(X)
    layer_output = X_scaled
    for i in range(len(model.coefs_) - 1):  # Stop before last layer
        layer_output = model.activation(layer_output @ model.coefs_[i] + model.intercepts_[i])

    # layer_output shape: (N, 32) - these are your meta-features
    return layer_output


# Usage in pipeline
meta_features = create_neural_meta_features(X, y, feature_names)

# Name them
meta_feature_names = [f'nn_meta_{i}' for i in range(meta_features.shape[1])]

# Concatenate with original features
X_augmented = np.column_stack([X, meta_features])
feature_names_augmented = feature_names + meta_feature_names

# Train XGBoost on augmented features
```

**Test:** Run multi-model selection on augmented features.

**Expected:** Meta-features capture interactions, boost XGBoost/LightGBM R² by 10-20%.

**Time:** 1 week

---

## Priority Ranking

| Feature Type | ROI | Complexity | Time | Do When |
|--------------|-----|------------|------|---------|
| **Regime detection** | | Low | 2 days | **Week 2** |
| **Target ranking** | | Done | 0 | **Week 1** |
| **VIX features** | | Low | 3 days | Week 3 |
| **Triple barriers** | Done | Done | 0 | **You have these** |
| **Fractional diff** | | Med | 3 days | Week 4 |
| **NN meta-features** | | High | 1 week | Week 5 |
| **Currency crosses** | | Med | 1 week | Month 2 |
| **Treasury yields** | | Med | 1 week | Month 2 |

---

## Integration with Multi-Model System

**The beauty of your new system:** Every new feature you add can be instantly evaluated!

```bash
# Add regime features to data
python add_regime_features.py

# Test if they help (2-10 hours)
python SCRIPTS/multi_model_feature_selection.py \
  --target-column y_will_peak_60m_0.8 \
  --top-n 60

# Check ranking
head -20 DATA_PROCESSING/data/features/multi_model/feature_importance_multi_model.csv
```

**If regime features rank in top 20** → Keep them
**If they rank bottom 40** → Remove them

**This is your alpha discovery pipeline:**
1. Add new feature family
2. Run multi-model selection
3. Check ranking
4. Keep winners, discard losers

---

## Recommended 4-Week Plan

### Week 1: Validation
- Run target ranking
- Run multi-model selection on top 3 targets
- Document baseline (R², Sharpe, top features)

### Week 2: Regime Features
- Implement regime detection
- Add regime-conditional features
- Run multi-model selection
- Measure improvement vs baseline

### Week 3: VIX Features (if available)
- Get VIX data
- Add VIX features
- Test with multi-model selection
- Compare: regime only vs regime + VIX

### Week 4: Fractional Differentiation
- Implement frac diff
- Test d values [0.3, 0.5, 0.7]
- Multi-model selection
- Keep best d value

### Month 2: Neural Meta-Features
- Extract NN penultimate layer
- Test with tree models
- Measure lift

---

## Success Metrics

**Baseline (Week 1):**
- Top target R²: 0.XX
- Top features: [list]
- Sharpe: X.XX

**After Regime (Week 2):**
- Target R² improvement: +10-15%
- Regime features in top 20? YES/NO
- Sharpe improvement: +0.2-0.4

**After VIX (Week 3):**
- R² improvement: +5-10%
- VIX features ranked: Top 30
- Works for: [list of stocks]

**After Frac Diff (Week 4):**
- R² improvement: +5-8%
- Best d value: 0.X
- LSTM performance: +20%

**After NN Meta (Month 2):**
- R² improvement: +15-25%
- XGBoost with meta features: Best model
- Production-ready

---

## What NOT to Do

 **Don't add all features at once** → Can't measure individual impact
 **Don't skip baseline validation** → Need comparison point
 **Don't ignore multi-model rankings** → If features rank low, remove them
 **Don't add features without your data** → VIX/currency needs data first
 **Don't optimize hyperparameters yet** → Add features first, tune later

---

## Quick Start Command

```bash
# Week 1: Baseline
python SCRIPTS/rank_target_predictability.py
python SCRIPTS/multi_model_feature_selection.py --target-column y_will_peak_60m_0.8 --top-n 60

# Week 2: Add regime features
# (implement regime_features.py first)
python SCRIPTS/multi_model_feature_selection.py --target-column y_will_peak_60m_0.8 --top-n 60

# Compare:
python SCRIPTS/compare_feature_sets.py \
  --set1 DATA_PROCESSING/data/features/multi_model_baseline/selected_features.txt \
  --set2 DATA_PROCESSING/data/features/multi_model_with_regime/selected_features.txt
```

---

**Bottom line:** Do regime detection in Week 2. It's the highest ROI, lowest complexity addition that will work multiplicatively with your existing features. Your multi-model system will tell you immediately if it helps!

