# Look-Ahead Bias Fix Plan

**Date**: 2025-12-14  
**Status**: ✅ **IMPLEMENTED** - All fixes complete, behind feature flags  
**Priority**: CRITICAL - These bugs likely explain "suspiciously high" scores  
**Branch**: `fix/lookahead-bias-fixes`

## Executive Summary

We've identified **4 critical look-ahead bias issues** that are likely causing inflated model scores:

1. **Rolling windows include current bar** (HIGH) - Features leak current price information
2. **Global normalization before train/test split** (CRITICAL) - Leaks future statistics
3. **pct_change() may include current bar** (MEDIUM) - Needs verification
4. **Misnamed features** (LOW) - Confusing but not a leak

**Root Cause**: The "suspiciously high" scores were likely from normalization leakage (#2) and current bar inclusion (#1), **NOT** the lookback length itself. Banning long-lookback features was treating the symptom, not the disease.

---

## Issue #1: Rolling Windows Include Current Bar

### Problem
Polars `rolling_mean()` includes the current row by default. This means features like `sma_20` at time `t` include `close[t]`, allowing the model to use current price information.

### Evidence
**File**: `DATA_PROCESSING/features/simple_features.py`

```python
# Lines 291-295: These include current bar!
pl.col("close").rolling_mean(5).alias("sma_5")
pl.col("close").rolling_mean(20).alias("sma_20")
pl.col("close").rolling_mean(50).alias("sma_50")

# Lines 603-608: Beta/correlation calculations
pl.col("close").pct_change().rolling_std(20).alias("beta_20d")
pl.col("close").pct_change().rolling_std(60).alias("market_correlation_60d")
```

### Impact
For a target like "will price go up in 10 mins", if `sma_20` at time `t` includes `close[t]`, the model can infer current price movement, creating look-ahead bias.

### Fix Required

**File**: `DATA_PROCESSING/features/simple_features.py`

**Change 1.1**: Fix rolling means (lines 291-295)
```python
# BEFORE:
pl.col("close").rolling_mean(5).alias("sma_5").cast(pl.Float32),
pl.col("close").rolling_mean(10).alias("sma_10").cast(pl.Float32),
pl.col("close").rolling_mean(20).alias("sma_20").cast(pl.Float32),
pl.col("close").rolling_mean(50).alias("sma_50").cast(pl.Float32),
pl.col("close").rolling_mean(200).alias("sma_200").cast(pl.Float32),

# AFTER:
pl.col("close").shift(1).rolling_mean(5).alias("sma_5").cast(pl.Float32),
pl.col("close").shift(1).rolling_mean(10).alias("sma_10").cast(pl.Float32),
pl.col("close").shift(1).rolling_mean(20).alias("sma_20").cast(pl.Float32),
pl.col("close").shift(1).rolling_mean(50).alias("sma_50").cast(pl.Float32),
pl.col("close").shift(1).rolling_mean(200).alias("sma_200").cast(pl.Float32),
```

**Change 1.2**: Fix rolling std (lines 603-608)
```python
# BEFORE:
pl.col("close").pct_change().rolling_std(20).alias("beta_20d").cast(pl.Float32),
pl.col("close").pct_change().rolling_std(60).alias("beta_60d").cast(pl.Float32),
pl.col("close").pct_change().rolling_std(20).alias("market_correlation_20d").cast(pl.Float32),
pl.col("close").pct_change().rolling_std(60).alias("market_correlation_60d").cast(pl.Float32),

# AFTER:
pl.col("close").pct_change().shift(1).rolling_std(20).alias("beta_20d").cast(pl.Float32),
pl.col("close").pct_change().shift(1).rolling_std(60).alias("beta_60d").cast(pl.Float32),
pl.col("close").pct_change().shift(1).rolling_std(20).alias("market_correlation_20d").cast(pl.Float32),
pl.col("close").pct_change().shift(1).rolling_std(60).alias("market_correlation_60d").cast(pl.Float32),
```

**Change 1.3**: Fix all other rolling operations in this file
- Search for all `.rolling_mean(`, `.rolling_std(`, `.rolling_sum(`, `.rolling_max(`, `.rolling_min(`
- Add `.shift(1)` before each rolling operation
- **Exception**: If the feature is explicitly meant to include current bar (e.g., "current_price"), document it

**Change 1.4**: Apply same fixes to `DATA_PROCESSING/features/comprehensive_builder.py`
- Lines 204-206: SMA calculations
- Lines 187-192: Volatility calculations
- All other rolling operations

### Verification
1. Create test: Generate sample data with known values
2. Verify: `sma_20[t]` should NOT include `close[t]` in calculation
3. Check: First 20 rows should be NaN (expected behavior after shift)

---

## Issue #2: Global Normalization Before Train/Test Split

### Problem
Scalers and imputers are being fit on the **entire dataset** before CV splits, leaking future statistics into training.

### Evidence

**File**: `TRAINING/ranking/multi_model_feature_selection.py:1158-1162`
```python
# ⚠️ LEAK: Fits on entire X before CV splits!
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)  # Uses future data for median!

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)  # Uses future data for mean/std!
```

**File**: `TRAINING/preprocessing/mega_script_sequential_preprocessor.py:90`
```python
# ⚠️ LEAK: Fits on entire dataset
X_imputed = self.imputer.fit_transform(X_float)
```

### Impact
If you normalize `beta_20d` across the entire dataset, a high value tells the model "this is the highest beta will be for the whole period." This is massive look-ahead bias.

### Fix Required

**File**: `TRAINING/ranking/multi_model_feature_selection.py`

**Change 2.1**: Move imputer/scaler inside CV loop (around line 2026)

**BEFORE** (lines 1151-1182):
```python
elif model_family == 'neural_network':
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    
    # Handle NaN values (neural networks can't handle them)
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)  # ⚠️ LEAK
    
    # Scale for neural networks
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)  # ⚠️ LEAK
    
    model = MLPRegressor(**nn_config, **extra)
    model.fit(X_scaled, y)
```

**AFTER**:
```python
elif model_family == 'neural_network':
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    
    # NOTE: This function is called INSIDE CV loop (line 2026)
    # Imputer/scaler should be fit on train, transform test
    # But this function receives X,y for a single fold already
    # So we need to check: is this called inside CV or outside?
    
    # If called OUTSIDE CV: Move to CV loop
    # If called INSIDE CV: This is correct (X is already train fold)
    
    # Handle NaN values (neural networks can't handle them)
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)  # OK if X is train fold only
    
    # Scale for neural networks
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)  # OK if X is train fold only
    
    model = MLPRegressor(**nn_config, **extra)
    model.fit(X_scaled, y)
```

**Action**: Need to verify where `extract_native_importance()` is called. If it's called OUTSIDE CV, we need to refactor.

**File**: `TRAINING/ranking/predictability/model_evaluation.py`

**Change 2.2**: Check line 2026 - ensure imputer/scaler are fit inside CV loop

**BEFORE** (if normalization happens before CV):
```python
# Somewhere before CV loop
X_scaled = scaler.fit_transform(X)  # ⚠️ LEAK

for train_idx, test_idx in tscv.split(X, y):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]  # Uses future stats
```

**AFTER**:
```python
# Inside CV loop
for train_idx, test_idx in tscv.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    
    # Fit on train only
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Transform test using train stats
```

**File**: `TRAINING/preprocessing/mega_script_sequential_preprocessor.py`

**Change 2.3**: This file may be used for preprocessing before training. If so, it should NOT fit on entire dataset.

**Action**: Check where this preprocessor is used. If it's called before train/test split, refactor to fit inside CV.

### Verification
1. Add logging: Log mean/std of features before and after scaling
2. Verify: Mean of scaled features should be ~0, std ~1 for **train set only**
3. Check: Test set mean/std should NOT be 0/1 (it's transformed using train stats)

---

## Issue #3: pct_change() May Include Current Bar

### Problem
Need to verify if Polars `pct_change(60)` includes current bar in calculation.

### Evidence
**File**: `DATA_PROCESSING/features/simple_features.py:280-284`
```python
pl.col("close").pct_change(60).alias("price_momentum_60d")
```

### Question
Does `pct_change(60)` calculate:
- `(close[t] - close[t-60]) / close[t-60]` ← Includes current bar (LEAK)
- `(close[t-1] - close[t-61]) / close[t-61]` ← Excludes current bar (CORRECT)

### Fix Required

**Action**: Test Polars `pct_change()` behavior

**Test Code**:
```python
import polars as pl

df = pl.DataFrame({
    "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
})

# Test pct_change(3)
result = df.with_columns([
    pl.col("close").pct_change(3).alias("pct_change_3")
])

print(result)
# Check: Does pct_change_3[3] use close[3] or close[2]?
```

**If pct_change includes current bar**:
```python
# BEFORE:
pl.col("close").pct_change(60).alias("price_momentum_60d")

# AFTER:
(pl.col("close").shift(1) / pl.col("close").shift(61) - 1).alias("price_momentum_60d")
```

**If pct_change excludes current bar**: No change needed.

### Verification
1. Run test code above
2. Verify: `pct_change_3[3]` should be `(close[3] - close[0]) / close[0]` or `(close[2] - close[-1]) / close[-1]`?
3. Document result

---

## Issue #4: Misnamed Features

### Problem
Features named `beta_20d` and `market_correlation_60d` are actually just rolling standard deviation of returns, not beta or correlation.

### Evidence
**File**: `DATA_PROCESSING/features/simple_features.py:603-608`
```python
# These are NOT beta/correlation - they're rolling std of returns!
pl.col("close").pct_change().rolling_std(20).alias("beta_20d")
pl.col("close").pct_change().rolling_std(60).alias("market_correlation_60d")
```

### Impact
- **Low**: Not a leak, just confusing naming
- Could mislead users about what the feature represents

### Fix Required

**Option A**: Rename to accurate names
```python
pl.col("close").pct_change().rolling_std(20).alias("volatility_20d_returns").cast(pl.Float32),
pl.col("close").pct_change().rolling_std(60).alias("volatility_60d_returns").cast(pl.Float32),
```

**Option B**: Implement actual beta/correlation (requires market data)
```python
# Would need market returns (e.g., SPY) to calculate actual beta
# beta = cov(stock_returns, market_returns) / var(market_returns)
```

**Recommendation**: Option A (rename) for now. Option B requires market data infrastructure.

### Verification
1. Update feature registry with new names
2. Update any configs that reference old names
3. Add migration script if needed

---

## Implementation Order

1. **Issue #2** (Normalization) - CRITICAL, easiest to fix
2. **Issue #1** (Rolling windows) - HIGH, straightforward
3. **Issue #3** (pct_change) - MEDIUM, needs testing first
4. **Issue #4** (Naming) - LOW, can be done later

## Testing Plan

### Unit Tests
1. Test rolling windows exclude current bar
2. Test normalization uses train stats only
3. Test pct_change behavior
4. Test feature alignment with targets

### Integration Tests
1. Run full pipeline with fixes
2. Compare scores before/after (should decrease if leaks were present)
3. Verify long-lookback features (beta_20d, etc.) can be re-enabled safely

### Validation
1. Scores should be lower (more realistic)
2. Long-lookback features should work without leaks
3. No performance degradation (just removing leaks)

---

## Rollback Plan

If fixes cause issues:
1. Git revert specific changes
2. Keep fixes isolated by issue
3. Can rollback normalization fix separately from rolling window fix

---

## Related Files

### Feature Calculation
- `DATA_PROCESSING/features/simple_features.py` - Main feature definitions
- `DATA_PROCESSING/features/comprehensive_builder.py` - Comprehensive features
- `DATA_PROCESSING/features/regime_features.py` - Regime features

### Normalization/Preprocessing
- `TRAINING/ranking/multi_model_feature_selection.py` - Feature selection normalization
- `TRAINING/preprocessing/mega_script_sequential_preprocessor.py` - Sequential preprocessing
- `TRAINING/preprocessing/mega_script_pipeline.py` - Pipeline preprocessing
- `TRAINING/ranking/predictability/model_evaluation.py` - Model evaluation normalization

### CV/Splitting
- `TRAINING/utils/purged_time_series_split.py` - CV splitter
- `TRAINING/ranking/shared_ranking_harness.py` - Shared harness with CV

---

## Notes

- **Don't remove long-lookback features yet** - Fix the bugs first, then re-enable them
- **Test incrementally** - Fix one issue at a time, test, then move to next
- **Document behavior** - Add comments explaining why shift(1) is needed
- **Update tests** - Add tests to prevent regression

---

## Questions to Resolve

1. Where is `extract_native_importance()` called? Inside or outside CV loop?
2. Where is `mega_script_sequential_preprocessor` used? Before or after train/test split?
3. Does Polars `pct_change()` include current bar? (Needs test)
4. Are there other places doing global normalization? (Search for `fit_transform` on full dataset)

---

**Next Steps**: 
1. Verify call sites for normalization functions
2. Create test for pct_change() behavior
3. Implement fixes in order of priority
4. Re-enable long-lookback features after fixes
