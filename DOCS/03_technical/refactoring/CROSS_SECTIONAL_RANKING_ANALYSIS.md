# Cross-Sectional Ranking Disable Analysis

**Date**: 2025-12-09  
**Purpose**: Analyze why cross-sectional ranking might be disabled or skipped

## Overview

Cross-sectional ranking is an optional feature in the feature selection pipeline that trains a panel model across all symbols simultaneously to identify universe-level feature importance. It complements per-symbol selection by answering "Does this feature work across the universe?" rather than just "Does this feature work on AAPL?"

## When Cross-Sectional Ranking is Skipped

Based on `TRAINING/ranking/feature_selector.py` (lines 199-257), cross-sectional ranking is **skipped** (not run) in these cases:

### 1. Config Disabled

**Condition**: `aggregation.cross_sectional_ranking.enabled = false`

**Code Location**: `feature_selector.py:202-203, 256-257`

```python
if (cs_config.get('enabled', False) and 
    len(symbols) >= cs_config.get('min_symbols', 5)):
    # ... run CS ranking
else:
    # ... skip
    if not cs_config.get('enabled', False):
        logger.debug("Cross-sectional ranking disabled in config")
```

**Current Config**: `CONFIG/feature_selection/multi_model.yaml:233` shows `enabled: true`

**Why it might be disabled**:
- Performance: CS ranking adds ~30-60 seconds per target (trains panel model)
- Not enough value: With <5 symbols, CS ranking adds little value
- Debugging: Temporarily disabled to isolate per-symbol selection issues

### 2. Insufficient Symbols

**Condition**: `len(symbols) < min_symbols` (default: 5, config shows: 2 for testing)

**Code Location**: `feature_selector.py:203, 254-255`

```python
if len(symbols) < cs_config.get('min_symbols', 5):
    logger.debug(f"Skipping cross-sectional ranking: only {len(symbols)} symbols (min: {cs_config.get('min_symbols', 5)})")
```

**Current Config**: `CONFIG/feature_selection/multi_model.yaml:234` shows `min_symbols: 2` (lowered for testing)

**Why it's skipped**:
- **Statistical validity**: Panel models need sufficient cross-sectional size to be meaningful
- **Value proposition**: With 1-2 symbols, CS ranking = per-symbol ranking (no added value)
- **Recommended**: Use `min_symbols: 5` for production (config comment says "increase to 5+ for production")

### 3. Exception During Execution

**Condition**: Any exception during CS ranking execution

**Code Location**: `feature_selector.py:247-250`

```python
except Exception as e:
    logger.warning(f"Cross-sectional ranking failed: {e}", exc_info=True)
    summary_df['cs_importance_score'] = 0.0
    summary_df['feature_category'] = 'UNKNOWN'
```

**Common failure modes**:

#### 3a. Data Loading Failures
- **Location**: `cross_sectional_feature_ranker.py:237-240`
- **Cause**: `load_mtf_data_for_ranking()` returns empty dict
- **Result**: Returns zero importance, continues execution

#### 3b. Data Preparation Failures
- **Location**: `cross_sectional_feature_ranker.py:243-252`
- **Cause**: `prepare_cross_sectional_data_for_ranking()` returns `None` for X or y
- **Common reasons**:
  - Insufficient cross-sectional size per timestamp (`min_cs` not met)
  - All features filtered out by leakage detection
  - Target column missing or all NaN
- **Result**: Returns zero importance, continues execution

#### 3c. Model Training Failures
- **Location**: `cross_sectional_feature_ranker.py:266-274`
- **Cause**: `train_panel_model()` returns `None` for model
- **Common reasons**:
  - LightGBM/XGBoost import failures
  - Memory errors on large datasets
  - Invalid model config parameters
- **Result**: That model family skipped, continues with others

#### 3d. All Models Fail
- **Location**: `cross_sectional_feature_ranker.py:276-278`
- **Condition**: All model families fail to train
- **Result**: Returns zero importance, continues execution

## Current Configuration

**File**: `CONFIG/feature_selection/multi_model.yaml`

```yaml
aggregation:
  cross_sectional_ranking:
    enabled: true  # ✅ Currently enabled
    min_symbols: 2  # ⚠️ Lowered to 2 for testing (should be 5+ for production)
    top_k_candidates: 50
    model_families: [lightgbm]
    min_cs: 10
    max_cs_samples: 1000
    normalization: null
    symbol_threshold: 0.1
    cs_threshold: 0.1
```

## Why It Might Appear Disabled

### 1. Silent Failures

**Problem**: Exceptions are caught and logged as warnings, but execution continues. If you're not watching logs, you might not notice CS ranking failed.

**Solution**: Check logs for:
- `"Cross-sectional ranking failed: ..."`
- `"Failed to prepare cross-sectional data"`
- `"All panel models failed"`

### 2. Debug Logging

**Problem**: Skip messages use `logger.debug()`, which might not appear in production logs.

**Code**: `feature_selector.py:255, 257`

```python
logger.debug(f"Skipping cross-sectional ranking: only {len(symbols)} symbols...")
logger.debug("Cross-sectional ranking disabled in config")
```

**Solution**: Check debug logs or temporarily increase log level.

### 3. Config Not Loaded

**Problem**: If `aggregation_config` is empty or missing `cross_sectional_ranking` key, `cs_config.get('enabled', False)` defaults to `False`.

**Code**: `feature_selector.py:201`

```python
cs_config = aggregation_config.get('cross_sectional_ranking', {})
```

**Solution**: Verify config is loaded correctly:
- Check `multi_model_config.get('aggregation', {})` is not empty
- Verify config file path is correct
- Check for config overlay issues

### 4. Typed Config Override

**Problem**: If using new typed config (`FeatureSelectionConfig`), the aggregation config comes from `feature_selection_config.aggregation`, which might not have CS ranking settings.

**Code**: `feature_selector.py:117`

```python
if feature_selection_config is not None:
    aggregation_config = feature_selection_config.aggregation
```

**Solution**: Verify typed config includes CS ranking settings.

## Diagnostic Steps

### 1. Check Config

```python
from TRAINING.ranking.feature_selector import load_multi_model_config
config = load_multi_model_config()
cs_config = config.get('aggregation', {}).get('cross_sectional_ranking', {})
print(f"Enabled: {cs_config.get('enabled', False)}")
print(f"Min symbols: {cs_config.get('min_symbols', 5)}")
```

### 2. Check Logs

Search for:
- `"Cross-sectional ranking"` - Should see either "Computing..." or "Skipping..."
- `"Cross-sectional ranking failed"` - Indicates exception
- `"Cross-sectional ranking complete"` - Success indicator

### 3. Check Symbol Count

```python
print(f"Symbols: {len(symbols)}, Min required: {cs_config.get('min_symbols', 5)}")
if len(symbols) < cs_config.get('min_symbols', 5):
    print("⚠️ Not enough symbols for CS ranking")
```

### 4. Check Data Availability

```python
from TRAINING.utils.cross_sectional_data import load_mtf_data_for_ranking
mtf_data = load_mtf_data_for_ranking(data_dir, symbols)
print(f"Loaded data for {len(mtf_data)} symbols")
if not mtf_data:
    print("⚠️ No data loaded - CS ranking will fail")
```

## Recommendations

### For Production

1. **Set `min_symbols: 5`** (not 2) - CS ranking adds little value with <5 symbols
2. **Monitor logs** - Watch for CS ranking failures
3. **Use INFO logging** - Change debug messages to info for visibility
4. **Validate data** - Ensure sufficient cross-sectional size per timestamp

### For Debugging

1. **Temporarily enable verbose logging**:
   ```python
   import logging
   logging.getLogger('TRAINING.ranking.feature_selector').setLevel(logging.DEBUG)
   ```

2. **Check exception details**:
   ```python
   # The exception is logged with exc_info=True, so full traceback should be in logs
   ```

3. **Test CS ranking in isolation**:
   ```python
   from TRAINING.ranking.cross_sectional_feature_ranker import compute_cross_sectional_importance
   cs_importance = compute_cross_sectional_importance(
       candidate_features=['feature1', 'feature2'],
       target_column='target',
       symbols=['AAPL', 'MSFT', 'GOOGL'],
       data_dir=Path('data/')
   )
   ```

## Code Flow Summary

```
feature_selector.select_features_for_target()
  ↓
Load aggregation_config (from multi_model_config or typed config)
  ↓
Check: cs_config.get('enabled', False) AND len(symbols) >= min_symbols
  ↓
If TRUE:
  → Try: compute_cross_sectional_importance()
    → load_mtf_data_for_ranking() → If fails: return zero
    → prepare_cross_sectional_data_for_ranking() → If fails: return zero
    → train_panel_model() for each family → If all fail: return zero
    → Aggregate importances → Success
  → If exception: catch, log warning, set scores to 0.0
  ↓
If FALSE:
  → Set cs_importance_score = 0.0
  → Set feature_category = 'SYMBOL_ONLY'
  → Log debug message (might not appear)
```

## Conclusion

Cross-sectional ranking can be disabled/skipped for:
1. **Config**: `enabled: false`
2. **Symbols**: `len(symbols) < min_symbols`
3. **Failures**: Data loading, preparation, or model training failures (silently caught)

**Most likely causes**:
- Not enough symbols (< min_symbols)
- Data preparation failures (insufficient CS size, missing features)
- Silent exceptions (check logs for warnings)

**To diagnose**: Check logs for "Cross-sectional ranking" messages and verify config is loaded correctly.
