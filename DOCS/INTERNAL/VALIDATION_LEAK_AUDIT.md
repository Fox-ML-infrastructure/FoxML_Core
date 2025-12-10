# Validation Split Leak Audit

**Date**: 2025-11-23
**Status**: **ALL CRITICAL LEAKS FIXED**

## Summary

Found and fixed **6 critical locations** using standard K-Fold or TimeSeriesSplit without purge gap. All production training and ranking scripts now use `PurgedTimeSeriesSplit` or manual purge logic to prevent temporal leakage in financial ML.

## Leak Locations

### FIXED (All Critical Leaks Resolved)

1. **`SCRIPTS/rank_target_predictability.py`** (line 565)
 - Now uses `PurgedTimeSeriesSplit` with horizon-based purge_overlap
 - Calculates purge from target horizon (e.g., 60m target → 17 bars purge)

2. **`SCRIPTS/rank_features_by_ic_and_predictive.py`** (line 731-733)
 - **FIXED**: Replaced `LogisticRegressionCV(cv=3)` and `LassoCV(cv=3)` with `PurgedTimeSeriesSplit`
 - Calculates purge_overlap from target horizon (extracted from target_column)
 - Uses horizon-based purge: `target_horizon_bars + 5` buffer

3. **`SCRIPTS/multi_model_feature_selection.py`** (line 549-551)
 - **FIXED**: Replaced `LogisticRegressionCV(cv=3)` and `LassoCV(cv=3)` with `PurgedTimeSeriesSplit`
 - Calculates purge_overlap from target horizon
 - Applied in `train_model_and_get_importance` function

4. **`TRAINING/unified_training_interface.py`** (line 185)
 - **FIXED**: Replaced `cross_val_score(..., cv=n_splits)` with `PurgedTimeSeriesSplit`
 - Uses conservative default purge (17 bars = 60m target)
 - Note: Could be enhanced to extract horizon from kwargs if available

5. **`TRAINING/model_fun/ensemble_trainer.py`** (line 163)
 - **FIXED**: Replaced `train_test_split(..., shuffle=False)` with manual chronological split + purge gap
 - Manually calculates split with purge_overlap (default: 17 bars)
 - Ensures no temporal leakage between train and validation sets

6. **`TRAINING/processing/cross_sectional.py`** (line 106)
 - **FIXED**: Enhanced `GroupKFold` splits with purge logic
 - Added `purge_overlap` parameter to `create_group_splits` (default: 17 groups)
 - Removes last `purge_overlap` timestamp groups from training set before validation

### ️ LOW PRIORITY (Diagnostic Tools - Less Critical)

7. **`SCRIPTS/diagnose_leakage.py`** (line 201)
 - ️ Uses `TimeSeriesSplit` (not purged)
 - **Problem**: Diagnostic tool, not production code
 - **Impact**: LOW - Only used for debugging
 - **Fix**: Optional - can update for consistency

## Visual Explanation

### Standard K-Fold (LEAKS):
```
Fold 1: [Train] [Test]  ← Train and Test touch (leak!)
Fold 2: [Train] [Test]  ← Train and Test touch (leak!)
```

### PurgedTimeSeriesSplit (SAFE):
```
Fold 1: [Train] [GAP] [Test]  ← Gap = target horizon (safe!)
Fold 2: [Train] [GAP] [Test]  ← Gap = target horizon (safe!)
```

## Implementation Details

### Purge Calculation
- **Default**: 17 bars = 60m target / 5m bars + 5 buffer
- **Horizon-based**: When target_column is available, extracts horizon and calculates: `target_horizon_bars + 5`
- **Fallback**: Uses conservative default (17 bars) when horizon cannot be extracted

### Files Modified
1. `SCRIPTS/utils/purged_time_series_split.py` - New PurgedTimeSeriesSplit class
2. `SCRIPTS/rank_target_predictability.py` - Already fixed (previous work)
3. `SCRIPTS/rank_features_by_ic_and_predictive.py` - Fixed stability_selection CV
4. `SCRIPTS/multi_model_feature_selection.py` - Fixed stability_selection CV
5. `TRAINING/unified_training_interface.py` - Fixed fallback CV
6. `TRAINING/model_fun/ensemble_trainer.py` - Fixed train/val split
7. `TRAINING/processing/cross_sectional.py` - Enhanced GroupKFold with purge

## Testing Recommendations

1. **Verify scores drop**: Previous scores were inflated due to leakage. Expect lower (but more realistic) scores.
2. **Check purge calculation**: Verify purge_overlap is calculated correctly for different target horizons.
3. **Test with different targets**: Ensure purge works for 15m, 30m, 60m, etc. targets.
4. **Monitor validation**: Watch for warnings about reduced purge due to small datasets.

## Notes

- All production training and ranking scripts now use purged validation
- Diagnostic tools (`diagnose_leakage.py`) still use standard TimeSeriesSplit (acceptable for debugging)
- Cross-sectional validation now includes purge logic for timestamp groups

