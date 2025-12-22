# Changelog: CatBoost Formatting Error and CV Skip Fixes

**Date**: 2025-12-21  
**Type**: Bug Fixes, Performance Fix  
**Impact**: High - Fixes runtime errors and 3-hour training times  
**Breaking**: No - Backward compatible

## Summary

Fixed two critical CatBoost issues in feature selection:
1. **Format specifier error**: Fixed `train_val_gap` formatting causing `ValueError: Invalid format specifier`
2. **3-hour training times**: Always skip CV for CatBoost in feature selection to prevent excessive training times

## Problems

### Issue 1: CatBoost Format Specifier Error

**Error**: `ValueError: Invalid format specifier` when logging CatBoost scores

**Root Cause**:
- Line 2031 attempted to format `train_val_gap` with `.4f` in a conditional expression
- When `train_val_gap` is `None`, the expression returns `'N/A'` (a string), but Python tries to apply `.4f` format specifier to it
- Same issue as the previous `val_score` fix, but `train_val_gap` wasn't fixed

**Error Message**:
```
❌ MSFT: catboost FAILED: ValueError: Invalid format specifier
Traceback (most recent call last):
  File ".../multi_model_feature_selection.py", line 2030, in train_model_and_get_importance
    logger.info(f"    CatBoost: Scores - Train: {train_score:.4f}, Val: {val_score_str}, "
ValueError: Invalid format specifier
```

### Issue 2: 3-Hour Training Times for Single Symbol

**Problem**: CatBoost training taking **3 hours for a single symbol** with only 988 samples and 148 features

**Root Cause**:
- CV runs when `cv_n_jobs > 1` (line 1891)
- `cross_val_score()` doesn't support early stopping - runs full 300 iterations per fold
- With 2 folds × 300 iterations without early stopping = ~1.5 hours per fold = 3 hours total
- Previous fixes added early stopping to final fit and reduced iteration cap, but CV overhead remained

**Previous Fixes Context**:
- Early stopping was added to final fit (reduces final fit from 3 hours to <30 min) - [2025-12-21-catboost-early-stopping-fix.md](2025-12-21-catboost-early-stopping-fix.md)
- Iterations cap reduced from 2000 to 300 - [2025-12-21-catboost-performance-diagnostics.md](2025-12-21-catboost-performance-diagnostics.md)
- But CV still runs when `cv_n_jobs > 1`, causing the 3-hour issue
- The comment at line 1886-1889 says "Skip CV for feature selection" but implementation only skips if `cv_n_jobs <= 1`

## Solutions

### Fix 1: CatBoost Format Specifier Error

**File**: `TRAINING/ranking/multi_model_feature_selection.py`

**Line 2030**: Extract `train_val_gap` formatting to separate variable before using in f-string:

```python
# Before:
logger.info(f"    CatBoost: Scores - Train: {train_score:.4f}, Val: {val_score_str}, "
          f"Gap: {train_val_gap:.4f if train_val_gap is not None else 'N/A'} ...")

# After:
train_val_gap_str = f"{train_val_gap:.4f}" if train_val_gap is not None else 'N/A'
logger.info(f"    CatBoost: Scores - Train: {train_score:.4f}, Val: {val_score_str}, "
          f"Gap: {train_val_gap_str} ...")
```

**Lines Changed**: 2030

### Fix 2: Always Skip CV for CatBoost in Feature Selection

**File**: `TRAINING/ranking/multi_model_feature_selection.py`

**Lines 1890-1904**: Always skip CV for CatBoost in feature selection, regardless of `cv_n_jobs` setting:

```python
# Before:
cv_scores = None
if cv_n_jobs > 1:
    cv_start_time = time.time()
    try:
        logger.info(f"    CatBoost: Starting CV phase (n_splits={n_splits}, cv_n_jobs={cv_n_jobs})")
        logger.info(f"    CatBoost: NOTE: CV in feature selection can be slow (no early stopping per fold). Consider skipping CV for faster feature selection.")
        cv_scores = cross_val_score(model, X_catboost, y, cv=tscv, scoring=scoring, n_jobs=cv_n_jobs, error_score=np.nan)
        cv_elapsed = time.time() - cv_start_time
        logger.info(f"    CatBoost: CV phase completed in {cv_elapsed/60:.2f} minutes "
                  f"(scores: mean={np.nanmean(cv_scores):.4f}, std={np.nanstd(cv_scores):.4f})")
    except Exception as e:
        cv_elapsed = time.time() - cv_start_time if cv_start_time else None
        logger.debug(f"    CatBoost: CV failed after {cv_elapsed/60:.2f} minutes (continuing with single fit): {e}")
else:
    logger.debug(f"    CatBoost: Skipping CV (cv_n_jobs={cv_n_jobs} <= 1) - using single fit like target ranking")

# After:
# PERFORMANCE FIX: Always skip CV in feature selection to prevent 3-hour training times
# CV doesn't use early stopping per fold and runs full iterations, causing excessive overhead (2-3 hours)
# Feature selection only needs feature importance, not CV scores - early stopping in final fit is sufficient
# This matches the intent of the comment above and previous performance fixes (early stopping + iteration cap)
# Backward compatible: Users with cv_n_jobs <= 1 already skip CV, so no change for them
cv_scores = None
logger.info(f"    CatBoost: Skipping CV for feature selection (CV adds 2-3 hours overhead with no early stopping per fold)")
logger.info(f"    CatBoost: Using single fit with early stopping instead (matches target ranking approach)")
```

**Lines Changed**: 1890-1904 (replaced conditional CV logic with always-skip logic)

## Impact

### Before Fix 1
- CatBoost logging would fail with `ValueError: Invalid format specifier` when `train_val_gap` is `None`
- Training would fail even though model training succeeded

### After Fix 1
- CatBoost logging works correctly even when `train_val_gap` is `None`
- No more format specifier errors

### Before Fix 2
- With `cv_n_jobs > 1`: CV runs → 2 folds × 300 iterations = **~3 hours**
- With `cv_n_jobs <= 1`: CV skipped → single fit with early stopping = **~minutes**
- Inconsistent behavior based on config

### After Fix 2
- Always: CV skipped → single fit with early stopping = **~minutes** (consistent fast performance)
- **Backward compatible**: No change for users with `cv_n_jobs <= 1` (they already skip CV)

## Expected Performance

### Format Specifier Fix
- **Before**: Training fails with ValueError when `train_val_gap` is `None`
- **After**: Training completes successfully with proper logging

### CV Skip Fix
- **Before**: 3 hours for 988 samples (with CV enabled via `cv_n_jobs > 1`)
- **After**: <5 minutes for 988 samples (CV always skipped, early stopping enabled)
- **Speedup**: ~36x faster

## Verification

After the fixes:

1. CatBoost logging should work correctly even when `train_val_gap` is `None`
2. No more `ValueError: Invalid format specifier` errors
3. CatBoost feature selection should complete in minutes, not hours
4. Logs should show "Skipping CV for feature selection" message
5. Single fit with early stopping should complete quickly (typically 10-30 iterations)
6. Feature importance should still be computed correctly
7. Users with `cv_n_jobs <= 1` should see no change in behavior

## Testing

- Run feature selection with CatBoost on a single symbol
- Verify training completes in <5 minutes instead of 3 hours
- Check logs confirm CV is skipped
- Verify no format specifier errors appear
- Verify feature importance is still computed correctly
- Test with both `cv_n_jobs = 1` and `cv_n_jobs > 1` to ensure consistent behavior

## Related Issues

- Related to previous fix: [2025-12-21-catboost-logging-and-n-features-extraction-fixes.md](2025-12-21-catboost-logging-and-n-features-extraction-fixes.md) (also fixed format specifier for `val_score`)
- Builds on previous performance fixes: [2025-12-21-catboost-early-stopping-fix.md](2025-12-21-catboost-early-stopping-fix.md), [2025-12-21-catboost-performance-diagnostics.md](2025-12-21-catboost-performance-diagnostics.md)

## Files Changed

- `TRAINING/ranking/multi_model_feature_selection.py`
  - Line 2030: Fixed `train_val_gap` format specifier error
  - Lines 1890-1904: Always skip CV for CatBoost in feature selection

