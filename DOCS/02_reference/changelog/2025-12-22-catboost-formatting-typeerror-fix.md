# Changelog: CatBoost Formatting TypeError Fix

**Date**: 2025-12-22  
**Type**: Bug Fix  
**Impact**: High - Prevents runtime errors in CatBoost logging  
**Breaking**: No - Backward compatible

## Summary

Fixed `TypeError: unsupported format string passed to NoneType.__format__` when `cv_mean` or `val_score` is `None` in CatBoost overfitting check logging. This error occurred when attempting to format `None` values directly within f-string conditional expressions.

## Problem

**Error**: `TypeError: unsupported format string passed to NoneType.__format__`

**Root Cause**:
- Line 2189-2190 in `multi_model_feature_selection.py` attempted to format `cv_mean` and `val_score` within f-string conditional expressions
- When these values are `None`, Python's format specifier (`.4f`) cannot be applied to `NoneType`
- The conditional expression `f"{cv_mean:.4f}" if cv_mean is not None else 'N/A'` was being evaluated incorrectly, causing the TypeError

**Error Message**:
```
TypeError: unsupported format string passed to NoneType.__format__
```

## Solution

**File**: `TRAINING/ranking/multi_model_feature_selection.py`

**Lines 2189-2190**: Pre-format `cv_mean` and `val_score` into strings before using them in the f-string:

```python
# Before (PROBLEMATIC):
logger.info(f"    CatBoost: Overfitting check - train={train_score:.4f}, cv={cv_mean:.4f if cv_mean is not None else 'N/A'}, "
          f"val={val_score:.4f if 'val_score' in locals() and val_score is not None else 'N/A'}, ...")

# After (FIXED):
cv_mean_str = f"{cv_mean:.4f}" if cv_mean is not None else 'N/A'
val_score_str = f"{val_score:.4f}" if 'val_score' in locals() and val_score is not None else 'N/A'
logger.info(f"    CatBoost: Overfitting check - train={train_score:.4f}, cv={cv_mean_str}, "
          f"val={val_score_str}, "
          f"decision={'SKIP' if should_skip else 'RUN'}, reason={skip_reason}")
```

**Key Changes**:
1. Extract `cv_mean` formatting to separate variable `cv_mean_str` before f-string
2. Extract `val_score` formatting to separate variable `val_score_str` before f-string
3. Use pre-formatted strings in the f-string to avoid format specifier errors

## Impact

### Before
- CatBoost overfitting check logging would crash with `TypeError` when `cv_mean` or `val_score` is `None`
- Training pipeline would fail at logging stage even if model training succeeded

### After
- CatBoost logging works correctly even when `cv_mean` or `val_score` is `None`
- Displays 'N/A' for missing values instead of crashing
- Training pipeline completes successfully regardless of CV or validation score availability

## Related

- Similar fix was done for `val_score` in previous changelog: `2025-12-21-catboost-logging-and-n-features-extraction-fixes.md`
- Part of ongoing CatBoost logging improvements and error handling

## Files Changed

- `TRAINING/ranking/multi_model_feature_selection.py` (2 lines changed: 2189-2190)

