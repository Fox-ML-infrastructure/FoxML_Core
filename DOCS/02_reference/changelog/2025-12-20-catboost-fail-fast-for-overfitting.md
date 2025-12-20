# CatBoost Fail-Fast for 100% Training Accuracy

**Date**: 2025-12-20  
**Type**: Bug Fix, Performance Improvement

## Summary

Added fail-fast mechanism for CatBoost when training accuracy reaches 100% (or >= 99.9% threshold). This prevents wasting 40+ minutes on expensive feature importance computation when the model is overfitting/memorizing training data.

## Problem

CatBoost was reaching 100% training accuracy (indicating overfitting/memorization) and had been running for 40+ minutes on a single thread in the feature selection phase. The warning was logged but expensive operations (feature importance computation) still ran, wasting significant time.

## Root Cause

After CatBoost fits and computes `train_score` at line 1740/1744, there was no check for suspiciously high training accuracy. The warning was logged later in `_check_for_perfect_correlation`, but by then expensive operations (like `get_feature_importance`, `extract_shap_importance`, `extract_permutation_importance`) had already started running.

**The issue:** When a tree-based model reaches 100% training accuracy, it's memorizing the data. Computing feature importance on such a model takes an extremely long time (40+ minutes) because the model has overfit to every detail.

## Solution

Add a fail-fast check immediately after `train_score` is computed:
1. Check if `train_score >= 0.999` (99.9% threshold, matches `_check_for_perfect_correlation` threshold)
2. If so, log a debug message and mark model as failed
3. Skip expensive operations (feature importance computation)
4. Return early with empty importance

## Implementation Details

**Before (buggy) code:**
```python
model.fit(X_catboost, y)
train_score = model.score(X_catboost, y)
# ... continues with expensive operations (extract_native_importance, etc.)
```

**After (fixed) code:**
```python
model.fit(X_catboost, y)
train_score = model.score(X_catboost, y)

# FAIL-FAST: Check for suspiciously high training accuracy (overfitting/memorization)
# Tree-based models can easily overfit to 100% training accuracy
# This indicates the model is memorizing data and will take a long time to compute importance
# Skip expensive operations (feature importance computation) to save time
if train_score >= 0.999:  # 99.9% threshold (matches _check_for_perfect_correlation threshold)
    logger.debug(f"    CatBoost: Training accuracy {train_score:.1%} >= 99.9% (overfitting detected, skipping expensive operations)")
    # Mark as failed and return early - skip expensive feature importance computation
    return None, pd.Series(0.0, index=feature_names), importance_method, train_score
```

**Note:** We return `train_score` in the tuple so it's still recorded, but we skip the expensive `get_feature_importance`, `extract_shap_importance`, and `extract_permutation_importance` calls.

## Files Changed

1. **TRAINING/ranking/multi_model_feature_selection.py** (lines 1746-1753)
   - Added fail-fast check after `train_score` is computed
   - Skip expensive operations if training accuracy >= 99.9%
   - Return early with empty importance

## Impact

- ✅ CatBoost fails fast when training accuracy >= 99.9%
- ✅ Skips expensive feature importance computation that can take 40+ minutes
- ✅ Process continues with other models instead of wasting time
- ✅ Consistent with Elastic Net fail-fast pattern
- ✅ Time savings: Prevents wasting 40+ minutes on overfitted models

## Testing

Verified that:
- CatBoost fails fast when train_score >= 99.9%
- Expensive operations are skipped
- Process continues with other models
- Model is marked as failed gracefully

