# CatBoost Verbosity Conflict and Feature Selection Reproducibility Fixes

**Date**: 2025-12-21  
**Type**: Bug Fix  
**Impact**: Medium - Fixes CatBoost training failures and reproducibility tracking warnings  
**Breaking**: No - Backward compatible

## Summary

Fixed two critical issues:
1. **CatBoost verbosity parameter conflict** causing training failures with `CatBoostError: Only one of parameters ['verbose', 'logging_level', 'verbose_eval', 'silent'] should be set`
2. **Missing n_features in feature selection reproducibility tracking** causing `Cannot create snapshot for FEATURE_SELECTION: Missing required fields for FEATURE_SELECTION: n_features` warnings

## Problem

### Issue 1: CatBoost Verbosity Parameter Conflict

CatBoost was failing during training with:
```
CatBoostError: Only one of parameters ['verbose', 'logging_level', 'verbose_eval', 'silent'] should be set
```

**Root Cause**:
- Line 1417 in `multi_model_feature_selection.py` already sets `verbose` from backend logging config
- Lines 1958-1960 attempted to add verbose logging for GPU verification but incorrectly set both `verbose=50` and `logging_level='Verbose'` simultaneously
- CatBoost does not allow multiple verbosity parameters to be set at the same time

**Impact**: CatBoost models failed to train, causing feature selection to fail for all symbols

### Issue 2: Missing n_features in Feature Selection Reproducibility Tracking

Feature selection reproducibility tracking was failing with:
```
⚠️  Diff telemetry failed (non-critical): Cannot create snapshot for FEATURE_SELECTION: Missing required fields for FEATURE_SELECTION: n_features
```

**Root Cause**:
- `diff_telemetry.py` requires `n_features` in `additional_data` for FEATURE_SELECTION stage validation (line 576-577)
- `feature_selector.py` adds `feature_names` to `additional_data_with_cohort` (line 2164) but does not add `n_features`
- When `log_comparison` is called (line 2192), the snapshot creation fails because `n_features` is missing

**Impact**: Reproducibility tracking warnings, potential issues with diff telemetry and trend analysis

## Solution

### Fix 1: CatBoost Verbosity Conflict

**File**: `TRAINING/ranking/multi_model_feature_selection.py` (lines 1958-1960)

**Changes**:
- Removed `logging_level='Verbose'` parameter (conflicts with `verbose`)
- Changed `verbose=50` to `verbose=1` (more reasonable level: 0=silent, 1=info, 2=debug)
- Added comment explaining why only `verbose` parameter is used

**Before**:
```python
if hasattr(model, 'base_model'):
    model.base_model.set_params(verbose=50, logging_level='Verbose')
elif hasattr(model, 'set_params'):
    model.set_params(verbose=50, logging_level='Verbose')
```

**After**:
```python
# Note: Only set 'verbose' parameter - 'logging_level' conflicts with it
if hasattr(model, 'base_model'):
    model.base_model.set_params(verbose=1)
elif hasattr(model, 'set_params'):
    model.set_params(verbose=1)
```

### Fix 2: Feature Selection Reproducibility

**File**: `TRAINING/ranking/feature_selector.py` (line 2165)

**Changes**:
- Added `n_features = len(selected_features)` to `additional_data_with_cohort`
- Added comment explaining it's required for diff_telemetry validation

**Before**:
```python
if selected_features:
    additional_data_with_cohort['feature_names'] = selected_features
```

**After**:
```python
if selected_features:
    additional_data_with_cohort['feature_names'] = selected_features
    additional_data_with_cohort['n_features'] = len(selected_features)  # Required for diff_telemetry validation
```

## Files Changed

1. `TRAINING/ranking/multi_model_feature_selection.py`
   - Lines 1957-1961: Fixed CatBoost verbosity parameter conflict

2. `TRAINING/ranking/feature_selector.py`
   - Line 2165: Added n_features to additional_data for reproducibility tracking

## Testing/Verification

### Expected Results

1. **CatBoost Training**:
   - CatBoost models should train successfully without parameter conflict errors
   - Verbose logging should work correctly (level 1 = info)
   - GPU verification logging should still function

2. **Feature Selection Reproducibility**:
   - Reproducibility tracking should complete without "Missing required fields" warnings
   - Diff telemetry snapshots should be created successfully
   - Trend analysis should work correctly with n_features data

### Verification Steps

1. Run feature selection with CatBoost enabled
2. Verify CatBoost training completes without errors
3. Check logs for reproducibility tracking - should not see "Missing required fields" warnings
4. Verify diff telemetry snapshots are created in `RESULTS/runs/.../targets/.../reproducibility/`

## Related Changes

- Preceded by: [2025-12-21-catboost-performance-diagnostics.md](2025-12-21-catboost-performance-diagnostics.md) - Performance diagnostics and iteration cap reduction
- Related to: [2025-12-21-catboost-early-stopping-fix.md](2025-12-21-catboost-early-stopping-fix.md) - Early stopping implementation

## Risk Assessment

- **Risk**: Low - Both fixes are simple parameter removals/additions that address clear errors
- **Impact**: 
  - CatBoost models will now train successfully instead of failing
  - Feature selection reproducibility tracking will work correctly without warnings
- **Rollback**: Simple git revert if needed (baseline committed at `8054461`)

