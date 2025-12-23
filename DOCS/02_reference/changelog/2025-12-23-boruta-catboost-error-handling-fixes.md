# Changelog: Boruta Timeout and CatBoost Pickle Error Fixes

**Date**: 2025-12-23
**Type**: Bug Fix
**Impact**: High - Prevents crashes during feature selection
**Breaking**: No - Backward compatible error handling improvements

## Summary

Fixed two critical errors in feature selection:
1. **Boruta timeout error handling**: Improved handling of Boruta timeouts to prevent confusing error messages and crashes
2. **CatBoost pickle error**: Fixed "Can't pickle local object" error when extracting CatBoost importance using multiprocessing

## Problems

### 1. Boruta Timeout Error Handling

**Issue**: When Boruta exceeded its time budget, the timeout handler raised a `TimeoutError`, but Boruta's internal code caught it and re-raised it as a `ValueError` with the message "Please check your X and y variable. The provided estimator cannot be fitted to your data." This made it appear as a data issue rather than a timeout.

**Evidence**:
```
TimeoutError: Boruta training exceeded 10 minute time budget
...
ValueError: Please check your X and y variable. The provided estimator cannot be fitted to your data.
Boruta training exceeded 10 minute time budget
```

**Impact**: 
- Confusing error messages that made debugging difficult
- Pipeline could crash instead of gracefully skipping Boruta when it times out

### 2. CatBoost Pickle Error

**Issue**: When extracting CatBoost importance using `PredictionValuesChange` (PVC), the code used a local nested function `compute_importance_worker` in a multiprocessing context. Local functions cannot be pickled, causing the error:

```
Can't pickle local object 'train_model_and_get_importance.<locals>.compute_importance_worker'
```

**Impact**:
- CatBoost importance extraction failed during feature selection
- Feature selection pipeline could not complete for CatBoost models

## Solutions

### 1. Boruta Timeout Error Handling

**File**: `TRAINING/ranking/predictability/model_evaluation.py`

**Changes**:
- Updated exception handler to catch both `TimeoutError` and `ValueError`
- Added detection logic to identify timeout errors even when wrapped as `ValueError`
- Improved error messages to clearly indicate timeout vs. actual data issues
- Added graceful degradation: when Boruta times out with no partial results, it's skipped with a warning instead of crashing

**Key changes**:
```python
except (TimeoutError, ValueError) as e:
    # Check if this is actually a timeout by examining the error message
    error_msg = str(e)
    is_timeout = (
        isinstance(e, TimeoutError) or
        "timeout" in error_msg.lower() or
        "time budget" in error_msg.lower() or
        "exceeded" in error_msg.lower() or
        ("Please check your X and y variable" in error_msg and 
         boruta_elapsed >= (boruta_max_time_seconds or 600))
    )
    
    if is_timeout:
        logger.warning(f"⚠️  Boruta skipped due to timeout/budget after {boruta_elapsed:.2f} seconds")
        # Skip gracefully instead of crashing
    else:
        # Re-raise actual data errors
        raise
```

### 2. CatBoost Pickle Error

**File**: `TRAINING/ranking/multi_model_feature_selection.py`

**Changes**:
- Moved `compute_importance_worker` from a local nested function to a module-level function `_compute_catboost_importance_worker`
- Module-level functions can be pickled for multiprocessing, fixing the pickle error

**Key changes**:
```python
# Module-level function for CatBoost importance computation (must be picklable for multiprocessing)
def _compute_catboost_importance_worker(model_data, X_data, feature_names_data, result_queue):
    """
    Worker process to compute CatBoost importance.
    
    This must be a module-level function (not nested) to be picklable for multiprocessing.
    """
    try:
        if hasattr(model_data, 'base_model'):
            importance_raw = model_data.base_model.get_feature_importance(data=X_data, type='PredictionValuesChange')
        else:
            importance_raw = model_data.get_feature_importance(data=X_data, type='PredictionValuesChange')
        result_queue.put(('success', pd.Series(importance_raw, index=feature_names_data)))
    except Exception as e:
        result_queue.put(('error', str(e)))

# Later in train_model_and_get_importance:
p = multiprocessing.Process(
    target=_compute_catboost_importance_worker,  # Use module-level function
    args=(model, X_train_fit, feature_names, result_queue)
)
```

## Impact

- **Boruta**: Timeout errors are now clearly identified and handled gracefully, preventing pipeline crashes
- **CatBoost**: Importance extraction now works correctly with multiprocessing, allowing feature selection to complete successfully
- **User Experience**: Clearer error messages make debugging easier
- **Pipeline Stability**: Feature selection can continue even when individual models fail or timeout

## Related

- Builds on previous Boruta optimizations in [2025-12-22-boruta-performance-optimizations.md](2025-12-22-boruta-performance-optimizations.md)
- Related to CatBoost improvements in [2025-12-22-catboost-formatting-typeerror-fix.md](2025-12-22-catboost-formatting-typeerror-fix.md)

