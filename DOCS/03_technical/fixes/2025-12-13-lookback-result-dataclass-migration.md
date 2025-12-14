# LookbackResult Dataclass Migration Fix

**Date**: 2025-12-13  
**Issue**: `AttributeError: 'tuple' object has no attribute 'max_minutes'`

## Problem

The `compute_feature_lookback_max()` wrapper function in `resolved_config.py` was returning a tuple `(max_minutes, top_offenders)` for backward compatibility, but the code was updated to expect a `LookbackResult` dataclass with attributes like `.max_minutes`, `.top_offenders`, `.fingerprint`.

## Root Cause

During the fingerprint tracking improvements, `leakage_budget.compute_feature_lookback_max()` was changed to return a `LookbackResult` dataclass, but the wrapper function in `resolved_config.py` was still converting it to a tuple for backward compatibility. This caused a mismatch when code tried to access `.max_minutes` on what it expected to be a dataclass.

## Solution

### 1. Updated Wrapper to Return Dataclass

Changed `resolved_config.compute_feature_lookback_max()` to return the `LookbackResult` dataclass directly instead of converting to a tuple.

**Before**:
```python
return result.max_minutes, result.top_offenders  # Tuple
```

**After**:
```python
return result  # LookbackResult dataclass
```

### 2. Added Backward Compatibility Handling

Added checks in call sites to handle both dataclass and tuple returns (for any remaining legacy code):

```python
if hasattr(lookback_result, 'max_minutes'):
    computed_lookback = lookback_result.max_minutes
    top_offenders = lookback_result.top_offenders
    lookback_fingerprint = lookback_result.fingerprint
else:
    # Tuple return (backward compatibility)
    computed_lookback, top_offenders = lookback_result
    lookback_fingerprint = None
```

### 3. Fixed All Call Sites

Updated all call sites to handle the dataclass return:
- `TRAINING/ranking/predictability/model_evaluation.py` (multiple locations)
- `TRAINING/utils/resolved_config.py` (internal usage)
- `TRAINING/ranking/shared_ranking_harness.py` (already using dataclass)
- `TRAINING/utils/feature_sanitizer.py` (already using dataclass)

## Files Modified

1. **`TRAINING/utils/resolved_config.py`**
   - Changed return type from tuple to `LookbackResult` dataclass
   - Updated internal usage to handle dataclass
   - Updated docstring

2. **`TRAINING/ranking/predictability/model_evaluation.py`**
   - Added backward compatibility checks
   - Fixed fingerprint access to use local variable
   - Updated all lookback_result attribute accesses

## Validation

All code now:
- Returns `LookbackResult` dataclass from wrapper
- Handles both dataclass and tuple returns (backward compatibility)
- Accesses attributes safely with `hasattr()` checks
- Logs fingerprints correctly

## Related Issues

- Fixes `AttributeError: 'tuple' object has no attribute 'max_minutes'`
- Ensures consistent return type across all lookback computation functions
- Maintains backward compatibility for any legacy code
