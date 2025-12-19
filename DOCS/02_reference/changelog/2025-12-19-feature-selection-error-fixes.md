# 2025-12-19: Feature Selection Error Fixes

## Problem

Two errors were occurring during feature selection:

1. **cohort_metadata undefined error**: In `feature_selector.py`, `cohort_metadata`, `cohort_metrics`, and `cohort_additional_data` were extracted in one code path but used in the `ImportError` fallback path without proper guards. If extraction failed or was skipped, these variables were undefined when referenced, causing `NameError: local variable 'cohort_metadata' referenced before assignment`.

2. **Insufficient data span warning noise**: The error message for insufficient data span was correct but didn't clarify that fallback to per-symbol processing is expected behavior for long-horizon targets. This caused unnecessary WARNING-level log noise for expected scenarios.

## Root Cause

### Issue 1: cohort_metadata undefined
The cohort metadata extraction (lines 1752-1774) happened in one code path, but the `ImportError` fallback path (lines 1954-2114) attempted to use `cohort_metrics` and `cohort_additional_data` without checking if they were defined. If:
- The extraction failed with an exception
- The extraction was skipped due to missing context
- An exception occurred before extraction completed

Then these variables would be undefined when referenced in the fallback path.

### Issue 2: Log noise
The insufficient data span error was logged at WARNING level even though it's expected behavior for long-horizon targets (e.g., 5-day targets) with limited data. The error message also didn't clearly indicate that fallback is acceptable.

## Solution

### Fix 1: cohort_metadata initialization and guards
- **Initialize variables to safe defaults** at the start of the reproducibility tracking block:
  - `cohort_metadata = None`
  - `cohort_metrics = {}`
  - `cohort_additional_data = {}`
- **Wrap extraction in try-except** to handle failures gracefully
- **Add conditional checks** before using `cohort_metrics` and `cohort_additional_data` in the ImportError fallback path
- **Use conditional `.update()` calls** instead of unpacking with `**` to safely handle empty dicts

### Fix 2: Improved error messaging and log levels
- **Updated error message** in `shared_ranking_harness.py` to clarify that fallback to per-symbol processing is expected for long-horizon targets
- **Changed log level** from WARNING to INFO for insufficient data span errors in `feature_selector.py` since this is expected behavior
- **Added detection** for insufficient data span errors to log them at appropriate level

## Files Changed

- `TRAINING/ranking/feature_selector.py`:
  - Lines 1754-1757: Initialize cohort metadata variables to safe defaults
  - Lines 1760-1786: Wrap extraction in try-except with proper error handling
  - Lines 1982-1984: Add conditional check before using `cohort_metrics` in fallback path
  - Lines 2022-2024: Add conditional check before using `cohort_additional_data` in fallback path
  - Lines 996-997: Change log level to INFO for insufficient data span errors

- `TRAINING/ranking/shared_ranking_harness.py`:
  - Lines 492-497: Update error message to clarify fallback is expected behavior

## Impact

- **Eliminates undefined variable errors**: Feature selection will no longer crash with `NameError` when cohort metadata extraction fails
- **Reduces log noise**: Expected fallback scenarios are now logged at INFO level instead of WARNING
- **Clearer messaging**: Users understand that fallback to per-symbol processing is acceptable for long-horizon targets
- **Backward compatible**: All changes are defensive and don't change existing behavior when extraction succeeds

## Testing

- Feature selection should complete without errors for targets with insufficient data span
- Reproducibility tracking should work in both shared harness and fallback paths
- Error messages should be clear and informative
- No undefined variable errors should occur even when extraction fails

