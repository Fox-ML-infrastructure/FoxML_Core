# Changelog: CatBoost Logging and n_features Extraction Fixes

**Date**: 2025-12-21  
**Type**: Bug Fixes  
**Impact**: Medium - Fixes runtime errors in feature selection and diff telemetry

## Summary

Fixed two critical bugs:
1. **CatBoost logging ValueError**: Fixed format specifier error when `val_score` is not available
2. **Missing n_features for FEATURE_SELECTION**: Fixed extraction logic to check nested `evaluation` dict where `n_features` is actually stored

## Changes

### 1. CatBoost Logging Fix

**File**: `TRAINING/ranking/multi_model_feature_selection.py`

**Problem**: 
- Line 2029 attempted to format `'N/A'` string with `.4f` format specifier
- Caused `ValueError: Invalid format specifier` when `val_score` was not in locals()
- Error: `❌ NVDA: catboost FAILED: ValueError: Invalid format specifier`

**Solution**:
- Conditionally format `val_score` before using it in the logging statement
- Extract formatting to a separate variable to avoid format specifier on non-numeric value

**Code Change**:
```python
# Before:
logger.info(f"    CatBoost: Scores - Train: {train_score:.4f}, Val: {val_score:.4f if 'val_score' in locals() else 'N/A'}, ...")

# After:
val_score_str = f"{val_score:.4f}" if 'val_score' in locals() else 'N/A'
logger.info(f"    CatBoost: Scores - Train: {train_score:.4f}, Val: {val_score_str}, ...")
```

**Lines Changed**: 2029

### 2. n_features Extraction Fix for FEATURE_SELECTION

**File**: `TRAINING/orchestration/utils/diff_telemetry.py`

**Problem**:
- `_build_resolved_context()` only checked `additional_data.get('n_features')` (flat path)
- But `n_features` is stored in `full_metadata['evaluation']['n_features']` (nested path)
- When `full_metadata` is passed as `resolved_metadata`, the extraction logic didn't check the nested `evaluation` dict
- Caused validation failure: `Cannot create snapshot for FEATURE_SELECTION: Missing required fields for FEATURE_SELECTION: n_features`

**Root Cause Analysis**:
1. `feature_selector.py` line 2165: `n_features` added to `additional_data['n_features']`
2. `_save_to_cohort()` line 1279: `n_features` stored in `full_metadata['evaluation']['n_features']` when `feature_names` exists
3. Line 1491: `full_metadata` passed as `resolved_metadata` to `telemetry.finalize_run()`
4. `_build_resolved_context()` prioritizes `resolved_metadata` but only checked flat paths
5. Extraction failed because nested `evaluation` dict wasn't checked

**Solution**:
- Updated extraction logic to check multiple sources in priority order:
  1. `resolved_metadata['evaluation']['n_features']` (nested - where it's actually stored)
  2. `resolved_metadata['n_features']` (top-level fallback)
  3. `metadata['evaluation']['n_features']` (from filesystem, nested)
  4. `metadata['n_features']` (from filesystem, top-level)
  5. `additional_data['n_features']` (direct pass-through)
- Also checks `n_features_selected` as alternative key name in all locations

**Code Change**:
```python
# Before:
ctx.n_features = (
    additional_data.get('n_features') if additional_data else None
)

# After:
# Check multiple sources in priority order:
# 1. resolved_metadata['evaluation']['n_features'] (nested - where it's actually stored)
# 2. resolved_metadata['n_features'] (top-level fallback)
# 3. metadata['evaluation']['n_features'] (from filesystem, nested)
# 4. metadata['n_features'] (from filesystem, top-level)
# 5. additional_data['n_features'] (direct pass-through)
# Also check n_features_selected as alternative key name
ctx.n_features = (
    (resolved_metadata.get('evaluation', {}).get('n_features') if resolved_metadata else None) or
    (resolved_metadata.get('n_features') if resolved_metadata else None) or
    (resolved_metadata.get('evaluation', {}).get('n_features_selected') if resolved_metadata else None) or
    (resolved_metadata.get('n_features_selected') if resolved_metadata else None) or
    metadata.get('evaluation', {}).get('n_features') or
    metadata.get('n_features') or
    metadata.get('evaluation', {}).get('n_features_selected') or
    metadata.get('n_features_selected') or
    (additional_data.get('n_features') if additional_data else None) or
    (additional_data.get('n_features_selected') if additional_data else None)
)
```

**Lines Changed**: 477-496

## Impact

### Before
- CatBoost training would fail with `ValueError: Invalid format specifier` when validation score not available
- FEATURE_SELECTION diff telemetry would fail with missing `n_features` field
- Warning: `⚠️ Diff telemetry failed for FEATURE_SELECTION:y_will_swing_high_60m_0.05: Cannot create snapshot for FEATURE_SELECTION: Missing required fields for FEATURE_SELECTION: n_features`

### After
- CatBoost logging works correctly even when `val_score` is not available
- FEATURE_SELECTION snapshots successfully extract `n_features` from nested `evaluation` dict
- No more missing field validation errors for FEATURE_SELECTION stage

## Testing

- CatBoost training should complete without ValueError when validation score is missing
- FEATURE_SELECTION stage should successfully create snapshots without missing field errors
- Diff telemetry should work correctly for FEATURE_SELECTION runs

## Related Issues

- Related to previous fix: `2025-12-21-catboost-verbosity-and-reproducibility-fixes.md` (also addressed n_features but didn't check nested paths)
- Part of ongoing diff telemetry validation improvements

## Files Changed

- `TRAINING/ranking/multi_model_feature_selection.py` (1 line changed)
- `TRAINING/orchestration/utils/diff_telemetry.py` (18 lines changed)

