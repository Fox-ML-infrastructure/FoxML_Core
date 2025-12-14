# Unknown Lookback Enforcement Fix

**Date**: 2025-12-13  
**Issue**: Unknown lookback (`inf`) features making it into POST_PRUNE despite being logged as errors

## Problem

Logs showed:
- `compute_budget(POST_PRUNE): 36 features have unknown lookback (inf)… These should have been dropped/quarantined.`
- Immediately after: `Budget computed: actual_max=150.0m`

This indicates:
1. Unknowns are being logged as errors but not enforced (not dropped/hard-stopped)
2. `actual_max` is computed only from finite values, so it doesn't reflect that unknowns exist
3. POST_PRUNE is calling `compute_feature_lookback_max()` directly, bypassing enforcement

## Root Cause

1. **POST_PRUNE bypasses enforcement**: Calls `compute_feature_lookback_max()` directly on full feature list (including unknowns) instead of using `apply_lookback_cap()`
2. **`compute_budget()` logs but doesn't enforce**: It logs errors about unknowns but doesn't hard-fail or drop them
3. **`actual_max` computed from finite only**: Unknowns excluded from max calculation, making cap check meaningless if unknowns survive

## Fixes Applied

### 1. POST_PRUNE Uses Enforcement ✅

**Location**: `TRAINING/ranking/predictability/model_evaluation.py` lines 984-1025

**Before**: Called `compute_feature_lookback_max()` directly on full feature list

**After**: Uses `apply_lookback_cap()` to enforce, then computes lookback from safe features only

```python
# Enforce cap (quarantines unknowns in strict mode, drops them in drop mode)
cap_result = apply_lookback_cap(
    features=feature_names,
    interval_minutes=data_interval_minutes,
    cap_minutes=effective_cap,
    policy=policy,
    stage="POST_PRUNE",
    registry=registry
)

# Update feature_names to safe features only (unknowns dropped)
feature_names = cap_result.safe_features
X = X[:, feature_indices]  # Slice X to match

# Now compute lookback from SAFE features only (no unknowns)
lookback_result = compute_feature_lookback_max(
    safe_features_post_prune, ...,
    canonical_lookback_map=canonical_map_from_post_prune
)
```

### 2. Enforcement Hard-Stops on Unknowns in Strict Mode ✅

**Location**: `TRAINING/utils/lookback_cap_enforcement.py` lines 254-265

**Before**: Unknowns quarantined but budget computed anyway

**After**: In strict mode, hard-stop BEFORE computing budget if unknowns exist

```python
# CRITICAL: If unknowns exist and policy is strict, hard-stop BEFORE computing budget
if unknown_features and policy == "strict":
    error_msg = (
        f"🚨 UNKNOWN LOOKBACK VIOLATION ({stage}): {len(unknown_features)} features have unknown lookback (inf). "
        f"In strict mode, unknown lookback is UNSAFE and must be dropped/quarantined."
    )
    raise RuntimeError(f"{error_msg} (policy: strict - training blocked)")
```

### 3. `compute_budget()` Hard-Fails on Unknowns ✅

**Location**: `TRAINING/utils/leakage_budget.py` lines 969-1000

**Before**: Logged errors but didn't hard-fail

**After**: Hard-fails in strict mode if unknowns reach `compute_budget()` (they shouldn't)

```python
if unknown_features:
    error_msg = (
        f"🚨 compute_budget({stage}): {len(unknown_features)} features have unknown lookback (inf). "
        f"This indicates a bug: compute_budget() was called on features that should have been quarantined."
    )
    if policy == "strict":
        raise RuntimeError(
            f"{error_msg} "
            f"(policy: strict - training blocked. Fix enforcement to quarantine unknowns before calling compute_budget)"
        )
```

### 4. Logging Shows Finite Max + Unknown Count ✅

**Location**: `TRAINING/utils/lookback_cap_enforcement.py` lines 286-290

**After**: Logs both finite max and unknown count for clarity

```python
if unknown_features:
    logger.info(
        f"   📊 {stage}: actual_max_finite={actual_max_lookback:.1f}m, "
        f"unknown_count={len(unknown_features)} (quarantined, not included in actual_max)"
    )
```

## Result

After these fixes:

✅ **Unknowns quarantined at POST_PRUNE**: Enforcement runs before lookback computation

✅ **Strict mode hard-stops**: Unknowns cause hard-stop in strict mode, not just logging

✅ **`compute_budget()` never sees unknowns**: Only called on safe features (unknowns already dropped)

✅ **Clear logging**: Shows both finite max and unknown count (if any survive)

## Testing

1. **Run with strict mode**:
   - POST_PRUNE should hard-stop if unknowns exist (or drop them in drop mode)
   - `compute_budget()` should never see unknowns (enforcement runs first)

2. **Check logs**:
   - Should see: `POST_PRUNE enforcement: N → M (quarantined=K)`
   - Should NOT see: `compute_budget(POST_PRUNE): X features have unknown lookback`

3. **Verify actual_max**:
   - Should be computed from safe features only (no unknowns)
   - If unknowns were quarantined, log shows both finite max and unknown count

## Related Files

- `TRAINING/ranking/predictability/model_evaluation.py`: POST_PRUNE enforcement
- `TRAINING/utils/lookback_cap_enforcement.py`: Hard-stop on unknowns in strict mode
- `TRAINING/utils/leakage_budget.py`: Hard-fail if unknowns reach compute_budget()
