# Leakage Validation Fix - Separate Purge/Embargo Constraints

**Date**: 2025-12-13  
**Related**: [Fingerprint Tracking](2025-12-13-lookback-fingerprint-tracking.md) | [Fingerprint Improvements](2025-12-13-fingerprint-improvements.md) | [Leakage Controls Evaluation](../architecture/LEAKAGE_CONTROLS_EVALUATION.md) | [Canary Test Guide](../testing/LEAKAGE_CANARY_TEST_GUIDE.md)

**Issue**: Hard-stop validation was incorrectly treating `purge` as needing to cover both lookback AND horizon, when design intent is:
- `purge` covers feature lookback
- `embargo` covers target horizon

## Root Cause

The validation code was using `required_gap_minutes = max_lookback + horizon` and checking `purge_minutes >= required_gap_minutes`, which is semantically incorrect.

**Example of the bug**:
- `max_lookback = 100m`
- `horizon = 60m`
- `purge = 105m` (correctly bumped to cover lookback)
- `embargo = 85m` (correctly set to cover horizon)
- **Bug**: Validation checked `105m >= 160m` (lookback + horizon) → FAILED ❌
- **Correct**: Should check `105m >= 100m` (lookback) AND `85m >= 60m` (horizon) → PASS ✅

## Fix Applied

Changed validation to enforce **two separate constraints**:

1. **Purge constraint**: `purge_minutes >= max_feature_lookback_minutes + buffer`
2. **Embargo constraint**: `embargo_minutes >= horizon_minutes + buffer`

### Code Changes

**File**: `TRAINING/ranking/predictability/model_evaluation.py`

**Before**:
```python
required_gap = budget.required_gap_minutes  # max_lookback + horizon
if purge_minutes < required_gap:
    raise RuntimeError(f"purge_minutes ({purge_minutes:.1f}m) < required_gap_minutes ({required_gap:.1f}m)")
```

**After**:
```python
buffer_minutes = 5.0
purge_required = budget.max_feature_lookback_minutes + buffer_minutes
embargo_required = budget.horizon_minutes + buffer_minutes
purge_violation = purge_minutes < purge_required
embargo_violation = embargo_minutes < embargo_required

if purge_violation or embargo_violation:
    violations = []
    if purge_violation:
        violations.append(f"purge ({purge_minutes:.1f}m) < lookback_requirement ({purge_required:.1f}m)")
    if embargo_violation:
        violations.append(f"embargo ({embargo_minutes:.1f}m) < horizon_requirement ({embargo_required:.1f}m)")
    raise RuntimeError(f"🚨 LEAKAGE VIOLATION: {'; '.join(violations)}")
```

### Locations Fixed

1. **After Final Gatekeeper** (line ~3846) - Main validation point
2. **After Pruning** (line ~689) - Post-pruning validation
3. **After Pruning (resolved_config)** (line ~764) - Final feature set validation

## Additional Fix: Top Offenders List Consistency

**Issue**: Top offenders list could show features with lookback > max_lookback (e.g., showing 86400m when max is 100m), suggesting the list comes from a different feature set scope.

**Fix**: 
1. Filter top_offenders to only include features from the actual `feature_names` list (current feature set)
2. Add sanity check: `max(top_offenders) == reported_max_feature_lookback_minutes` (within 1 minute tolerance)
3. Verify all top features are in the current feature set before logging
4. Log feature set size to make scope explicit

**File**: `TRAINING/utils/leakage_budget.py`

```python
# SANITY CHECK: Verify max_lookback matches actual max from feature set
if feature_lookbacks and max_lookback is not None:
    actual_max = feature_lookbacks[0][1]
    if abs(actual_max - max_lookback) > 1.0:
        logger.warning(f"⚠️  Lookback mismatch: reported max={max_lookback:.1f}m but actual max={actual_max:.1f}m")
```

**File**: `TRAINING/ranking/predictability/model_evaluation.py`

- Added sanity check before logging top offenders
- Verify all top features are in current feature set
- Log feature set size: `"Top lookback features (from {len(feature_names)} features)"`

## Definition of Done ✅

- ✅ A target with `max_lookback=100m`, `horizon=60m`, `purge=105m`, `embargo=85m` does NOT hard-stop under `policy=strict`
- ✅ A target fails strict only when:
  - `purge < lookback_requirement` OR
  - `embargo < horizon_requirement`
- ✅ Error message explicitly says which requirement failed and by how much
- ✅ Top offenders list only shows features that contribute to max_lookback (respects cap)

## Testing

**Test Case 1**: Normal case (should PASS)
- `max_lookback = 100m`, `horizon = 60m`
- `purge = 105m` (100m + 5m buffer)
- `embargo = 85m` (60m + 25m buffer)
- **Expected**: ✅ PASS (both constraints satisfied)

**Test Case 2**: Purge violation (should FAIL)
- `max_lookback = 100m`, `horizon = 60m`
- `purge = 95m` (< 100m + buffer)
- `embargo = 85m`
- **Expected**: ❌ FAIL with message: "purge (95.0m) < lookback_requirement (105.0m)"

**Test Case 3**: Embargo violation (should FAIL)
- `max_lookback = 100m`, `horizon = 60m`
- `purge = 105m`
- `embargo = 55m` (< 60m + buffer)
- **Expected**: ❌ FAIL with message: "embargo (55.0m) < horizon_requirement (65.0m)"
