# Single Source of Truth for Lookback Computation - Fix Summary

**Date**: 2025-12-13  
**Related**: [Single Source of Truth Changelog](../../02_reference/changelog/2025-12-13-single-source-of-truth.md)

## Problem Statement

The system had a critical split-brain issue where different code paths computed different lookback values for the same feature set:

- **Gatekeeper/sanitizer**: Missing long-lookback features (1440m, 86400m), reporting `actual_max=150m`
- **POST_PRUNE**: Reporting `actual_max=86400m` (correctly finding 60d features)
- **POST_PRUNE_policy_check**: Reporting `actual_max=150m` (missing 60d features)
- **Final enforcement**: Hard-stopping with `actual_max=1440m > cap=240m`

This caused:
- Long-lookback features slipping through gatekeeper/sanitizer
- Contradictory `actual_max` values for the same fingerprint
- Late-stage CAP VIOLATION crashes after gatekeeper claimed safety
- CV splitter built on wrong reality (purge based on 150m but features needed 86405m)

## Root Causes

1. **Duplicate lookback computation**: `compute_feature_lookback_max()` built canonical map but then recomputed lookbacks for `feature_lookbacks` list
2. **Sanitizer ignoring canonical map**: Called `compute_feature_lookback_max()` but then recomputed using `infer_lookback_minutes()` directly
3. **Unknown lookback = 0.0 (safe)**: Sanitizer treated unknown as `0.0` instead of `inf` (unsafe)
4. **No diagnostic logging**: Gatekeeper/sanitizer didn't log 1440m offenders, making split-brain invisible
5. **_Xd pattern inference missing**: Day-suffix features (`_60d`, `_20d`, `_3d`) were treated as `0.0m` lookback instead of pattern-matched

## Solution Architecture

### Single Source of Truth: `compute_feature_lookback_max()`

All stages must use this function and extract the canonical map:

```python
# All stages follow this pattern:
lookback_result = compute_feature_lookback_max(
    feature_names,
    interval_minutes,
    max_lookback_cap_minutes=cap,
    stage="STAGE_NAME"
)

# Extract canonical map (the truth)
canonical_map = lookback_result.canonical_lookback_map

# Use canonical map directly (NO recomputation)
for feat_name in feature_names:
    feat_key = _feat_key(feat_name)
    lookback = canonical_map.get(feat_key)
    if lookback is None:
        lookback = float("inf")  # Missing = unsafe
```

### Canonical Map Flow

```
compute_feature_lookback_max()
  ↓
Build canonical_lookback_map (ALL features, even if inf)
  ↓
Pass to compute_budget() → returns LeakageBudget
  ↓
Return LookbackResult with canonical_lookback_map
  ↓
All stages extract canonical_lookback_map and use it directly
  ↓
No recomputation = no split-brain
```

## Key Fixes

### 1. Eliminated Duplicate Computation

**File**: `TRAINING/utils/leakage_budget.py` (lines 1162-1196)

- **Before**: Recomputed lookbacks for `feature_lookbacks` using `infer_lookback_minutes()` again
- **After**: Uses `canonical_lookback_map` directly (single source of truth)

### 2. Fixed Sanitizer

**File**: `TRAINING/utils/feature_sanitizer.py` (lines 137-178)

- **Before**: Called `compute_feature_lookback_max()` but then ignored canonical map and recomputed
- **After**: Extracts canonical map from result and uses it directly
- **Before**: Unknown lookback → `0.0` (safe)
- **After**: Unknown lookback → `inf` (unsafe, quarantined)

### 3. Added Diagnostic Logging

**File**: `TRAINING/utils/leakage_budget.py` (lines 1010-1032)

- Logs ALL features exceeding cap for gatekeeper/sanitizer stages
- Shows top 10 offenders with lookback values
- Makes split-brain visible immediately

### 4. Fixed _Xd Pattern Inference

**File**: `TRAINING/utils/leakage_budget.py` (lines 353-394, 465-472)

- **Before**: `_Xd` features treated as `0.0m` lookback
- **After**: Pattern-matched to `days * 1440` minutes (e.g., `60d → 86400m`)

### 5. POST_PRUNE Invariant Check

**File**: `TRAINING/ranking/predictability/model_evaluation.py` (lines 1117-1160)

- Hard-fail check: `max(canonical_map[features]) == computed_lookback` (within 1.0 minute tolerance)
- Prevents regression and ensures canonical map consistency
- Hard-fails in strict mode if violated

### 6. Readline Library Conflict Fix

**File**: `TRAINING/common/subprocess_utils.py` (lines 34-67)

- Filters `LD_LIBRARY_PATH` to remove AppImage mount paths
- Prevents subprocess readline conflicts

## Testing

### Unit Tests

**File**: `TRAINING/utils/test_xd_pattern_inference.py` (NEW)

Tests verify:
- `_Xd` pattern inference (e.g., `price_momentum_60d → 86400m`)
- Canonical map includes `_Xd` features
- Gatekeeper drops `_Xd` offenders correctly

### Integration Tests

Run with `lookback_budget_minutes: 240` and verify:

1. **Sanitizer quarantines 1440m features**:
   ```
   feature_sanitizer DIAGNOSTIC: 126 features exceed cap (240.0m): macd_signal(1440m), ...
   ACTIVE SANITIZATION: Quarantined 126 feature(s) ...
   ```

2. **Gatekeeper diagnostic shows offenders**:
   ```
   🔍 GATEKEEPER DIAGNOSTIC: 47 features exceed cap (240.0m): macd_signal(1440m), ...
   ```

3. **POST_GATEKEEPER sanity check passes**:
   ```
   ✅ POST_GATEKEEPER sanity check PASSED: actual_max_from_features=150.0m <= cap=240.0m
   ```
   (Because 1440m features were already quarantined)

4. **POST_PRUNE invariant check passes**:
   ```
   ✅ INVARIANT CHECK (POST_PRUNE): max(canonical_map[features])=150.0m == computed_lookback=150.0m ✓
   ```

5. **No late-stage CAP VIOLATION**: Offenders already dropped

## Expected Behavior

### Before Fix
- `POST_GATEKEEPER: actual_max=150m` (missing 1440m features)
- `Final enforcement: actual_max=1440m > cap=240m` (CAP VIOLATION crash)

### After Fix
- `feature_sanitizer DIAGNOSTIC: 126 features exceed cap (240.0m): macd_signal(1440m), ...`
- `ACTIVE SANITIZATION: Quarantined 126 feature(s) ...`
- `POST_GATEKEEPER: actual_max=150m <= cap=240m` (1440m features already quarantined)
- `POST_PRUNE: max(canonical_map[features])=150.0m == computed_lookback=150.0m ✓`
- No CAP VIOLATION (offenders already dropped)

## Files Modified

1. `TRAINING/utils/leakage_budget.py` - Eliminated duplicate computation, added diagnostic logging, fixed _Xd inference
2. `TRAINING/utils/feature_sanitizer.py` - Use canonical map, unknown = inf
3. `TRAINING/ranking/predictability/model_evaluation.py` - Hard-fail on sanity check mismatch, diagnostic logging, POST_PRUNE invariant
4. `TRAINING/common/subprocess_utils.py` - Filter LD_LIBRARY_PATH to prevent readline conflicts

## Verification Checklist

- [x] Single function = source of truth (`compute_feature_lookback_max()`)
- [x] Unknown lookback = unsafe (`inf`, not `0.0`)
- [x] No duplicate computation (use canonical map directly)
- [x] Sanitizer uses canonical map (no recompute)
- [x] Diagnostic logging shows 1440m offenders in gatekeeper/sanitizer
- [x] Split-brain detection (hard-fail on mismatch)
- [x] POST_GATEKEEPER sanity check uses exact same oracle
- [x] POST_PRUNE invariant check prevents regression
- [x] _Xd pattern inference works correctly
- [x] Readline library conflict fixed

## Related Documentation

- [Single Source of Truth Changelog](../../02_reference/changelog/2025-12-13-single-source-of-truth.md)
