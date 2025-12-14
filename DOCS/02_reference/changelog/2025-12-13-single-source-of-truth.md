# Single Source of Truth for Lookback Computation - 2025-12-13

**Related**: [Single Source of Truth Fix Summary](../../../../TRAINING/utils/SINGLE_SOURCE_OF_TRUTH_COMPLETE.md) | [POST_PRUNE Invariant Check](../../../../TRAINING/utils/POST_PRUNE_INVARIANT_CHECK.md) | [XD Pattern Fix](../../../../TRAINING/utils/XD_PATTERN_FIX_SUMMARY.md)

## Overview

Fixed critical split-brain issue where different code paths computed different lookback values for the same feature set, causing:
- Gatekeeper/sanitizer missing long-lookback features (1440m, 86400m)
- Contradictory `actual_max` values (150m vs 1440m vs 86400m) for the same fingerprint
- Late-stage CAP VIOLATION crashes after gatekeeper claimed safety
- CV splitter built on wrong reality (purge based on 150m but features needed 86405m)

## Root Causes

1. **Duplicate lookback computation**: `compute_feature_lookback_max()` built canonical map but then recomputed lookbacks for `feature_lookbacks` list
2. **Sanitizer ignoring canonical map**: Called `compute_feature_lookback_max()` but then recomputed using `infer_lookback_minutes()` directly
3. **Unknown lookback = 0.0 (safe)**: Sanitizer treated unknown as `0.0` instead of `inf` (unsafe)
4. **No diagnostic logging**: Gatekeeper/sanitizer didn't log 1440m offenders, making split-brain invisible
5. **_Xd pattern inference missing**: Day-suffix features (`_60d`, `_20d`, `_3d`) were treated as `0.0m` lookback instead of pattern-matched

## Fixes Applied

### 1. Eliminated Duplicate Computation in `compute_feature_lookback_max()`

**File**: `TRAINING/utils/leakage_budget.py` (lines 1162-1196)

**Before**: Recomputed lookbacks for `feature_lookbacks` using `infer_lookback_minutes()` again
**After**: Uses `canonical_lookback_map` directly (single source of truth)

```python
# OLD (split-brain):
for feat_name in feature_names:
    lookback = infer_lookback_minutes(...)  # Recompute!
    feature_lookbacks.append((feat_name, lookback))

# NEW (single source of truth):
for feat_name in feature_names:
    feat_key = _feat_key(feat_name)
    lookback = canonical_lookback_map.get(feat_key)  # Use canonical map!
    if lookback is None:
        lookback = float("inf")  # Missing = unsafe
    feature_lookbacks.append((feat_name, lookback))
```

### 2. Fixed Sanitizer to Use Canonical Map

**File**: `TRAINING/utils/feature_sanitizer.py` (lines 137-178)

**Before**: Called `compute_feature_lookback_max()` but then ignored canonical map and recomputed
**After**: Extracts canonical map from result and uses it directly

**Before**: Unknown lookback → `0.0` (safe)
**After**: Unknown lookback → `inf` (unsafe, quarantined)

### 3. Added Diagnostic Logging for Gatekeeper/Sanitizer

**File**: `TRAINING/utils/leakage_budget.py` (lines 1010-1032)

Added diagnostic logging that shows ALL features exceeding cap for gatekeeper/sanitizer stages:
- Logs top 10 offenders with lookback values
- Helps verify gatekeeper/sanitizer see same offenders as final enforcement

**File**: `TRAINING/ranking/predictability/model_evaluation.py` (lines 506-530)

Added gatekeeper diagnostic that logs:
- `_Xd` features with their lookback values
- Top offenders exceeding cap (with lookback values)

### 4. Fixed _Xd Pattern Inference

**File**: `TRAINING/utils/leakage_budget.py` (lines 353-394, 465-472)

**Before**: `_Xd` features (e.g., `price_momentum_60d`, `volume_momentum_20d`) were treated as `0.0m` lookback
**After**: Pattern-matched to `days * 1440` minutes (e.g., `60d → 86400m`, `20d → 28800m`, `3d → 4320m`)

**Key changes**:
- Registry `lag_bars=0` for `_Xd` features now falls through to pattern matching
- Pattern `r".*_(\d+)d$"` matches and computes `days * 1440.0`
- Debug logging shows when `_Xd` patterns are matched

### 5. Consistent Unknown Lookback Rule

**Rule**: Unknown lookback = `inf` (unsafe)
- **Strict mode**: Log error (unknown should have been dropped/quarantined)
- **Drop mode**: Drop unknown features
- **Max calculation**: Exclude `inf` from max (they should have been dropped), but log warning

**Implementation**:
- Missing from canonical map → `inf` (unsafe)
- Unknown features quarantined by sanitizer
- Unknown features dropped by gatekeeper (if `over_budget_action=drop`)
- Warning logged if unknown features exist in max calculation

### 6. Hard-Fail on Split-Brain Detection

**File**: `TRAINING/utils/leakage_budget.py` (lines 1198-1230)

Added invariant check: `budget.max_feature_lookback_minutes` MUST match `actual_max_uncapped` (both use same canonical map).

**File**: `TRAINING/ranking/predictability/model_evaluation.py` (lines 4706-4740)

POST_GATEKEEPER sanity check now:
- Hard-fails on mismatch (split-brain detection) in strict mode
- Hard-fails if `actual_max_from_features > cap` in strict mode
- Uses exact same oracle as final enforcement

### 7. POST_PRUNE Invariant Check

**File**: `TRAINING/ranking/predictability/model_evaluation.py` (lines 1117-1160)

Added hard-fail invariant check at POST_PRUNE:
- Verifies `max(canonical_map[features]) == computed_lookback` (within 1.0 minute tolerance)
- Prevents regression and ensures canonical map consistency
- Hard-fails in strict mode if violated

### 8. Readline Library Conflict Fix

**File**: `TRAINING/common/subprocess_utils.py` (lines 34-67)

**Problem**: Cursor AppImage's `LD_LIBRARY_PATH` caused subprocesses to load readline from wrong location, causing `sh: symbol lookup error: ... rl_print_keybinding`

**Fix**: Enhanced `get_safe_subprocess_env()` to filter `LD_LIBRARY_PATH`:
- Removes all AppImage mount paths (`/tmp/.mount_*`)
- Keeps system/conda paths
- If all paths are AppImage mounts, removes `LD_LIBRARY_PATH` entirely

## Single Source of Truth Architecture

### The Oracle: `compute_feature_lookback_max()`

**All stages must use this function**:
1. **Sanitizer**: Calls `compute_feature_lookback_max()` → extracts canonical map → uses it directly
2. **Gatekeeper**: Calls `compute_feature_lookback_max()` → extracts canonical map → uses it directly
3. **POST_GATEKEEPER sanity check**: Calls `compute_feature_lookback_max()` → validates against budget
4. **POST_PRUNE**: Calls `compute_feature_lookback_max()` → validates with invariant check
5. **Final enforcement**: Calls `compute_feature_lookback_max()` → hard-stops if violation

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

## Files Modified

1. `TRAINING/utils/leakage_budget.py` - Eliminated duplicate computation, added diagnostic logging, fixed _Xd inference
2. `TRAINING/utils/feature_sanitizer.py` - Use canonical map, unknown = inf
3. `TRAINING/ranking/predictability/model_evaluation.py` - Hard-fail on sanity check mismatch, diagnostic logging, POST_PRUNE invariant
4. `TRAINING/common/subprocess_utils.py` - Filter LD_LIBRARY_PATH to prevent readline conflicts

## Testing

### Unit Tests

**File**: `TRAINING/utils/test_xd_pattern_inference.py` (NEW)

Tests verify:
- `_Xd` pattern inference (e.g., `price_momentum_60d → 86400m`)
- Canonical map includes `_Xd` features
- Gatekeeper drops `_Xd` offenders correctly

### Integration Tests

Run with `lookback_budget_minutes: 240` and verify:
- Sanitizer quarantines 1440m features (count increases)
- Gatekeeper diagnostic shows 1440m offenders
- POST_GATEKEEPER sanity check shows `actual_max=150m` (1440m features already gone)
- No late-stage CAP VIOLATION (offenders already dropped)
- POST_PRUNE invariant check passes

### Expected Behavior

**Before fix**:
- `POST_GATEKEEPER: actual_max=150m` (missing 1440m features)
- `Final enforcement: actual_max=1440m > cap=240m` (CAP VIOLATION crash)

**After fix**:
- `feature_sanitizer DIAGNOSTIC: 126 features exceed cap (240.0m): macd_signal(1440m), ...`
- `ACTIVE SANITIZATION: Quarantined 126 feature(s) ...`
- `POST_GATEKEEPER: actual_max=150m <= cap=240m` (1440m features already quarantined)
- `POST_PRUNE: max(canonical_map[features])=150.0m == computed_lookback=150.0m ✓`
- No CAP VIOLATION (offenders already dropped)

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

## Breaking Changes

None. All changes are backward compatible.

## Migration Notes

No migration required. The fix is automatic and transparent.

## Related Documentation

- [Single Source of Truth Fix Summary](../../../../TRAINING/utils/SINGLE_SOURCE_OF_TRUTH_COMPLETE.md)
- [POST_PRUNE Invariant Check](../../../../TRAINING/utils/POST_PRUNE_INVARIANT_CHECK.md)
- [XD Pattern Fix Summary](../../../../TRAINING/utils/XD_PATTERN_FIX_SUMMARY.md)
- [Gatekeeper Lookback Fix](../../../../TRAINING/utils/GATEKEEPER_LOOKBACK_FIX.md)
- [Lookback Inference Consistency Fix](../../../../TRAINING/utils/LOOKBACK_INFERENCE_CONSISTENCY_FIX.md)
