# Single Source of Truth - Complete Fix

## Problem Fixed

Split-brain issue where different code paths computed different lookback values:
- `resolved_config`: `max=1440m` (correctly finds `vwap_dev_low`, `market_impact`, etc.)
- Gatekeeper/POST_GATEKEEPER: `actual_max=150m` (missing 1440m features)
- Final enforcement: `actual_max=1440m > 240m` (hard-stop)

## Root Causes Identified

1. **Duplicate computation in `compute_feature_lookback_max()`**: Built canonical map, then recomputed lookbacks for `feature_lookbacks` list
2. **Sanitizer ignoring canonical map**: Called `compute_feature_lookback_max()` but then recomputed using `infer_lookback_minutes()` directly
3. **Unknown lookback = 0.0 (safe)**: Sanitizer treated unknown as `0.0` instead of `inf` (unsafe)
4. **No diagnostic logging**: Gatekeeper/sanitizer didn't log 1440m offenders, making split-brain invisible

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

**File**: `TRAINING/utils/leakage_budget.py` (lines 948-963)

Added diagnostic logging that shows ALL features exceeding cap for gatekeeper/sanitizer stages:
- Logs top 10 offenders with lookback values
- Helps verify gatekeeper/sanitizer see same offenders as final enforcement

**File**: `TRAINING/ranking/predictability/model_evaluation.py` (lines 506-530)

Added gatekeeper diagnostic that logs:
- `_Xd` features with their lookback values
- Top offenders exceeding cap (with lookback values)

### 4. Consistent Unknown Lookback Rule

**Rule**: Unknown lookback = `inf` (unsafe)
- **Strict mode**: Log error (unknown should have been dropped/quarantined)
- **Drop mode**: Drop unknown features
- **Max calculation**: Exclude `inf` from max (they should have been dropped), but log warning

**Implementation**:
- Missing from canonical map → `inf` (unsafe)
- Unknown features quarantined by sanitizer
- Unknown features dropped by gatekeeper (if `over_budget_action=drop`)
- Warning logged if unknown features exist in max calculation

### 5. Hard-Fail on Split-Brain Detection

**File**: `TRAINING/utils/leakage_budget.py` (lines 1198-1230)

Added invariant check: `budget.max_feature_lookback_minutes` MUST match `actual_max_uncapped` (both use same canonical map).

**File**: `TRAINING/ranking/predictability/model_evaluation.py` (lines 4706-4740)

POST_GATEKEEPER sanity check now:
- Hard-fails on mismatch (split-brain detection) in strict mode
- Hard-fails if `actual_max_from_features > cap` in strict mode
- Uses exact same oracle as final enforcement

## Single Source of Truth Architecture

### The Oracle: `compute_feature_lookback_max()`

**All stages must use this function**:
1. **Sanitizer**: Calls `compute_feature_lookback_max()` → extracts canonical map → uses it directly
2. **Gatekeeper**: Calls `compute_feature_lookback_max()` → extracts canonical map → uses it directly
3. **POST_GATEKEEPER sanity check**: Calls `compute_feature_lookback_max()` → validates against budget
4. **POST_PRUNE**: Calls `compute_feature_lookback_max()` → validates
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

## Verification

### Sanitizer Test
```
✅ Input: ['rsi_21', 'beta_60d', 'vwap_dev_low', 'market_impact']
✅ Quarantined: ['beta_60d', 'vwap_dev_low', 'market_impact'] (all > 240m)
✅ Safe: ['rsi_21'] (105m <= 240m)
```

### Consistency Test
```
✅ compute_feature_lookback_max max_minutes: 86400.0m (consistent across calls)
✅ Canonical map includes ALL features with correct lookbacks
```

## Expected Behavior After Fix

### Next Run Should Show

1. **Gatekeeper diagnostic shows 1440m features**:
   ```
   🔍 GATEKEEPER DIAGNOSTIC: 47 features exceed cap (240.0m): 
   macd_signal(1440m), vwap_dev_low(1440m), market_impact(1440m), ...
   ```

2. **Sanitizer quarantines 1440m features**:
   ```
   👻 ACTIVE SANITIZATION: Quarantined 47 feature(s) with lookback > 240.0m
   ```

3. **POST_GATEKEEPER sanity check passes with correct max**:
   ```
   ✅ POST_GATEKEEPER sanity check PASSED: actual_max_from_features=150.0m <= cap=240.0m
   ```
   (Because 1440m features were already quarantined/dropped)

4. **No late-stage CAP VIOLATION**: Gatekeeper/sanitizer already caught offenders

## Key Changes Summary

| Component | Change | Impact |
|-----------|--------|--------|
| `compute_feature_lookback_max()` | Use canonical map for `feature_lookbacks` (no recompute) | Eliminates split-brain |
| `feature_sanitizer.py` | Use canonical map directly (no recompute) | Sanitizer sees same lookbacks |
| `feature_sanitizer.py` | Unknown lookback = `inf` (not `0.0`) | Unknown treated as unsafe |
| `compute_budget()` | Diagnostic logging for gatekeeper/sanitizer | Makes split-brain visible |
| Gatekeeper | Diagnostic logging for top offenders | Verifies gatekeeper sees 1440m features |
| POST_GATEKEEPER | Hard-fail on mismatch + cap violation | Catches split-brain immediately |

## Files Modified

1. `TRAINING/utils/leakage_budget.py` - Eliminated duplicate computation, added diagnostic logging
2. `TRAINING/utils/feature_sanitizer.py` - Use canonical map, unknown = inf
3. `TRAINING/ranking/predictability/model_evaluation.py` - Hard-fail on sanity check mismatch, diagnostic logging

## Definition of Done ✅

- [x] Single function = source of truth (`compute_feature_lookback_max()`)
- [x] Unknown lookback = unsafe (`inf`, not `0.0`)
- [x] No duplicate computation (use canonical map directly)
- [x] Sanitizer uses canonical map (no recompute)
- [x] Diagnostic logging shows 1440m offenders in gatekeeper/sanitizer
- [x] Split-brain detection (hard-fail on mismatch)
- [x] POST_GATEKEEPER sanity check uses exact same oracle

## Next Run Verification Checklist

- [ ] Gatekeeper diagnostic shows 1440m features in offenders list
- [ ] Sanitizer quarantines 1440m features (count increases)
- [ ] POST_GATEKEEPER sanity check shows `actual_max=150m` (1440m features already gone)
- [ ] No late-stage CAP VIOLATION (offenders already dropped)
- [ ] Diagnostic logs match between gatekeeper and final enforcement
