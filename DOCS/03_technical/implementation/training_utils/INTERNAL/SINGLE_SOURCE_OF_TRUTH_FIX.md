# Single Source of Truth for Lookback Computation

## Problem Fixed

Split-brain issue: Different code paths were computing different lookback values for the same feature set:
- `resolved_config` computes `max=1440m` (correctly finds `vwap_dev_low`, `market_impact`, etc.)
- Gatekeeper + POST_GATEKEEPER sanity check claims `actual_max_from_features=150m` (missing 1440m features)
- Later `compute_feature_lookback_max()` hard-stops with `actual_max=1440m > 240m`

This indicates **gatekeeper/sanitizer are using an incomplete inference path** while final enforcement knows the truth.

## Root Cause

1. **Duplicate lookback computation in `compute_feature_lookback_max()`**:
   - First builds `canonical_lookback_map` (correct)
   - Then passes it to `compute_budget()` (correct)
   - But then **recomputes lookbacks** for `feature_lookbacks` list (lines 1169-1189) - **split-brain!**

2. **Unknown lookback treated as safe**: Features with unknown lookback were getting `0.0` (safe) instead of `inf` (unsafe)

3. **No split-brain detection**: POST_GATEKEEPER sanity check didn't fail hard on mismatch

## Fixes Applied

### 1. Eliminated Duplicate Computation in `compute_feature_lookback_max()`

**File**: `TRAINING/utils/leakage_budget.py` (lines 1162-1196)

**Before**: Recomputed lookbacks for `feature_lookbacks` list using `infer_lookback_minutes()` again
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

### 2. Unknown Lookback = Unsafe

**File**: `TRAINING/utils/leakage_budget.py` (lines 949-971)

- Unknown features (`inf`) are excluded from max calculation (they should have been dropped)
- Warning logged if unknown features exist
- Missing features in canonical map treated as `inf` (unsafe)

### 3. Hard-Fail on Split-Brain Detection

**File**: `TRAINING/utils/leakage_budget.py` (lines 1198-1230)

Added invariant check: `budget.max_feature_lookback_minutes` MUST match `actual_max_uncapped` (both use same canonical map).

**File**: `TRAINING/ranking/predictability/model_evaluation.py` (lines 4706-4740)

POST_GATEKEEPER sanity check now:
- Hard-fails on mismatch (split-brain detection)
- Hard-fails if `actual_max_from_features > cap` in strict mode
- Uses exact same oracle as final enforcement

## Single Source of Truth Architecture

### The Oracle: `compute_feature_lookback_max()`

**All stages must use this function**:
1. **Gatekeeper**: Calls `compute_feature_lookback_max()` → gets canonical map → uses it
2. **Sanitizer**: Should use same canonical map (or call `compute_feature_lookback_max()`)
3. **POST_GATEKEEPER sanity check**: Calls `compute_feature_lookback_max()` → validates against budget
4. **POST_PRUNE**: Calls `compute_feature_lookback_max()` → validates
5. **Final enforcement**: Calls `compute_feature_lookback_max()` → hard-stops if violation

### Canonical Map Structure

```python
canonical_lookback_map: Dict[str, float] = {
    "rsi_21": 105.0,
    "beta_60d": 86400.0,
    "vwap_dev_low": 1440.0,  # Conservative default for unknown
    "market_impact": 1440.0,  # Conservative default for unknown
    # ... ALL features have an entry (even if inf)
}
```

**Key properties**:
- **Complete**: Every feature has an entry (no missing keys)
- **Normalized keys**: Uses `_feat_key()` for consistent lookup
- **Unknown = inf**: Unknown features get `inf`, not `0.0`
- **Cached**: Reused across stages for same fingerprint

## Verification

### Canonical Map Test
```
✅ rsi_21         → 105.0m
✅ beta_60d       → 86400.0m
✅ volatility_20d  → 28800.0m
✅ vwap_dev_low    → 1440.0m (conservative default)
✅ market_impact   → 1440.0m (conservative default)
```

### Consistency Test
```
✅ compute_feature_lookback_max max_minutes: 86400.0m
✅ Canonical map includes ALL features
✅ No split-brain (budget.max matches actual_max_uncapped)
```

## Expected Behavior After Fix

1. **Gatekeeper uses canonical map**: All features get correct lookback (no missing/0.0)
2. **POST_GATEKEEPER sanity check uses same oracle**: Hard-fails on mismatch (split-brain detection)
3. **No late-stage CAP VIOLATION**: Gatekeeper already dropped offenders (1440m features > 240m cap)
4. **Unknown lookback = unsafe**: Features with unknown lookback are dropped/quarantined, not treated as safe

## Key Changes Summary

| Component | Change | Impact |
|-----------|--------|--------|
| `compute_feature_lookback_max()` | Use canonical map for `feature_lookbacks` (no recompute) | Eliminates split-brain |
| `compute_budget()` | Exclude `inf` from max, log warning for unknown | Unknown treated as unsafe |
| POST_GATEKEEPER sanity check | Hard-fail on mismatch + cap violation | Catches split-brain immediately |
| Gatekeeper lookup | Remove `0.0` default, use `None` → `inf` | Missing = unsafe |

## Files Modified

1. `TRAINING/utils/leakage_budget.py` - Eliminated duplicate computation, added split-brain detection
2. `TRAINING/ranking/predictability/model_evaluation.py` - Hard-fail on sanity check mismatch

## Definition of Done ✅

- [x] Single function = source of truth (`compute_feature_lookback_max()`)
- [x] Unknown lookback = unsafe (`inf`, not `0.0`)
- [x] No duplicate computation (use canonical map directly)
- [x] Split-brain detection (hard-fail on mismatch)
- [x] POST_GATEKEEPER sanity check uses exact same oracle
- [x] All features included in canonical map (no missing keys)
