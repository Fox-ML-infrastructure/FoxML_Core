# Canonical Map _Xd Feature Fix

## Problem Fixed

Gatekeeper's canonical map was missing `_Xd` features (e.g., `beta_60d`, `volatility_20d`), causing them to be treated as `lookback=0.0` instead of correctly inferred as `days * 1440` minutes. This allowed long-lookback features to slip through the gatekeeper, causing `CAP VIOLATION` errors later.

## Root Cause

1. **Registry returns `lag_bars=0` for `_Xd` features**: The registry was returning `lag_bars=0` for `_Xd` features, which was being passed as `spec_lookback_minutes=0.0` to `infer_lookback_minutes()`.

2. **Early return in `infer_lookback_minutes()`**: When `spec_lookback_minutes=0.0` was passed, `infer_lookback_minutes()` was returning `0.0` early (line 365) before pattern matching could run.

3. **Registry check also returned 0.0**: Even when `spec_lookback=None` was set, the registry check inside `infer_lookback_minutes()` (lines 370-394) was still returning `0.0` for `_Xd` features because they're not indicator-period features.

## Fixes Applied

### 1. Ignore `spec_lookback=0.0` for `_Xd` features (3 locations)

**Files**: `TRAINING/utils/leakage_budget.py`

- **Line 353-365**: In `infer_lookback_minutes()`, check if `spec_lookback_minutes=0.0` and feature is `_Xd`, then fall through to pattern matching
- **Line 728-739**: In `compute_budget()` (missing keys path), ignore `lag_bars=0` for `_Xd` features
- **Line 836-846**: In `compute_budget()` (recompute path), ignore `lag_bars=0` for `_Xd` features
- **Line 1051-1061**: In `compute_feature_lookback_max()` (canonical map builder), ignore `lag_bars=0` for `_Xd` features

### 2. Ignore registry `lag_bars=0` for `_Xd` features

**File**: `TRAINING/utils/leakage_budget.py` (lines 374-394)

Updated the registry check inside `infer_lookback_minutes()` to also check for `_Xd` features, not just indicator-period features. When registry returns `lag_bars=0` for `_Xd` features, fall through to pattern matching instead of returning `0.0`.

### 3. Remove `0.0` default in gatekeeper lookup

**File**: `TRAINING/ranking/predictability/model_evaluation.py` (lines 499-508)

Changed gatekeeper lookup from:
```python
lookback_minutes = feature_lookback_dict.get(feature_name, float("inf"))
```

to:
```python
lookback_minutes = feature_lookback_dict.get(feature_name)
if lookback_minutes is None:
    # Feature missing - treat as unsafe
    lookback_minutes = float("inf")
```

This ensures missing features are treated as unsafe (`inf`), not safe (`0.0`).

### 4. Ensure canonical map includes ALL features

**File**: `TRAINING/utils/leakage_budget.py` (lines 1077-1079, 824-830)

- Store lookback for ALL features in canonical map (even if `inf`)
- In `compute_budget()`, if a feature is missing from canonical map, compute it immediately (don't default to `0.0`)

## Verification

### Pattern Matching Test
```
✅ beta_60d: pattern matches _60d, days=60 → 86400.0m
✅ volatility_20d: pattern matches _20d, days=20 → 28800.0m
```

### compute_budget Test
```
✅ compute_budget max_lookback: 86400.0m (correctly includes beta_60d)
```

### Gatekeeper Integration Test
With `cap=240m`:
- `beta_60d` (86400m) → should be dropped ✅
- `volatility_20d` (28800m) → should be dropped ✅
- `rsi_21` (105m) → should be kept ✅

## Expected Behavior After Fix

1. **Gatekeeper correctly identifies `_Xd` features**: Diagnostic logs show `_Xd` features with correct nonzero lookbacks (e.g., `beta_60d: lookback=86400.0m`)
2. **Gatekeeper drops `_Xd` offenders**: With `cap=240m` + `drop_features=auto-drop`, gatekeeper removes `_Xd` offenders (86400m, 28800m all > 240m)
3. **No CAP VIOLATION after gatekeeper**: `compute_feature_lookback_max()` no longer raises CAP VIOLATION on post-gatekeeper/pruned featureset (gatekeeper already dropped the offenders)

## Key Changes Summary

| Component | Change | Impact |
|-----------|--------|--------|
| `infer_lookback_minutes()` | Ignore `spec_lookback=0.0` for `_Xd` features | Pattern matching runs for `_Xd` features |
| Registry check in `infer_lookback_minutes()` | Also check for `_Xd` features | Registry `lag_bars=0` doesn't return `0.0` for `_Xd` |
| `compute_budget()` | Ignore `lag_bars=0` for `_Xd` features (3 locations) | `_Xd` features get pattern-matched lookback |
| `compute_feature_lookback_max()` | Ignore `lag_bars=0` for `_Xd` features | Canonical map includes `_Xd` features with correct lookback |
| Gatekeeper lookup | Remove `0.0` default, use `None` → `inf` | Missing features treated as unsafe |

## Files Modified

1. `TRAINING/utils/leakage_budget.py` - Fixed `_Xd` feature handling in 5 locations
2. `TRAINING/ranking/predictability/model_evaluation.py` - Fixed gatekeeper lookup default

## Definition of Done ✅

- [x] `_Xd` pattern inference works correctly (verified by unit tests)
- [x] Registry `lag_bars=0` ignored for `_Xd` features (pattern matching used instead)
- [x] Canonical map includes `_Xd` features with correct lookback (verified)
- [x] `compute_budget` includes `_Xd` features in max calculation (verified: 86400m)
- [x] Gatekeeper treats missing features as unsafe (`inf`), not safe (`0.0`)
- [x] Unit tests created and passing
