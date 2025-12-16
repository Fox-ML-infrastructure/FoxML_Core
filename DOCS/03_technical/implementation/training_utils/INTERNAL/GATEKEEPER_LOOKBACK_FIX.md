# Gatekeeper Lookback Inference Fix

## Critical Bug Fixed

### Problem
Gatekeeper was using a **different lookback inference path** than `compute_feature_lookback_max`, causing it to miss `_60d/_20d/_3d` features that the audit system correctly identified.

**Symptoms**:
- Gatekeeper: "✅ After post-prune gatekeeper: 107 features remaining" (dropped zero)
- Immediately after: `compute_feature_lookback_max` throws `CAP VIOLATION: actual_max=86400m ... offenders: price_momentum_60d, volume_*_20d`

**Root Cause**:
1. Gatekeeper called `infer_lookback_minutes()` directly for each feature
2. `compute_feature_lookback_max()` uses a canonical map built once
3. Different code paths = different results for `_60d` features

### Fix

**Changed gatekeeper to use the EXACT SAME function as audit system**:
- Gatekeeper now calls `compute_feature_lookback_max()` to build canonical map
- Uses canonical map for all lookback lookups (same as audit)
- Added diagnostic logging for `_Xd` suffix features

**Location**: `TRAINING/ranking/predictability/model_evaluation.py` lines 403-484

## Additional Fixes

### 1. Unknown Lookback = UNSAFE (Fixed)

**Problem**: Unknown lookback was treated as safe (0.0), allowing features to slip through.

**Fix**:
- Unknown lookback (`inf`) now treated as UNSAFE
- In strict mode: raises `RuntimeError` immediately
- In drop mode: feature is dropped
- In warn mode: feature is warned but kept

**Location**: `TRAINING/ranking/predictability/model_evaluation.py` lines 458-471, 499-508

### 2. Strict Mode Exception Handling (Fixed)

**Problem**: Strict mode exceptions were being swallowed in pruning try/except.

**Fix**:
- Pruning exception handler now re-raises `RuntimeError` if it contains "policy: strict" or "training blocked"
- Only non-critical exceptions are swallowed

**Location**: `TRAINING/ranking/predictability/model_evaluation.py` lines 1292-1295

### 3. Diagnostic Logging (Added)

**Added diagnostic logging**:
- Count of features with `_Xd` suffix pattern
- Sample `_Xd` features
- Lookback values for `_Xd` features

**Location**: `TRAINING/ranking/predictability/model_evaluation.py` lines 407-412, 473-478

## Expected Behavior After Fix

1. **Gatekeeper uses same canonical map as audit** → sees same lookbacks
2. **`_60d` features correctly identified** → gatekeeper drops them if they exceed cap
3. **Unknown lookback = unsafe** → features with unknown lookback are dropped/blocked
4. **Strict mode violations abort** → no training with violating features
5. **Diagnostic logs show `_Xd` features** → easy to verify gatekeeper is evaluating them

## Testing Checklist

- [ ] Run target ranking and verify gatekeeper diagnostic shows `_Xd` features
- [ ] Verify gatekeeper drops `_60d` features if they exceed cap
- [ ] Verify `CAP VIOLATION` no longer appears after gatekeeper (gatekeeper should have caught them)
- [ ] Verify strict mode violations abort training (no "using all features" after strict error)
- [ ] Verify unknown lookback features are dropped/blocked

## Key Changes

1. **Gatekeeper lookback computation**:
   - **Before**: `infer_lookback_minutes()` per feature
   - **After**: `compute_feature_lookback_max()` → canonical map → lookup

2. **Unknown lookback handling**:
   - **Before**: `inf → 0.0` (assumed safe)
   - **After**: `inf → unsafe` (dropped/blocked)

3. **Strict mode**:
   - **Before**: Exceptions swallowed, training continues
   - **After**: Exceptions re-raised, training aborts
