# Lookback Fingerprint Tracking Fix

**Date**: 2025-12-13  
**Issue**: Lookback accounting using different feature sets at different steps, causing mismatches between reported max and actual max lookback.

## Problem

The system was computing lookback from different feature sets at different stages:

1. **Lookback computed from pre-filter feature set** → reported max=100.0m
2. **Top offenders list from different scope** → showing 60d features (86400m)
3. **Gatekeeper using different feature list** → missing features like `price_momentum_60d`

This caused:
- "Lookback mismatch: reported max=100.0m but actual max from features=86400.0m" warnings
- Gatekeeper not catching all violating features
- Inconsistent lookback values between audit, gatekeeper, and CV

## Root Cause

Multiple feature lists in flight:
- SAFE_CANDIDATES (300)
- After gatekeeper (285)
- After pruner (108; `fingerprint=c9d40b50`)

Lookback analyzer was sometimes using:
- An earlier list (pre-filter / pre-schema-add / pre-pruner)
- A cached list
- A list derived from registry/schema rather than runtime "MODEL_TRAIN_INPUT"

## Solution

### 1. Added Fingerprint Tracking to `leakage_budget.py`

- Added `_compute_feature_fingerprint()` function (consistent with `cross_sectional_data`)
- Modified `compute_budget()` to return `(LeakageBudget, fingerprint)` tuple
- Modified `compute_feature_lookback_max()` to return `(max_lookback, top_offenders, fingerprint)` tuple
- Added `expected_fingerprint` parameter for validation
- Added `stage` parameter for logging context

### 2. Fingerprint Validation at All Lookback Computations

- All `compute_budget()` and `compute_feature_lookback_max()` calls now:
  - Compute fingerprint from input feature list
  - Validate against expected fingerprint (if provided)
  - Log fingerprint with lookback computation
  - Error on mismatch

### 3. Invariant Checks

- **MODEL_TRAIN_INPUT fingerprint** computed in `train_and_evaluate_models()` (post gatekeeper)
- **Post-gatekeeper fingerprint** computed and validated
- **Pre-training validation** ensures features haven't changed between gatekeeper and training
- All lookback computations validate against MODEL_TRAIN_INPUT fingerprint

### 4. Updated All Call Sites

Updated all call sites to handle new return signatures:
- `TRAINING/ranking/predictability/model_evaluation.py` (multiple locations)
- `TRAINING/utils/resolved_config.py` (wrapper function)
- `TRAINING/ranking/shared_ranking_harness.py`
- `TRAINING/utils/feature_sanitizer.py`

## Changes Made

### Files Modified

1. **`TRAINING/utils/leakage_budget.py`**
   - Added fingerprint computation
   - Modified return signatures to include fingerprints
   - Added validation logic

2. **`TRAINING/ranking/predictability/model_evaluation.py`**
   - Added fingerprint tracking at MODEL_TRAIN_INPUT
   - Added fingerprint validation at all lookback computations
   - Added invariant checks between gatekeeper and training

3. **`TRAINING/utils/resolved_config.py`**
   - Updated wrapper to handle new return signature

4. **`TRAINING/ranking/shared_ranking_harness.py`**
   - Updated to handle new return signature

5. **`TRAINING/utils/feature_sanitizer.py`**
   - Updated to handle new return signature

## Expected Behavior

### Before Fix

```
⚠️  Lookback mismatch: reported max=100.0m but actual max from features=86400.0m
Top lookback features: ..._60d(86400m), ..._20d(28800m)
```

### After Fix

```
📊 MODEL_TRAIN_INPUT fingerprint=c9d40b50 (n_features=108)
📊 compute_budget(post_gatekeeper): max_lookback=100.0m, n_features=108, fingerprint=c9d40b50
✅ Fingerprint validated: all lookback computations use same feature set
```

## Validation

1. **Fingerprint consistency**: All lookback computations use same fingerprint as MODEL_TRAIN_INPUT
2. **Gatekeeper alignment**: Gatekeeper uses same feature→lookback mapping as analyzer
3. **Top offenders accuracy**: Top offenders list only includes features from current feature set
4. **Hard-stop on mismatch**: System errors if fingerprints don't match

## Related Issues

- Fixes "weird embargo/purge filtering" caused by different feature sets
- Fixes "Lookback mismatch" warnings
- Ensures gatekeeper catches all violating features (not just naming patterns)

## Testing

Run with canary test config to validate:
```bash
python -m TRAINING.orchestration.intelligent_trainer \
  --output-dir "fingerprint_validation_test" \
  --experiment-config leakage_canary_test
```

Expected: No fingerprint mismatch warnings, consistent lookback values throughout.
