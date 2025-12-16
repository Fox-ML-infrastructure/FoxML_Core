# Lookback Inference Consistency Fix

## Critical Bugs Fixed

### 1. ✅ POST_PRUNE vs POST_PRUNE_policy_check Inconsistency (Fixed)

**Problem**: Same fingerprint, two different `actual_max` values:
- `POST_PRUNE`: `actual_max=86400.0m` (correctly finds 60d features)
- `POST_PRUNE_policy_check`: `actual_max=150.0m` (missing 60d features)

**Root Cause**: `POST_PRUNE_policy_check` was calling `compute_budget()` without the canonical map from `POST_PRUNE`, causing it to recompute and potentially miss features or use different inference logic.

**Fix**:
- Added `canonical_lookback_map` field to `LookbackResult` dataclass
- `POST_PRUNE_policy_check` now reuses the canonical map from `POST_PRUNE`
- Ensures both stages use the exact same lookback values

**Location**: 
- `TRAINING/utils/leakage_budget.py` lines 291, 946, 1221-1226
- `TRAINING/ranking/predictability/model_evaluation.py` lines 1095-1125

### 2. ✅ Hard-Stop on CAP VIOLATION (Fixed)

**Problem**: CAP VIOLATION was only a warning, allowing training to proceed with features exceeding the cap.

**Fix**:
- `compute_feature_lookback_max()` now raises `RuntimeError` in strict mode when `actual_max > cap`
- `POST_PRUNE` now raises `RuntimeError` in strict mode when lookback exceeds cap
- Includes top offenders in error message for debugging

**Location**:
- `TRAINING/utils/leakage_budget.py` lines 1115-1120
- `TRAINING/ranking/predictability/model_evaluation.py` lines 1028-1040

### 3. ✅ Re-Run Gatekeeper After Pruning (Fixed)

**Problem**: Pruning can surface long-lookback features that were previously masked by low-importance features. Gatekeeper only ran once, before pruning.

**Fix**:
- Gatekeeper now re-runs immediately after pruning (if features were pruned)
- Ensures long-lookback features surfaced by pruning are caught

**Location**: `TRAINING/ranking/predictability/model_evaluation.py` lines 836-840

### 4. ✅ Unified Lookback Inference (Verified)

**Problem**: Different stages might use different inference logic.

**Status**: All stages now use:
- `compute_feature_lookback_max()` → builds canonical map
- `compute_budget()` → uses canonical map (if provided) or recomputes using same logic
- Both use `infer_lookback_minutes()` with same parameters

**Pattern Matching Verified**: `_60d` and `_20d` patterns correctly match and return 86400m and 28800m respectively.

## Remaining Issues (Next Steps)

### 1. Sanitizer Missing Long-Lookback Features

**Problem**: Sanitizer only quarantines `volume_sma_50` but misses `price_momentum_60d`, `volume_*_20d`.

**Possible Causes**:
- Sanitizer runs before these features are added to the feature set
- Sanitizer uses different lookback inference (should use same as gatekeeper)
- Features are added after sanitizer runs

**Fix Needed**: Ensure sanitizer uses the same `infer_lookback_minutes()` logic and runs on the complete feature set.

### 2. Registry Metadata Issues

**Problem**: Registry returns `lag_bars=0` for indicator-period features (e.g., `stoch_d_21`).

**Status**: Already handled - code ignores `lag_bars=0` for indicator-period features and uses pattern matching instead.

**Enhancement Needed**: Fix registry upstream to return correct `lag_bars` for indicator-period features.

## Testing Checklist

- [ ] Run target ranking and verify POST_PRUNE and POST_PRUNE_policy_check show same `actual_max`
- [ ] Verify CAP VIOLATION raises RuntimeError in strict mode
- [ ] Verify gatekeeper re-runs after pruning and catches long-lookback features
- [ ] Verify sanitizer catches all long-lookback features (if they exist at sanitizer stage)
- [ ] Verify CV splitter purge is computed from final featureset (post-gatekeeper + post-prune)

## Expected Behavior After Fixes

1. **POST_PRUNE** computes canonical map → finds 86400m if 60d features exist
2. **POST_PRUNE_policy_check** reuses same canonical map → same 86400m result
3. **CAP VIOLATION** → hard-stop in strict mode (no training with violating features)
4. **Gatekeeper after pruning** → catches any long-lookback features surfaced by pruning
5. **CV splitter** → purge computed from final featureset (should match POST_PRUNE max)

If 60d features still exist after gatekeeper, the system will either:
- **State A**: Hard-stop (strict mode) - training blocked
- **State B**: Increase purge to ~86405m (warn mode) - but this is basically unusable

The system should no longer be in the impossible hybrid state: "features exceed cap" AND "budget max is 150m".
