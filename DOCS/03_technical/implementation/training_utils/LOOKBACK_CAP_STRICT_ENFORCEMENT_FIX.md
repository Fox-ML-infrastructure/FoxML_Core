# Lookback Cap Strict Enforcement Fix

**Date**: 2025-12-13  
**Issue**: Strict lookback-cap enforcement failing after gatekeeper/pruning with contradictory results

## Problem

Logs showed an internal contradiction:
- **Gatekeeper** (cap=240m) on pruned set: `n_features=102 → safe=102 quarantined=0 … actual_max=150m` ✅
- **Strict check** immediately after: `actual_max=1440m > cap=240m … 37 features exceeding cap` ❌

Top offenders: `day_x_volume, negative_volume_index, roc_x_volatility, bb_x_vol, … cmf, n_trades, …`

## Root Causes Identified

### 1. Featureset Mis-Wire (Most Likely)
The feature list passed to the strict check differs from what gatekeeper validated. This can happen if:
- `X.columns` is used from a dataframe that wasn't re-sliced to the pruned list
- "candidate_features" is passed instead of "selected_features"
- Feature list is modified between POST_PRUNE and strict check

### 2. Lookback Oracle Mismatch (Also Plausible)
Unknown/suffixless features (e.g., `cmf`, `negative_volume_index`) get different lookback values:
- **Gatekeeper**: Uses canonical map with `unknown_policy="conservative"` → 1440m
- **Strict check**: Uses same canonical map, but unknown features should be `inf` (unsafe) in strict mode

## Fixes Implemented

### 1. Invariant Fingerprint Check ✅

**Location**: `TRAINING/ranking/predictability/model_evaluation.py` lines 1339-1362

Added invariant check right before strict check in `train_and_evaluate_models()`:
- Computes fingerprint of `feature_names` passed to strict check
- Compares against `POST_PRUNE` fingerprint
- **Hard-fails** with detailed error if mismatch detected
- Logs sample differences (added/removed features) for debugging

**Code**:
```python
# CRITICAL INVARIANT CHECK: Verify featureset matches POST_PRUNE (if it exists)
if 'post_prune_fp' in locals() and post_prune_fp is not None:
    if current_fp != post_prune_fp:
        logger.error(f"🚨 FEATURESET MIS-WIRE DETECTED: ...")
        raise RuntimeError(f"FEATURESET MIS-WIRE: ...")
```

### 2. Consistent Unknown Lookback Policy ✅

**Location**: `TRAINING/utils/leakage_budget.py` lines 1118-1155

Fixed canonical map builder to use consistent `unknown_policy`:
- **Strict mode**: `unknown_policy="drop"` → unknown features get `inf` (unsafe)
- **Other modes**: `unknown_policy="conservative"` → unknown features get 1440m (backward compatibility)

**Code**:
```python
# CRITICAL: Determine unknown_policy based on safety policy (for consistency)
unknown_policy_for_canonical = "conservative"  # Default
if safety_policy == "strict":
    unknown_policy_for_canonical = "drop"  # inf = unsafe, will be quarantined

lookback = infer_lookback_minutes(
    feat_name,
    interval_minutes,
    spec_lookback_minutes=spec_lookback,
    registry=registry,
    unknown_policy=unknown_policy_for_canonical,  # CRITICAL: Use consistent policy
    feature_time_meta=feat_meta
)
```

**Impact**:
- Unknown features (suffixless, no registry metadata) now get `inf` in strict mode
- Gatekeeper quarantines `inf` features (unknown = unsafe)
- Strict check sees same `inf` values (no 1440m vs inf mismatch)

### 3. Reduced Log Spam ✅

**Location**: `TRAINING/utils/lookback_cap_enforcement.py` lines 242-303

**Changes**:
- **One summary line** per stage at INFO: `n_features → safe quarantined cap actual_max`
- **Top 5 offenders** logged at INFO (with lookback values)
- **Full drop list** written to JSON artifact file (if `output_dir` provided)
- Artifact path logged once

**Artifact Format** (JSON):
```json
{
  "stage": "GATEKEEPER",
  "cap_minutes": 240.0,
  "interval_minutes": 5.0,
  "n_quarantined": 37,
  "n_safe": 102,
  "actual_max_lookback": 150.0,
  "quarantined_features": [
    {
      "feature_name": "cmf",
      "lookback_minutes": null,
      "reason": "unknown lookback (cannot infer - treated as unsafe)"
    },
    {
      "feature_name": "day_x_volume",
      "lookback_minutes": 1440.0,
      "reason": "lookback (1440.0m) > cap (240.0m)"
    }
  ]
}
```

**Log Output**:
```
📊 GATEKEEPER: n_features=139 → safe=102 quarantined=37 cap=240.0m actual_max=150.0m
   Top offenders: day_x_volume(1440m), negative_volume_index(1440m), roc_x_volatility(1440m), bb_x_vol(1440m), cmf(unknown)
   📄 Full drop list written to: /path/to/output/lookback_cap_quarantined_GATEKEEPER.json
```

### 4. POST_PRUNE Feature Names Storage ✅

**Location**: `TRAINING/ranking/predictability/model_evaluation.py` line 969

Stored `post_prune_feature_names` for invariant check comparison:
```python
post_prune_feature_names = feature_names.copy()  # Store for later comparison
```

## Definition of Done

After these fixes, for the same run:

✅ **Fingerprint invariant**: `POST_PRUNE` fingerprint == `MODEL_TRAIN_INPUT` fingerprint == fingerprint before strict check

✅ **Gatekeeper consistency**: Gatekeeper summary `actual_max <= cap` and strict check never finds additional offenders

✅ **Unknown feature handling**: Suffixless/unknown features either:
- Have registry lookback metadata, OR
- Are consistently quarantined at gatekeeper (unknown = inf = unsafe)

✅ **Log hygiene**: 1-3 lines per stage + top offenders, full lists only in artifacts

## Testing

To verify the fix:

1. **Run with strict mode** and check logs:
   - Gatekeeper should show `actual_max <= cap`
   - Strict check should never find additional offenders
   - Fingerprint invariant check should pass (or fail with clear error)

2. **Check artifact files**:
   - `lookback_cap_quarantined_GATEKEEPER.json` should contain all quarantined features
   - Lookback values should match what's logged

3. **Verify unknown features**:
   - Suffixless features (e.g., `cmf`) should have `lookback_minutes: null` in artifact
   - Should be quarantined at gatekeeper (unknown = unsafe)

## Rollback

If issues arise, revert these changes:
```bash
git checkout HEAD -- TRAINING/ranking/predictability/model_evaluation.py
git checkout HEAD -- TRAINING/utils/leakage_budget.py
git checkout HEAD -- TRAINING/utils/lookback_cap_enforcement.py
```

## Related Files

- `TRAINING/ranking/predictability/model_evaluation.py`: Invariant check, POST_PRUNE storage
- `TRAINING/utils/leakage_budget.py`: Unknown policy consistency
- `TRAINING/utils/lookback_cap_enforcement.py`: Log spam reduction, artifact writing
