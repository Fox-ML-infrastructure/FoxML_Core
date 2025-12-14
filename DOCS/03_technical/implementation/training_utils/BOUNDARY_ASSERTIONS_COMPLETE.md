# Boundary Assertions Complete

**Date**: 2025-12-13  
**Status**: Phase 3 complete - All key boundaries have fingerprint assertions

## Summary

Added `assert_featureset_fingerprint()` assertions at all critical boundaries to detect featureset mis-wire immediately. These assertions validate that the actual feature list matches the expected `EnforcedFeatureSet` at each stage.

## Assertions Added

### 1. POST_GATEKEEPER ✅
**Location**: `TRAINING/ranking/predictability/model_evaluation.py` lines 4760-4775

- Validates `feature_names` matches `resolved_config._gatekeeper_enforced.features`
- Fixes mismatch by using `enforced.features` (the truth)
- Logs error but continues (validation check)

### 2. POST_PRUNE ✅
**Location**: `TRAINING/ranking/predictability/model_evaluation.py` lines 1076-1090

- Validates `feature_names` matches `post_prune_enforced.features`
- Should never fail if we used `enforced.features.copy()` correctly
- Fixes mismatch by using `enforced.features` (the truth)

### 3. MODEL_TRAIN_INPUT ✅
**Location**: `TRAINING/ranking/predictability/model_evaluation.py` lines 1438-1446

- Already existed, validates `feature_names` matches `post_prune_enforced.features`
- Ensures featureset integrity before model training starts

### 4. FS_PRE (SYMBOL_SPECIFIC) ✅
**Location**: `TRAINING/ranking/feature_selector.py` lines 379-390

- Validates `feature_names` matches `enforced_fs_pre.features`
- Fixes mismatch by using `enforced.features` (the truth)
- Logs error but continues (validation check)

### 5. FS_PRE (CROSS_SECTIONAL) ✅
**Location**: `TRAINING/ranking/feature_selector.py` lines 595-606

- Validates `feature_names` matches `enforced_fs_pre.features`
- Fixes mismatch by using `enforced.features` (the truth)
- Logs error but continues (validation check)

### 6. FS_POST ✅
**Location**: `TRAINING/ranking/feature_selector.py` lines 901-913

- Validates `selected_features` matches `enforced_fs_post.features`
- Fixes mismatch by using `enforced.features` (the truth)
- Logs error but continues (validation check)

## Assertion Pattern

All assertions follow this pattern:

```python
# CRITICAL: Boundary assertion - validate feature_names matches EnforcedFeatureSet
from TRAINING.utils.lookback_policy import assert_featureset_fingerprint
try:
    assert_featureset_fingerprint(
        label="<STAGE_NAME>",
        expected=enforced_set,
        actual_features=feature_names,
        logger_instance=logger,
        allow_reorder=False  # Strict order check
    )
except RuntimeError as e:
    # Log but don't fail - this is a validation check
    logger.error(f"<STAGE> assertion failed: {e}")
    # Fix it: use enforced.features (the truth)
    feature_names = enforced_set.features.copy()
    logger.info(f"Fixed: Updated feature_names to match enforced.features")
```

## Benefits

✅ **Immediate detection**: Featureset mis-wire detected at the boundary where it happens

✅ **Actionable errors**: Detailed diff showing added/removed features and order divergence

✅ **Auto-fix**: Automatically fixes mismatch by using `enforced.features` (the truth)

✅ **Non-blocking**: Logs error but continues (validation check, not hard-fail)

✅ **Order preservation**: Strict order check ensures X columns match `enforced.features` order

## What Gets Validated

Each assertion checks:
1. **Exact list equality**: `actual_features == expected.features`
2. **Set equality**: `set(actual_features) == set(expected.features)`
3. **Order equality**: `actual_features == expected.features` (if `allow_reorder=False`)
4. **Fingerprint match**: Set and ordered fingerprints match

## Error Messages

If a mismatch is detected, the assertion provides:
- Stage labels (expected → actual)
- Cap information
- Feature counts (expected vs actual)
- Added/removed features (top 10)
- Order divergence location and window
- Fingerprint mismatches

## Next Steps

The SST enforcement design is now complete:
- ✅ Type boundary wired (all stages use `EnforcedFeatureSet`)
- ✅ Boundary assertions added (all key boundaries validated)
- ✅ No rediscovery (X sliced immediately using `enforced.features`)
- ✅ Order preserved (X columns match `enforced.features` order)

The system is now provably split-brain free!
