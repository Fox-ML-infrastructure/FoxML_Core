# Type Boundary Wiring Progress

**Date**: 2025-12-13  
**Status**: Phase 2 in progress - Gatekeeper and POST_PRUNE wired

## Completed

### 1. Gatekeeper Uses EnforcedFeatureSet ✅

**Location**: `TRAINING/ranking/predictability/model_evaluation.py` lines 426-625

**Changes**:
- Converts `cap_result` to `EnforcedFeatureSet` via `.to_enforced_set()`
- Slices X immediately using `enforced.features` (no rediscovery)
- Updates `feature_names` to match `enforced.features` (the truth)
- Stores `enforced` set on `resolved_config._gatekeeper_enforced` for downstream access

**Key code**:
```python
# Convert to EnforcedFeatureSet (SST contract)
enforced = cap_result.to_enforced_set(stage="GATEKEEPER", cap_minutes=safe_lookback_max)

# Slice X immediately using enforced.features (no rediscovery)
feature_indices = [i for i, name in enumerate(feature_names) if name in enforced.features]
X = X[:, feature_indices]
feature_names = enforced.features.copy()  # Use enforced.features (the truth)
```

### 2. POST_PRUNE Uses EnforcedFeatureSet ✅

**Location**: `TRAINING/ranking/predictability/model_evaluation.py` lines 1016-1075

**Changes**:
- Converts `cap_result` to `EnforcedFeatureSet` via `.to_enforced_set()`
- Slices X immediately using `enforced_post_prune.features`
- Updates `feature_names` to match `enforced_post_prune.features`
- Stores `post_prune_enforced` for downstream use (invariant checks)

**Key code**:
```python
# Convert to EnforcedFeatureSet (SST contract)
enforced_post_prune = cap_result.to_enforced_set(stage="POST_PRUNE", cap_minutes=effective_cap)

# Slice X immediately using enforced.features (no rediscovery)
feature_indices = [i for i, f in enumerate(feature_names) if f in enforced_post_prune.features]
X = X[:, feature_indices]
feature_names = enforced_post_prune.features.copy()  # Use enforced.features (the truth)
```

### 3. Post-Prune Gatekeeper Validation ✅

**Location**: `TRAINING/ranking/predictability/model_evaluation.py` lines 860-880

**Changes**:
- Validates that `feature_names` matches `enforced.features` after post-prune gatekeeper
- Fixes `feature_names` if mismatch detected (uses `enforced.features` as truth)

## Remaining Work

### Phase 2 (Type Boundary Wiring)
- [ ] **FS_PRE/FS_POST**: Update feature selection to use `EnforcedFeatureSet`
  - Location: `TRAINING/ranking/feature_selector.py`
  - Should consume `EnforcedFeatureSet`, return `EnforcedFeatureSet`

### Phase 3 (Boundary Assertions)
- [ ] Add `assert_featureset_fingerprint()` at SAFE_CANDIDATES
- [ ] Add `assert_featureset_fingerprint()` at AFTER_LEAK_REMOVAL
- [ ] Add `assert_featureset_fingerprint()` at POST_GATEKEEPER
- [ ] Add `assert_featureset_fingerprint()` at POST_PRUNE
- [ ] Add `assert_featureset_fingerprint()` at MODEL_TRAIN_INPUT (already done, but can use helper)

## Benefits So Far

✅ **No rediscovery**: X sliced immediately after enforcement using `enforced.features`

✅ **Single source of truth**: `enforced.features` is the authoritative feature list

✅ **Order preserved**: X columns match `enforced.features` order exactly

✅ **Ready for assertions**: `EnforcedFeatureSet` available at boundaries for validation

## Next Steps

1. **Wire FS_PRE/FS_POST** (feature selection) to use `EnforcedFeatureSet`
2. **Add boundary assertions** using `assert_featureset_fingerprint()` helper
3. **Test end-to-end** to verify no split-brain
