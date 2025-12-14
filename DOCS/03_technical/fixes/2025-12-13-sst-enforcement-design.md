# SST Enforcement Design Implementation Fix

**Date**: 2025-12-13  
**Status**: Complete  
**Related**: [SST Enforcement Design Changelog](../../02_reference/changelog/2025-12-13-sst-enforcement-design.md) | [Single Source of Truth Fix](2025-12-13-single-source-of-truth-complete.md)

## Overview

Implemented a comprehensive Single Source of Truth (SST) enforcement design that eliminates split-brain across all training paths. The design uses `EnforcedFeatureSet` as the authoritative contract, immediate X slicing to prevent rediscovery, and boundary assertions to detect mis-wire immediately.

## Problem Statement

Previous implementation had split-brain risks:
- Different code paths could derive feature lists from different sources (X.columns, feature_names, importance keys)
- X matrix columns could drift from authoritative feature list
- No validation that featureset matches expected state at boundaries
- Order drift could occur during DataFrame operations

## Solution

### 1. EnforcedFeatureSet Contract

**File**: `TRAINING/utils/lookback_cap_enforcement.py`

New dataclass that represents the authoritative state after enforcement:

```python
@dataclass
class EnforcedFeatureSet:
    features: List[str]  # Safe, ordered features (the truth)
    fingerprint_set: str  # Set-invariant fingerprint (for cache keys)
    fingerprint_ordered: str  # Order-sensitive fingerprint (for validation)
    cap_minutes: Optional[float]  # Cap that was enforced
    actual_max_minutes: float  # Actual max lookback from safe features
    canonical_map: Dict[str, float]  # Canonical lookback map (SST)
    quarantined: Dict[str, float]  # Quarantined features (over-cap)
    unknown: List[str]  # Unknown lookback features
    stage: str  # Enforcement stage name
```

### 2. Type Boundary Wiring

All enforcement stages now follow this pattern:

```python
# 1. Call apply_lookback_cap()
cap_result = apply_lookback_cap(...)

# 2. Convert to EnforcedFeatureSet (SST contract)
enforced = cap_result.to_enforced_set(stage=..., cap_minutes=...)

# 3. Slice X immediately using enforced.features (no rediscovery)
feature_indices = [i for i, f in enumerate(feature_names) if f in enforced.features]
X = X[:, feature_indices]

# 4. Update feature_names to match enforced.features (the truth)
feature_names = enforced.features.copy()

# 5. Store EnforcedFeatureSet for downstream use
resolved_config._<stage>_enforced = enforced
```

### 3. Boundary Assertions

**File**: `TRAINING/utils/lookback_policy.py`

Reusable invariant check function validates featureset integrity:

```python
def assert_featureset_fingerprint(
    label: str,
    expected: 'EnforcedFeatureSet',
    actual_features: list[str],
    logger_instance: Optional[logging.Logger] = None,
    allow_reorder: bool = False
) -> None:
```

**Validation Checks**:
1. Exact list equality: `actual_features == expected.features`
2. Set equality: `set(actual_features) == set(expected.features)`
3. Order equality: `actual_features == expected.features` (if `allow_reorder=False`)
4. Fingerprint match: Set and ordered fingerprints match

## Implementation

### Target Ranking

- **Gatekeeper**: Uses `EnforcedFeatureSet`, slices X immediately
- **POST_GATEKEEPER**: Assertion validates against gatekeeper_enforced
- **POST_PRUNE**: Uses `EnforcedFeatureSet`, slices X immediately, assertion validates
- **MODEL_TRAIN_INPUT**: Assertion validates against post_prune_enforced

### Feature Selection

- **FS_PRE (CROSS_SECTIONAL)**: Uses `EnforcedFeatureSet`, slices X immediately, assertion validates
- **FS_PRE (SYMBOL_SPECIFIC)**: Uses `EnforcedFeatureSet`, slices X immediately, assertion validates
- **FS_POST**: Uses `EnforcedFeatureSet`, updates selected_features, assertion validates

## Key Benefits

✅ **No Split-Brain**: All stages use the same `EnforcedFeatureSet` contract  
✅ **No Rediscovery**: X is sliced immediately using `enforced.features`  
✅ **Order Preservation**: X columns match `enforced.features` order exactly  
✅ **Immediate Detection**: Boundary assertions catch mis-wire at the boundary where it happens  
✅ **Auto-Fix**: Mismatches automatically corrected using `enforced.features`  
✅ **Full Coverage**: Implemented across all training paths and views

## Files Changed

### Core Infrastructure
- `TRAINING/utils/lookback_cap_enforcement.py`: Added `EnforcedFeatureSet` dataclass, `to_enforced_set()` method
- `TRAINING/utils/lookback_policy.py`: Added `LookbackPolicy` dataclass, `resolve_lookback_policy()`, `assert_featureset_fingerprint()`
- `TRAINING/utils/leakage_budget.py`: Added `canonical_lookback_map` parameter support

### Target Ranking
- `TRAINING/ranking/predictability/model_evaluation.py`: 
  - Gatekeeper uses `EnforcedFeatureSet` (lines 426-625)
  - POST_PRUNE uses `EnforcedFeatureSet` (lines 1016-1090)
  - POST_GATEKEEPER assertion (lines 4760-4775)
  - POST_PRUNE assertion (lines 1076-1090)
  - MODEL_TRAIN_INPUT assertion (lines 1438-1446)

### Feature Selection
- `TRAINING/ranking/feature_selector.py`:
  - FS_PRE (SYMBOL_SPECIFIC) uses `EnforcedFeatureSet` (lines 344-390)
  - FS_PRE (CROSS_SECTIONAL) uses `EnforcedFeatureSet` (lines 558-606)
  - FS_POST uses `EnforcedFeatureSet` (lines 868-913)
  - All stages have boundary assertions

### Supporting Changes
- `TRAINING/utils/resolved_config.py`: Pre-enforcement purge guard, `canonical_lookback_map` parameter
- `TRAINING/utils/cross_sectional_data.py`: Order drift clamping

## Verification

### Test Results
- ✅ Gatekeeper drops 39 over-cap features correctly
- ✅ POST_GATEKEEPER sanity check passes: `actual_max=150m <= cap=240m`
- ✅ POST_PRUNE enforcement: all 102 features safe
- ✅ MODEL_TRAIN_INPUT fingerprint matches POST_PRUNE: `4cfee217`
- ✅ Purge correctly computed: `155m` (150m lookback + 5m buffer)
- ✅ No assertion failures (would have logged errors)
- ✅ No split-brain detected

## Related Documentation

- [SST Enforcement Design Changelog](../../02_reference/changelog/2025-12-13-sst-enforcement-design.md) - Complete changelog
- [SST Enforcement Design Document](../../../TRAINING/utils/SST_ENFORCEMENT_DESIGN.md) - Design specification
- [Type Boundary Wiring Complete](../../../TRAINING/utils/TYPE_BOUNDARY_WIRING_COMPLETE.md) - Implementation details
- [Boundary Assertions Complete](../../../TRAINING/utils/BOUNDARY_ASSERTIONS_COMPLETE.md) - Assertion details
- [SST Implementation Coverage](../../../TRAINING/utils/SST_IMPLEMENTATION_COVERAGE.md) - Coverage matrix
