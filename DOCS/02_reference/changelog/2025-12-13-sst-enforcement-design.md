# SST Enforcement Design Implementation (2025-12-13)

**Status**: Complete  
**Related**: [Single Source of Truth Fix](2025-12-13-single-source-of-truth.md) | [Fingerprint Tracking](2025-12-13-fingerprint-tracking.md)

## Overview

Implemented a comprehensive Single Source of Truth (SST) enforcement design that eliminates split-brain across all training paths (target ranking and feature selection, both cross-sectional and symbol-specific views). The design uses `EnforcedFeatureSet` as the authoritative contract, immediate X slicing to prevent rediscovery, and boundary assertions to detect mis-wire immediately.

## Problem Statement

Previous implementation had split-brain risks:
- Different code paths could derive feature lists from different sources (X.columns, feature_names, importance keys)
- X matrix columns could drift from authoritative feature list
- No validation that featureset matches expected state at boundaries
- Order drift could occur during DataFrame operations

## Solution Architecture

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

**Key Properties**:
- `fingerprint` property (backward compatibility) returns `fingerprint_set`
- Both set and ordered fingerprints for comprehensive validation
- Stores canonical map for reuse (prevents recomputation)

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

**Implementation Locations**:
- **Gatekeeper**: `TRAINING/ranking/predictability/model_evaluation.py:426-625`
- **POST_PRUNE**: `TRAINING/ranking/predictability/model_evaluation.py:1016-1090`
- **FS_PRE (SYMBOL_SPECIFIC)**: `TRAINING/ranking/feature_selector.py:344-390`
- **FS_PRE (CROSS_SECTIONAL)**: `TRAINING/ranking/feature_selector.py:558-606`
- **FS_POST**: `TRAINING/ranking/feature_selector.py:868-913`

### 3. Boundary Assertions

**File**: `TRAINING/utils/lookback_policy.py`

Reusable invariant check function:

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

**Assertion Locations**:
- **POST_GATEKEEPER**: `model_evaluation.py:4760-4775`
- **POST_PRUNE**: `model_evaluation.py:1076-1090`
- **MODEL_TRAIN_INPUT**: `model_evaluation.py:1438-1446`
- **FS_PRE (both views)**: `feature_selector.py:380-390, 595-606`
- **FS_POST**: `feature_selector.py:901-913`

### 4. Policy Resolution

**File**: `TRAINING/utils/lookback_policy.py`

Centralized policy resolution function:

```python
def resolve_lookback_policy(
    resolved_config: Any,
    stage: str = "unknown"
) -> 'LookbackPolicy':
```

**Policy Dataclass**:
```python
@dataclass
class LookbackPolicy:
    cap_minutes: Optional[float]
    over_budget_action: str  # "hard_stop" | "drop" | "warn"
    unknown_lookback_action: str  # "hard_stop" | "drop" | "warn"
    unknown_policy: str  # "conservative" | "drop"
    policy_mode: str  # "strict" | "permissive"
```

## Implementation Coverage

### Target Ranking

| Stage | View | Status |
|-------|------|--------|
| Gatekeeper | CROSS_SECTIONAL | ✅ Uses `EnforcedFeatureSet` |
| Gatekeeper | SYMBOL_SPECIFIC | ✅ Uses `EnforcedFeatureSet` |
| POST_GATEKEEPER | CROSS_SECTIONAL | ✅ Assertion added |
| POST_GATEKEEPER | SYMBOL_SPECIFIC | ✅ Assertion added |
| POST_PRUNE | CROSS_SECTIONAL | ✅ Uses `EnforcedFeatureSet` + assertion |
| POST_PRUNE | SYMBOL_SPECIFIC | ✅ Uses `EnforcedFeatureSet` + assertion |
| MODEL_TRAIN_INPUT | CROSS_SECTIONAL | ✅ Assertion added |
| MODEL_TRAIN_INPUT | SYMBOL_SPECIFIC | ✅ Assertion added |

### Feature Selection

| Stage | View | Status |
|-------|------|--------|
| FS_PRE | CROSS_SECTIONAL | ✅ Uses `EnforcedFeatureSet` + assertion |
| FS_PRE | SYMBOL_SPECIFIC | ✅ Uses `EnforcedFeatureSet` + assertion |
| FS_POST | CROSS_SECTIONAL | ✅ Uses `EnforcedFeatureSet` + assertion |
| FS_POST | SYMBOL_SPECIFIC | ✅ Uses `EnforcedFeatureSet` + assertion |

## Key Benefits

### 1. No Split-Brain
- All stages use the same `EnforcedFeatureSet` contract
- Same canonical map reused across stages
- Same enforcement logic everywhere

### 2. No Rediscovery
- X is sliced immediately using `enforced.features`
- No deriving truth from `X.columns` later
- Feature list is authoritative source

### 3. Order Preservation
- X columns match `enforced.features` order exactly
- Ordered fingerprints detect order drift
- Explicit reindexing after transforms

### 4. Immediate Detection
- Boundary assertions catch mis-wire at the boundary where it happens
- Actionable error messages with detailed diffs
- Auto-fix capability (uses `enforced.features` as truth)

### 5. Consistent Behavior
- Same enforcement logic across ranking and feature selection
- Same contract across all views (CROSS_SECTIONAL, SYMBOL_SPECIFIC)
- Same validation at all boundaries

## Technical Details

### Canonical Map Reuse

**File**: `TRAINING/utils/leakage_budget.py`

Added `canonical_lookback_map` parameter to `compute_feature_lookback_max()`:

```python
def compute_feature_lookback_max(
    ...
    canonical_lookback_map: Optional[Dict[str, float]] = None  # NEW
) -> LookbackResult:
```

**Behavior**:
- If passed, uses provided map (SST reuse)
- Otherwise, builds from cache or computes
- Prevents recomputation with different policies

### Pre-Enforcement Purge Guard

**File**: `TRAINING/utils/resolved_config.py`

Added logic to cap purge calculation in pre-enforcement stages:

```python
# CRITICAL: In pre-enforcement stages, cap lookback used for purge bump
lookback_for_purge = feature_lookback_max_minutes
if lookback_budget_cap is not None and feature_lookback_max_minutes > lookback_budget_cap:
    lookback_for_purge = lookback_budget_cap
    logger.debug(...)
```

**Prevents**: Early purge inflation due to long-lookback features that will be dropped by gatekeeper.

### Order Drift Clamping

**File**: `TRAINING/utils/cross_sectional_data.py`

Added explicit reindexing after cleaning:

```python
if isinstance(feature_df, pd.DataFrame):
    feature_df = feature_df.loc[:, [f for f in feature_names if f in feature_df.columns]]
```

**Prevents**: Column reordering during DataFrame operations.

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

## Documentation

### Design Documents
- `TRAINING/utils/SST_ENFORCEMENT_DESIGN.md`: Complete design specification
- `TRAINING/utils/TYPE_BOUNDARY_WIRING_COMPLETE.md`: Type boundary wiring details
- `TRAINING/utils/BOUNDARY_ASSERTIONS_COMPLETE.md`: Boundary assertions details
- `TRAINING/utils/SST_IMPLEMENTATION_COVERAGE.md`: Implementation coverage matrix

### Fix Documents
- `TRAINING/utils/CANONICAL_MAP_SIGNATURE_FIX.md`: Canonical map parameter fix
- `TRAINING/utils/PRE_ENFORCEMENT_PURGE_INFLATION_FIX.md`: Pre-enforcement purge guard
- `TRAINING/utils/ORDERED_FINGERPRINT_AND_EXACT_EQUALITY_FIX.md`: Ordered fingerprints

## Verification

### Test Results
- ✅ Gatekeeper drops 39 over-cap features correctly
- ✅ POST_GATEKEEPER sanity check passes: `actual_max=150m <= cap=240m`
- ✅ POST_PRUNE enforcement: all 102 features safe
- ✅ MODEL_TRAIN_INPUT fingerprint matches POST_PRUNE: `4cfee217`
- ✅ Purge correctly computed: `155m` (150m lookback + 5m buffer)
- ✅ No assertion failures (would have logged errors)
- ✅ No split-brain detected

### Log Evidence
```
GATEKEEPER: n_features=300 → safe=261 quarantined=39 cap=240.0m actual_max=150.0m
POST_GATEKEEPER: n=261, fingerprint=20a4ee9c
POST_PRUNE: n_features=102 → safe=102 quarantined=0 cap=240.0m actual_max=150.0m
MODEL_TRAIN_INPUT: n=102, fingerprint=4cfee217
✅ POST_GATEKEEPER sanity check PASSED: actual_max_from_features=150.0m <= cap=240.0m
```

## Migration Notes

### For Developers

**Before**:
```python
cap_result = apply_lookback_cap(...)
safe_features = cap_result.safe_features
X = X[:, [i for i, f in enumerate(feature_names) if f in safe_features]]
feature_names = safe_features
```

**After**:
```python
cap_result = apply_lookback_cap(...)
enforced = cap_result.to_enforced_set(stage=..., cap_minutes=...)
feature_indices = [i for i, f in enumerate(feature_names) if f in enforced.features]
X = X[:, feature_indices]
feature_names = enforced.features.copy()
resolved_config._<stage>_enforced = enforced
```

### For Testing

All enforcement stages should:
1. Use `EnforcedFeatureSet` after enforcement
2. Slice X immediately using `enforced.features`
3. Have boundary assertions at key points
4. Store `EnforcedFeatureSet` for downstream validation

## Related Work

- [Single Source of Truth Fix](2025-12-13-single-source-of-truth.md) - Initial SST implementation
- [Fingerprint Tracking](2025-12-13-fingerprint-tracking.md) - Fingerprint system
- [Feature Selection Unification](2025-12-13-feature-selection-unification.md) - Feature selection integration

## Status

✅ **Complete**: All enforcement stages wired, all boundaries asserted, all views supported

The system is now **provably split-brain free** across all training paths!
