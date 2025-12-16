# SST Design Comparison: Current vs. Recommended

**Date**: 2025-12-13  
**Purpose**: Compare current implementation to recommended tightening suggestions

## 1. Fingerprint Spec: Set vs Ordered

### Current Implementation
```python
@dataclass
class EnforcedFeatureSet:
    fingerprint: str  # Set-invariant fingerprint (for validation)
    features: List[str]  # Safe, ordered feature list (the truth)
```

**Issue**: Only stores set-invariant fingerprint, but `features` is "ordered (the truth)". This creates ambiguity:
- Set fingerprint catches "same members?"
- But we claim order matters, yet don't validate it

**Function exists**: `_compute_feature_fingerprint()` returns both:
- `set_fingerprint`: hash(sorted(features))
- `order_fingerprint`: hash(tuple(features))

But we only store one.

### Recommended
```python
@dataclass
class EnforcedFeatureSet:
    fingerprint_set: str        # hash(sorted(features))
    fingerprint_ordered: str    # hash(tuple(features))
    features: List[str]  # Safe, ordered feature list (the truth)
```

**Boundary checks**: Should enforce **ordered** equality by default (unless stage explicitly allows reorder)

**Cache keys**: Use **set** fingerprint + policy + interval signature

**Status**: ❌ Not implemented - only storing set fingerprint

---

## 2. `assert_featureset_fingerprint()` Validation Depth

### Current Implementation
```python
def assert_featureset_fingerprint(...):
    actual_fp, _ = _compute_feature_fingerprint(actual_features, set_invariant=True)
    
    if actual_fp != expected.fingerprint:
        # Log sample differences
        expected_set = set(expected.features)
        actual_set = set(actual_features)
        added = actual_set - expected_set
        removed = expected_set - actual_set
        # ... log and raise
```

**Issues**:
- Only checks hash equality (can have collisions, though unlikely)
- Doesn't validate exact list equality
- Doesn't detect order divergence
- Doesn't show first index of divergence
- Doesn't include stage/cap/policy in failure message

### Recommended
```python
def assert_featureset_fingerprint(...):
    # 1. Exact list equality check (not just hash)
    if actual_features != expected.features:
        # 2. Compute actionable diff:
        #    - missing features
        #    - unexpected features
        #    - first index of divergence (order)
        #    - short "nearby" window around divergence
        # 3. Include stage, cap, policy in error message
```

**Status**: ⚠️ Partial - checks hash and shows set differences, but:
- No exact list equality check
- No order divergence detection
- No first divergence index
- Missing stage/cap/policy in error message

---

## 3. "No Rediscovery" Enforcement

### Current Implementation
```python
# Comment in SST_ENFORCEMENT_DESIGN.md:
# "After enforcement, slice X immediately:
#  X = X.loc[:, enforced.features]  # NO rediscovery from X.columns"
```

**Issue**: Just a comment. Someone can still do:
```python
feature_names = X.columns.tolist()  # Bypass - split-brain reintroduced
```

### Recommended
Two options:

**Option A**: Wrap sliced matrix in container
```python
@dataclass
class EnforcedMatrix:
    X: np.ndarray
    enforced: EnforcedFeatureSet
    
    # Expose .X but NOT .columns directly (or log if accessed)
```

**Option B**: Make downstream APIs accept `EnforcedFeatureSet` only
```python
def train_model(enforced: EnforcedFeatureSet, ...):
    # Never accepts List[str]
```

**Status**: ❌ Not enforced - only documented as comment

---

## 4. `canonical_map` + `budget` Weight

### Current Implementation
```python
@dataclass
class EnforcedFeatureSet:
    canonical_map: Dict[str, float]  # Canonical lookback map (single source of truth)
    budget: Any  # LeakageBudget object (for purge/embargo computation)
```

**Issue**: Can get heavy if passed everywhere, makes serialization/debugging annoying

### Recommended
```python
@dataclass
class EnforcedFeatureSet:
    canonical_map_fingerprint: str  # Hash of canonical map
    budget_id: str  # Lightweight summary or cache handle
    # Keep canonical_map optional or in cache
```

**Status**: ⚠️ Current - stores full objects (works but could be optimized)

---

## 5. Cache Key Multi-Interval Reality

### Current Implementation
**Phase 4 (TODO)**: Cache key mentioned as `(fingerprint, features, policy, interval)`

**Issue**: Single `interval` not sufficient if per-feature intervals exist (FeatureTimeMeta)

### Recommended
Cache key should include:
- `base_interval_minutes`
- Fingerprint of `FeatureTimeMeta` mapping (feature → interval/effective lookback basis)
- Policy hash (unknown_policy + actions + any defaults)

**Status**: ❌ Not implemented (Phase 4 TODO)

---

## 6. `unknown_policy="conservative"` Deterministic Semantics

### Current Implementation
```python
@dataclass
class LookbackPolicy:
    unknown_policy: str  # "drop" (inf) | "conservative" (1440m) - for inference only
```

**Issue**: String-based, not deterministic. "conservative" → 1440m is implicit.

### Recommended
```python
@dataclass
class LookbackPolicy:
    unknown_assumed_minutes: float  # e.g., 1440.0 for conservative
    # Still run through exact same "> cap → quarantine" logic
```

**Status**: ⚠️ Partial - has `unknown_policy` string but not explicit `unknown_assumed_minutes`

---

## 7. Naming: "quarantined" vs "dropped" vs "excluded"

### Current Implementation
```python
@dataclass
class EnforcedFeatureSet:
    quarantined: Dict[str, float]  # Feature → lookback for quarantined features
    unknown: List[str]  # Features with unknown lookback (inf) - tracked separately
```

**Issue**: "quarantined" reads like "set aside for review", but flow implies "removed from training"

### Recommended
```python
@dataclass
class EnforcedFeatureSet:
    excluded_over_cap: Dict[str, float]  # Feature → lookback for over-cap features
    excluded_unknown: List[str]  # Features with unknown lookback (inf)
    excluded_total: int  # Total excluded count
    # Keep "quarantined" only if truly intend "rehabilitation" path
```

**Status**: ⚠️ Current naming could be clearer

---

## 8. Migration Plan Order

### Current Plan (SST_ENFORCEMENT_DESIGN.md)
```
Phase 2: Update Enforcement (TODO)
- [ ] Update apply_lookback_cap() to return EnforcedFeatureSet directly
- [ ] Update gatekeeper to use EnforcedFeatureSet and slice X immediately
- [ ] Update POST_PRUNE to use EnforcedFeatureSet
- [ ] Update feature selection (FS_PRE, FS_POST) to use EnforcedFeatureSet

Phase 3: Add Invariant Checks (TODO)
- [ ] Add assert_featureset_fingerprint() at SAFE_CANDIDATES
- [ ] Add assert_featureset_fingerprint() at AFTER_LEAK_REMOVAL
- [ ] Add assert_featureset_fingerprint() at POST_GATEKEEPER
- [ ] Add assert_featureset_fingerprint() at POST_PRUNE
- [ ] Add assert_featureset_fingerprint() at MODEL_TRAIN_INPUT (already done)
- [ ] Add assert_featureset_fingerprint() at FS_PRE/FS_POST
```

### Recommended Order
1. **Gatekeeper returns `EnforcedFeatureSet` and slices X immediately**
2. **POST_PRUNE consumes/returns `EnforcedFeatureSet`**
3. **FS_PRE/FS_POST use it**
4. **Then add boundary assertions everywhere**

**Rationale**: If you add assertions before the enforced type boundary is fully wired, you'll create noisy failures that encourage "temporary bypasses" (aka split-brain reintroduced).

**Status**: ⚠️ Order needs adjustment - assertions should come AFTER type boundary is wired

---

## Summary: What's Strongest

✅ **Centralized `resolve_lookback_policy(resolved_config)`** - Right SST policy root

✅ **Treating `inf` as first-class violation** - Fixes most common real-world failure mode

✅ **Forcing "slice X immediately"** - Single highest-leverage move

## What Needs Tightening

### Mandatory (per user)
1. **Add ordered vs set fingerprints** - Store both, validate ordered by default
2. **Make boundary checks validate exact list equality** - Not just hash, with surgical diffs

### High Priority
3. **Enforce "no rediscovery"** - Wrap matrix or make APIs accept `EnforcedFeatureSet` only
4. **Fix migration order** - Wire type boundary first, then add assertions
5. **Clarify naming** - "excluded" vs "quarantined"

### Nice to Have
6. **Lightweight canonical_map/budget** - Optional or cached
7. **Multi-interval cache key** - Include FeatureTimeMeta fingerprint
8. **Explicit `unknown_assumed_minutes`** - Not just string

---

## Next Steps

1. **Immediate**: Implement ordered + set fingerprints, exact list equality checks
2. **Phase 2 (revised)**: Wire type boundary (gatekeeper → POST_PRUNE → FS) BEFORE adding assertions
3. **Phase 3**: Add boundary assertions everywhere (now that type boundary is enforced)
4. **Phase 4**: Optimize (lightweight maps, multi-interval cache keys)
