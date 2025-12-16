# Infrastructure Complexity Analysis

**Date**: 2025-12-15  
**Context**: Analysis of lookback enforcement, budget computation, and unknown lookback handling to ensure "dumb infra" principles are followed.

---

## Executive Summary

The system implements a clean 3-step contract for feature safety:
1. **Candidate set construction** (registry + schema/pattern allowances + target-conditional exclusions)
2. **Lookback accounting** (`compute_budget`)
3. **Enforcement** (gatekeeper quarantine + purge adjustment)

This is **boring, explicit invariant enforcement**—exactly the kind of "dumb infra" that prevents silent corruption.

---

## Analysis Results

### ✅ What's Working Well (Keep)

1. **Feature fingerprints at each stage** (`SAFE_CANDIDATES`, `POST_GATEKEEPER`, `POST_PRUNE`, `MODEL_TRAIN_INPUT`)
   - ✅ "Dumb" observability - prevents split-brain
   - ✅ Location: `TRAINING/utils/cross_sectional_data.py` (`_compute_feature_fingerprint`)

2. **Gatekeeper actually drops offenders**
   - ✅ Printed drop list + reasons
   - ✅ Location: `TRAINING/ranking/predictability/model_evaluation.py` (gatekeeper section)
   - ✅ Uses `apply_lookback_cap()` which quarantines unknowns and over-cap features

3. **Audit rule enforcement**
   - ✅ `purge < feature_lookback_max → increase purge`
   - ✅ Prevents accidental leakage when features have longer memory than horizon
   - ✅ Location: `TRAINING/ranking/predictability/model_evaluation.py` (POST_PRUNE_policy_check)

4. **Re-run gatekeeper after pruning**
   - ✅ Correct: pruning changes the set, so re-validate invariants
   - ✅ Location: `TRAINING/ranking/predictability/model_evaluation.py` (post-prune gatekeeper)

---

## Issues Found & Recommendations

### 🔴 Issue 1: Auto-Detected Interval Not Artifacted

**Problem**: `base_interval_minutes` and `base_interval_source` are logged but not saved to metadata.json.

**Current State**:
- ✅ Logged: `base_interval_source` is logged in `create_resolved_config()`
- ❌ **NOT artifacted**: `ResolvedConfig.to_dict()` does NOT include `base_interval_minutes` or `base_interval_source`
- ❌ **NOT in metadata.json**: These fields are not persisted for replayability

**Location**: `TRAINING/utils/resolved_config.py` lines 138-156

**Fix Required**:
```python
def to_dict(self) -> Dict[str, Any]:
    """Convert to dictionary for reproducibility tracking."""
    return {
        # ... existing fields ...
        "base_interval_minutes": self.base_interval_minutes,  # NEW
        "base_interval_source": self.base_interval_source,    # NEW
        # ... rest of fields ...
    }
```

**Impact**: Without this, runs cannot be fully replayed because the auto-detected interval is lost.

---

### 🟡 Issue 2: Unknown Lookback Policy Ambiguity

**Problem**: Unknown lookback (`inf`) handling has inconsistent messaging:
- Pre-enforcement: "expected, will be quarantined"
- Post-enforcement: "should have been dropped/quarantined" (warning)

**Current State**:
- ✅ **Pre-enforcement stages**: Unknowns allowed, logged as INFO/DEBUG
- ✅ **Post-enforcement stages**: Hard-fail in strict mode (correct)
- ⚠️ **Policy clarity**: The messaging could be clearer about the contract

**Location**: 
- `TRAINING/utils/lookback_cap_enforcement.py` lines 233-247
- `TRAINING/utils/leakage_budget.py` lines 996-1065

**Current Behavior**:
```python
# Pre-enforcement (SAFE_CANDIDATES, FS_PRE, FS_POST)
if is_pre_enforcement:
    logger.debug(f"unknown lookback (will be quarantined at gatekeeper)")

# Post-enforcement (POST_GATEKEEPER, POST_PRUNE)
else:
    logger.warning(f"unknown lookback (quarantined - violation)")
```

**Recommendation**: Make contract explicit:
- **Pre-enforcement**: Tag as `UNKNOWN_LOOKBACK` status, force through sanitizer/gatekeeper
- **Enforcement stage**: Unknowns must become (a) inferred, (b) mapped, or (c) quarantined
- **No "inf survives" past PRE_GATEKEEPER**: This is already enforced via hard-fail, but messaging could be clearer

**Status**: ✅ **Working as intended** - hard-fail prevents unknowns from surviving, but messaging could be improved.

---

### 🟡 Issue 3: Multiple Budget Computations

**Problem**: `compute_budget()` is called multiple times per stage:
- `create_resolved_config` → `compute_budget()`
- `GATEKEEPER` → `compute_budget()`
- `POST_GATEKEEPER` → `compute_budget()`
- `POST_PRUNE_policy_check` → `compute_budget()`
- `CV_SPLITTER_CREATION` → `compute_budget()`

**Current State**:
- ✅ **Cache exists**: `_budget_cache` in `leakage_budget.py` reduces recomputation
- ⚠️ **Still noisy**: Multiple calls create log spam even with cache
- ✅ **FeatureSetArtifact exists**: Can store budget, but not fully integrated

**Location**: 
- `TRAINING/utils/leakage_budget.py` lines 686-690 (cache)
- `TRAINING/utils/feature_set_artifact.py` (artifact structure exists)

**Recommendation**: 
1. **Phase 2 Enhancement**: Create `ResolvedFeatureSetPolicy` dataclass:
```python
@dataclass
class ResolvedFeatureSetPolicy:
    featureset_fingerprint: str
    interval_minutes: float
    actual_max_lookback_minutes: float
    cap_minutes: Optional[float]
    purge_minutes: float
    embargo_minutes: float
    unknown_lookback_count: int
    canonical_lookback_map: Dict[str, float]
    stage: str
```

2. **Pass through pipeline**: Stages consume this object instead of recomputing
3. **Still deterministic**: Same inputs → same outputs, just computed once

**Status**: 🟡 **Partially addressed** - cache exists, but could be better with explicit policy object.

---

### 🟡 Issue 4: Horizon Logging Clarity

**Problem**: Horizon is logged inconsistently - sometimes in minutes, sometimes in bars, not always both.

**Current State**:
- ✅ **Some places log both**: `leakage_filtering.py` logs `horizon={target_horizon_minutes}m = {target_horizon_bars} bars`
- ❌ **Inconsistent**: Not all places log both formats
- ⚠️ **Confusing**: "horizon=12" (bars) vs "horizon=60m" (minutes) can be confusing

**Location**: 
- `TRAINING/utils/leakage_filtering.py` line 631 (logs both)
- `TRAINING/ranking/predictability/model_evaluation.py` line 4801 (logs bars only)
- `TRAINING/common/feature_registry.py` (uses bars internally)

**Recommendation**: Always log both:
```python
horizon_info = f"horizon_minutes={horizon_minutes:.1f}m, horizon_bars={horizon_bars} bars @ interval={interval_minutes:.1f}m"
```

**Status**: 🟡 **Needs improvement** - some places log both, but not consistently.

---

### 🟢 Issue 5: Naming Clarity

**Current Names**:
- `Gatekeeper` → Could be `LookbackCapEnforcer` (more explicit)
- `Sanitizer` → Could be `QuarantineUnknownLookback` (more explicit)
- `Budget` → Could be `LookbackReport` (less ambiguous)

**Recommendation**: 
- **Low priority**: Current names are fine, but renaming would make "dumb infra" principle more obvious
- **Consider**: Rename in future refactor if it improves clarity

**Status**: 🟢 **Optional** - current names work, but could be more explicit.

---

## Detailed Findings

### 1. Auto-Detected Interval Artifacting

**Code Location**: `TRAINING/utils/resolved_config.py`

**Issue**: `base_interval_minutes` and `base_interval_source` are computed and logged but not saved to metadata.

**Evidence**:
```python
# Line 64: Field exists in ResolvedConfig
base_interval_minutes: Optional[float] = None
base_interval_source: str = "auto"

# Line 138-156: to_dict() does NOT include these fields
def to_dict(self) -> Dict[str, Any]:
    return {
        # ... missing base_interval_minutes and base_interval_source ...
    }
```

**Fix**: Add to `to_dict()` method.

---

### 2. Unknown Lookback Enforcement

**Code Location**: 
- `TRAINING/utils/lookback_cap_enforcement.py` (apply_lookback_cap)
- `TRAINING/ranking/predictability/model_evaluation.py` (POST_GATEKEEPER, POST_PRUNE)

**Status**: ✅ **Working correctly**

**Evidence**:
- ✅ Pre-enforcement: Unknowns allowed, logged as INFO/DEBUG
- ✅ Post-enforcement: Hard-fail in strict mode (lines 1097-1118, 5174-5211)
- ✅ Gatekeeper quarantines unknowns (line 247 in lookback_cap_enforcement.py)

**Contract**:
- Pre-enforcement: Unknowns expected, will be quarantined
- Post-enforcement: Unknowns = bug, hard-fail in strict mode

**Recommendation**: Improve messaging clarity (already working, just needs better docs).

---

### 3. Budget Computation Frequency

**Code Location**: `TRAINING/ranking/predictability/model_evaluation.py`

**Count**: 103 matches for `compute_budget` in model_evaluation.py

**Stages**:
1. `create_resolved_config` (initial setup)
2. `GATEKEEPER` (enforcement)
3. `POST_GATEKEEPER` (validation)
4. `POST_PRUNE_policy_check` (post-prune validation)
5. `CV_SPLITTER_CREATION` (CV setup)

**Current Mitigation**:
- ✅ Cache exists: `_budget_cache` in `leakage_budget.py`
- ✅ FeatureSetArtifact exists: Can store budget

**Recommendation**: 
- Create `ResolvedFeatureSetPolicy` object
- Pass through pipeline instead of recomputing
- Still deterministic (same inputs → same outputs)

---

### 4. Horizon Logging

**Code Location**: Multiple files

**Current State**:
- ✅ `leakage_filtering.py` line 631: Logs both `horizon={target_horizon_minutes}m = {target_horizon_bars} bars`
- ❌ `model_evaluation.py` line 4801: Logs only `horizon={target_horizon_bars} bars`
- ❌ Registry logs: Use bars only

**Recommendation**: Standardize to always log both formats.

---

## Recommendations Summary

### High Priority

1. **Artifact base_interval_minutes** in `ResolvedConfig.to_dict()`
   - Impact: Prevents full replayability
   - Effort: Low (add 2 fields to dict)

2. **Standardize horizon logging** to always show both minutes and bars
   - Impact: Reduces confusion
   - Effort: Low (update log statements)

### Medium Priority

3. **Create ResolvedFeatureSetPolicy** object to reduce budget recomputation
   - Impact: Reduces log noise, clearer contract
   - Effort: Medium (new dataclass + pipeline integration)

4. **Improve unknown lookback messaging** clarity
   - Impact: Better understanding of contract
   - Effort: Low (update log messages)

### Low Priority

5. **Rename for clarity** (Gatekeeper → LookbackCapEnforcer, etc.)
   - Impact: Makes "dumb infra" principle more obvious
   - Effort: Medium (rename across codebase)

---

## Conclusion

**Overall Assessment**: ✅ **System is working as intended**

The infrastructure is doing the right thing:
- ✅ Unknown lookback enforcement is correct (hard-fail prevents survival)
- ✅ Gatekeeper actually drops offenders
- ✅ Budget computation is deterministic (cache prevents recomputation)
- ✅ Feature fingerprints prevent split-brain

**Main Issues**:
1. Auto-detected interval not artifacted (prevents full replayability)
2. Horizon logging inconsistent (minor confusion)
3. Budget recomputation creates log noise (but cache mitigates)

**Recommendation**: Fix high-priority items (artifacting + horizon logging), then consider policy object consolidation in future refactor.

