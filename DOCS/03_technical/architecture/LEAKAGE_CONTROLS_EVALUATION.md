# Leakage Controls Evaluation

**Date**: 2025-12-13  
**Task**: Fix structural contradictions in leakage controls (purge/lookback + messaging) and make CV/time-safety + smoke tests enforceable and self-consistent.

## Executive Summary

The current implementation has **critical structural contradictions** that allow the system to appear leak-safe while still being wrong. The primary issue is that **audit and gatekeeper use different feature sets** (pre vs post filtering), causing audit to report `feature_lookback_max_minutes=1440` while gatekeeper claims `safe_lookback_max=34.6m` and proceeds with training.

### Critical Issues Found

1. ❌ **No unified lookback calculator** - Audit and gatekeeper both call `compute_feature_lookback_max()` but on different feature sets
2. ❌ **Audit runs on wrong feature set** - Computes lookback BEFORE gatekeeper drops features
3. ❌ **Calendar features misclassified** - `day_of_week`, `holiday_dummy` treated as 1440m lookback (should be 0m)
4. ❌ **No hard-stop on audit violations** - Violations logged but training continues
5. ⚠️ **CV splitter exists but not consistently logged** - `PurgedTimeSeriesSplit` is used but splitter identity not in run summary
6. ⚠️ **Smoke tests exist but not wired as gates** - Leakage diagnostics exist but don't block "PASS" status
7. ❌ **Contradictory reason strings** - `"overfit_likely; cv_not_suspicious"` is self-contradictory

---

## 1. Unified Lookback Calculator

### Current State

**Location**: `TRAINING/utils/resolved_config.py::compute_feature_lookback_max()`

**Issues**:
- Function exists but is called from multiple places with different feature sets
- Calendar features (`day_of_week`, `holiday_dummy`, `trading_day_of_month`) are incorrectly classified as 1440m lookback
- Pattern matching has precedence issues (keyword heuristics can override explicit suffixes)

**Code Analysis**:
```python
# Lines 327-333: Calendar features incorrectly treated as daily patterns
elif re.search(r'.*day.*', feat_name, re.I):
    # Very aggressive: catch "day" anywhere
    if interval_minutes > 0:
        lag_bars = int(1440 / interval_minutes)  # 1 day in bars
    else:
        lag_bars = 288  # Fallback: assume 5m bars
```

**Problem**: `day_of_week` matches `.*day.*` pattern and gets 1440m lookback, but it's an **exogenous calendar feature** with 0m lookback.

### Required Fix

Create `TRAINING/utils/leakage_budget.py` with:
- Single `LeakageBudget` dataclass
- `infer_lookback_minutes()` that prioritizes:
  1. Schema/registry metadata (if available)
  2. Calendar features whitelist (0m lookback)
  3. Explicit time suffixes (`_15m`, `_24h`, `_1d`)
  4. Bar-based patterns (`_288`)
  5. Conservative default (1440m) or drop policy

**Status**: ❌ **NOT IMPLEMENTED**

---

## 2. Audit Runs on Wrong Feature Set

### Current State

**Execution Order** (from `FEATURE_FILTERING_EXECUTION_ORDER.md`):
1. Phase 8: Final Gatekeeper drops features (line ~3688 in `model_evaluation.py`)
2. Phase 9: Feature Pruning (optional)
3. Phase 11: Audit validation (line ~87 in doc)

**Actual Code Flow**:
```python
# model_evaluation.py:3688-3714
X, feature_names = _enforce_final_safety_gate(...)  # Drops features
# Line 3700-3714: Recompute lookback AFTER gatekeeper
max_lookback_after_gatekeeper, _ = compute_feature_lookback_max(
    feature_names,  # FINAL features
    interval_minutes=detected_interval
)
resolved_config.feature_lookback_max_minutes = max_lookback_after_gatekeeper
```

**Good**: Code DOES recompute lookback after gatekeeper (lines 3700-3714).

**Problem**: Audit enforcer (`audit_enforcer.py`) reads `resolved_config.feature_lookback_max_minutes` from metadata, but:
- Metadata might be written BEFORE gatekeeper runs
- Or metadata might be written with stale lookback value

**Status**: ⚠️ **PARTIALLY FIXED** - Lookback recomputed after gatekeeper, but audit timing unclear

### Required Fix

- Ensure audit runs **after** `MODEL_TRAIN_INPUT` is finalized (post gatekeeper + pruning)
- Audit must read from `resolved_config.feature_lookback_max_minutes` that was computed from **final** features
- Add explicit check: `assert final_features == context["featuresets"]["MODEL_TRAIN_INPUT"]`

---

## 3. Hard-Stop on Audit Violations

### Current State

**Location**: `TRAINING/utils/audit_enforcer.py::_validate_feature_lookback()`

**Code**:
```python
# Lines 227-238: Violation detected
if purge_d < lookback_d:
    self.violations.append({
        "rule": "purge_minutes >= feature_lookback_max_minutes",
        "message": f"purge_minutes ({purge_str}) < feature_lookback_max_minutes ({lookback_str})",
        "severity": "critical"
    })

# Lines 139-144: Strict mode raises, warn mode logs
if self.mode == AuditMode.STRICT and not is_valid:
    raise ValueError(f"Audit validation failed (strict mode): {violation_summary}")
```

**Problem**: 
- In `warn` mode (default), violations are logged but training continues
- No hard-stop in the training pipeline itself
- Violations are detected but not enforced at the point of use

**Status**: ❌ **NOT ENFORCED** - Violations logged but training continues in warn mode

### Required Fix

```python
budget = compute_budget(final_features, interval_minutes, horizon_minutes, registry)
required = budget.required_gap_minutes
if purge_minutes < required:
    msg = f"purge_minutes ({purge_minutes}) < required_gap_minutes ({required})"
    if cfg.audit_hard_fail:
        raise RuntimeError("🚨 AUDIT VIOLATION: " + msg)
    else:
        logger.warning("🚨 AUDIT VIOLATION: " + msg)
        # Still raise in warn mode for critical violations
        raise RuntimeError("🚨 CRITICAL AUDIT VIOLATION: " + msg)
```

---

## 4. One Behavior: Drop Features OR Increase Purge

### Current State

**Location**: `TRAINING/ranking/predictability/model_evaluation.py::_enforce_final_safety_gate()`

**Behavior**: **Mode B (Exploratory)** - Drops features that violate purge limit

```python
# Lines 382-393: Drop features that violate purge
if is_daily_name:
    should_drop = True
    reason = "daily/24h naming pattern"
elif lookback_minutes > safe_lookback_max:
    should_drop = True
    reason = f"lookback ({lookback_minutes:.1f}m) > safe_limit ({safe_lookback_max:.1f}m)"
```

**Problem**: 
- Features are dropped silently (logged but no explicit policy decision)
- No deterministic "increase purge" mode
- No clear policy: strict (hard-stop) vs exploratory (drop features)

**Status**: ⚠️ **PARTIALLY IMPLEMENTED** - Drops features but no explicit policy choice

### Required Fix

Add config flag:
```yaml
safety:
  leakage_detection:
    purge_insufficient_policy: "strict"  # or "drop_features"
```

- **strict**: Hard-stop if purge insufficient
- **drop_features**: Drop violating features, recompute budget, ensure pass

---

## 5. Fix Contradictory Reason Strings

### Current State

**Location**: `TRAINING/ranking/predictability/model_evaluation.py:4292`

```python
auto_fix_reason=None if should_auto_fix else "overfit_likely; cv_not_suspicious",
```

**Problem**: `"overfit_likely; cv_not_suspicious"` is contradictory - if CV is not suspicious, why is overfit likely?

**Status**: ❌ **CONTRADICTORY** - Reason string built ad-hoc

### Required Fix

Create `LeakageAssessment` dataclass:
```python
@dataclass(frozen=True)
class LeakageAssessment:
    leak_scan_pass: bool
    cv_suspicious: bool
    overfit_likely: bool
    auc_too_high_models: list[str]

    def reason(self) -> str:
        flags = []
        if self.cv_suspicious: flags.append("cv_suspicious")
        if self.overfit_likely: flags.append("overfit_likely")
        if self.auc_too_high_models: flags.append(f"auc>0.90:{','.join(self.auc_too_high_models)}")
        return "; ".join(flags) if flags else "none"
```

---

## 6. CV Time-Safety and Logging

### Current State

**Location**: `TRAINING/utils/purged_time_series_split.py`

**Implementation**: ✅ **GOOD** - `PurgedTimeSeriesSplit` uses time-based purging for panel data

**Usage**: 
- `TRAINING/ranking/predictability/leakage_detection.py:639` - Creates `PurgedTimeSeriesSplit` with `purge_overlap_time`
- `TRAINING/ranking/shared_ranking_harness.py:326` - `split_policy()` creates splitter

**Problem**: 
- Splitter identity not logged in run summary
- No explicit `PurgedGroupTimeSeriesSplit` for cross-sectional data (though `PurgedTimeSeriesSplit` handles panel data correctly)

**Status**: ⚠️ **IMPLEMENTED BUT NOT LOGGED** - Splitter exists but not in summary

### Required Fix

Add to run summary:
```python
logger.info(f"splitter=PurgedTimeSeriesSplit")
logger.info(f"purge_minutes={purge_minutes}")
logger.info(f"embargo_minutes={embargo_minutes}")
logger.info(f"max_feature_lookback_minutes={max_lookback}")
logger.info(f"horizon_minutes={horizon_minutes}")
```

**Note**: For cross-sectional data, `PurgedTimeSeriesSplit` already handles panel data correctly (groups by timestamp). No need for separate `PurgedGroupTimeSeriesSplit` if current implementation groups correctly.

---

## 7. Smoke Tests as Gates

### Current State

**Location**: `TRAINING/utils/leakage_diagnostics.py`

**Tests Available**:
- `placebo_label_test()` - Shuffle labels, expect AUC ~0.5
- `time_shift_label_test()` - Shift labels, expect AUC drop
- `univariate_feature_auc_scan()` - Scan individual features
- `raw_ohlcv_only_test()` - Test with only OHLCV features

**Problem**: 
- Tests exist but not wired as gates
- Tests don't block "leak_scan: PASS" status
- No explicit "permutation y-test" or "feature shift test" as specified

**Status**: ⚠️ **EXISTS BUT NOT WIRED** - Tests available but not enforced

### Required Fix

1. Add explicit tests:
   - **Permutation y-test**: Shuffle labels, AUC should be ~0.5 (hard ceiling: 0.58)
   - **Feature shift test**: Shift features by +1 bar, AUC should crater

2. Wire as gates:
```python
permutation_result = permutation_y_test(X, y, cv_splitter)
if permutation_result['auc'] > 0.58:
    leak_scan_pass = False
    leak_scan_reason = "permutation_test_failed"

shift_result = feature_shift_test(X, y, cv_splitter)
if shift_result['auc'] > threshold:
    leak_scan_pass = False
    leak_scan_reason = "feature_shift_test_failed"

if not leak_scan_pass:
    # Block "PASS" status
    result.leak_scan_verdict = "LEAK_SUSPECT"
```

---

## 8. Calendar Features Classification

### Current State

**Location**: `TRAINING/utils/resolved_config.py::compute_feature_lookback_max()`

**Problem**: Calendar features like `day_of_week`, `holiday_dummy`, `trading_day_of_month` are matched by `.*day.*` pattern and assigned 1440m lookback.

**Required Fix**: Add calendar features whitelist:
```python
CALENDAR_FEATURES = {
    "day_of_week", "trading_day_of_month", "trading_day_of_quarter",
    "holiday_dummy", "pre_holiday_dummy", "post_holiday_dummy",
    "_weekday",  # if you really have this as a feature name
}

if feature_name in CALENDAR_FEATURES:
    return 0.0  # Exogenous, no lookback
```

**Status**: ❌ **NOT IMPLEMENTED**

---

## Summary of Required Changes

### Critical (Must Fix)

1. ✅ **Create `leakage_budget.py`** - Unified lookback calculator
2. ✅ **Fix calendar features** - Whitelist calendar features as 0m lookback
3. ✅ **Hard-stop on violations** - Raise exception in both strict and warn modes for critical violations
4. ✅ **Audit timing** - Ensure audit runs on final feature set (post gatekeeper + pruning)

### Important (Should Fix)

5. ✅ **Policy decision** - Add config flag for "strict" vs "drop_features" mode
6. ✅ **LeakageAssessment dataclass** - Fix contradictory reason strings
7. ✅ **CV splitter logging** - Add splitter identity + purge/embargo to run summary
8. ✅ **Smoke tests as gates** - Wire permutation + shift tests to block "PASS" status

### Implementation Priority

**Phase 1 (Tonight)**:
1. Create `leakage_budget.py` with unified calculator
2. Fix calendar features classification
3. Move audit to run after final feature set
4. Add hard-stop on violations

**Phase 2 (Next)**:
5. Add `LeakageAssessment` dataclass
6. Wire smoke tests as gates
7. Add CV splitter logging
8. Add policy config flag

---

## Files to Modify

1. **NEW**: `TRAINING/utils/leakage_budget.py` - Unified calculator
2. **MODIFY**: `TRAINING/utils/resolved_config.py` - Use unified calculator, fix calendar features
3. **MODIFY**: `TRAINING/ranking/predictability/model_evaluation.py` - Use unified calculator, add hard-stop
4. **MODIFY**: `TRAINING/utils/audit_enforcer.py` - Ensure runs on final features, hard-stop
5. **MODIFY**: `TRAINING/utils/leakage_diagnostics.py` - Add permutation + shift tests
6. **MODIFY**: `TRAINING/ranking/predictability/model_evaluation.py` - Wire smoke tests as gates
7. **MODIFY**: `TRAINING/utils/reproducibility_tracker.py` - Add CV splitter logging

---

## Definition of Done Checklist

- [ ] Run summary prints: `splitter=PurgedTimeSeriesSplit`, `purge_minutes`, `embargo_minutes`, `max_feature_lookback_minutes`, `horizon_minutes`
- [ ] Same values appear in audit + gatekeeper logs
- [ ] If purge insufficient, run either (a) hard-fails or (b) deterministically increases purge and logs `effective_purge_minutes`
- [ ] Smoke tests fail → run marked `LEAK_SUSPECT` and target ranking refuses to emit "PASS"
- [ ] Calendar features have 0m lookback (not 1440m)
- [ ] No contradictory reason strings (`overfit_likely; cv_not_suspicious`)
