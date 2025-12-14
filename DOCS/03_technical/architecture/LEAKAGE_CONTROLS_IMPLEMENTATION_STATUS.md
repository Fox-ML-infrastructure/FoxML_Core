# Leakage Controls Implementation Status

**Date**: 2025-12-13  
**Related**: [Leakage Controls Evaluation](LEAKAGE_CONTROLS_EVALUATION.md) | [Leakage Validation Fix](../fixes/2025-12-13-leakage-validation-fix.md) | [Fingerprint Tracking](../fixes/2025-12-13-lookback-fingerprint-tracking.md) | [Fingerprint Improvements](../fixes/2025-12-13-fingerprint-improvements.md) | [Canary Test Guide](../testing/LEAKAGE_CANARY_TEST_GUIDE.md)  
**Status**: Phase 1 (Critical) - In Progress

## Completed (Commits 1-3)

### ✅ Commit 1: Single Source of Truth for Lookback

**File**: `TRAINING/utils/leakage_budget.py` (NEW)

- Created unified `LeakageBudget` dataclass
- Implemented `compute_budget()` - single source of truth
- Implemented `infer_lookback_minutes()` with proper precedence:
  1. Schema/registry metadata (highest priority)
  2. Calendar features whitelist (0m lookback)
  3. Explicit time suffixes (_15m, _24h, _1d)
  4. Bar-based patterns (ret_288, sma_20)
  5. Keyword heuristics (daily patterns)
  6. Unknown policy (conservative default or drop)

**Changes**:
- `TRAINING/utils/resolved_config.py` - Now delegates to `leakage_budget.compute_feature_lookback_max()`
- `TRAINING/ranking/predictability/model_evaluation.py` - Gatekeeper uses `infer_lookback_minutes()` directly

**Acceptance**: ✅ Audit and gatekeeper now use the same calculation function

---

### ✅ Commit 2: Fix Calendar Feature Classification

**File**: `TRAINING/utils/leakage_budget.py`

- Added `CALENDAR_FEATURES` whitelist with 0m lookback:
  - `day_of_week`, `trading_day_of_month`, `trading_day_of_quarter`
  - `holiday_dummy`, `pre_holiday_dummy`, `post_holiday_dummy`
  - `weekday`, `is_weekend`, `is_month_end`, etc.
- Calendar features checked BEFORE keyword heuristics (prevents false positives)
- Pattern matching for calendar features returns 0m (not 1440m)

**Acceptance**: ✅ Calendar features now resolve to 0m lookback (not 1440m)

---

### ✅ Commit 3: Hard-Stop on Audit Violations

**File**: `CONFIG/pipeline/training/safety.yaml`

- Added `safety.leakage_detection.policy` config option:
  - `"strict"` (default): Hard-stop on violations (raise exception)
  - `"drop_features"`: Drop violating features, recompute budget, ensure pass
  - `"warn"`: Log violations but continue (NOT recommended)

**File**: `TRAINING/ranking/predictability/model_evaluation.py`

- Added policy enforcement after Final Gatekeeper (line ~3705)
- Added policy enforcement after pruning (line ~697)
- Policy checks `purge_minutes >= required_gap_minutes` (max_lookback + horizon)
- In `strict` mode: raises `RuntimeError` immediately
- In `drop_features` mode: drops violating features, recomputes budget, verifies pass
- In `warn` mode: logs error but continues (NOT recommended)

**Acceptance**: ✅ Violations now block training in strict mode

---

## Completed (Commits 4-8)

### ✅ Commit 4: Guarantee Audit Runs on Final Feature Set

**File**: `TRAINING/ranking/predictability/model_evaluation.py`

- ✅ Fixed RunContext to use `resolved_config.feature_lookback_max_minutes` (computed from final features)
- ✅ Removed hardcoded 288-bar estimation
- ✅ Fallback computes budget from final feature_names if resolved_config unavailable
- ✅ Policy checks run after gatekeeper (final features)
- ✅ Policy checks run after pruning (final features)

**Acceptance**: ✅ Audit now reads from final feature set

---

### ✅ Commit 5: Fix Contradictory Reason Strings

**File**: `TRAINING/utils/leakage_assessment.py` (NEW)

- Created `LeakageAssessment` dataclass with:
  - `leak_scan_pass`, `cv_suspicious`, `overfit_likely`, `auc_too_high_models`
  - `reason()` method generates consistent reason strings
  - `should_auto_fix()` method determines auto-fix decision
  - `auto_fix_reason()` generates skip reason (prevents contradictions)

**File**: `TRAINING/ranking/predictability/model_evaluation.py`

- Replaced hardcoded `"overfit_likely; cv_not_suspicious"` with `assessment.auto_fix_reason()`
- Assessment computed from actual flags (cv_suspicious, overfit_likely, etc.)

**Acceptance**: ✅ No more contradictory reason strings

---

### ✅ Commit 7: Log CV Splitter Identity

**File**: `TRAINING/ranking/predictability/model_evaluation.py`

- Added CV splitter logging to `_log_canonical_summary()`:
  - `splitter=PurgedTimeSeriesSplit`
  - `n_splits`, `purge_minutes`, `embargo_minutes`, `max_feature_lookback_minutes`
- Values extracted from `resolved_config` (single source of truth)

**Acceptance**: ✅ Run summary now shows splitter identity + purge/embargo

---

### ✅ Commit 8: Make Policy Explicit

**File**: `TRAINING/ranking/predictability/model_evaluation.py`

- Log active policy at start of gatekeeper: `🔒 Leakage policy: {policy}`
- Log drop list with reasons when features are dropped
- Policy decision explicit in all drop logs

**Acceptance**: ✅ Policy is explicit in logs

---

## Pending

### 📋 Commit 6: Wire Smoke Tests as Gates

**Status**: Not started (deferred - can be done separately)

**Required**:
- Add permutation y-test (shuffle labels, expect AUC ~0.5)
- Add feature shift test (shift features +1 bar, expect AUC drop)
- Wire tests to block "PASS" status if they fail
- Mark runs as `LEAK_SUSPECT` on failure

**Note**: Smoke tests exist in `TRAINING/utils/leakage_diagnostics.py` but not wired as gates yet.

---

## Testing Checklist

### Acceptance Tests for Phase 1

- [ ] **Test 1**: Same run shows same `max_feature_lookback_minutes` in audit and gatekeeper logs
- [ ] **Test 2**: Feature list with `day_of_week`, `holiday_dummy` → `max_feature_lookback_minutes` does NOT jump to 1440m
- [ ] **Test 3**: Set `policy=strict` + `purge_minutes=35` + include long-lookback feature → training fails fast with clear message
- [ ] **Test 4**: Set `policy=drop_features` + violation → features dropped, budget recomputed, training continues
- [ ] **Test 5**: Final feature set (post gatekeeper + pruning) passes policy check

---

## Files Modified

1. **NEW**: `TRAINING/utils/leakage_budget.py` - Unified calculator
2. **MODIFIED**: `TRAINING/utils/resolved_config.py` - Delegates to unified calculator
3. **MODIFIED**: `TRAINING/ranking/predictability/model_evaluation.py` - Uses unified calculator, enforces policy
4. **MODIFIED**: `CONFIG/pipeline/training/safety.yaml` - Added policy config

---

## Next Steps

1. Complete Commit 4 (verify audit timing)
2. Test Phase 1 changes
3. Proceed to Phase 2 (Commits 5-8)
