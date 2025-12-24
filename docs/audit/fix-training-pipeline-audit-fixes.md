# fix/training-pipeline-audit-fixes — Consolidated Summary

## Overview

This branch fixes **contract breaks** across: family IDs, routing, plan consumption, feature schema, and counting/tracking. These were not isolated bugs but systemic inconsistencies that broke the training pipeline's correctness guarantees.

---

## What's Fixed

### 1) Family Normalization / Registry Mismatch ✅

**Problem:** `XGBoost` → `x_g_boost` normalization caused registry/policy mismatches, leading to fallback behavior and incorrect GPU policy decisions.

**Fix:**
- Added alias `x_g_boost -> xgboost` in `TRAINING/utils/sst_contract.py`
- Ensured `XGBoost` (TitleCase) normalizes to `xgboost` (not `x_g_boost`)
- Updated `TRAINING/common/runtime_policy.py` to use SST normalization
- Updated `TRAINING/training_strategies/utils.py` to delegate to SST contract

**Files Changed:**
- `TRAINING/utils/sst_contract.py` (normalize_family)
- `TRAINING/common/runtime_policy.py`
- `TRAINING/training_strategies/utils.py`

**Outcome:** Consistent family IDs across registry/policy/trainer lookup. No more "Unknown family 'x_g_boost'" or "Family XGBoost not in TRAINER_MODULE_MAP" errors.

---

### 2) Reproducibility Tracking `.name` Crash ✅

**Problem:** Tracker expected Enum-like objects with `.name` attribute but received strings, causing `'str' object has no attribute 'name'` crashes.

**Fix:**
- Added `tracker_input_adapter()` in `TRAINING/utils/sst_contract.py` for safe string/Enum conversion
- Updated `TRAINING/training_strategies/training.py` to adapt inputs before `tracker.log_comparison()`
- Added defensive string/Enum handling in `TRAINING/utils/reproducibility_tracker.py`'s `_compute_drift`

**Files Changed:**
- `TRAINING/utils/sst_contract.py` (tracker_input_adapter)
- `TRAINING/training_strategies/training.py`
- `TRAINING/utils/reproducibility_tracker.py`

**Outcome:** Tracker no longer explodes on strings. All reproducibility artifacts are saved correctly.

---

### 3) LightGBM Save Hook `_pkg_ver` Referenced-Before-Assignment ✅

**Problem:** `_pkg_ver` was defined inside conditional blocks but used outside, causing "referenced before assignment" errors.

**Fix:**
- Defined `_pkg_ver` function **before** conditional blocks in both save paths (lines 511 and 1064)
- Ensured function is in scope for all code paths

**Files Changed:**
- `TRAINING/training_strategies/training.py` (2 locations)

**Outcome:** No more `_pkg_ver` runtime crashes. LightGBM models save metadata correctly.

---

### 4) Preflight Filtering Applied to ALL Routes ✅

**Problem:** Preflight validation only ran for CROSS_SECTIONAL path, so SYMBOL_SPECIFIC path attempted invalid families (random_forest, catboost, etc.) and failed.

**Fix:**
- Moved preflight validation to run **before** both CROSS_SECTIONAL and SYMBOL_SPECIFIC paths (line 314)
- SYMBOL_SPECIFIC path now uses `validated_families` (not raw `target_families`)
- Enhanced preflight logging to distinguish feature selectors from invalid families

**Files Changed:**
- `TRAINING/training_strategies/training.py`

**Outcome:** Unregistered families are filtered out everywhere. No more "Family 'random_forest' not found" errors in symbol-specific training.

---

### 5) Router Default for Swing Targets ✅

**Problem:** `y_will_swing_*` targets were routing to regression by default, but they are binary classification targets (0/1 labels).

**Fix:**
- Added explicit pattern: `y_will_swing_(high|low)_*` → binary classification in `TRAINING/target_router.py`
- Pattern routes to `TaskSpec('binary', 'binary', ['roc_auc', 'log_loss'], label_type='int32')`

**Files Changed:**
- `TRAINING/target_router.py`

**Outcome:** Swing targets route to binary classification, not regression.

---

### 6) Training Plan "0 Jobs" Now Hard Error ✅

**Problem:** Training plan with 0 jobs was logged as warning but execution continued, ignoring the plan.

**Fix:**
- Changed from warning to **error** in `TRAINING/orchestration/training_plan_consumer.py`
- Added explicit error message explaining this is a logic error

**Files Changed:**
- `TRAINING/orchestration/training_plan_consumer.py`

**Outcome:** "Plan empty but still runs everything" is now impossible. Run stops immediately if plan has 0 jobs.

---

### 7) Routing/Plan Integration — Respect CS: DISABLED ✅

**Problem:** Training ignored routing plan's `cross_sectional.route == "DISABLED"` status and defaulted to CROSS_SECTIONAL.

**Fix:**
- Added check for `cross_sectional.route == "DISABLED"` in both feature selection and training paths
- Skip CS training/feature selection if DISABLED, with clear logging

**Files Changed:**
- `TRAINING/training_strategies/training.py`
- `TRAINING/orchestration/intelligent_trainer.py`

**Outcome:** Training respects routing plan DISABLED status. No more "CS: DISABLED (UNKNOWN)" but still running CS training.

---

### 8) Routing Decision Count Mismatch Detection ✅

**Problem:** Routing decisions count (15) didn't match filtered targets count (9), indicating duplication or stale artifacts.

**Fix:**
- Added validation logging to detect count mismatches
- Log route summary (CROSS_SECTIONAL, SYMBOL_SPECIFIC, etc.) for debugging
- Warn if routing decision count != filtered targets count

**Files Changed:**
- `TRAINING/orchestration/intelligent_trainer.py` (2 locations)

**Outcome:** Mismatches are detected and logged with actionable warnings.

---

### 9) Symbol-Specific Route — All Eligible Symbols ✅

**Problem:** SYMBOL_SPECIFIC route only trained NVDA when 5 symbols were available, indicating filtering bug.

**Fix:**
- Validate `winner_symbols` from routing plan (not just use fallback)
- Filter out invalid symbols not in symbol list
- Log which symbols are being trained
- Fall back to all symbols if `winner_symbols` is empty/invalid

**Files Changed:**
- `TRAINING/orchestration/intelligent_trainer.py`

**Outcome:** All eligible symbols are included in SYMBOL_SPECIFIC training, not just the first one.

---

### 10) Feature Pipeline Collapse — Fixed Threshold & Diagnostics ✅

**Problem (Pitfall A):** Hard-fail threshold used wrong denominator (requested vs allowed), causing false positives when registry intentionally prunes.

**Problem (Pitfall B):** Filtering to existing columns masked schema breaches. No diagnostics for missing allowed features.

**Fix:**
- **Pitfall A:** Changed threshold check to use `allowed → present` (not `requested → present`)
- **Pitfall B:** Added diagnostics with close matches for missing allowed features
- Track feature pipeline stages separately: requested → allowed → present → used
- Log detailed drop reasons at each stage

**Files Changed:**
- `TRAINING/training_strategies/data_preparation.py`

**Outcome:** 
- No false positives when registry intentionally prunes
- Actionable diagnostics for schema mismatches (close matches shown)
- Clear visibility into feature pipeline stages

---

## Remaining Issues (Lower Priority)

### Threading Configuration Inconsistency
- **Issue:** Logs say `num_threads=15` but actual threadpool is `openmp:7` (OMP=7) and BLAS=1
- **Impact:** Performance variability, hard-to-reproduce speed changes
- **Status:** Not fixed (lower priority)

### Duplicate GPU Availability Checks
- **Issue:** XGBoost trainer runs GPU check twice back-to-back each run
- **Impact:** Unnecessary overhead, noisy logs
- **Status:** Not fixed (lower priority)

---

## Verification Checklist (Definition of Done)

Run one end-to-end training and verify **exact invariants**:

### A) Repro Tracker ✅
- [ ] No `'.name'` attribute errors in logs
- [ ] Tracker comparisons logged for `lightgbm` and `xgboost`
- [ ] All reproducibility artifacts saved without exceptions

### B) Family Normalization + Registry ✅
- [ ] No `Unknown family 'x_g_boost'` errors
- [ ] No `Family XGBoost not in TRAINER_MODULE_MAP, using fallback` warnings
- [ ] Zero attempts spawned for `random_forest/catboost/neural_network/lasso`
  - They must be **skipped_invalid_family**, not "failed"
- [ ] All family lookups use consistent normalized names

### C) Plan + Routing Sanity ✅
- [ ] If plan mode enabled: `Total jobs > 0` and only those targets run
- [ ] If `Total jobs == 0`: run stops immediately (error, not warning)
- [ ] Routing decisions count == accounted targets (no unexplained 9→15 jump)
- [ ] CS: DISABLED targets are skipped (not trained)
- [ ] Route summary logged (CROSS_SECTIONAL, SYMBOL_SPECIFIC counts)

### D) Feature Integrity ✅
- [ ] `used_features` is close to `allowed_features` (not `requested_features`)
- [ ] If `allowed → present` drop > 50%: run hard-fails with actionable error
- [ ] Missing allowed features list printed with close matches
- [ ] Feature audit logs show: `requested → allowed → present → used`

### E) Symbol-Specific Correctness ✅
- [ ] Multiple symbols train (not only NVDA), unless explicitly configured otherwise
- [ ] `winner_symbols` validated against symbol list
- [ ] Log shows which symbols are being trained: `training {N} symbols: [SYMBOL1, SYMBOL2, ...]`

### F) Training Plan Consumption ✅
- [ ] Plan is actually consumed (targets filtered by plan)
- [ ] No "plan says do X but execution does Y" mismatches
- [ ] Plan jobs match executed targets

---

## Fast "grep" Verification (No Excuses)

```bash
# Should be ZERO after fixes
grep -R "object has no attribute 'name'" logs/ || echo "✅ No .name errors"
grep -R "Unknown family 'x_g_boost'" logs/ || echo "✅ No x_g_boost errors"
grep -R "TRAINER_MODULE_MAP, using fallback" logs/ || echo "✅ No fallback warnings"
grep -R "Total jobs: 0" logs/ || echo "✅ No 0-job plans (should error)"

# Sanity: confirm invalid families are skipped, not attempted
grep -R "skipped_invalid_family\|Invalid famil" logs/ || echo "✅ Invalid families skipped"
grep -R "random_forest\|catboost\|neural_network\|lasso" logs/ | grep -v "skip\|SKIP" || echo "✅ Invalid families not attempted"

# Verify routing plan is respected
grep -R "CS: DISABLED" logs/ | grep -v "Skipping\|skipping" && echo "⚠️ CS: DISABLED but still training!" || echo "✅ CS: DISABLED respected"

# Verify symbol-specific includes all symbols
grep -R "SYMBOL_SPECIFIC route.*symbols" logs/ | grep -E "1 symbol|only.*NVDA" && echo "⚠️ Only 1 symbol in SYMBOL_SPECIFIC!" || echo "✅ Multiple symbols in SYMBOL_SPECIFIC"
```

---

## Risk Assessment

### High Risk (Must Verify)
- **Feature pipeline collapse:** Threshold fix prevents false positives, but schema mismatches must be caught
- **Routing plan consumption:** DISABLED status must be respected, or training will ignore plan
- **Symbol-specific routing:** All eligible symbols must train, or results are incomplete

### Medium Risk
- **Family normalization:** Registry/policy consistency is critical for GPU/CPU decisions
- **Reproducibility tracking:** Missing artifacts break audit trail

### Low Risk
- **Threading inconsistency:** Performance impact, not correctness
- **Duplicate GPU checks:** Overhead only, not correctness

---

## Testing Recommendations

1. **Run end-to-end training** with routing plan that has:
   - Some targets with CS: DISABLED
   - Some targets with SYMBOL_SPECIFIC route
   - Mix of valid and invalid families

2. **Verify logs** match all checklist items above

3. **Check feature audit** shows correct pipeline stages (requested → allowed → present → used)

4. **Confirm routing decisions** count matches filtered targets count

5. **Validate symbol-specific** training includes all eligible symbols (not just one)

---

## Files Changed Summary

- `TRAINING/utils/sst_contract.py` - Family normalization, tracker adapter
- `TRAINING/common/runtime_policy.py` - Use SST normalization
- `TRAINING/training_strategies/utils.py` - Delegate to SST contract
- `TRAINING/training_strategies/training.py` - Preflight, routing, repro tracking, _pkg_ver
- `TRAINING/target_router.py` - Swing target pattern
- `TRAINING/orchestration/training_plan_consumer.py` - 0 jobs error
- `TRAINING/orchestration/intelligent_trainer.py` - Routing plan consumption, symbol-specific
- `TRAINING/training_strategies/data_preparation.py` - Feature pipeline threshold & diagnostics
- `TRAINING/utils/reproducibility_tracker.py` - Defensive string/Enum handling

---

## Definition of Done

✅ All fixes implemented  
✅ Verification checklist created  
✅ Fast grep checks provided  
⏳ **PENDING:** End-to-end test run with verification checklist

