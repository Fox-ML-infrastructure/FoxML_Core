# Training Pipeline Audit Fixes - 2025-12-17

## Overview

This changelog documents critical fixes to training pipeline contract breaks across: family IDs, routing, plan consumption, feature schema, and counting/tracking. These were not isolated bugs but systemic inconsistencies that broke the training pipeline's correctness guarantees.

**Branch:** `fix/training-pipeline-audit-fixes`

---

## Critical Fixes

### 1) Family Normalization / Registry Mismatch ✅

**Problem:** `XGBoost` → `x_g_boost` normalization caused registry/policy mismatches, leading to fallback behavior and incorrect GPU policy decisions.

**Solution:**
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

**Solution:**
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

**Solution:**
- Defined `_pkg_ver` function **before** conditional blocks in both save paths (lines 511 and 1064)
- Ensured function is in scope for all code paths

**Files Changed:**
- `TRAINING/training_strategies/training.py` (2 locations)

**Outcome:** No more `_pkg_ver` runtime crashes. LightGBM models save metadata correctly.

---

### 4) Preflight Filtering Applied to ALL Routes ✅

**Problem:** Preflight validation only ran for CROSS_SECTIONAL path, so SYMBOL_SPECIFIC path attempted invalid families (random_forest, catboost, etc.) and failed.

**Solution:**
- Moved preflight validation to run **before** both CROSS_SECTIONAL and SYMBOL_SPECIFIC paths (line 314)
- SYMBOL_SPECIFIC path now uses `validated_families` (not raw `target_families`)
- Enhanced preflight logging to distinguish feature selectors from invalid families

**Files Changed:**
- `TRAINING/training_strategies/training.py`

**Outcome:** Unregistered families are filtered out everywhere. No more "Family 'random_forest' not found" errors in symbol-specific training.

---

### 5) Router Default for Swing Targets ✅

**Problem:** `y_will_swing_*` targets were routing to regression by default, but they are binary classification targets (0/1 labels).

**Solution:**
- Added explicit pattern: `y_will_swing_(high|low)_*` → binary classification in `TRAINING/target_router.py`
- Pattern routes to `TaskSpec('binary', 'binary', ['roc_auc', 'log_loss'], label_type='int32')`

**Files Changed:**
- `TRAINING/target_router.py`

**Outcome:** Swing targets route to binary classification, not regression.

---

### 6) Training Plan "0 Jobs" Now Hard Error ✅

**Problem:** Training plan with 0 jobs was logged as warning but execution continued, ignoring the plan.

**Solution:**
- Changed from warning to **error** in `TRAINING/orchestration/training_plan_consumer.py`
- Added explicit error message explaining this is a logic error

**Files Changed:**
- `TRAINING/orchestration/training_plan_consumer.py`

**Outcome:** "Plan empty but still runs everything" is now impossible. Run stops immediately if plan has 0 jobs.

---

### 7) Routing/Plan Integration — Respect CS: DISABLED ✅

**Problem:** Training ignored routing plan's `cross_sectional.route == "DISABLED"` status and defaulted to CROSS_SECTIONAL.

**Solution:**
- Added check for `cross_sectional.route == "DISABLED"` in both feature selection and training paths
- Skip CS training/feature selection if DISABLED, with clear logging

**Files Changed:**
- `TRAINING/training_strategies/training.py`
- `TRAINING/orchestration/intelligent_trainer.py`

**Outcome:** Training respects routing plan DISABLED status. No more "CS: DISABLED (UNKNOWN)" but still running CS training.

---

### 8) Routing Decision Count Mismatch Detection ✅

**Problem:** Routing decisions count (15) didn't match filtered targets count (9), indicating duplication or stale artifacts.

**Solution:**
- Added validation logging to detect count mismatches
- Log route summary (CROSS_SECTIONAL, SYMBOL_SPECIFIC, etc.) for debugging
- Warn if routing decision count != filtered targets count

**Files Changed:**
- `TRAINING/orchestration/intelligent_trainer.py` (2 locations)

**Outcome:** Mismatches are detected and logged with actionable warnings.

---

### 9) Symbol-Specific Route — All Eligible Symbols ✅

**Problem:** SYMBOL_SPECIFIC route only trained NVDA when 5 symbols were available, indicating filtering bug.

**Solution:**
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

**Solution:**
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

## Files Changed Summary

**Core Fixes:**
- `TRAINING/utils/sst_contract.py` - Family normalization, tracker adapter
- `TRAINING/common/runtime_policy.py` - Use SST normalization
- `TRAINING/training_strategies/utils.py` - Delegate to SST contract
- `TRAINING/training_strategies/training.py` - Preflight, routing, repro tracking, _pkg_ver
- `TRAINING/target_router.py` - Swing target pattern
- `TRAINING/orchestration/training_plan_consumer.py` - 0 jobs error
- `TRAINING/orchestration/intelligent_trainer.py` - Routing plan consumption, symbol-specific
- `TRAINING/training_strategies/data_preparation.py` - Feature pipeline threshold & diagnostics
- `TRAINING/utils/reproducibility_tracker.py` - Defensive string/Enum handling

**Documentation:**
- `docs/audit/fix-training-pipeline-audit-fixes.md` - Consolidated summary with verification checklist
- `scripts/verify_training_pipeline_fixes.sh` - Automated verification script

---

## Verification

See `docs/audit/fix-training-pipeline-audit-fixes.md` for complete verification checklist.

**Quick Verification:**
```bash
# Should be ZERO after fixes
grep -R "object has no attribute 'name'" logs/ || echo "✅ No .name errors"
grep -R "Unknown family 'x_g_boost'" logs/ || echo "✅ No x_g_boost errors"
grep -R "TRAINER_MODULE_MAP, using fallback" logs/ || echo "✅ No fallback warnings"
grep -R "Total jobs: 0" logs/ || echo "✅ No 0-job plans (should error)"
```

---

## Impact

**Before:** Training pipeline had multiple contract breaks causing:
- Family registry mismatches (XGBoost → x_g_boost)
- Reproducibility tracking crashes
- Invalid families attempted in symbol-specific training
- Routing plan ignored (CS: DISABLED still trained)
- Feature pipeline false positives
- Symbol-specific training only used one symbol

**After:** All contract breaks fixed:
- Consistent family IDs everywhere
- Reproducibility tracking works correctly
- Invalid families properly skipped
- Routing plan respected
- Feature pipeline has correct threshold and diagnostics
- All eligible symbols trained in symbol-specific route

---

### 11) Diff Telemetry Severity Logic Fix ✅

**Problem:** Severity was incorrectly set to "minor" when `total_changes=0`, `changed_keys=[]`, `patch=[]`, and `metric_deltas={}`. The bug was that `all()` on an empty list returns `True`, causing the "metrics only" check to incorrectly return MINOR.

**Solution:**
- Made severity **purely derived** from the report (SST-style)
- Added `severity_reason` field to explain why severity was set
- Fixed logic: `total_changes==0` and `metric_deltas_count==0` → `severity="none"` (not "minor")
- `comparable=false` → `severity="critical"` with `severity_reason`
- Only excluded factors changed → `severity="minor"` with `excluded_factors_summary`
- Output/metric changes → `severity="major"`
- Only input/process changes → `severity="minor"`

**Files Changed:**
- `TRAINING/utils/diff_telemetry.py` (`_determine_severity`, `DiffResult` dataclass, `compute_diff`)

**Outcome:** Severity is now consistent with the actual changes. No more "minor" severity when nothing changed. `severity_reason` explains every severity assignment.

---

### 12) Output Digests for Full Determinism Verification ✅

**Problem:** The diff telemetry system could verify that inputs/process were identical across reruns, but couldn't prove that outputs/metrics/artifacts were also identical. This left a gap in the full determinism claim.

**Solution:**
- Added three output digest fields to `NormalizedSnapshot`:
  - `metrics_sha256`: SHA256 of metrics dict (proves metric determinism)
  - `artifacts_manifest_sha256`: SHA256 of artifacts manifest (proves artifact determinism)
  - `predictions_sha256`: SHA256 of predictions (if available, proves prediction determinism)
- Implemented digest computation:
  - `_compute_metrics_digest()`: Hashes normalized metrics dict from outputs/resolved_metadata
  - `_compute_artifacts_manifest_digest()`: Creates manifest of artifact files (feature_importances.parquet, etc.) with sizes/mtimes, hashes manifest
  - `_compute_predictions_digest()`: Hashes predictions files if they exist (with size optimization for large files)
- Updated diff logic to check output digests:
  - If any digest differs → `severity="critical"` with reason "non-determinism detected"
  - Added `output_digest_changes` to diff summary for visibility
- Digest computation happens during snapshot normalization (uses `cohort_dir` to scan for artifacts)

**Files Changed:**
- `TRAINING/utils/diff_telemetry.py` (`NormalizedSnapshot`, `normalize_snapshot`, `compute_diff`, `_determine_severity`, digest computation methods)

**Outcome:** Full determinism verification - can now prove that outputs/metrics/artifacts are identical across reruns with same inputs/process. Non-deterministic outputs are flagged as CRITICAL severity.

---

## Related Documentation

- [Consolidated Fix Summary](../../../docs/audit/fix-training-pipeline-audit-fixes.md)
- [Verification Script](../../../scripts/verify_training_pipeline_fixes.sh)

