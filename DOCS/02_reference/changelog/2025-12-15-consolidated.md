# Consolidated Changelog - 2025-12-15

## Summary

Multiple improvements and fixes on 2025-12-15, including metrics system rename, seed tracking fixes, feature selection improvements, and CatBoost GPU fixes.

---

## Metrics System Rename

Renamed telemetry system to metrics throughout the codebase for better branding and clarity. All functionality remains identical with full backward compatibility.

**Changes:**
- Renamed `telemetry.py` → `metrics.py`, `TelemetryWriter` → `MetricsWriter`
- Updated config section: `safety.telemetry` → `safety.metrics` (backward compatible)
- Updated file outputs: `telemetry_metrics.json` → `metrics.json`, `telemetry_drift.json` → `metrics_drift.json`, etc.
- Updated all method names: `write_cohort_telemetry()` → `write_cohort_metrics()`, etc.
- Backward compatibility: Code checks `safety.metrics.*` first, falls back to `safety.telemetry.*` for existing configs
- No breaking changes: All existing functionality preserved

**Privacy Clarification:**
- Updated all documentation to clarify metrics system tracks MODEL PERFORMANCE METRICS only (ROC-AUC, R², feature importance)
- All metrics stored locally on your infrastructure - no external data transmission
- NOT user data collection - it's model performance tracking for reproducibility

**Files Changed:**
- `TRAINING/utils/telemetry.py` → `TRAINING/utils/metrics.py` (renamed)
- `TRAINING/utils/reproducibility_tracker.py` (all references updated)
- `TRAINING/orchestration/intelligent_trainer.py` (method call updated)
- `CONFIG/pipeline/training/safety.yaml` (updated to use `metrics:` section)
- `LEGAL/PRIVACY_POLICY.md` (clarified local-only metrics)
- `LEGAL/DATA_PROCESSING_ADDENDUM.md` (clarified metrics are ML performance measurements)
- `README.md` (added note about local metrics tracking)

→ [Detailed Changelog](2025-12-15-metrics-rename.md)

---

## Seed Tracking in Metadata

Fixed missing seed field in metadata.json for target ranking and feature selection runs. Seed is now properly extracted from config (`pipeline.determinism.base_seed`) and included in all reproducibility metadata.

**Changes:**
- Seed now included in `metadata.json` for both CROSS_SECTIONAL and SYMBOL_SPECIFIC views
- Seed extracted from config and set on RunContext before logging
- Ensures full reproducibility tracking across all stages (target ranking, feature selection)
- Fixes issue where metadata.json showed `seed: null` for cross-sectional runs

**Files Changed:**
- `TRAINING/ranking/predictability/model_evaluation.py` (seed extraction and logging)
- `TRAINING/utils/reproducibility_tracker.py` (seed inclusion in metadata)

---

## Feature Selection Output Structure Refactor

Refactored feature selection output to write directly to all-caps folder structure (FEATURE_SELECTION/) instead of legacy `feature_selections/` folder.

**Changes:**
- Removed intermediate `feature_selections/` folder structure
- Output now writes directly to `REPRODUCIBILITY/FEATURE_SELECTION/` structure
- Cleaner directory organization aligned with target ranking structure
- Improved consistency across all reproducibility outputs

**Files Changed:**
- `TRAINING/ranking/feature_selector.py` (output path updates)
- `TRAINING/utils/reproducibility_tracker.py` (path handling)

---

## Model Family Name Normalization

Fixed model family name normalization for capabilities map lookup to ensure consistent family name handling across the system.

**Changes:**
- Normalized model family names for reliable capabilities map lookup
- Prevents lookup failures due to case/format inconsistencies

**Files Changed:**
- `TRAINING/ranking/predictability/model_evaluation.py` (normalization logic)

---

## Experiment Config Documentation

Made experiment configs self-contained with comprehensive documentation and clear structure.

**Changes:**
- Experiment configs now include detailed inline documentation
- Self-contained configs reduce need to cross-reference multiple files
- Improved clarity for experiment configuration

**Files Changed:**
- `CONFIG/experiments/*.yaml` (added documentation)

---

## Symbol-Specific Evaluation Fixes

Fixed indentation and evaluation loop issues for symbol-specific target ranking, enabling proper SYMBOL_SPECIFIC evaluation.

**Changes:**
- Fixed indentation bug in symbol-specific evaluation loop
- Enabled SYMBOL_SPECIFIC evaluation for classification targets
- Fixed CatBoost importance extraction for symbol-specific runs
- Improved CatBoost verbosity and feature importance snapshot generation

**Files Changed:**
- `TRAINING/ranking/predictability/model_evaluation.py` (evaluation loop fixes)

---

## CatBoost GPU Fixes

Critical fixes for CatBoost GPU mode compatibility and feature importance output.

**Changes:**
- Fixed CatBoost GPU requiring Pool objects instead of numpy arrays (automatic conversion via wrapper)
- Fixed sklearn clone compatibility for CatBoost wrapper (implements get_params/set_params)
- Fixed missing CatBoost feature importances in results directory (now saves to catboost_importances.csv)
- CatBoost GPU training now works correctly with cross-validation

**Files Changed:**
- `TRAINING/ranking/predictability/model_evaluation.py` (CatBoost wrapper and GPU handling)
- `TRAINING/utils/model_wrappers.py` (CatBoost wrapper improvements)

→ [Detailed Changelog](2025-12-15-catboost-gpu-fixes.md)

