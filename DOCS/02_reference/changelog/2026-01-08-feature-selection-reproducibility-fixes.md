# 2026-01-08: FEATURE_SELECTION Reproducibility Fixes

## Summary

Fixed critical reproducibility issues in FEATURE_SELECTION stage that prevented proper tracking, diff telemetry, and model result visibility. Issues included missing signatures in metadata, CatBoost disappearing from results, training snapshots not being validated, and duplicate cohort directories due to inconsistent config hashing.

## Issues Fixed

### 1. CatBoost Missing from Results

**Problem**: CatBoost family was disappearing from feature selection results even when enabled in config. Failed models were being filtered out during aggregation.

**Root Cause**: 
- Line 4096 in `multi_model_feature_selection.py` had `if importance is not None and importance.sum() > 0:` which excluded failed models with zero importance
- Empty importance dicts `{}` created empty Series when converted, which were filtered out during aggregation
- Failed models weren't creating `ImportanceResult` objects, so they never appeared in results

**Fix**:
- Removed `importance.sum() > 0` filter - always create `ImportanceResult` even if importance is zero
- Handle empty dicts by creating zero importance Series with proper `feature_names` index
- Ensure failed models appear in aggregation (with zero consensus score)
- Track failed models in `all_family_statuses` with proper error details

**Files Changed**:
- `TRAINING/ranking/multi_model_feature_selection.py:4096-4117` - Always create ImportanceResult, handle empty/None importance
- `TRAINING/ranking/multi_model_feature_selection.py:4317-4328` - Create ImportanceResult for failed models
- `TRAINING/ranking/multi_model_feature_selection.py:4483-4530` - Don't skip families with empty importance, create zero Series
- `TRAINING/ranking/feature_selector.py:1163-1192` - Handle empty dicts in CROSS_SECTIONAL view
- `TRAINING/ranking/feature_selector.py:873-902` - Handle empty dicts in SYMBOL_SPECIFIC view

**Impact**: CatBoost (and other failed models) now appear in results even if they failed, making failures visible for debugging. Empty importance is preserved as zero importance Series instead of being filtered out.

---

### 2. Training Snapshot Validation

**Problem**: Training snapshots were being created but failures were silent - no validation that files actually existed after creation.

**Root Cause**: 
- `create_and_save_training_snapshot()` caught exceptions and logged warnings, but didn't verify file existence
- Silent failures made it unclear if snapshots were actually created
- No validation that snapshot file exists after save operation

**Fix**:
- Added validation after `save_training_snapshot()` to verify file exists
- Improved error logging with full traceback at warning level (not just debug)
- Added file existence check in both creation and save functions
- Verify snapshot path exists even if exception occurred (might have been partially created)

**Files Changed**:
- `TRAINING/training_strategies/execution/training.py:980-1000` - Add validation and improve error handling
- `TRAINING/training_strategies/reproducibility/io.py:645-663` - Add validation that snapshot was created, verify file exists

**Impact**: Training snapshot failures are now visible and validated. Users can see if snapshots were actually created, making debugging easier.

---

### 3. Duplicate Cohort Directories

**Problem**: Multiple cohort directories existed with same `universe_sig` but different `config_hash` values, causing confusion about which cohort to use.

**Root Cause**: 
- `cs_config` dict structure varied between calls - keys were conditionally added (only if not None)
- `config_hash` computed from `cs_config` dict, so different structures produced different hashes
- Example: `{'min_cs': 10}` vs `{'min_cs': 10, 'universe_sig': None, 'leakage_filter_version': None}` produced different hashes

**Fix**:
- Normalize `cs_config` to always include all keys (even if None) for consistent hashing
- Ensure `extract_cohort_metadata()` always produces consistent structure
- Normalize `cs_config` before hashing in `reproducibility_tracker.py` to ensure all expected keys present

**Files Changed**:
- `TRAINING/orchestration/utils/cohort_metadata_extractor.py:250-259` - Always include all cs_config keys (with None if not provided)
- `TRAINING/orchestration/utils/reproducibility_tracker.py:1461-1471` - Normalize cs_config before hashing (ensure all keys present)

**Impact**: Same logical config now produces same `config_hash` regardless of None vs missing key differences. Duplicate cohort directories resolved.

---

### 4. Missing universe_sig in Metadata

**Problem**: `universe_sig` was computed but not making it into metadata that reproducibility_tracker reads.

**Root Cause**: Duplicate assignment of `universe_sig` in `_save_to_cohort()` - first assignment was overwritten.

**Fix**: Consolidated duplicate assignment into single line with proper fallback chain.

**Files Changed**:
- `TRAINING/orchestration/utils/reproducibility_tracker.py:1456-1458` - Single assignment with proper fallback

**Impact**: `universe_sig` now appears in metadata.json for FEATURE_SELECTION stage.

---

### 5. Missing snapshot.json and Diff Files

**Problem**: Cohort directories only had `metadata.json` and `metrics.json`, but were missing `snapshot.json`, `diff_prev.json`, etc.

**Root Cause**: `finalize_run()` might not be called or was failing silently for FEATURE_SELECTION stage.

**Fix**:
- Added validation after `finalize_run()` to verify required files are created
- Improved error logging from warning to error level for critical failures
- Added file existence checks for `snapshot.json` and `diff_prev.json`

**Files Changed**:
- `TRAINING/orchestration/utils/reproducibility_tracker.py:2053-2075` - Add validation after finalize_run()

**Impact**: Missing snapshot/diff files are now detected and logged prominently, making failures visible.

---

### 6. Duplicate Universe Scopes (SST Violation)

**Problem**: Two different universe scopes existed for same target/view: `universe=ALL` vs `universe=ef91e9db233a`.

**Root Cause**: Hardcoded `universe_sig="ALL"` default in cross-sectional panel methods.

**Fix**:
- Removed hardcoded "ALL" default, use SST universe signature passed as parameter
- Updated metadata to use SST universe signature consistently

**Files Changed**:
- `TRAINING/ranking/cross_sectional_feature_ranker.py:596` - Remove hardcoded "ALL" default
- `TRAINING/ranking/feature_selector.py:1837, 2168` - Use SST universe_sig parameter

**Impact**: Only one universe scope exists per target/view, maintaining SST consistency.

---

### 7. Missing Per-Model Snapshots

**Problem**: Per-model snapshots were being saved but failures were invisible (logged at debug level).

**Root Cause**: Per-model snapshot save failures were logged at debug level, making them invisible.

**Fix**: Improved error logging from debug to warning level with traceback.

**Files Changed**:
- `TRAINING/ranking/feature_selector.py:861-868` - Improve error logging
- `TRAINING/stability/feature_importance/hooks.py:227-228` - Improve error logging
- `TRAINING/stability/feature_importance/io.py:645-651` - Improve error handling

**Impact**: Per-model snapshot failures are now visible, making debugging easier.

---

### 8. Missing deterministic_config_fingerprint

**Problem**: `deterministic_config_fingerprint` was null in `fs_snapshot_index.json` even though `config.resolved.json` existed.

**Root Cause**: Code tried to load from `output_dir / "globals" / "config.resolved.json"` but `output_dir` might not be run root.

**Fix**: Modified path resolution to walk up directory tree to find run root before attempting to load config.

**Files Changed**:
- `TRAINING/stability/feature_importance/schema.py:520-532` - Walk up to find run root

**Impact**: `deterministic_config_fingerprint` now populated correctly in FS snapshots.

---

## Additional Improvements

- **Empty Importance Dict Handling**: Empty `{}` dicts now create zero importance Series instead of being filtered out
- **Failed Model Visibility**: Failed models appear in results with zero importance, making failures visible
- **Error Logging**: Improved error logging throughout (debug → warning level for critical failures)
- **File Validation**: Added validation that files were actually created after save operations
- **Config Normalization**: Consistent `cs_config` structure prevents hash differences

## Testing

After fixes, verify:
- CatBoost appears in results (even if with zero importance if it failed)
- Training snapshots are created and validated in cohort directories
- Only one universe scope exists per target/view (not both `universe=ALL` and actual hash)
- Per-model snapshots exist in directory structure (lightgbm, xgboost, etc., not just multi_model_aggregated)
- Duplicate cohort directories are resolved (same config produces same config_hash)
- `deterministic_config_fingerprint` is populated in fs_snapshot_index.json
- `snapshot.json`, `diff_prev.json`, etc. exist in cohort directories

## Documentation Updates

- **NEW**: Created `DOCS/03_technical/implementation/FEATURE_SELECTION_SNAPSHOTS.md` - Comprehensive guide explaining which snapshots exist, their purposes, and which one to use
- **Updated**: `FEATURE_SELECTION_GUIDE.md` - Added snapshot structure section with link to detailed docs
- **Updated**: `REPRODUCIBILITY_STRUCTURE.md` - Added snapshot directory structure explanation
- **Updated**: `DOCS/INDEX.md` - Added link to new snapshot documentation

## Cohort Directory Consolidation

**Issue**: Two cohort directories existed for FEATURE_SELECTION stage (main feature selection and cross-sectional panel), creating confusion and duplicate outputs.

**Fix**: 
- Main feature selection passes `cohort_id` to cross-sectional panel computation
- Cross-sectional panel writes `metrics_cs_panel.json` to existing cohort directory instead of creating new one
- Both metrics files (`metrics.json` and `metrics_cs_panel.json`) now in same cohort directory
- Added `find_cohort_dir_by_id()` helper function to locate cohort directories

**Files Modified**:
- `TRAINING/ranking/feature_selector.py` - Extract and pass `cohort_id` to CS panel, fix null pointer checks for `audit_result`
- `TRAINING/ranking/cross_sectional_feature_ranker.py` - Accept `cohort_id`, write `metrics_cs_panel.json` instead of calling `log_run()`, fix null pointer check for `cohort_dir`
- `TRAINING/orchestration/utils/target_first_paths.py` - Add `find_cohort_dir_by_id()` helper

**Bug Fixes**:
- Fixed `AttributeError` when `cohort_dir` is `None` - now checks `if cohort_dir and cohort_dir.exists()` before accessing
- Fixed potential `AttributeError` when `audit_result` is `None` - now checks `if audit_result` before accessing
- Improved error messages to distinguish between `cohort_dir is None` vs `cohort_dir doesn't exist`

**Backward Compatibility**: If `cohort_id` is not provided, CS panel falls back to creating its own cohort (legacy behavior).

## Backward Compatibility

All fixes maintain backward compatibility:
- Empty importance dicts still work (now create zero Series instead of being filtered)
- Failed models still tracked (now visible in results instead of silently excluded)
- Config normalization doesn't change existing behavior (just ensures consistency)
- File validation doesn't break existing code (only adds checks)
- Cohort consolidation: CS panel falls back to creating own cohort if `cohort_id` not provided
