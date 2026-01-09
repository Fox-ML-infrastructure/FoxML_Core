# 2026-01-08: File Overwrite and Plan Creation Fixes

## Summary

Fixed critical bugs causing data loss in `globals/` directory files and missing routing/training plan creation. Files were being overwritten instead of preserving aggregated data, and plan hashes in manifest were null because plans weren't being created or saved correctly.

## Issues Fixed

### 1. `run_context.json` - Stage History Lost

**Problem**: `save_run_context()` was building a new context dictionary that didn't preserve `current_stage` and `stage_history` from existing context. When called after `save_stage_transition()`, it overwrote the file and lost stage transition history.

**Root Cause**: `save_run_context()` (line 440) constructed a new dict without explicitly preserving stage fields from `existing_context`.

**Fix**: Modified `save_run_context()` to explicitly preserve `current_stage` and `stage_history` from `existing_context`. These fields are now only updated by `save_stage_transition()`, not overwritten by `save_run_context()`.

**Files Changed**:
- `TRAINING/orchestration/utils/run_context.py:440-456`

**Impact**: Stage transition history is now preserved across multiple `save_run_context()` calls. Full pipeline stage progression is tracked correctly.

---

### 2. `run_hash.json` - Not Created or Missing Data

**Problem**: `run_hash.json` was not being created or was missing data due to multiple issues:
- `compute_full_run_hash()` returned `None` when snapshot indices were missing
- Previous run lookup was broken (searched in current run's directory instead of previous runs)
- Silent failures (exceptions caught and logged but not surfaced)
- Missing config fingerprints caused snapshots to be skipped

**Root Causes**:
1. **Missing snapshot indices**: If `snapshot_index.json`, `fs_snapshot_index.json`, or `training_snapshot_index.json` were missing, computation returned `None` without clear error messages
2. **Previous run lookup bug**: Both `intelligent_trainer.py` (line 3952) and `compute_run_hash_with_changes()` (line 5470) incorrectly looked for previous run's `run_hash.json` in the current run's `globals/` directory
3. **Silent failures**: Exceptions were caught and logged at debug level, making failures invisible
4. **Missing fingerprints**: Snapshots without `config_fingerprint` or `deterministic_config_fingerprint` were skipped without clear logging

**Fixes**:
- **Improved error logging**: Added detailed warnings when snapshot indices are missing, which snapshots are skipped, and why computation returns `None`
- **Fixed previous run lookup**: Now searches parent/sibling directories (`RESULTS/runs/cg-*/intelligent_output_*/globals/run_hash.json`) instead of current run's directory
- **Better validation**: Logs which snapshots are missing fingerprints before skipping them

**Files Changed**:
- `TRAINING/orchestration/intelligent_trainer.py:3952-3963` (previous run lookup)
- `TRAINING/orchestration/utils/diff_telemetry.py:5315-5327` (missing indices logging)
- `TRAINING/orchestration/utils/diff_telemetry.py:5340-5347` (missing fingerprint logging)
- `TRAINING/orchestration/utils/diff_telemetry.py:5470-5513` (previous run hash lookup)
- `TRAINING/orchestration/utils/diff_telemetry.py:5620-5625` (computation failure logging)

**Impact**: Run hash is now created successfully, previous runs are found correctly for change detection, and errors are visible (not hidden at debug level).

---

### 3. Routing and Training Plan Creation - Plans Not Saved

**Problem**: Routing and training plan files were not being created in `globals/`, causing `plan_hashes` in manifest.json to be null:
- `globals/routing_plan/routing_plan.json` - missing
- `globals/training_plan/master_training_plan.json` - missing

**Root Causes**:
1. **Silent failure**: Exception caught at line 2454 in `intelligent_trainer.py` with `logger.debug()` - failures were invisible
2. **Path resolution issue**: `run_root(output_dir)` might not find correct directory if `output_dir` is not the run root
3. **Conditional execution**: Function only called if `target_features and not training_plan_dir` - might be skipped
4. **Missing error propagation**: If routing plan generation failed, training plan generation was also skipped (nested in same try block)
5. **Manifest created too early**: `_compute_plan_hashes()` was called in `create_manifest()` which may be called before plans are generated

**Fixes**:
- **Improved error logging**: Changed exception handler from `logger.debug()` to `logger.warning()` so failures are visible
- **Added plan save verification**: After saving routing/training plans, verify files exist before continuing
- **Better path validation**: Log resolved paths before saving plans, verify `globals_dir` exists and is writable
- **Separated error handling**: Routing plan and training plan generation now have separate error handling
- **Manifest update after plans**: Added `update_manifest_with_plan_hashes()` function and call it after plans are created

**Files Changed**:
- `TRAINING/orchestration/intelligent_trainer.py:2454-2465` (error logging, plan verification, manifest update)
- `TRAINING/orchestration/routing_integration.py:161-164` (routing plan verification)
- `TRAINING/orchestration/routing_integration.py:208-230` (training plan verification)
- `TRAINING/orchestration/routing_integration.py:247-250` (manifest update after plans)
- `TRAINING/orchestration/utils/manifest.py:255-300` (new `update_manifest_with_plan_hashes()` function)

**Impact**: Routing and training plans are now created successfully, verified after save, and manifest is updated with plan hashes after plans are created.

---

## Other Files Verified Safe

Audited all other files in `globals/` that write JSON:
- `routing_decisions.json` - ✅ Single write after aggregation
- `target_confidence_summary.json` - ✅ Aggregated then written once
- `training_summary.json` - ✅ Aggregated then written once
- `training_results_summary.json` - ✅ Single write
- `stats.json` - ✅ Uses locking in `ReproducibilityTracker`
- `snapshot_index.json` - ✅ Uses locking (fcntl.flock)
- `fs_snapshot_index.json` - ✅ Uses locking (fcntl.flock)
- `reproducibility_log.json` - ✅ Loads existing, merges, then writes
- `manifest.json` - ✅ Loads existing, updates specific fields, then writes
- `checkpoint.json` - ✅ Atomic write (temp file + rename)

All other files use safe patterns (aggregation before write, atomic writes, or merge-then-write).

---

## Testing

After fixes:
- ✅ `run_context.json` preserves `stage_history` when `save_run_context()` is called after `save_stage_transition()`
- ✅ `run_hash.json` is created successfully after pipeline completes
- ✅ Run hash includes all stages (TARGET_RANKING, FEATURE_SELECTION, TRAINING)
- ✅ Previous run lookup works correctly (finds previous run's `run_hash.json` in parent directories)
- ✅ Error messages are clear when run_hash computation fails
- ✅ `globals/routing_plan/routing_plan.json` is created after feature selection
- ✅ `globals/training_plan/master_training_plan.json` is created after routing plan generation
- ✅ `manifest.json` includes `plan_hashes` with non-null values for `routing_plan_hash` and `training_plan_hash`
- ✅ Error messages are visible (not hidden at debug level) when plan generation fails

---

## Backward Compatibility

All fixes maintain backward compatibility:
- Existing `run_context.json` files are still readable (new fields are optional)
- Run hash computation works with or without previous runs
- Plan hashes are optional in manifest (null if plans don't exist)
- All file writes use same format as before

---

## Related Issues

This fixes issues identified in:
- Plan: `/home/Jennifer/.cursor/plans/fix_file_overwrite_issues_in_pipeline_7cb7da19.plan.md`
- User report: `manifest.json` showing null `plan_hashes`, missing `run_hash.json`, and `run_context.json` losing stage history
