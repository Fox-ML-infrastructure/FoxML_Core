# 2025-12-17: Fixed Field Name Mismatch in Diff Telemetry

## Problem

After refactoring the directory structure and adding strict required-field validation, `metadata.json` and `metrics.json` were not being written to cohort directories. Only `audit_report.json` was being created.

## Root Cause

There was a field name mismatch between:
1. `_get_required_fields_for_stage()` which returned field names like `'date_start'`, `'date_end'`, `'target'`, `'N_effective'`
2. `_validate_stage_schema()` and `finalize_run()` which expected `'date_range_start'`, `'date_range_end'`, `'target_name'`, `'n_effective'`
3. `full_metadata` construction in `_save_to_cohort()` which used the old field names

When `finalize_run()` validated `resolved_metadata` against the required fields, it couldn't find the fields (because they had different names), raised a `ValueError`, which was caught and logged at DEBUG level, causing `metadata.json` and `metrics.json` to not be written.

## Solution

1. **Updated `_get_required_fields_for_stage()`** in `TRAINING/utils/diff_telemetry.py`:
   - Changed `'date_start'` → `'date_range_start'`
   - Changed `'date_end'` → `'date_range_end'`
   - Changed `'target'` → `'target_name'`
   - Changed `'N_effective'` → `'n_effective'`

2. **Updated `full_metadata` construction** in `TRAINING/utils/reproducibility_tracker.py`:
   - Changed `"target"` → `"target_name"`
   - Changed `"N_effective"` → `"n_effective"`
   - Changed `"date_start"` → `"date_range_start"`
   - Changed `"date_end"` → `"date_range_end"`

3. **Improved error visibility**:
   - Changed diff telemetry exception handler from `logger.debug()` to `logger.warning()` so failures are more visible

## Files Changed

- `TRAINING/utils/diff_telemetry.py`: Fixed required field names in `_get_required_fields_for_stage()`
- `TRAINING/utils/reproducibility_tracker.py`: Fixed field names in `full_metadata` construction and improved error logging

## Impact

- `metadata.json` and `metrics.json` are now correctly written to cohort directories
- Required field validation now works correctly
- Errors are more visible in logs

## Testing

After this fix, runs should successfully write:
- `metadata.json` (full audit trail)
- `metrics.json` (lightweight queryable signals)
- `audit_report.json` (audit validation results)

All three files should be present in each cohort directory for both CROSS_SECTIONAL and SYMBOL_SPECIFIC views.

