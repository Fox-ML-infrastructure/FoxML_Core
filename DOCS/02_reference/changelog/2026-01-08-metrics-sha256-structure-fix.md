# Metrics SHA256 Structure Fix

**Date**: 2026-01-08  
**Type**: Bug Fix  
**Impact**: High - Fixes misleading error logs and ensures metrics digest computation works correctly

## Problem

The `metrics_sha256 cannot be computed` error was being logged despite the hash being successfully computed and present in `snapshot.json`. Root cause: structural mismatch between how metrics were stored in `run_data` and how `_compute_metrics_digest()` attempted to retrieve them.

### Root Cause

1. **Metrics Structure Mismatch**: In `reproducibility_tracker.py`, when `run_data` was constructed, `metrics_dict` was spread directly into `run_data` as top-level keys (e.g., `run_data['primary_metric']`, `run_data['coverage']`) instead of being nested under a `'metrics'` key.

2. **Lookup Failure**: `_normalize_outputs()` and `_compute_metrics_digest()` in `diff_telemetry.py` explicitly looked for a nested `metrics` dictionary using `run_data.get('metrics')` or `outputs.get('metrics')`. Since metrics were top-level, these lookups failed.

3. **Timing/Fallback**: The error was logged because in-memory lookup failed, but the hash was eventually computed successfully from disk (`cohort_dir/metrics.json`), leading to confusing logs.

## Solution

Fixed the structural mismatch by ensuring `run_data` includes a dedicated `'metrics'` key containing the full metrics dictionary, while maintaining backward compatibility by also spreading metrics to the top level.

### Key Changes

1. **Added `'metrics'` Key to `run_data`** (`reproducibility_tracker.py`)
   - **Location**: Lines 3425, 3755, 3912 (3 locations in `log_run()` method)
   - **Change**: Added `"metrics": metrics` to `run_data` dict construction
   - **Backward Compatibility**: Kept spreading metrics to top level (`**{k: v for k, v in metrics.items()}`)
   - **Benefit**: `_normalize_outputs()` and `_compute_metrics_digest()` can now find metrics in expected location

2. **Enhanced Metrics Population in `full_metadata`** (`reproducibility_tracker.py`)
   - **Location**: Lines 1807-1820 (in `_save_to_cohort()`)
   - **Change**: Added fallback to reconstruct metrics from top-level keys if `run_data['metrics']` doesn't exist
   - **Benefit**: Handles legacy data and edge cases where metrics might be spread

3. **Added Fallback in `_normalize_outputs()`** (`diff_telemetry.py`)
   - **Location**: Lines 1937-1950
   - **Change**: Added fallback to reconstruct metrics from top-level keys in `run_data` if `run_data.get('metrics')` is empty
   - **Benefit**: Prevents error logs when metrics exist but are structured differently

4. **Fixed Decision Engine Metrics Extraction** (`reproducibility_tracker.py`)
   - **Location**: Lines 2356-2391
   - **Change**: Extract metrics from `run_data['metrics']` with fallback to top-level keys, then update `run_data['metrics']` after adding decision fields
   - **Benefit**: Ensures decision engine receives properly structured metrics

## Files Changed

- `TRAINING/orchestration/utils/reproducibility_tracker.py`
  - Added `"metrics": metrics` to `run_data` in 3 locations (lines 3425, 3755, 3912)
  - Added fallback metrics reconstruction in `_save_to_cohort()` (lines 1807-1820)
  - Fixed decision engine metrics extraction (lines 2356-2391)
  - Fixed trend analysis code cleanup (lines 3773-3795)

- `TRAINING/orchestration/utils/diff_telemetry.py`
  - Added fallback metrics reconstruction in `_normalize_outputs()` (lines 1937-1950)

## Impact

- **Before**: Error logged "metrics_sha256 cannot be computed" even though hash was successfully computed from disk
- **After**: Metrics found in-memory via `run_data['metrics']`, error no longer logged, hash computed correctly
- **Backward Compatibility**: Maintained - code still handles top-level metrics via fallback logic

## Testing

- Verified `metrics_sha256` is computed successfully for both CROSS_SECTIONAL and SYMBOL_SPECIFIC views
- Verified no error logs when metrics are properly structured
- Verified fallback logic works for legacy data with top-level metrics
