# Run Comparison Fixes for Target-First Structure

**Date**: 2025-12-21  
**Type**: Bug Fix / Architecture  
**Impact**: High - Enables proper run comparison and trend analysis  
**Breaking**: No - Backward compatible

## Summary

Fixed diff telemetry and trend analyzer to properly find and compare runs across the new target-first directory structure. Both systems were only searching within a single run or looking in the wrong location for snapshot indices, preventing similar runs from being compared.

## Problem

### Issue 1: Diff Telemetry Not Finding Previous Runs
- **Location**: `TRAINING/orchestration/utils/diff_telemetry.py` - `find_previous_comparable()` method
- **Problem**: 
  - Searched for `REPRODUCIBILITY/METRICS/snapshot_index.json` (legacy structure)
  - New structure stores snapshots in `globals/snapshot_index.json` (target-first structure)
  - Result: Previous comparable runs were not found, so no diffs were generated

### Issue 2: Trend Analyzer Only Searches Single Run
- **Location**: `TRAINING/common/utils/trend_analyzer.py` - `load_artifact_index()` method
- **Problem**:
  - Only searched within the single `reproducibility_dir` passed to it
  - Didn't search across multiple runs in the same comparison group directory (`RESULTS/runs/cg-*/`)
  - Result: Couldn't compare trends across different runs in the same comparison group

## Solution

### Fix 1: Update Diff Telemetry to Check globals/snapshot_index.json

**File**: `TRAINING/orchestration/utils/diff_telemetry.py`

**Changes**:
1. In `find_previous_comparable()` (line ~3405), added check for `globals/snapshot_index.json`:
   - Checks `run_subdir / "globals" / "snapshot_index.json"` in addition to legacy path
   - Prioritizes target-first structure (globals/) over legacy (REPRODUCIBILITY/METRICS/)
   
2. Updated snapshot file verification logic (line ~3443):
   - Updated path construction to use target-first structure: `targets/<target>/reproducibility/<view>/cohort=<cohort_id>/snapshot.json`
   - Also checks legacy path for backward compatibility
   - Handles both CROSS_SECTIONAL and SYMBOL_SPECIFIC views correctly

3. Updated metrics loading:
   - Uses `get_metrics_path_from_cohort_dir()` to find metrics in target-first structure
   - Falls back to legacy location if target-first path not found
   - Supports both JSON and parquet metrics files

4. Applied same fix to `get_or_establish_baseline()` for consistency

### Fix 2: Update Trend Analyzer to Search Across Comparison Group

**File**: `TRAINING/common/utils/trend_analyzer.py`

**Changes**:
1. In `load_artifact_index()` (line ~143), detects if `reproducibility_dir` is in a comparison group structure:
   - Checks if we're in `RESULTS/runs/cg-*/run_name/` structure
   - If so, finds the comparison group directory and searches all runs in it
   - Builds artifact index from all runs in the comparison group, not just one run

2. Processes all runs in comparison group:
   - Iterates through all run directories in the comparison group
   - Processes each run's `targets/` structure to build complete artifact index
   - Maintains backward compatibility for single-run directories

### Fix 3: Update ReproducibilityTracker Trend Summary

**File**: `TRAINING/orchestration/utils/reproducibility_tracker.py`

**Changes**:
1. In `generate_trend_summary()` (line ~3935), detects comparison group directory:
   - Walks up from run directory to find `RESULTS/runs/cg-*/` structure
   - Passes comparison group directory to TrendAnalyzer instead of single run directory
   - This allows trend analyzer to search across all runs in the comparison group

## Implementation Details

### Diff Telemetry Changes

**Before**:
```python
run_snapshot_index = run_subdir / "REPRODUCIBILITY" / "METRICS" / "snapshot_index.json"
```

**After**:
```python
# Check both target-first (globals/) and legacy (REPRODUCIBILITY/METRICS/)
run_snapshot_index = None
globals_snapshot_index = run_subdir / "globals" / "snapshot_index.json"
legacy_snapshot_index = run_subdir / "REPRODUCIBILITY" / "METRICS" / "snapshot_index.json"

if globals_snapshot_index.exists():
    run_snapshot_index = globals_snapshot_index
elif legacy_snapshot_index.exists():
    run_snapshot_index = legacy_snapshot_index
```

### Trend Analyzer Changes

**Before**: Only processed single run directory

**After**: Detects comparison group and processes all runs:
```python
# Detect if we're in RESULTS/runs/cg-*/run_name/ structure
temp_dir = run_dir
for _ in range(5):
    if temp_dir.parent.name == "runs" and temp_dir.parent.parent.name == "RESULTS":
        comparison_group_dir = temp_dir.parent
        runs_to_process = [
            d for d in comparison_group_dir.iterdir()
            if d.is_dir() and (d / "targets").exists() or (d / "globals").exists()
        ]
        break
    temp_dir = temp_dir.parent
```

## Testing Strategy

1. **Verify diff telemetry finds previous runs**:
   - Run two similar runs in the same comparison group
   - Check that `diff_prev.json` is created in the second run
   - Verify it references the first run

2. **Verify trend analyzer compares across runs**:
   - Run multiple similar runs in the same comparison group
   - Run trend analysis
   - Verify trends show multiple runs, not just one

3. **Backward compatibility**:
   - Test with legacy structure (REPRODUCIBILITY/METRICS/)
   - Verify both old and new structures work

## Files Changed

### Modified Files

1. **`TRAINING/orchestration/utils/diff_telemetry.py`**
   - `find_previous_comparable()`: Added check for `globals/snapshot_index.json`
   - `find_previous_comparable()`: Updated snapshot file verification to use target-first paths
   - `find_previous_comparable()`: Updated metrics loading to use target-first paths
   - `get_or_establish_baseline()`: Added check for `globals/snapshot_index.json`

2. **`TRAINING/common/utils/trend_analyzer.py`**
   - `load_artifact_index()`: Added detection of comparison group directory
   - `load_artifact_index()`: Added logic to process all runs in comparison group
   - Updated all path references to use `current_run_dir` instead of `run_dir`

3. **`TRAINING/orchestration/utils/reproducibility_tracker.py`**
   - `generate_trend_summary()`: Added detection of comparison group directory
   - `generate_trend_summary()`: Updated to pass comparison group directory to TrendAnalyzer

## Impact

- **Diff Telemetry**: Now properly finds previous comparable runs and generates diffs
- **Trend Analyzer**: Now compares trends across all runs in the same comparison group
- **Backward Compatible**: All changes support both new target-first structure and legacy structure
- **No Breaking Changes**: Existing runs continue to work without modification

## Related Documentation

- [Target-First Structure Migration](2025-12-19-target-first-structure-migration.md)
- [Diff Telemetry Integration](2025-12-16-diff-telemetry-integration.md)
- [Results Organization Options](../../03_technical/implementation/RESULTS_ORGANIZATION_OPTIONS.md)

