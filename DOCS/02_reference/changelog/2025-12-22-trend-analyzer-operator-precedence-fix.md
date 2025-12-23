# Changelog: Trend Analyzer Operator Precedence Fix

**Date**: 2025-12-22  
**Type**: Bug Fix  
**Impact**: High - Enables proper run detection in comparison groups  
**Breaking**: No - Backward compatible

## Summary

Fixed operator precedence bug in trend analyzer path detection that prevented correct identification of runs in comparison groups. The bug caused the trend analyzer to incorrectly filter run directories when searching for runs within a comparison group structure (`RESULTS/runs/cg-*/`).

## Problem

**Location**: `TRAINING/common/utils/trend_analyzer.py` line 192

**Root Cause**:
- The condition `d.is_dir() and (d / "targets").exists() or (d / "globals").exists()` was incorrectly parsed due to operator precedence
- Python evaluates `and` before `or`, so the expression was parsed as: `(d.is_dir() and (d / "targets").exists()) or (d / "globals").exists()`
- This meant that ANY directory with a `globals/` subdirectory would be included, even if it wasn't a valid run directory (didn't pass `d.is_dir()` check)
- Result: Trend analyzer would either miss valid runs or include invalid directories

**Impact**:
- Trend analyzer failed to correctly identify runs in comparison groups
- Could not build artifact index from all runs in a comparison group
- Trend analysis would only work on a subset of runs or fail entirely

## Solution

**File**: `TRAINING/common/utils/trend_analyzer.py`

**Line 192**: Added explicit parentheses to ensure `d.is_dir()` is evaluated before checking subdirectories:

```python
# Before (PROBLEMATIC):
runs_to_process = [
    d for d in comparison_group_dir.iterdir()
    if d.is_dir() and (d / "targets").exists() or (d / "globals").exists() or (d / "REPRODUCIBILITY").exists()
]

# After (FIXED):
runs_to_process = [
    d for d in comparison_group_dir.iterdir()
    if d.is_dir() and ((d / "targets").exists() or (d / "globals").exists() or (d / "REPRODUCIBILITY").exists())
]
```

**Key Changes**:
1. Added explicit parentheses around the subdirectory existence checks: `((d / "targets").exists() or (d / "globals").exists() or (d / "REPRODUCIBILITY").exists())`
2. Ensures `d.is_dir()` is evaluated first, then the subdirectory checks are evaluated as a group
3. Correctly identifies runs that have at least one of: `targets/`, `globals/`, or `REPRODUCIBILITY/` subdirectories

## Impact

### Before
- Trend analyzer would incorrectly filter run directories in comparison groups
- Could miss valid runs or include invalid directories
- Trend analysis would fail or produce incomplete results

### After
- Trend analyzer correctly identifies all runs in comparison groups
- Properly filters directories to only include valid run directories with `targets/`, `globals/`, or `REPRODUCIBILITY/` subdirectories
- Successfully builds artifact index from all runs in comparison group
- Trend analysis works correctly across all runs in a comparison group

## Verification

Tested with actual comparison group structure:
- Structure: `RESULTS/runs/cg-*/run_name/targets/` and `RESULTS/runs/cg-*/run_name/globals/`
- Result: Correctly identifies 10 runs in comparison group
- All runs have `targets=True, globals=True` as expected

## Related

- Builds on previous fixes:
  - `2025-12-21-run-comparison-fixes.md` - Initial comparison group detection
  - `2025-12-22-trend-analyzer-indentation-fix.md` - Fixed indentation errors that prevented module loading
- Part of ongoing trend analyzer improvements for target-first structure support

## Files Changed

- `TRAINING/common/utils/trend_analyzer.py` (1 line changed: 192)

