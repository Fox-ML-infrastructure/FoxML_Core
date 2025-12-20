# Path Resolution Fix - Stop at Run Directory, Not RESULTS Root

**Date**: 2025-12-20  
**Type**: Bug Fix

## Summary

Fixed path resolution logic that incorrectly stopped at `RESULTS/` directory instead of continuing to find the actual run directory. This caused `RESULTS/targets/` to be created outside of run directories instead of inside `RESULTS/runs/{run}/targets/`.

## Root Cause Analysis

**Why it broke now:** The path resolution bug was always present, but it didn't matter before because:
1. We wrote to BOTH legacy locations AND target-first structure
2. Even if path resolution stopped at `RESULTS/`, the legacy writes still went to the correct location
3. In the Elastic Net commit, we removed legacy root-level writes
4. Now that we ONLY write to target-first structure, if path resolution is wrong, files go to the wrong place

**The actual bug:** When `output_dir` is something like:
```
RESULTS/runs/cg-xxx/intelligent_output_xxx/REPRODUCIBILITY/FEATURE_SELECTION/CROSS_SECTIONAL/{target}/
```

The walk-up logic stopped at `RESULTS/` because of this check:
```python
if base_output_dir.name == "RESULTS" or (base_output_dir / "targets").exists():
    break
```

Then `ensure_target_structure(RESULTS/, target)` created `RESULTS/targets/<target>/` instead of the correct location.

## Fix

Changed the path resolution to only stop when it finds a directory that has `targets/`, `globals/`, or `cache/` subdirectories (indicating it's a run directory). Don't stop at `RESULTS/` itself.

**Before:**
```python
if base_output_dir.name == "RESULTS" or (base_output_dir / "targets").exists():
    break
```

**After:**
```python
# Only stop if we find a run directory (has targets/, globals/, or cache/)
# Don't stop at RESULTS/ - continue to find actual run directory
if (base_output_dir / "targets").exists() or (base_output_dir / "globals").exists() or (base_output_dir / "cache").exists():
    break
```

## Files Changed

1. **TRAINING/ranking/multi_model_feature_selection.py** (line 3546)
   - Fixed path resolution in `save_multi_model_results()`

2. **TRAINING/ranking/predictability/model_evaluation.py** (lines 909, 1336, 5329)
   - Fixed path resolution in 3 locations where artifacts are saved to target-first structure

3. **TRAINING/ranking/feature_selection_reporting.py** (lines 121, 333)
   - Fixed path resolution in 2 locations where feature selection results are saved

4. **TRAINING/orchestration/target_routing.py** (lines 118, 245)
   - Fixed path resolution in `load_target_confidence()` and `save_target_routing_metadata()`

## Impact

- ✅ No more `RESULTS/targets/` created outside run directories
- ✅ All target-first structure files now correctly created inside `RESULTS/runs/{run}/targets/<target>/`
- ✅ Path resolution correctly identifies run directories by checking for `targets/`, `globals/`, or `cache/` subdirectories

## Testing

Verified that:
- Path resolution continues past `RESULTS/` directory
- Only stops when it finds a run directory (has `targets/`, `globals/`, or `cache/`)
- All files are created in the correct location: `RESULTS/runs/{run}/targets/<target>/`

