# Training Plan Model Families and Feature Summary Fixes

**Date**: 2025-12-21  
**Type**: Bug Fix / Enhancement  
**Impact**: High - Ensures correct model families are used for training and provides feature auditing  
**Breaking**: No - Backward compatible

## Summary

Fixed training plan to use correct trainer families from experiment config (filtering out feature selectors), enhanced global feature summary with actual feature lists for auditing, fixed REPRODUCIBILITY directory location, and added comprehensive documentation for feature storage locations.

## Problems

### Issue 1: Training Plan Includes Feature Selector Families
- **Location**: `TRAINING/orchestration/training_plan_generator.py` - `__init__()` method
- **Problem**: 
  - Training plan metadata included feature selector families (random_forest, catboost, lasso, mutual_information, univariate_selection) instead of trainer families
  - These are used for feature selection only, not for training
  - Result: Training plan showed incorrect model families, potentially causing confusion about what models are actually trained

### Issue 2: Missing Global Feature Summary
- **Location**: `TRAINING/orchestration/intelligent_trainer.py` - `_aggregate_feature_selection_summaries()` method
- **Problem**:
  - No global summary of selected features per target per view in globals/ folder
  - Unlike routing decisions which have `globals/feature_selection_routing.json`, features had no centralized audit location
  - Result: Difficult to audit what features were selected for each target/view combination

### Issue 3: REPRODUCIBILITY Directory at Wrong Level
- **Location**: `TRAINING/orchestration/utils/diff_telemetry.py` - `_organize_run_by_comparison_group()` method
- **Problem**:
  - `RESULTS/REPRODUCIBILITY` was being created at RESULTS root level
  - REPRODUCIBILITY should only exist within run directories
  - Target-first structure should use `globals/` instead of legacy `REPRODUCIBILITY/` structure
  - Result: Incorrect directory structure, potential confusion about where data is stored

### Issue 4: Missing Feature Storage Documentation
- **Location**: `TRAINING/orchestration/intelligent_trainer.py` - Feature selection section
- **Problem**:
  - No clear documentation of where features are stored (memory, disk) and how they're passed from phase 2 to phase 3
  - Result: Difficult to understand feature flow through the pipeline

## Solutions

### Fix 1: Filter Feature Selectors from Training Plan

**File**: `TRAINING/orchestration/training_plan_generator.py`

**Changes**:
1. Added `FEATURE_SELECTORS` set containing known feature selector families:
   - `random_forest`, `catboost`, `lasso`, `mutual_information`, `univariate_selection`, `elastic_net`, `ridge`, `lasso_cv`
   
2. In `__init__()` method (line 74-102):
   - Filter out feature selectors from `model_families` before storing in `self.model_families`
   - Log warning when feature selectors are filtered out
   - Apply filtering to both provided `model_families` and `default_families`
   
3. Added logging to show which families are being used after filtering

**Before**:
```python
if model_families is not None:
    self.model_families = model_families
```

**After**:
```python
FEATURE_SELECTORS = {
    'random_forest', 'catboost', 'lasso', 'mutual_information', 
    'univariate_selection', 'elastic_net', 'ridge', 'lasso_cv'
}

if model_families is not None:
    filtered_families = [f for f in model_families if f not in FEATURE_SELECTORS]
    removed = set(model_families) - set(filtered_families)
    
    if removed:
        logger.warning(f"⚠️ Filtered out {len(removed)} feature selector(s): {sorted(removed)}")
    
    self.model_families = filtered_families
```

### Fix 2: Enhanced Global Feature Summary with Actual Feature Lists

**File**: `TRAINING/orchestration/intelligent_trainer.py`

**Changes**:
1. Enhanced `_aggregate_feature_selection_summaries()` method (line 1324-1499):
   - Added logic to read actual feature lists from `targets/{target}/reproducibility/selected_features.txt`
   - Load routing decisions to determine view (CROSS_SECTIONAL or SYMBOL_SPECIFIC) per target
   - Create `feature_selections` dict with structure: `{target_name:view: {target_name, view, n_features, features, paths}}`
   
2. Create new global summary file: `globals/selected_features_summary.json`:
   - Contains actual feature lists per target per view
   - Includes summary statistics (total_targets, by_view, total_features)
   - Structure similar to `globals/feature_selection_routing.json` for consistency
   
3. Separate from existing `globals/feature_selection_summary.json` (which contains metadata only)

**New File Structure**:
```json
{
  "feature_selections": {
    "target_name:view": {
      "target_name": "...",
      "view": "CROSS_SECTIONAL" | "SYMBOL_SPECIFIC",
      "n_features": 100,
      "features": ["feature1", "feature2", ...],
      "selected_features_path": "targets/.../reproducibility/selected_features.txt",
      "feature_selection_summary_path": "targets/.../reproducibility/feature_selection_summary.json"
    }
  },
  "summary": {
    "total_targets": 4,
    "by_view": {
      "CROSS_SECTIONAL": 3,
      "SYMBOL_SPECIFIC": 1
    },
    "total_features": 400
  }
}
```

### Fix 3: Fixed REPRODUCIBILITY Directory Location

**File**: `TRAINING/orchestration/utils/diff_telemetry.py`

**Changes**:
1. In `_organize_run_by_comparison_group()` method (line 2338-2347):
   - Changed from creating `target_dir / "REPRODUCIBILITY" / "METRICS"` (legacy structure)
   - To using `get_globals_dir(target_dir)` (target-first structure)
   - Ensures REPRODUCIBILITY is only created within run directories, not at RESULTS root level

**Before**:
```python
self.run_metrics_dir = target_dir / "REPRODUCIBILITY" / "METRICS"
```

**After**:
```python
# Use target-first structure (globals/) instead of legacy REPRODUCIBILITY/
from TRAINING.orchestration.utils.target_first_paths import get_globals_dir
globals_dir = get_globals_dir(target_dir)
globals_dir.mkdir(parents=True, exist_ok=True)
self.run_metrics_dir = globals_dir
```

### Fix 4: Added Feature Storage Documentation

**File**: `TRAINING/orchestration/intelligent_trainer.py`

**Changes**:
1. Added comprehensive documentation comments (line 1805-1821):
   - Documents where features are stored (memory, disk, routing decisions, global summary)
   - Explains structure of `target_features` dict for different route types
   - Documents how features are passed from phase 2 to phase 3
   - Includes file paths and line number references

### Fix 5: Enhanced Logging for Families Parameter

**File**: `TRAINING/orchestration/intelligent_trainer.py`

**Changes**:
1. Added detailed logging (line 1920-1926):
   - Log families parameter value, type, and length before passing to routing integration
   - Log families contents for debugging
   - Helps trace where families come from (config vs CLI)

## Implementation Details

### Feature Selector Filtering

The filtering happens at two levels:
1. **TrainingPlanGenerator**: Filters feature selectors when initializing with model_families
2. **Automatic**: No manual intervention needed - feature selectors are automatically excluded

### Global Feature Summary

The aggregation process:
1. Scans all `targets/{target}/reproducibility/selected_features.txt` files
2. Loads routing decisions to determine view per target
3. Creates key: `{target_name}:{view}` for each target/view combination
4. Aggregates into single JSON file in `globals/selected_features_summary.json`
5. Provides summary statistics for quick auditing

### REPRODUCIBILITY Location Fix

The fix ensures:
- REPRODUCIBILITY is only created within run directories
- Target-first structure uses `globals/` instead of legacy `REPRODUCIBILITY/`
- No more `RESULTS/REPRODUCIBILITY` at root level
- Backward compatible with existing runs

## Testing Strategy

1. **Verify feature selector filtering**:
   - Create training plan with feature selectors in model_families
   - Verify they are filtered out from training plan metadata
   - Verify warning is logged

2. **Verify global feature summary**:
   - Run feature selection for multiple targets
   - Check that `globals/selected_features_summary.json` is created
   - Verify it contains actual feature lists per target per view
   - Verify summary statistics are correct

3. **Verify REPRODUCIBILITY location**:
   - Run training pipeline
   - Verify no `RESULTS/REPRODUCIBILITY` is created at root level
   - Verify `globals/` is used instead of legacy structure

4. **Verify logging**:
   - Check logs show families parameter being passed correctly
   - Verify feature selector filtering warnings appear when applicable

## Files Changed

### Modified Files

1. **`TRAINING/orchestration/training_plan_generator.py`**
   - `__init__()`: Added feature selector filtering and improved logging
   - Lines 74-102: Filtering logic and warnings

2. **`TRAINING/orchestration/intelligent_trainer.py`**
   - `_aggregate_feature_selection_summaries()`: Enhanced to include actual feature lists
   - Lines 1324-1499: Feature aggregation and global summary creation
   - Lines 1805-1821: Feature storage documentation
   - Lines 1920-1926: Enhanced logging for families parameter

3. **`TRAINING/orchestration/utils/diff_telemetry.py`**
   - `_organize_run_by_comparison_group()`: Fixed REPRODUCIBILITY location
   - Lines 2341-2347: Use globals/ instead of legacy REPRODUCIBILITY/

## Impact

- **Training Plan**: Now correctly shows only trainer families, not feature selectors
- **Feature Auditing**: Global summary provides single location to audit selected features
- **Directory Structure**: REPRODUCIBILITY only created within run directories
- **Documentation**: Clear documentation of feature storage and flow
- **Logging**: Better visibility into families parameter flow
- **Backward Compatible**: All changes support existing runs

## Related Documentation

- [Target-First Structure Migration](2025-12-19-target-first-structure-migration.md)
- [Feature Selection Routing and Training View Tracking](2025-12-21-feature-selection-routing-and-training-view-tracking.md)
- Training Plan Generation Issues

