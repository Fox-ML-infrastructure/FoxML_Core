# Feature Selection Output Organization and Elastic Net Fail-Fast

**Date**: 2025-12-20  
**Type**: Bug Fix, Performance Improvement, Refactoring

## Summary

Fixed syntax error in feature selection reporting, reorganized feature selection outputs to prevent overwriting at run root, updated decision routing to use target-first structure, and added fail-fast mechanism for Elastic Net to prevent long-running fits when model is over-regularized.

## Issues Fixed

### 1. Syntax Error in Feature Selection Reporting
- **Issue**: Missing `except`/`finally` block in `feature_selection_reporting.py` at line 343-353 causing `SyntaxError: expected 'except' or 'finally' block`
- **Fix**: Added `except Exception as e:` block to handle errors when setting up target-first structure
- **File**: `TRAINING/ranking/feature_selection_reporting.py`

### 2. Output Overwriting at Run Root
- **Issue**: Files like `feature_selection_summary.json`, `target_confidence.json`, and `model_family_status.json` were written to run root and overwritten by each target
- **Fix**: Removed all root-level writes, now only writes to target-first structure (`targets/<target>/reproducibility/`)
- **Files**: `TRAINING/ranking/multi_model_feature_selection.py`

### 3. Decision Routing Path Updates
- **Issue**: `load_target_confidence()` and `save_target_routing_metadata()` used legacy paths instead of target-first structure
- **Fix**: 
  - Updated `load_target_confidence()` to look in `targets/<target>/reproducibility/target_confidence.json` with legacy fallback
  - Updated `save_target_routing_metadata()` to save per-target to `targets/<target>/decision/routing_decision.json` and create lightweight summary in `globals/feature_selection_routing.json` (separate from `globals/routing_decisions.json`)
  - Ensures merging (not overwriting) when updating global summary
- **Files**: `TRAINING/orchestration/target_routing.py`, `TRAINING/orchestration/intelligent_trainer.py`

### 4. Reproducibility Tracker Warning Noise
- **Issue**: Warning logged for expected fallback scenarios when RunContext is missing X, y, time_vals
- **Fix**: Changed warning to debug level when all core data fields (X, y, time_vals) are None (expected fallback scenario)
- **File**: `TRAINING/orchestration/utils/reproducibility_tracker.py`

### 5. Elastic Net Performance (Fail-Fast)
- **Issue**: Elastic Net could run for 30+ minutes before detecting all-zero coefficients (over-regularized or no signal)
- **Fix**: 
  - Added quick pre-check with `max_iter=50` to detect obvious failures early (~1-2 minutes instead of 30+)
  - Reduced max_iter cap from 1000 to 500 for faster failure
  - Immediate exception when all-zero coefficients detected (no warning log first)
  - Applied to both per-symbol feature selection and cross-sectional feature selection
- **Files**: `TRAINING/ranking/multi_model_feature_selection.py`, `TRAINING/ranking/predictability/model_evaluation.py`

## Changes Made

### Files Modified

1. **TRAINING/ranking/feature_selection_reporting.py**
   - Fixed syntax error: Added `except Exception as e:` block around try statement (lines 343-353)

2. **TRAINING/ranking/multi_model_feature_selection.py**
   - Removed root-level writes: `output_dir / "feature_selection_summary.json"`, `output_dir / "model_family_status.json"`, `output_dir / "target_confidence.json"`
   - Now only writes to target-first structure: `targets/<target>/reproducibility/`
   - Added fail-fast mechanism for Elastic Net:
     - Quick pre-check with max_iter=50
     - Reduced max_iter cap to 500
     - Immediate exception on all-zero coefficients

3. **TRAINING/orchestration/target_routing.py**
   - Updated `load_target_confidence()`: Looks in target-first structure first, then legacy fallback
   - Updated `save_target_routing_metadata()`:
     - Saves per-target to `targets/<target>/decision/routing_decision.json`
     - Creates/updates lightweight summary in `globals/feature_selection_routing.json` (separate from `routing_decisions.json`)
     - Merges with existing entries (doesn't overwrite)

4. **TRAINING/orchestration/intelligent_trainer.py**
   - Updated routing calls to pass target-specific directories using `get_target_reproducibility_dir()`
   - Added `_aggregate_feature_selection_summaries()` function to collect per-target summaries and write to `globals/`
   - Calls aggregation after all targets are processed

5. **TRAINING/orchestration/utils/reproducibility_tracker.py**
   - Changed warning to debug level when all core data fields (X, y, time_vals) are None (expected fallback)

6. **TRAINING/ranking/predictability/model_evaluation.py**
   - Added fail-fast mechanism for Elastic Net (same as multi_model_feature_selection.py)
   - Applied to cross-sectional feature selection path

## Output Structure

### Before
```
RESULTS/{run_id}/
  feature_selection_summary.json  # Overwritten by each target
  target_confidence.json          # Overwritten by each target
  model_family_status.json         # Overwritten by each target
```

### After
```
RESULTS/{run_id}/
  targets/<target>/
    reproducibility/
      feature_selection_summary.json
      target_confidence.json
      model_family_status.json
      selected_features.txt
      feature_importances/
    decision/
      routing_decision.json
  globals/
    feature_selection_routing.json      # Lightweight summary (separate from routing_decisions.json)
    feature_selection_summary.json      # Aggregated across all targets
    model_family_status_summary.json    # Aggregated across all targets
    target_confidence_summary.json     # Already existed
```

## Benefits

1. **No More Overwriting**: Each target's files are in separate directories
2. **Better Organization**: Target-first structure matches target ranking structure
3. **Fail-Fast Performance**: Elastic Net fails in ~1-2 minutes instead of 30+ minutes when over-regularized
4. **Reduced Log Noise**: Expected fallback scenarios logged at debug level instead of warning
5. **Separation of Concerns**: Feature selection routing decisions separate from target ranking routing decisions
6. **Reference-Based**: Global summaries reference per-target files instead of duplicating data

## Testing

- ✅ Syntax error fixed (import works)
- ✅ Per-target files written to `targets/<target>/reproducibility/`
- ✅ No files written to run root (except `globals/`)
- ✅ Aggregated summaries appear in `globals/` after all targets complete
- ✅ `load_target_confidence()` finds files in target-first structure
- ✅ `save_target_routing_metadata()` saves to target-first structure and updates `globals/feature_selection_routing.json`
- ✅ `globals/routing_decisions.json` NOT modified by feature selection (only target ranking modifies it)
- ✅ Merging works (multiple targets don't overwrite each other)
- ✅ Elastic Net fail-fast works (quick pre-check detects all-zero coefficients early)

## Related Issues

- Syntax error causing import failures
- Output files being overwritten at run root
- Decision routing using legacy paths
- Elastic Net taking 30+ minutes to fail when over-regularized
- Warning noise for expected fallback scenarios

