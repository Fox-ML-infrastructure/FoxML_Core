# 2025-12-20: Threading, Feature Pruning, and Path Resolution Fixes

## Summary

Fixed three critical issues: added threading/parallelization to feature selection to match target ranking performance, excluded `ret_zscore_*` targets from features to prevent leakage, and fixed path resolution errors causing permission denied when saving files.

## Changes

### Fix 1: Threading/Parallelization in Feature Selection

**Problem**: CatBoost and Elastic Net in feature selection were running single-threaded, taking 2-4x longer than target ranking which uses multithreading.

**Root Cause**: Feature selection's `train_model_and_get_importance()` function only did a single fit without using `cross_val_score` with `n_jobs` parameter, while target ranking uses parallel CV folds.

**Solution**:
- Added `cv_n_jobs` loading from config (same logic as `model_evaluation.py`)
- Added `cross_val_score` calls for both CatBoost and Elastic Net before single fit
- Uses `n_jobs=cv_n_jobs` for parallel CV folds (matching target ranking behavior)
- Keeps single fit for importance extraction (as required)

**Files Changed**:
- `TRAINING/ranking/multi_model_feature_selection.py`:
  - Added `cv_n_jobs` loading (line 962-975)
  - Added `cross_val_score` for CatBoost (lines 1748-1786)
  - Added `cross_val_score` for Elastic Net (lines 1972-2009)
  - Imports `cross_val_score` and `PurgedTimeSeriesSplit` for CV

**Expected Performance**: 2-4x speedup for CatBoost and Elastic Net in feature selection, matching target ranking performance.

### Fix 2: Feature Pruning - Exclude `ret_zscore_*` Targets

**Problem**: `ret_zscore_60m` was appearing as a feature with 37.5% importance, causing data leakage. This is a target column (like `fwd_ret_*`), not a feature.

**Root Cause**: `ret_zscore_*` pattern was not in the hardcoded exclusion patterns. Only `fwd_ret_*` was excluded, but `ret_zscore_*` is also a target type that should be excluded.

**Solution**:
- Added `'ret_zscore_'` to `hardcoded_leaky_patterns['prefix_patterns']` in `leakage_filtering.py`
- Updated exclusion logging to include `ret_zscore_` in the list of checked prefixes
- `ret_zscore_*` targets will now be automatically excluded from features

**Files Changed**:
- `TRAINING/ranking/utils/leakage_filtering.py`:
  - Added `'ret_zscore_'` to exclusion patterns (line 702)
  - Updated logging to include `ret_zscore_` (line 711)

**Impact**: Prevents target columns from being used as features, eliminating data leakage.

### Fix 3: Path Resolution Errors - Permission Denied

**Problem**: Trying to write to `/targets` (absolute root path) instead of run directory, causing `[Errno 13] Permission denied: '/targets'` errors.

**Root Cause**: Path resolution in `feature_selection_reporting.py` was walking up the directory tree but could resolve to root (`/`) if it didn't find the run directory, causing absolute paths like `/targets` to be created.

**Solution**:
- Added validation to check if `base_output_dir` resolves to root (`/`) or is invalid
- Added fallback logic to use original `output_dir` if path resolution fails
- Added error handling and detailed logging for all path operations
- Fixed path resolution in multiple locations:
  - Target decision directory creation
  - Feature rankings CSV save
  - Selected features save
  - Feature importances save

**Files Changed**:
- `TRAINING/ranking/feature_selection_reporting.py`:
  - Added root path validation (lines 134, 187, 288, 383)
  - Added fallback logic for failed path resolution
  - Added detailed error logging with path information

**Impact**: Files now save correctly to `RESULTS/runs/{run}/targets/{target}/` instead of failing with permission errors.

## Files Changed

### Modified
- `TRAINING/ranking/multi_model_feature_selection.py` - Added threading/parallelization
- `TRAINING/ranking/utils/leakage_filtering.py` - Added `ret_zscore_` exclusion
- `TRAINING/ranking/feature_selection_reporting.py` - Fixed path resolution

**Total**: 3 files modified

## Testing Recommendations

1. **Threading**: Verify feature selection runs faster (check logs for CV scores)
2. **Feature Pruning**: Verify `ret_zscore_*` columns are excluded from features
3. **Path Resolution**: Verify files save to correct locations without permission errors

## Notes

- Threading improvements only activate if `cv_n_jobs > 1` in config
- Feature pruning fix prevents future leakage from `ret_zscore_*` targets
- Path resolution fixes ensure robust file saving even if directory structure is unexpected

