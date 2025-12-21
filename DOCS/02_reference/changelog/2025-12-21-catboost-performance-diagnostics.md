# CatBoost Performance Diagnostics and Comprehensive Fixes

**Date**: 2025-12-21  
**Type**: Performance Fix, Diagnostics Enhancement  
**Impact**: High - Comprehensive diagnostics and performance optimizations  
**Breaking**: No - Backward compatible

## Summary

Implemented comprehensive performance diagnostics, timing logs, and additional performance fixes for CatBoost in feature selection. Added detailed comparison analysis between feature selection and target ranking stages to identify root causes of performance issues. Reduced iterations cap from 2000 to 300 to match target ranking behavior.

## Problem

After implementing early stopping (see [2025-12-21-catboost-early-stopping-fix.md](2025-12-21-catboost-early-stopping-fix.md)), CatBoost was still taking hours in feature selection while working fine in target ranking (~10 minutes). Needed comprehensive diagnostics to identify bottlenecks and systematic comparison between the two stages.

**Key Questions**:
- Why does CatBoost work fine in target ranking (~10 min) but take hours in feature selection?
- What are the exact differences between the two implementations?
- Where is time being spent (CV, fit, importance computation)?

## Solution

### Phase 1: Deep Comparison Analysis

**File**: `docs/analysis/catboost_feature_selection_vs_target_ranking_comparison.md`

Created comprehensive comparison document identifying key differences:

1. **Training Process Differences**:
   - Feature Selection: CV (3 folds) + train_test_split (80/20) + fit with eval_set
   - Target Ranking: CV + fit on full dataset (no eval_set)

2. **Iterations Cap**:
   - Feature Selection: Capped at 2000 iterations (too high)
   - Target Ranking: Uses config default (300 iterations)

3. **Early Stopping**:
   - Feature Selection: Configured and used in final fit
   - Target Ranking: Not configured in config (may not be used)

4. **CV Usage**:
   - Both stages use CV, but feature selection does CV + train_test_split + fit
   - CV overhead can be significant (3 folds × iterations without early stopping)

### Phase 2: Performance Timing and Diagnostics

**File**: `TRAINING/ranking/multi_model_feature_selection.py`

#### Added Performance Timing Logs

1. **CV Phase Timing** (lines 1844-1856):
   - Logs CV start time and duration
   - Reports CV scores (mean, std)
   - Identifies if CV is the bottleneck

2. **Final Fit Timing** (lines 1954-1970):
   - Logs fit duration in minutes
   - Reports actual vs max iterations
   - Verifies early stopping triggered
   - Logs train/val scores and gap

3. **Feature Importance Timing** (lines 2780-2795):
   - Logs importance computation duration
   - Identifies if importance computation is slow

4. **Total Time Breakdown** (lines 1970-1975):
   - Shows CV time, fit time, other time, total time
   - Helps identify which phase is the bottleneck

#### Added Diagnostic Logging

1. **Iteration Tracking** (lines 1958-1970):
   - Logs actual iterations completed vs max
   - Reports best_iteration from early stopping
   - Verifies early stopping status

2. **Score Logging** (lines 1970-1975):
   - Logs train score, val score, and gap
   - Warns if gap > 0.3 (overfitting indicator)
   - Reports CV scores per fold

3. **Feature Importance Analysis** (lines 2795-2810):
   - Logs top 10 features by importance
   - Shows importance percentage for each feature
   - Warns if top 5 features account for >50% of importance (potential leakage)

### Phase 3: Pre-Training Data Quality Checks

**File**: `TRAINING/ranking/multi_model_feature_selection.py` (lines 1804-1836)

Added quick verification checks (extensive checks skipped - feature pruning already filtered low-quality features):

1. **Constant Features Check**:
   - Detects features with only one unique value
   - Warns if found (should have been filtered by pruning)

2. **Perfect Separability Check**:
   - Detects features that perfectly separate classes
   - Warns if found (may explain 100% accuracy)

3. **Feature-to-Sample Ratio Check**:
   - Warns if ratio > 0.2 (high risk of overfitting)
   - Reports dataset size and feature count

### Phase 4: Overfitting Detection

**File**: `TRAINING/ranking/multi_model_feature_selection.py` (lines 1983-2010)

Enhanced overfitting detection beyond existing fail-fast:

1. **Early 100% Accuracy Detection** (already existed, enhanced):
   - Checks if train_score >= 0.999
   - Skips expensive operations immediately
   - Returns early with empty importance

2. **Train/Val Gap Analysis** (new):
   - Computes train_score - val_score
   - Warns if gap > 0.3 (severe overfitting)
   - Logs moderate gaps (> 0.1) for monitoring

3. **CV vs Train Gap Analysis** (new):
   - Compares CV mean score vs train score
   - Warns if gap > 0.3 (model overfitting to training data)
   - Logs moderate gaps for monitoring

### Phase 5: Performance Fixes

**File**: `TRAINING/ranking/multi_model_feature_selection.py`

#### Fix 1: Reduce Iterations Cap (lines 1558-1565)

**Before**: Capped at 2000 iterations (too high, matches old behavior)

**After**: Capped at 300 iterations (matches target ranking config)

```python
# 5. CRITICAL: Cap iterations for feature selection to prevent 2-3 hour training times
# Match target ranking approach: use 300 iterations (same as target ranking config)
max_iterations_feature_selection = 300  # Match target ranking (was 2000, too high)
```

#### Fix 2: CV Overhead Warnings (lines 1844-1856)

Added warnings about CV overhead:
- Notes that CV can be slow (no early stopping per fold)
- Suggests skipping CV for faster feature selection
- Logs CV duration for visibility

## Implementation Details

### Performance Timing Structure

```python
# Total timing structure
total_start_time = time.time()
cv_start_time = None
cv_elapsed = None

# CV phase (if enabled)
if cv_n_jobs > 1:
    cv_start_time = time.time()
    cv_scores = cross_val_score(...)
    cv_elapsed = time.time() - cv_start_time
    logger.info(f"CV phase completed in {cv_elapsed/60:.2f} minutes")

# Final fit phase
fit_start_time = time.time()
model.fit(X_train_fit, y_train_fit, eval_set=[(X_val_fit, y_val_fit)])
fit_elapsed = time.time() - fit_start_time
logger.info(f"Final fit completed in {fit_elapsed/60:.2f} minutes")

# Total breakdown
total_elapsed = time.time() - total_start_time
logger.info(f"Total time breakdown - CV: {cv_time} min, Fit: {fit_time} min, Other: {other_time} min, Total: {total_time} min")
```

### Diagnostic Logging Structure

```python
# Iteration tracking
logger.info(f"iterations: {actual_iterations}/{max_iterations}, "
          f"best_iteration: {best_iteration}, "
          f"early_stopping: {'triggered' if triggered else 'not triggered'}")

# Score logging
logger.info(f"Scores - Train: {train_score:.4f}, Val: {val_score:.4f}, "
          f"Gap: {train_val_gap:.4f} {'(OVERFITTING)' if gap > 0.3 else ''}")

# Feature importance
logger.info(f"Top 10 features by importance:")
for idx, (feat, imp) in enumerate(top_10.items(), 1):
    logger.info(f"  {idx:2d}. {feat}: {imp:.6f} ({imp/total*100:.2f}%)")
```

### Overfitting Detection Structure

```python
# Check 1: Early 100% accuracy
if train_score >= 0.999:
    logger.warning("OVERFITTING DETECTED - Training accuracy >= 99.9%")
    return None, pd.Series(0.0, index=feature_names), importance_method, train_score

# Check 2: Train/Val gap
if train_val_gap > 0.3:
    logger.warning(f"OVERFITTING WARNING - Large train/val gap: {train_val_gap:.4f}")

# Check 3: CV vs Train gap
if cv_train_gap > 0.3:
    logger.warning(f"OVERFITTING WARNING - Large CV/Train gap: {cv_train_gap:.4f}")
```

## Expected Performance Impact

### Before
- **Iterations Cap**: 2000 (too high)
- **No Timing Logs**: Can't identify bottlenecks
- **Limited Diagnostics**: Hard to debug performance issues
- **No Overfitting Warnings**: Wastes time on memorized models

### After
- **Iterations Cap**: 300 (matches target ranking)
- **Comprehensive Timing**: Identifies CV vs fit vs importance bottlenecks
- **Detailed Diagnostics**: Logs iterations, scores, importance, gaps
- **Overfitting Detection**: Early warnings prevent wasted time

## Testing Recommendations

1. **Verify Timing Logs**:
   - Check logs for CV phase duration
   - Check logs for fit phase duration
   - Check logs for importance computation duration
   - Verify total time breakdown is accurate

2. **Verify Iterations Reduction**:
   - Check that iterations cap is now 300 (not 2000)
   - Verify early stopping still works with lower cap
   - Confirm training completes faster

3. **Verify Diagnostic Logs**:
   - Check that train/val scores are logged
   - Check that CV scores are logged
   - Check that top 10 features are logged
   - Verify overfitting warnings appear when appropriate

4. **Compare with Target Ranking**:
   - Run both stages and compare timing logs
   - Verify feature selection now matches target ranking performance
   - Check that diagnostics help identify any remaining differences

## Files Changed

### Modified Files

1. **`TRAINING/ranking/multi_model_feature_selection.py`**
   - Added performance timing logs (CV, fit, importance, total)
   - Added diagnostic logging (iterations, scores, importance, gaps)
   - Added pre-training data quality checks
   - Enhanced overfitting detection
   - Reduced iterations cap from 2000 to 300
   - Added CV overhead warnings

2. **`docs/analysis/catboost_feature_selection_vs_target_ranking_comparison.md`** (new)
   - Comprehensive comparison document
   - Identifies all differences between stages
   - Provides hypothesis for performance issues
   - Recommends fixes

## Impact

- **Performance**: Reduced iterations cap should improve training time
- **Diagnostics**: Comprehensive timing and logging help identify bottlenecks
- **Overfitting Detection**: Early warnings prevent wasted time
- **Debugging**: Detailed logs make it easier to diagnose issues
- **Comparison**: Documented differences help understand why target ranking is faster

## Related Documentation

- [CatBoost Early Stopping Fix](2025-12-21-catboost-early-stopping-fix.md) - Previous fix that this builds on
- [CatBoost Fail-Fast for Overfitting](2025-12-20-catboost-fail-fast-for-overfitting.md) - Original overfitting detection
- [Comparison Analysis](../../../docs/analysis/catboost_feature_selection_vs_target_ranking_comparison.md) - Detailed comparison document

