# Look-Ahead Bias Fixes (2025-12-14)

**Status**: ✅ Implemented (behind feature flags)  
**Branch**: `fix/lookahead-bias-fixes`  
**Priority**: CRITICAL - Fixes likely explain "suspiciously high" scores

## Summary

Implemented comprehensive look-ahead bias fixes to prevent data leakage in feature engineering and model training. All fixes are behind feature flags (default: OFF) for safe gradual rollout.

## Issues Fixed

### Fix #1: Rolling Windows Include Current Bar ✅
**Problem**: Polars `rolling_mean()` includes the current row by default, allowing features to leak current price information.

**Solution**: Conditional `.shift(1)` before all rolling operations when `exclude_current_bar_from_rolling: true`.

**Files Changed**:
- `DATA_PROCESSING/features/simple_features.py`: All rolling operations (rolling_mean, rolling_std, ewm_mean)
- `DATA_PROCESSING/features/comprehensive_builder.py`: All rolling operations

**Impact**: Features now exclude current bar when flag enabled, preventing look-ahead bias.

### Fix #2: Global Normalization Before Train/Test Split ✅
**Problem**: Scalers and imputers were being fit on the entire dataset before CV splits, leaking future statistics into training.

**Solution**: Added optional `X_train`, `X_test`, `y_train`, `y_test` parameters to `train_model_and_get_importance()` for CV-based normalization.

**Files Changed**:
- `TRAINING/ranking/multi_model_feature_selection.py`: Neural network normalization with CV support

**Impact**: When train/test splits provided with `normalize_inside_cv: true`, normalization fits only on training data.

### Fix #3: pct_change() Verification ✅
**Problem**: Needed to verify `pct_change()` behavior regarding current bar inclusion.

**Solution**: Verified that Fix #1's shifted columns already handle this correctly. Documented behavior.

**Files Changed**:
- `DATA_PROCESSING/features/simple_features.py`: Added documentation

**Impact**: Confirmed no additional fix needed - Fix #1 covers this.

### Fix #4: Misnamed Features ✅
**Problem**: Features named `beta_20d` and `market_correlation_60d` are actually just rolling standard deviation of returns, not true beta/correlation.

**Solution**: Renamed to accurate names with backward-compatible aliases.

**Files Changed**:
- `DATA_PROCESSING/features/simple_features.py`: Renamed features, added legacy aliases

**Changes**:
- `beta_20d`/`beta_60d` → `volatility_20d_returns`/`volatility_60d_returns`
- `market_correlation_20d`/`market_correlation_60d` → `volatility_20d_returns`/`volatility_60d_returns`
- Legacy aliases preserved for backward compatibility

**Impact**: Clearer feature naming, no breaking changes.

## Additional Fixes

### Symbol-Specific Evaluation Logging ✅
**Problem**: Symbol-specific evaluations were running but results were being filtered out with no visibility into why.

**Solution**: Enhanced logging to show:
- When cross-sectional fails (with reason)
- Each symbol evaluation attempt and result
- When results are filtered (with reason)
- Summary of stored vs filtered results

**Files Changed**:
- `TRAINING/ranking/target_ranker.py`: Enhanced logging throughout symbol-specific evaluation flow

**Impact**: Better visibility into why symbol-specific results aren't being stored.

### Feature Selection Bug Fix ✅
**Problem**: `task_type` variable collision - CatBoost GPU config was overwriting `task_type` (TaskType enum) with string `"GPU"`, causing `"Unknown task_type: GPU"` errors for ridge and elastic_net.

**Solution**: Renamed GPU config variable to `catboost_task_type` to avoid collision.

**Files Changed**:
- `TRAINING/ranking/predictability/model_evaluation.py`: Renamed `task_type` → `catboost_task_type`
- `TRAINING/ranking/multi_model_feature_selection.py`: Renamed `task_type` → `catboost_task_type`

**Impact**: Fixed feature selection errors for ridge and elastic_net models.

## Configuration

All fixes controlled via `CONFIG/pipeline/training/safety.yaml`:

```yaml
safety:
  leakage_detection:
    lookahead_bias_fixes:
      exclude_current_bar_from_rolling: false  # Fix #1
      normalize_inside_cv: false                # Fix #2
      verify_pct_change_shift: false            # Fix #3
      migration_mode: "off"                     # "off" | "test" | "warn" | "enforce"
```

Per-experiment control available in experiment config files (e.g., `CONFIG/experiments/e2e_ranking_test.yaml`).

## Implementation Details

### Feature Flag System
- **Utility**: `TRAINING/utils/lookahead_bias_config.py`
- **Default Behavior**: All flags OFF (maintains current behavior)
- **Safety**: Can enable/disable per experiment
- **Rollback**: Instant disable via config

### Code Changes Summary
- **Files Modified**: 5
- **Lines Added**: ~400
- **Lines Removed**: ~150
- **New Files**: 2 (utility + config section)
- **Breaking Changes**: None (all behind flags)

## Testing Status

- ✅ Implementation complete
- ✅ All fixes behind feature flags
- ✅ Default behavior unchanged
- ⏳ Ready for experimental validation
- ⏳ Unit tests recommended for future
- ⏳ Integration tests recommended for future

## Expected Impact

### With Flags OFF (Current Behavior)
- Scores: Same as current (potentially inflated)
- Features: Include current bar
- Normalization: Global (leaks future stats)

### With Flags ON (Fixed Behavior)
- Scores: Lower, more realistic (removing leaks) - **Needs validation**
- Features: Exclude current bar (properly causal)
- Normalization: Per-fold (no future leakage)

## Migration Path

1. **Current**: All flags OFF (default) - no behavior change
2. **Testing**: Enable flags in test experiments, compare results
3. **Validation**: Verify scores decrease (more realistic) but models still train
4. **Production**: Gradually enable in production experiments
5. **Future**: Consider making flags default ON after validation

## Related Documentation

- [Look-Ahead Bias Fix Plan](../../03_technical/fixes/LOOKAHEAD_BIAS_FIX_PLAN.md) - Complete analysis and fix details
- [Safe Implementation Plan](../../03_technical/fixes/LOOKAHEAD_BIAS_SAFE_IMPLEMENTATION.md) - Implementation strategy and migration guide

## Commits

- `9d13261` - feat(config): Add look-ahead bias fix configuration flags
- `d4d4502` - feat(utils): Add look-ahead bias fix configuration utility
- `685884c` - fix(features): Implement Fix #1 - Exclude current bar from rolling windows
- `0d6107a` - fix(feature-selection): Add warning for Fix #2 - Normalization leak detection
- `4e3cbab` - fix(feature-selection): Implement Fix #2 Full - CV-based normalization support
- `da5fad0` - fix(features): Implement Fix #3 and Fix #4 - pct_change verification and rename misnamed features
- `1fe75e9` - fix(ranking): Fix symbol-specific evaluation logging and task_type collision
- `ae39102` - fix(feature-selection): Fix task_type variable collision in multi_model_feature_selection
- `3807925` - docs(fixes): Update look-ahead bias fix documentation with implementation status
