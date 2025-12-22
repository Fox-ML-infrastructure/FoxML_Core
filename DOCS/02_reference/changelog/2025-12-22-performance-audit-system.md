# 2025-12-22: Performance Audit System for Multiplicative Work Detection

## Summary

Added a comprehensive performance audit system to detect "accidental multiplicative work" - expensive operations called multiple times in loops, CV folds, or across stages. This helps identify performance bottlenecks where the same expensive computation is repeated unnecessarily.

## Problem

Multi-stage orchestration (targets → feature select → train → report) makes "accidental multiplicative work" extremely common. One expensive operation sneaks into a loop and suddenly you're paying it 3-20×. Examples:
- CatBoost `PredictionValuesChange` called 4× (3 CV folds + 1 final) - **already fixed**
- Feature matrix building called multiple times for same data
- Importance computation in selection AND reporting stages
- Permutation importance in feature loops inside CV loops

## Solution

### Performance Auditor Class

**File**: `TRAINING/common/utils/performance_audit.py`

New `PerformanceAuditor` class that tracks:
- **Call counts**: How many times each function is called
- **Timing**: Duration of each call
- **Input fingerprints**: Hash of function inputs to detect duplicate calls
- **Cache hits/misses**: Whether results were cached
- **Nested loop patterns**: Consecutive calls to same function

**Key Methods**:
- `track_call()`: Record a function call with metadata
- `report_multipliers()`: Find functions called multiple times with same input
- `report_nested_loops()`: Detect consecutive calls (nested loop patterns)
- `report_summary()`: Generate summary statistics
- `save_report()`: Save JSON report to disk

### Instrumentation Added

Instrumented the following heavy functions:

1. **CatBoost `get_feature_importance`** (PredictionValuesChange):
   - `TRAINING/ranking/multi_model_feature_selection.py` (feature selection)
   - `TRAINING/ranking/predictability/model_evaluation.py` (target ranking)
   - `TRAINING/ranking/predictability/leakage_detection.py` (leakage detection)

2. **`RankingHarness.build_panel`**:
   - `TRAINING/ranking/shared_ranking_harness.py`
   - Tracks panel data building calls

3. **`train_model_and_get_importance`**:
   - `TRAINING/ranking/multi_model_feature_selection.py`
   - Core model training and importance extraction

4. **Neural network permutation importance**:
   - `TRAINING/ranking/predictability/model_evaluation.py`
   - Tracks permutation importance computation in feature loops

### Automatic Report Generation

**File**: `TRAINING/orchestration/intelligent_trainer.py`

At the end of each training run:
- Automatically generates performance audit report
- Saves to: `<output_dir>/globals/performance_audit_report.json`
- Logs summary to console showing:
  - Total calls tracked
  - Multipliers found (functions called 2+ times with same input)
  - Nested loop patterns
  - Cache opportunities

## Report Format

The audit report includes:

1. **Summary**: Total calls, unique functions, timing breakdown per function
2. **Multipliers**: Functions called multiple times with same fingerprint
   - Call count, total duration, wasted duration (could be saved with caching)
   - Stage name, cache hit/miss counts
3. **Nested Loops**: Consecutive calls to same function
   - Time span, total duration, potential multiplier
4. **All Calls**: Complete log of all tracked calls with timestamps

## Example Output

```
📊 PERFORMANCE AUDIT SUMMARY
================================================================================
Total function calls tracked: 45
Unique functions: 8

⚠️  MULTIPLIERS FOUND: 2 functions called multiple times with same input
  - catboost.get_feature_importance: 4× calls, 180.5s total (wasted: 135.4s, stage: feature_selection)
  - RankingHarness.build_panel: 3× calls, 12.3s total (wasted: 8.2s, stage: rank_targets)

⚠️  NESTED LOOP PATTERNS: 1 potential nested loop issues
  - neural_network.permutation_importance: 10 consecutive calls in 5.2s (stage: target_ranking)

💾 Full audit report saved to: <output_dir>/globals/performance_audit_report.json
```

## Usage

The audit system is **enabled by default** and runs automatically during training. No configuration needed.

To disable (not recommended):
```python
from TRAINING.common.utils.performance_audit import get_auditor
auditor = get_auditor(enabled=False)
```

## Files Changed

1. **`TRAINING/common/utils/performance_audit.py`** (NEW):
   - `PerformanceAuditor` class
   - `track_performance` decorator
   - `get_auditor()` singleton function

2. **`TRAINING/orchestration/intelligent_trainer.py`**:
   - Added automatic report generation at end of `train_with_intelligence()`
   - Logs summary to console

3. **`TRAINING/ranking/multi_model_feature_selection.py`**:
   - Instrumented CatBoost importance computation
   - Instrumented `train_model_and_get_importance` function

4. **`TRAINING/ranking/shared_ranking_harness.py`**:
   - Instrumented `build_panel` method

5. **`TRAINING/ranking/predictability/model_evaluation.py`**:
   - Instrumented CatBoost importance computation
   - Instrumented neural network permutation importance

6. **`TRAINING/ranking/predictability/leakage_detection.py`**:
   - Instrumented CatBoost importance computation

7. **`TRAINING/common/utils/PERFORMANCE_AUDIT_README.md`** (NEW):
   - Documentation for the audit system

## Impact

- **Proactive detection**: Automatically identifies multipliers during runs
- **Actionable insights**: Reports show exactly which functions are called multiple times
- **Zero overhead when disabled**: Can be disabled if needed (not recommended)
- **Reusable**: Easy to add instrumentation to new functions

## Next Steps

1. Run training pipeline and review audit reports
2. Fix high-priority multipliers (functions called 10+ times)
3. Add caching for duplicate work
4. Restructure loops to move expensive operations outside
5. Continue adding instrumentation to other heavy functions

## Related Issues

- CatBoost PredictionValuesChange called 4× - **FIXED** (removed from CV folds)
- Performance audit will help identify other similar issues automatically

