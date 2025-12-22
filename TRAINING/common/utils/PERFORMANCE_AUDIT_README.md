# Performance Audit System

## Overview

The Performance Audit system tracks call counts and timing for heavy functions to identify "accidental multiplicative work" - expensive operations called multiple times in loops, CV folds, or across stages.

## How It Works

The audit system automatically tracks:
- **Function calls**: Name, duration, input dimensions, stage
- **Input fingerprints**: Hash of function inputs to detect duplicate calls
- **Cache hits/misses**: Whether results were cached
- **Call patterns**: Consecutive calls (nested loop detection)

## Enabled by Default

The audit system is **enabled by default** and runs automatically during training. No configuration needed.

## Instrumented Functions

The following functions are currently instrumented:

1. **`catboost.get_feature_importance`** (PredictionValuesChange)
   - Feature selection stage
   - Target ranking stage
   - Leakage detection stage

2. **`RankingHarness.build_panel`**
   - Panel data building for ranking/selection

3. **`train_model_and_get_importance`**
   - Core model training and importance extraction

4. **`neural_network.permutation_importance`**
   - Permutation importance computation (feature loop)

## Report Location

After a training run, the audit report is automatically saved to:
```
<output_dir>/globals/performance_audit_report.json
```

## Report Contents

The report includes:

1. **Summary**: Total calls, unique functions, timing breakdown
2. **Multipliers**: Functions called multiple times with same input fingerprint
3. **Nested Loops**: Consecutive calls to same function (potential nested loops)
4. **All Calls**: Complete log of all tracked calls

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

## Interpreting Results

### Multipliers

If a function appears in the multipliers list:
- **High priority**: 10+ calls with same input → likely accidental multiplier
- **Medium priority**: 3-10 calls → may be intentional but worth checking
- **Low priority**: 2 calls → may be intentional (e.g., train + validate)

### Nested Loops

If a function appears in nested loop patterns:
- Check if it's inside a feature loop, CV loop, or target loop
- Consider moving expensive operations outside loops
- Use caching if same inputs are processed multiple times

### Cache Opportunities

Functions with:
- Multiple calls with same fingerprint
- High total duration
- Low cache hit rate

→ Good candidates for caching

## Disabling Audit

To disable the audit system (not recommended):

```python
from TRAINING.common.utils.performance_audit import get_auditor
auditor = get_auditor(enabled=False)
```

## Adding New Instrumentation

To instrument a new function:

```python
from TRAINING.common.utils.performance_audit import get_auditor
import time

auditor = get_auditor()
start_time = time.time()

# ... your function code ...

duration = time.time() - start_time
auditor.track_call(
    func_name='your_function_name',
    duration=duration,
    rows=X.shape[0] if hasattr(X, 'shape') else None,
    cols=X.shape[1] if hasattr(X, 'shape') and len(X.shape) > 1 else None,
    stage='your_stage_name',
    cache_hit=False,
    input_fingerprint=fingerprint  # Optional: pre-computed
)
```

## Next Steps

1. Run a training pipeline
2. Check the audit report in `globals/performance_audit_report.json`
3. Identify multipliers and nested loops
4. Fix high-priority issues (functions called 10+ times)
5. Add caching for duplicate work
6. Restructure loops to move expensive operations outside

