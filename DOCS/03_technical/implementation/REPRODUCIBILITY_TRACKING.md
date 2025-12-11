# Reproducibility Tracking Guide

## Overview

The `ReproducibilityTracker` module provides automatic comparison of pipeline run results to verify deterministic behavior across executions. It tracks metrics, compares them to previous runs, and logs differences to help identify non-reproducibility issues.

**Location:** `TRAINING/utils/reproducibility_tracker.py`

## Quick Start

```python
from pathlib import Path
from TRAINING.utils.reproducibility_tracker import ReproducibilityTracker

# Initialize tracker
tracker = ReproducibilityTracker(output_dir=Path("results"))

# Log comparison after computing results
tracker.log_comparison(
    stage="target_ranking",
    item_name="y_will_swing_low_15m_0.05",
    metrics={
        "metric_name": "ROC-AUC",
        "mean_score": 0.751,
        "std_score": 0.029,
        "mean_importance": 0.23,
        "composite_score": 0.764
    }
)
```

## Features

- **Automatic comparison**: Compares current run to most recent previous run
- **Tolerance-based verification**: Configurable tolerances for floating-point differences
- **Multi-stage support**: Track different pipeline stages separately (target_ranking, feature_selection, model_training, etc.)
- **Comprehensive coverage**: Integrated into all deterministic pipeline stages (target ranking, feature selection, model training)
- **Architectural design**: Tracking is integrated into computation modules, not entry points
  - Works regardless of which entry point calls the computation functions
  - Single source of tracking logic (no duplication)
  - Computation functions handle their own tracking
- **Run history**: Keeps last N runs per item (configurable, default: 10)
- **Structured logging**: Clear ✅/⚠️ indicators for reproducible vs different runs
- **JSON storage**: Human-readable JSON format for easy analysis
- **Visibility**: Logs appear in main script output for immediate feedback

## API Reference

### ReproducibilityTracker

#### Initialization

```python
tracker = ReproducibilityTracker(
    output_dir: Path,                    # Directory for storing logs
    log_file_name: str = "reproducibility_log.json",  # Log file name
    max_runs_per_item: int = 10,         # Max runs to keep per item
    score_tolerance: float = 0.001,      # 0.1% tolerance for scores
    importance_tolerance: float = 0.01   # 1% tolerance for importance
)
```

#### Methods

##### `log_comparison(stage, item_name, metrics, additional_data=None)`

Main method for logging and comparing results.

**Parameters:**
- `stage` (str): Pipeline stage name (e.g., "target_ranking", "feature_selection")
- `item_name` (str): Name of the item being tracked (e.g., target name, symbol name)
- `metrics` (dict): Dictionary of metrics to track. Must include:
  - `metric_name` (str): Name of the metric (e.g., "ROC-AUC", "R²", "Consensus Score")
  - `mean_score` (float): Mean score value
  - `std_score` (float): Standard deviation of scores
  - `mean_importance` (float): Mean importance value (optional)
  - `composite_score` (float): Composite score (optional, defaults to mean_score)
- `additional_data` (dict, optional): Extra data to store with the run

**Returns:** None (logs comparison and saves run)

**Example:**
```python
tracker.log_comparison(
    stage="target_ranking",
    item_name="y_will_swing_low_15m_0.05",
    metrics={
        "metric_name": "ROC-AUC",
        "mean_score": 0.751,
        "std_score": 0.029,
        "mean_importance": 0.23,
        "composite_score": 0.764,
        "n_models": 3,
        "leakage_flag": "OK"
    },
    additional_data={
        "model_scores": {"lightgbm": 0.787, "random_forest": 0.751},
        "timestamp": "2025-12-11T06:10:01"
    }
)
```

##### `load_previous_run(stage, item_name)`

Load the most recent previous run for a stage/item combination.

**Parameters:**
- `stage` (str): Pipeline stage name
- `item_name` (str): Name of the item

**Returns:** Dictionary with previous run results, or `None` if no previous run exists

##### `save_run(stage, item_name, metrics, additional_data=None)`

Save a run without comparison (useful for first runs or manual tracking).

**Parameters:** Same as `log_comparison()`

## Integration Examples

**Note**: These examples show how tracking is integrated into computation functions. The actual integration is already done in the codebase - these examples are for reference if you need to add tracking to new computation functions.

### Target Ranking

Tracking is integrated into `evaluate_target_predictability()` in `TRAINING/ranking/predictability/model_evaluation.py`:

```python
# Inside evaluate_target_predictability() function, after result is created
if output_dir and result.mean_score != -999.0:
    try:
        from TRAINING.utils.reproducibility_tracker import ReproducibilityTracker
        
        tracker = ReproducibilityTracker(output_dir=output_dir)
        
        # Determine metric name based on task type
        if result.task_type == TaskType.REGRESSION:
            metric_name = "R²"
        elif result.task_type == TaskType.BINARY_CLASSIFICATION:
            metric_name = "ROC-AUC"
        else:
            metric_name = "Accuracy"
        
        tracker.log_comparison(
            stage="target_ranking",
            item_name=target_name,
            metrics={
                "metric_name": metric_name,
                "mean_score": result.mean_score,
                "std_score": result.std_score,
                "mean_importance": result.mean_importance,
                "composite_score": result.composite_score
            },
            additional_data={
                "n_models": result.n_models,
                "leakage_flag": result.leakage_flag,
                "task_type": result.task_type.name
            }
        )
    except Exception as e:
        logger.debug(f"Reproducibility tracking failed for {target_name}: {e}")
```

**Why in computation module**: This ensures tracking works whether called from `intelligent_trainer`, standalone scripts, or programmatic calls.

### Model Training

```python
from TRAINING.utils.reproducibility_tracker import ReproducibilityTracker

# After training a model successfully
model_result = train_model_comprehensive(...)

if model_result and model_result.get('success', False):
    strategy_manager = model_result.get('strategy_manager')
    
    # Extract CV scores if available
    if strategy_manager and hasattr(strategy_manager, 'cv_scores'):
        cv_scores = strategy_manager.cv_scores
        if cv_scores and len(cv_scores) > 0:
            tracker = ReproducibilityTracker(output_dir=output_dir)
            tracker.log_comparison(
                stage="model_training",
                item_name=f"{target}:{family}",
                metrics={
                    "metric_name": "CV Score",
                    "mean_score": float(np.mean(cv_scores)),
                    "std_score": float(np.std(cv_scores)),
                    "composite_score": float(np.mean(cv_scores))
                },
                additional_data={
                    "strategy": strategy,
                    "n_features": len(feature_names)
                }
            )
```

### Feature Selection

Tracking is integrated into `select_features_for_target()` in `TRAINING/ranking/feature_selector.py`:

```python
# Inside select_features_for_target() function, after results are computed
if output_dir and summary_df is not None and len(summary_df) > 0:
    try:
        from TRAINING.utils.reproducibility_tracker import ReproducibilityTracker
        
        tracker = ReproducibilityTracker(output_dir=output_dir)
        
        # Calculate summary metrics
        top_feature_score = summary_df.iloc[0]['consensus_score'] if not summary_df.empty else 0.0
        mean_consensus = summary_df['consensus_score'].mean()
        std_consensus = summary_df['consensus_score'].std()
        n_features_selected = len(selected_features)
        n_successful_families = len([s for s in all_family_statuses if s.get('status') == 'success'])
        
        tracker.log_comparison(
            stage="feature_selection",
            item_name=target_column,
            metrics={
                "metric_name": "Consensus Score",
                "mean_score": mean_consensus,
                "std_score": std_consensus,
                "mean_importance": top_feature_score,
                "composite_score": mean_consensus,
                "n_features_selected": n_features_selected,
                "n_successful_families": n_successful_families
            },
            additional_data={
                "top_feature": summary_df.iloc[0]['feature'] if not summary_df.empty else None,
                "top_n": top_n or len(selected_features),
                "n_symbols": len(symbols)
            }
        )
    except Exception as e:
        logger.debug(f"Reproducibility tracking failed for {target_column}: {e}")
```

**Why in computation module**: This ensures tracking works whether called from `intelligent_trainer`, standalone scripts, or programmatic calls.

### Model Training

Tracking is integrated into the training loop in `TRAINING/training_strategies/training.py`:

```python
# Inside training loop, after successful model training
if output_dir and model_result.get('success', False):
    try:
        from TRAINING.utils.reproducibility_tracker import ReproducibilityTracker
        tracker = ReproducibilityTracker(output_dir=output_dir)
        
        # Extract metrics from strategy_manager if available
        strategy_manager = model_result.get('strategy_manager')
        metrics = {}
        if strategy_manager and hasattr(strategy_manager, 'cv_scores'):
            cv_scores = strategy_manager.cv_scores
            if cv_scores and len(cv_scores) > 0:
                metrics = {
                    "metric_name": "CV Score",
                    "mean_score": float(np.mean(cv_scores)),
                    "std_score": float(np.std(cv_scores)),
                    "composite_score": float(np.mean(cv_scores))
                }
        
        if metrics:
            tracker.log_comparison(
                stage="model_training",
                item_name=f"{target}:{family}",
                metrics=metrics,
                additional_data={
                    "strategy": strategy,
                    "n_features": len(feature_names) if feature_names else 0
                }
            )
    except Exception as e:
        logger.debug(f"Reproducibility tracking failed for {family}:{target}: {e}")
```

### Custom Pipeline Stage (Example for New Code)

If you're adding tracking to a new computation function:
summary_df, selected_features = aggregate_multi_model_importance(...)

# Calculate summary metrics
top_feature_score = summary_df.iloc[0]['consensus_score']
mean_consensus = summary_df['consensus_score'].mean()
std_consensus = summary_df['consensus_score'].std()
n_features_selected = len(selected_features)
n_successful_families = len([s for s in family_statuses if s.get('status') == 'success'])

# Track reproducibility
if output_dir:
    tracker = ReproducibilityTracker(output_dir=output_dir)
    tracker.log_comparison(
        stage="feature_selection",
        item_name=target_column,
        metrics={
            "metric_name": "Consensus Score",
            "mean_score": mean_consensus,
            "std_score": std_consensus,
            "mean_importance": top_feature_score,
            "composite_score": mean_consensus,
            "n_features_selected": n_features_selected,
            "n_successful_families": n_successful_families
        },
        additional_data={
            "top_feature": summary_df.iloc[0]['feature'],
            "top_n": args.top_n,
            "n_symbols": len(symbols)
        }
    )
```

### Custom Pipeline Stage

```python
from TRAINING.utils.reproducibility_tracker import ReproducibilityTracker

# In your custom pipeline stage
def my_custom_stage(symbol: str, output_dir: Path):
    # ... perform computation ...
    
    results = compute_results()
    
    # Track reproducibility
    tracker = ReproducibilityTracker(output_dir=output_dir)
    tracker.log_comparison(
        stage="my_custom_stage",
        item_name=symbol,
        metrics={
            "metric_name": "Custom Metric",
            "mean_score": results.mean_value,
            "std_score": results.std_value,
            "mean_importance": results.importance,
            "composite_score": results.composite
        },
        additional_data={
            "custom_field": results.custom_data,
            "config_hash": hash(str(config))
        }
    )
```

## Log Output

### First Run

```
📊 Reproducibility: First run for target_ranking:y_will_swing_low_15m_0.05 (no previous run to compare)
```

### Reproducible Run (within tolerance)

```
✅ Reproducibility (REPRODUCIBLE):
   Previous: ROC-AUC=0.751±0.029, importance=0.23, composite=0.764
   Current:  ROC-AUC=0.751±0.029, importance=0.23, composite=0.764
   Diff:     ROC-AUC=+0.0000 (+0.00%), composite=+0.0000 (+0.00%), importance=+0.00
```

### Different Run (outside tolerance)

```
⚠️ Reproducibility (DIFFERENT):
   Previous: ROC-AUC=0.751±0.029, importance=0.23, composite=0.764
   Current:  ROC-AUC=0.755±0.031, importance=0.24, composite=0.768
   Diff:     ROC-AUC=+0.0040 (+0.53%), composite=+0.0040 (+0.52%), importance=+0.01
   ⚠️  Results differ from previous run - check for non-deterministic behavior
```

## Storage Format

Results are stored in `reproducibility_log.json` in the output directory:

```json
{
  "target_ranking:y_will_swing_low_15m_0.05": [
    {
      "timestamp": "2025-12-11T06:10:01.930000",
      "stage": "target_ranking",
      "item_name": "y_will_swing_low_15m_0.05",
      "metric_name": "ROC-AUC",
      "mean_score": 0.751,
      "std_score": 0.029,
      "mean_importance": 0.23,
      "composite_score": 0.764,
      "n_models": 3,
      "model_scores": {
        "lightgbm": 0.787,
        "random_forest": 0.751,
        "neural_network": 0.716
      },
      "leakage_flag": "OK"
    }
  ],
  "feature_selection:y_will_swing_low_15m_0.05": [
    {
      "timestamp": "2025-12-11T06:15:30.123000",
      "stage": "feature_selection",
      "item_name": "y_will_swing_low_15m_0.05",
      "metric_name": "Consensus Score",
      "mean_score": 0.145,
      "std_score": 0.023,
      "mean_importance": 0.284,
      "composite_score": 0.145,
      "n_features_selected": 50,
      "n_successful_families": 4,
      "additional_data": {
        "top_feature": "volume",
        "top_n": 50,
        "n_symbols": 5
      }
    }
  ]
}
```

## Configuration

### Tolerance Settings

Default tolerances are conservative to catch even small non-determinism:

- **Score tolerance**: 0.001 (0.1%) - For mean_score and composite_score
- **Importance tolerance**: 0.01 (1%) - For mean_importance

Adjust when initializing:

```python
tracker = ReproducibilityTracker(
    output_dir=output_dir,
    score_tolerance=0.005,      # 0.5% tolerance (more lenient)
    importance_tolerance=0.02   # 2% tolerance (more lenient)
)
```

### Run History Limits

By default, keeps last 10 runs per item. Adjust if needed:

```python
tracker = ReproducibilityTracker(
    output_dir=output_dir,
    max_runs_per_item=20  # Keep last 20 runs
)
```

## Best Practices

### 1. Use Descriptive Stage Names

Use clear, consistent stage names:
- ✅ `"target_ranking"`
- ✅ `"feature_selection"`
- ✅ `"model_training"`
- ❌ `"stage1"`, `"process"`, `"run"`

### 2. Use Unique Item Names

Item names should uniquely identify what's being tracked:
- ✅ Target name: `"y_will_swing_low_15m_0.05"`
- ✅ Symbol name: `"AAPL"`
- ✅ Model name: `"lightgbm_v1"`
- ❌ Generic: `"result"`, `"data"`

### 3. Include All Relevant Metrics

Include all metrics that should be reproducible:
- Mean scores
- Standard deviations
- Importance values
- Composite scores
- Counts (n_models, n_features, etc.)

### 4. Store Additional Context

Use `additional_data` for debugging context:
```python
additional_data={
    "config_hash": hash(str(config)),
    "data_version": data_version,
    "git_sha": git_sha,
    "environment": os.environ.get("CONDA_DEFAULT_ENV")
}
```

### 5. Handle Missing Output Directory

Always check if `output_dir` is available:
```python
if output_dir is not None:
    tracker = ReproducibilityTracker(output_dir=output_dir)
    tracker.log_comparison(...)
```

## Troubleshooting

### "Results differ from previous run"

**Possible causes:**
1. **Non-deterministic random seeds**: Check that all models use deterministic seeds
2. **Data changes**: Verify input data hasn't changed
3. **Config changes**: Check if configuration changed between runs
4. **Library version changes**: Different library versions may produce different results
5. **Hardware differences**: GPU vs CPU, different CPU architectures

**Debugging steps:**
1. Check `reproducibility_log.json` to see exact differences
2. Verify deterministic seeds are set correctly
3. Compare configs between runs
4. Check library versions
5. Review additional_data for environment differences

### Log File Not Created

**Possible causes:**
1. `output_dir` is `None` - Check that output directory is provided
2. Permission issues - Check write permissions for output directory
3. Disk space - Check available disk space

**Solution:**
```python
# Always check output_dir before using tracker
if output_dir is not None:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        tracker = ReproducibilityTracker(output_dir=output_dir)
        tracker.log_comparison(...)
    except Exception as e:
        logger.warning(f"Could not track reproducibility: {e}")
```

## Visibility & Logging

The reproducibility tracker ensures its logs are visible in the main script output:

- **Logger propagation**: The tracker's internal logger has `propagate = True` to ensure messages reach parent loggers
- **Main logger fallback**: The tracker attempts to find and use the main script's logger (e.g., 'rank_target_predictability') if available, ensuring comparison messages appear in the main output stream
- **Dual logging**: Critical comparison messages are logged to both the tracker's logger and the main logger for maximum visibility

This ensures that reproducibility comparisons (✅/⚠️ indicators) are always visible in your script output, not just in log files.

## Current Integrations

The reproducibility tracker is currently integrated into:

1. **Target Ranking** (`SCRIPTS/rank_target_predictability.py`)
   - Tracks: ROC-AUC/R², importance, composite score per target
   - Stage: `"target_ranking"`

2. **Feature Selection** (`TRAINING/ranking/multi_model_feature_selection.py`)
   - Tracks: Consensus scores, top feature, number of features selected
   - Stage: `"feature_selection"`

3. **Model Training** (`TRAINING/training_strategies/training.py`)
   - Tracks: CV scores (mean/std), composite scores per target:family combination
   - Stage: `"model_training"`
   - Item name format: `"{target}:{family}"` (e.g., `"y_will_peak_60m_0.8:lightgbm"`)

## Extending to New Stages

**Architecture Principle**: Add tracking to computation functions, not entry points. This ensures tracking works regardless of how the function is called.

To add reproducibility tracking to a new computation function:

1. **Identify the computation function** where results are computed (not the entry point)

2. **Import the tracker** inside the function (lazy import to avoid circular dependencies):
   ```python
   from TRAINING.utils.reproducibility_tracker import ReproducibilityTracker
   ```

3. **Add tracking after results are computed**, before returning:
   ```python
   # After computing final results
   if output_dir and results_are_valid:
       try:
           tracker = ReproducibilityTracker(output_dir=output_dir)
           tracker.log_comparison(
               stage="your_stage_name",
           item_name=unique_item_identifier,
           metrics={
               "metric_name": "Your Metric",
               "mean_score": your_mean_value,
               "std_score": your_std_value,
               # ... other metrics
           }
       )
   ```

4. **Test with two runs** to verify it works correctly.

## Related Documentation

- [Deterministic Training](../../00_executive/DETERMINISTIC_TRAINING.md) - Overview of determinism system
- [Configuration Guide](../../02_reference/configuration/README.md) - Config system for reproducibility
- [Training Optimization Guide](./TRAINING_OPTIMIZATION_GUIDE.md) - Performance and reproducibility
