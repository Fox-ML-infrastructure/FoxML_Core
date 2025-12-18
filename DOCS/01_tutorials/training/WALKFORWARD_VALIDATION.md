# Walk-Forward Validation

Use walk-forward validation to simulate real trading conditions and get realistic performance estimates.

## Overview

Walk-forward validation prevents look-ahead bias by:
1. Training on historical data
2. Testing on future data
3. Rolling forward in time

## Basic Walk-Forward

> **Note**: Walk-forward validation is currently planned but not yet implemented.

The walkforward configuration is available in the config:

```yaml
# CONFIG/training_config/walkforward.yaml
walkforward:
  fold_length: 252
  step_size: 63
  allow_truncated_final_fold: false
```

For now, use time-series aware validation methods available in the codebase:

```python
from TRAINING.unified_training_interface import UnifiedTrainingInterface
from scripts.utils.purged_time_series_split import PurgedTimeSeriesSplit

# Use PurgedTimeSeriesSplit for temporal validation
interface = UnifiedTrainingInterface()
model, cv_results = interface.train_with_cross_validation(
    trainer, X, y, timestamps=timestamps, n_splits=5
)
```

## Configuration

Walk-forward configuration is defined in YAML (all parameters load from config - Single Source of Truth):

```yaml
walkforward:
  fold_length: 252           # Training window size
  step_size: 63              # Step forward size
  allow_truncated_final_fold: false  # Allow shorter final fold
```

## Results

Walk-forward validation returns:

```python
results = {
    'folds': [
        {
            'train_start': '2020-01-01',
            'train_end': '2020-12-31',
            'test_start': '2021-01-01',
            'test_end': '2021-03-31',
            'metrics': {...},
            'predictions': [...]
        },
        ...
    ],
    'aggregate_metrics': {...}
}
```

## Best Practices

1. **Use realistic windows**: 252 days (1 year) training, 63 days (1 quarter) testing
2. **Check minimum size**: Ensure enough data for training
3. **Monitor overfitting**: Compare train vs test performance
4. **Track stability**: Check if performance is consistent across folds

## Example

> **Note**: This example shows the planned API. The `TRAINING.walkforward` module is not yet implemented.

For now, use temporal validation with `PurgedTimeSeriesSplit`:

```python
from TRAINING.unified_training_interface import UnifiedTrainingInterface
from TRAINING.model_fun import LightGBMTrainer
from CONFIG.config_loader import load_model_config
from scripts.utils.purged_time_series_split import PurgedTimeSeriesSplit
from sklearn.model_selection import cross_val_score

# Load data
labeled_data = pd.read_parquet("data/labeled/AAPL_labeled.parquet")
X = labeled_data[feature_cols]
y = labeled_data["target_fwd_ret_5m"]
timestamps = labeled_data.index.values

# Use temporal cross-validation
purged_cv = PurgedTimeSeriesSplit(n_splits=5, purge_overlap=17)
trainer = LightGBMTrainer()
config = load_model_config("lightgbm", variant="conservative")

# Train with temporal validation
cv_scores = cross_val_score(trainer, X, y, cv=purged_cv)
print(f"Average R²: {cv_scores.mean():.4f}")
print(f"Number of folds: {len(cv_scores)}")
```

## Next Steps

- [Model Training Guide](MODEL_TRAINING_GUIDE.md) - Training basics
- [Experiments Workflow](../../LEGACY/EXPERIMENTS_WORKFLOW.md) - 3-phase optimized training workflow (Legacy)

