# Walk-Forward Validation

Use walk-forward validation to simulate real trading conditions and get realistic performance estimates.

## Overview

Walk-forward validation prevents look-ahead bias by:
1. Training on historical data
2. Testing on future data
3. Rolling forward in time

## Basic Walk-Forward

```python
from TRAINING.walkforward import WalkForwardValidator

validator = WalkForwardValidator(
    fold_length=252,  # 1 year of trading days
    step_size=63      # 1 quarter step forward
)

results = validator.validate(
    X, y,
    trainer_class=LightGBMTrainer,
    config=config
)
```

## Configuration

```python
from TRAINING.walkforward import WalkForwardConfig

config = WalkForwardConfig(
    fold_length=252,           # Training window size
    step_size=63,              # Step forward size
    min_train_size=120,        # Minimum training data
    allow_truncated_final=True  # Allow shorter final fold
)

validator = WalkForwardValidator(config)
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

```python
from TRAINING.walkforward import WalkForwardValidator
from TRAINING.model_fun import LightGBMTrainer
from CONFIG.config_loader import load_model_config

# Load data
labeled_data = pd.read_parquet("data/labeled/AAPL_labeled.parquet")
X = labeled_data[feature_cols]
y = labeled_data["target_fwd_ret_5m"]

# Configure walk-forward
validator = WalkForwardValidator(fold_length=252, step_size=63)

# Train and validate
config = load_model_config("lightgbm", variant="conservative")
results = validator.validate(X, y, LightGBMTrainer, config)

# Analyze results
print(f"Average R²: {results['aggregate_metrics']['r2']}")
print(f"Number of folds: {len(results['folds'])}")
```

## Next Steps

- [Model Training Guide](MODEL_TRAINING_GUIDE.md) - Training basics
- [3-Phase Training Workflow](../../../TRAINING/EXPERIMENTS/README.md) - Optimized workflow

