# Strategy Updates - Training Optimization

## Overview

Updates made to the training strategies to fix overfitting and implement proper regularization.

## Changes Made

### 1. SingleTaskStrategy (`single_task.py`)

#### Updated: `_create_lightgbm_model()`
- Before: Minimal parameters, no regularization
  ```python
  lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
  ```

- After: Spec 2 high regularization defaults
  ```python
  lgb.LGBMRegressor(
      num_leaves=96,              # 64-128 range
      max_depth=8,                # 7-9 range (was unlimited)
      min_child_weight=0.5,       # NEW: prevents tiny leaf nodes
      learning_rate=0.03,         # 0.01-0.05 range
      n_estimators=1000,          # Use with early stopping
      subsample=0.75,             # Row sampling (was 1.0)
      colsample_bytree=0.75,      # Feature sampling (was 1.0)
      subsample_freq=1,           # Enable bagging
      reg_alpha=0.1,              # L1 regularization
      reg_lambda=0.1,             # L2 regularization
      random_state=42,
      verbose=-1,
  )
  ```

#### Updated: `_create_xgboost_model()`
- Before: Minimal parameters
- After: Spec 2 regularization + gamma parameter
  ```python
  xgb.XGBRegressor(
      max_depth=7,                # 5-8 range
      min_child_weight=0.5,       # Reduced from 10
      gamma=0.3,                  # NEW: min_split_gain
      learning_rate=0.03,         # 0.01-0.05 range
      n_estimators=1000,
      subsample=0.75,
      colsample_bytree=0.75,
      reg_alpha=0.1,
      reg_lambda=0.1,
      random_state=42,
      verbosity=0,
  )
  ```

#### Updated: `train()` method
- Added: Automatic early stopping detection
- Process:
 1. Check if model supports `eval_set` parameter
 2. If yes: split data into train/val
 3. Train with early stopping (50 rounds default)
 4. Log best iteration

```python
# Auto-detect early stopping support
if 'eval_set' in model.fit.__code__.co_varnames:
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    logger.info(f"Early stopping at iteration {model.best_iteration_}")
```

### 2. MultiTaskStrategy (`multi_task.py`)

#### Fixed: Dropout Activation
- Before: Risk of `training=False` disabling Dropout
- After: Explicit train()/eval() mode switching

```python
# Training: Dropout ACTIVE
self.shared_model.train()
for batch_X, batch_y in dataloader:
    predictions = self.shared_model(batch_X)  # Dropout drops units

# Inference: Dropout DISABLED
self.shared_model.eval()
with torch.no_grad():
    predictions = self.shared_model(X_test)  # All units active
```

#### Added: Early Stopping
- Before: Trained for full `n_epochs` (could overfit)
- After: Early stopping with patience

```python
best_loss = float('inf')
patience_counter = 0
patience = 10

for epoch in range(n_epochs):
    epoch_loss = train_one_epoch()

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        patience_counter = 0
        # Save best model state
        best_model_state = save_state()
    else:
        patience_counter += 1

    if patience_counter >= patience:
        logger.info(f"Early stopping at epoch {epoch}")
        restore_best_model(best_model_state)
        break
```

## Configuration Support

Both strategies now read configuration from `self.config`:

```python
# Example configuration
config = {
    'models': {
        'lightgbm': {
            'num_leaves': 96,
            'max_depth': 8,
            # ... other params from first_batch_specs.yaml
        },
        'xgboost': {
            'max_depth': 7,
            'gamma': 0.3,
            # ... other params
        }
    },
    'early_stopping_rounds': 50,
    'patience': 10,  # For neural networks
}
```

## Usage Examples

### Example 1: Train LightGBM with SingleTaskStrategy
```python
from strategies.single_task import SingleTaskStrategy
import yaml

# Load configuration
with open('config/first_batch_specs.yaml') as f:
    config = yaml.safe_load(f)

# Create strategy
strategy = SingleTaskStrategy(config)

# Train separate models for each target
results = strategy.train(X, y_dict, feature_names)

# Results include:
# - Trained models with Spec 2 regularization
# - Early stopping automatically applied
# - Feature importances
```

### Example 2: Train MultiTask with Correlated Targets
```python
from strategies.multi_task import MultiTaskStrategy

config = {
    'model': {
        'shared_dim': 128,
        'learning_rate': 0.001,
    },
    'batch_size': 32,
    'n_epochs': 100,
    'patience': 10,  # Early stopping patience
    'loss_weights': {
        'tth': 1.0,
        'mdd': 0.5,
        'mfe': 1.0,
    }
}

strategy = MultiTaskStrategy(config)

# Train with multiple targets
y_dict = {
    'tth': y_tth,
    'mdd': y_mdd,
    'mfe': y_mfe,
}
results = strategy.train(X, y_dict, feature_names)

# Model will:
# - Use active Dropout during training
# - Apply early stopping with patience=10
# - Learn shared representation for correlated targets
```

## Benefits

### 1. Reduced Overfitting
- Tree models: Regularization parameters prevent trees from memorizing training data
- Neural networks: Active dropout + early stopping improve generalization

### 2. Faster Training
- Early stopping saves time by not training full epochs/iterations
- Models stop when validation loss stops improving

### 3. Better Generalization
- Lower `max_depth`, higher `min_child_weight`: simpler trees
- Row/feature sampling: reduces correlation between trees
- L1/L2 regularization: prevents extreme weights

### 4. Automatic Behavior
- Strategies auto-detect if models support early stopping
- No manual intervention required
- Falls back gracefully for models without early stopping support

## Verification

### Check Parameters Were Applied
```python
# For LightGBM
model = strategy.models['fwd_ret_1m']
params = model.get_params()
assert params['max_depth'] == 8
assert params['subsample'] == 0.75
assert params['reg_alpha'] == 0.1

# For XGBoost
model = strategy.models['fwd_ret_5m']
params = model.get_params()
assert params['max_depth'] == 7
assert params['gamma'] == 0.3
```

### Check Early Stopping Worked
```python
# Check iteration count
if hasattr(model, 'best_iteration_'):
    print(f"Stopped at iteration {model.best_iteration_} (out of {model.n_estimators})")
    # Good: Should stop well before n_estimators
```

## Migration Guide

### From Old Code
```python
# OLD: No regularization, no early stopping
config = {}
strategy = SingleTaskStrategy(config)
strategy.train(X, y_dict, feature_names)
```

### To New Code
```python
# NEW: With regularization and early stopping
import yaml

with open('config/first_batch_specs.yaml') as f:
    config = yaml.safe_load(f)

strategy = SingleTaskStrategy(config)
strategy.train(X, y_dict, feature_names)
# Models now use Spec 2 parameters automatically
```

## Files Modified

1. `TRAINING/strategies/single_task.py`
   - `_create_lightgbm_model()`: Added Spec 2 regularization
   - `_create_xgboost_model()`: Added Spec 2 regularization + gamma
   - `train()`: Added automatic early stopping detection

2. `TRAINING/strategies/multi_task.py`
   - `_train_multi_task_model()`: Added early stopping with patience
   - Ensured Dropout is active during training, disabled during inference
   - Added best model state saving and restoration

3. `TRAINING/config/first_batch_specs.yaml` (NEW)
   - Complete configuration with all recommended parameters

4. [Training Optimization Guide](TRAINING_OPTIMIZATION_GUIDE.md) (NEW)
   - Comprehensive guide for training optimization

## Next Steps

1. Test with your data: Run training with the new configurations
2. Monitor metrics: Compare train vs validation scores
3. Tune if needed: Adjust parameters in `first_batch_specs.yaml`
4. Implement two-stage pipeline: VAE/GMM → LightGBM/XGBoost

## Related Documentation

- `TRAINING_OPTIMIZATION_GUIDE.md`: Detailed explanations
- `FIRST_BATCH_SPECS_IMPLEMENTATION.md`: Trainer-level changes
- `first_batch_specs.yaml`: Parameter recommendations
