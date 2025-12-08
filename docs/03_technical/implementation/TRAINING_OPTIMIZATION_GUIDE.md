# Training Optimization Guide

Actionable guidance for optimizing the model training pipeline. Addresses overfitting in tree models and inactive regularization in neural networks.

## Problem 1: Overfitting in Tree Models

### Symptoms
- Training accuracy >> Validation accuracy
- `max_depth=-1` (unlimited depth)
- `subsample=1.0` (no row sampling)
- Low or zero `min_child_weight`
- No L1/L2 regularization

### Solution: Spec 2 High Regularization

Trainer files (`lightgbm_trainer.py`, `xgboost_trainer.py`) use Spec 2 defaults:

#### LightGBM
```python
{
    'num_leaves': 96,              # 64-128 range (was 255)
    'max_depth': 8,                # 7-9 range (was -1)
    'min_child_weight': 0.5,       # 0.1-1.0 range (was 0)
    'learning_rate': 0.03,         # 0.01-0.05 range (was 0.05)
    'subsample': 0.75,             # 0.7-0.8 range (was 1.0)
    'colsample_bytree': 0.75,      # 0.7-0.8 range (was 1.0)
    'reg_alpha': 0.1,              # L1 reg (was 1.0)
    'reg_lambda': 0.1,             # L2 reg (was 2.0)
    'n_estimators': 1000,          # Use with early stopping
    'early_stopping_rounds': 50,   # Stop when no improvement
}
```

#### XGBoost
```python
{
    'max_depth': 7,                # 5-8 range (was 8)
    'min_child_weight': 0.5,       # 0.1-1.0 range (was 10)
    'gamma': 0.3,                  # min_split_gain (NEW)
    'learning_rate': 0.03,         # 0.01-0.05 range (was 0.05)
    'subsample': 0.75,             # 0.7-0.8 range (was 0.7)
    'colsample_bytree': 0.75,      # 0.7-0.8 range (was 0.7)
    'reg_alpha': 0.1,              # L1 reg (was 1.0)
    'reg_lambda': 0.1,             # L2 reg (was 2.0)
    'n_estimators': 1000,          # Use with early stopping
    'early_stopping_rounds': 50,   # Stop when no improvement
}
```

## Problem 2: Inactive Dropout in Neural Networks

### Symptoms
- Dropout layers set with `training=False` (disables dropout)
- No early stopping (training runs full epochs)
- Models train well but generalize poorly

### Solution: Active Dropout + Early Stopping

#### PyTorch (multi_task.py)
```python
# Dropout is ACTIVE during training
self.shared_model.train()  # Enables Dropout
for epoch in range(n_epochs):
    # Dropout randomly drops units here
    predictions = self.shared_model(batch_X)

# Dropout is DISABLED during inference
self.shared_model.eval()  # Disables Dropout
predictions = self.shared_model(X_test)

# Early stopping added
patience = 10
if patience_counter >= patience:
    logger.info(f"Early stopping at epoch {epoch}")
    # Restore best model weights
    model.load_state_dict(best_model_state)
    break
```

#### TensorFlow/Keras (multi_task_trainer.py)
```python
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
]
model.fit(X, y, validation_data=(X_val, y_val), callbacks=callbacks)
```

## Problem 3: Early Stopping Not Applied

### Solution: Auto-detect and Apply

The `single_task.py` strategy automatically detects if models support early stopping:

```python
# Auto-detect early stopping support
if hasattr(model, 'fit') and 'eval_set' in model.fit.__code__.co_varnames:
    # Split data for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    # Fit with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    logger.info(f"Early stopping at iteration {model.best_iteration_}")
else:
    # Standard fit for models without early stopping
    model.fit(X, y)
```

## Two-Stage Pipeline Architecture

### Stage 1: Feature Engineering

Models: `VAE`, `GMMRegime`, `ChangePoint`

Purpose: Create new, informative features

Process:
1. Train on X only (your 421 features) or price series
2. Extract outputs:
   - `VAE`: Latent features (e.g., 20 compressed features)
   - `GMMRegime`: Regime labels (e.g., `regime=0,1,2,3`)
   - `ChangePoint`: Changepoint features (e.g., `days_since_last_break`)

Example:
```python
# Stage 1: Feature Engineering
vae = VAETrainer(config)
vae.train(X)  # Train on features only
vae_features = vae.encode(X)  # Get latent features (N, 20)

gmm = GMMRegimeTrainer(config)
gmm.train(X)
regime_labels = gmm.predict(X)  # Get regime (N,)

# Concatenate new features
X_enriched = np.column_stack([X, vae_features, regime_labels])
# Shape: (N, 421 + 20 + 1) = (N, 442)
```

### Stage 2: Prediction

Models: `LightGBM`, `XGBoost`, `MultiTask`, `Ensemble`

Purpose: Predict targets (TTH, MDD, MFE, returns)

Process:
1. Use X_enriched (original + engineered features)
2. Train prediction models using strategies:
   - `SingleTaskStrategy`: Separate model per target
   - `MultiTaskStrategy`: One model for correlated targets
   - `CascadeStrategy`: Barrier models + return models

Example:
```python
# Stage 2: Prediction
from strategies.single_task import SingleTaskStrategy

strategy = SingleTaskStrategy(config)
results = strategy.train(X_enriched, y_dict, feature_names)

# Or use MultiTask for correlated targets
from strategies.multi_task import MultiTaskStrategy

y_multi = np.column_stack([y_tth, y_mdd, y_mfe])
strategy = MultiTaskStrategy(config)
results = strategy.train(X_enriched, y_multi, feature_names)
```

## Model-to-Strategy Mapping

| Model Family | Strategy | Purpose |
|--------------|----------|---------|
| `LightGBM`, `XGBoost` | `SingleTaskStrategy` | Independent models per target |
| `MultiTask`, `MLP` | `MultiTaskStrategy` | Shared encoder for correlated targets |
| `Ensemble`, `MetaLearning` | `SingleTaskStrategy` | Robust predictions via stacking |
| `QuantileLightGBM`, `NGBoost` | `SingleTaskStrategy` | Probabilistic predictions (ranges) |
| `VAE`, `GMMRegime`, `ChangePoint` | Stage 1 only | Feature engineering |
| `CascadeStrategy` | Special use case | Barrier models + return models |
| `RewardBased`, `FTRL`, `GAN` | R&D projects | Advanced, separate frameworks |

## Configuration Examples

### Example 1: Single-Task with LightGBM
```yaml
# config/single_task_lightgbm.yaml
strategy: single_task
models:
  lightgbm:
    num_leaves: 96
    max_depth: 8
    min_child_weight: 0.5
    learning_rate: 0.03
    n_estimators: 1000
    bagging_fraction: 0.75
    feature_fraction: 0.75
    lambda_l1: 0.1
    lambda_l2: 0.1
early_stopping_rounds: 50
```

### Example 2: Multi-Task with Correlated Targets
```yaml
# config/multi_task_correlated.yaml
strategy: multi_task
target_names: ['tth', 'mdd', 'mfe']
loss_weights:
  tth: 1.0
  mdd: 0.5
  mfe: 1.0
model:
  shared_dim: 128
  learning_rate: 0.0001
batch_size: 32
n_epochs: 100
patience: 10
```

### Example 3: Two-Stage Pipeline
```python
# Stage 1: Feature Engineering
from model_fun.vae_trainer import VAETrainer
from model_fun.gmm_regime_trainer import GMMRegimeTrainer

# Train VAE
vae = VAETrainer(config)
vae.train(X)
vae_features = vae.encode(X)

# Train GMM
gmm = GMMRegimeTrainer({'n_components': 4})
gmm.train(X)
regime_labels = gmm.predict(X)

# Enrich features
X_enriched = np.column_stack([X, vae_features, regime_labels])

# Stage 2: Prediction
from strategies.single_task import SingleTaskStrategy

strategy = SingleTaskStrategy(config)
results = strategy.train(X_enriched, y_dict, feature_names)
```

## Verification

### Check 1: Overfitting Fixed
```python
# After training, check train vs validation scores
train_score = model.score(X_train, y_train)
val_score = model.score(X_val, y_val)
print(f"Train: {train_score:.4f}, Val: {val_score:.4f}")
# Good: val_score should be close to train_score (within 0.05)
```

### Check 2: Early Stopping Working
```python
# Check if early stopping triggered
if hasattr(model, 'best_iteration_'):
    print(f"Early stopping at iteration {model.best_iteration_}")
    # Good: Should stop well before max_estimators
```

### Check 3: Dropout Active
```python
# PyTorch: Check model mode
model.train()  # Should enable Dropout
print(model.encoder.training)  # Should be True

model.eval()   # Should disable Dropout
print(model.encoder.training)  # Should be False
```

## References

- Spec 1: Multi-task Learning (MTL) with multiple output heads
- Spec 2: High Regularization for Gradient Boosted Trees
- Spec 3: Stacking Regressor with Cross-Validation
- Configuration: `TRAINING/config/first_batch_specs.yaml`
- Implementation Summary: [First Batch Specs](FIRST_BATCH_SPECS_IMPLEMENTATION.md)
