# First Batch Training Specifications - Implementation Summary

## Overview

This document summarizes the implementation of the "1st batch" training specifications for the model zoo. The changes address overfitting issues and add support for multi-target training with correlated targets (TTH, MDD, MFE).

## Changes Implemented

### 1. LightGBM Trainer (Spec 2: High Regularization)

File: `TRAINING/model_fun/lightgbm_trainer.py`

Changes:
- Reduced `num_leaves` from 255 to 96 (64-128 range)
- Added `min_child_weight: 0.5` (0.1-1.0 range)
- Increased `feature_fraction` and `bagging_fraction` to 0.75 (0.7-0.8 range)
- Reduced `lambda_l1` and `lambda_l2` from 1.0/2.0 to 0.1 (0.1-1.0 range)
- Reduced `learning_rate` to 0.03 (0.01-0.05 range)
- Reduced `early_stopping_rounds` from 200 to 50
- Set `n_estimators` to 1000 (rely on early stopping)

Purpose: Fix overfitting while maintaining high model complexity.

### 2. XGBoost Trainer (Spec 2: High Regularization)

File: `TRAINING/model_fun/xgboost_trainer.py`

Changes:
- Reduced `max_depth` from 8 to 7 (5-8 range)
- Reduced `min_child_weight` from 10 to 0.5 (0.1-1.0 range)
- Added `gamma: 0.3` (min_split_gain: 0.1-0.5 range)
- Increased `subsample` and `colsample_bytree` to 0.75 (0.7-0.8 range)
- Reduced `reg_alpha` and `reg_lambda` from 1.0/2.0 to 0.1 (0.1-1.0 range)
- Reduced `eta` (learning_rate) to 0.03 (0.01-0.05 range)
- Reduced `early_stopping_rounds` from 200 to 50

Purpose: Same as LightGBM - fix overfitting with proper regularization.

### 3. MultiTask Trainer (Spec 1: MTL with Multiple Output Heads)

File: `TRAINING/model_fun/multi_task_trainer.py`

Changes:
- Added support for multiple output heads for correlated targets
- Auto-detects multi-target mode from `y` shape (2D with multiple columns)
- Configurable `target_names` (e.g., ["tth", "mdd", "mfe"])
- Configurable `loss_weights` per target (default: all 1.0)
- Shared hidden layers: Dense(256, ReLU), BN, Dropout(0.2), Dense(128, ReLU), BN, Dropout(0.2)
- Separate output head per target: Dense(1, linear)
- Learning rate set to 3e-4 (1e-4 to 5e-4 range)
- Backward compatible: still works with single-target data

Purpose: Train one model for correlated targets (TTH, MDD, MFE) instead of separate models, leveraging shared representation learning.

Usage Example:
```python
# Multi-target training (auto-detected from y shape)
y_multi = np.column_stack([y_tth, y_mdd, y_mfe])  # Shape: (N, 3)
trainer = MultiTaskTrainer(config={
    "target_names": ["tth", "mdd", "mfe"],
    "loss_weights": {"tth": 1.0, "mdd": 0.5, "mfe": 1.0}
})
trainer.train(X, y_multi)
```

### 4. Ensemble Trainer (Spec 3: Stacking Regressor with CV)

File: `TRAINING/model_fun/ensemble_trainer.py`

Changes:
- Added `StackingRegressor` with cross-validation (default: K=5)
- Base estimators: HistGradientBoosting, RandomForest, LinearRegression
- Final estimator: Ridge with L2 regularization (alpha: 1.0-10.0, tunable)
- Reduced RF `max_depth` from 18 to 15 (Spec 3 recommendation)
- Configurable: `use_stacking=True` (new) vs `use_stacking=False` (legacy weighted blend)
- Proper CV prevents data leakage in meta-learning

Purpose: Robust ensemble predictions using proper stacking with cross-validation instead of simple weighted averaging.

Configuration:
```python
config = {
    "use_stacking": True,      # Use StackingRegressor (Spec 3)
    "stacking_cv": 5,          # K-fold CV (5 or 10)
    "final_estimator_alpha": 1.0  # Ridge alpha for final estimator
}
```

### 5. Configuration File

File: `TRAINING/config/first_batch_specs.yaml`

Contents:
- Complete specifications for all model families
- Spec 1 (MTL), Spec 2 (High Regularization), Spec 3 (Stacking)
- Specialized boosters (QuantileLightGBM, NGBoost)
- Time series/regime models (GMMRegime, ChangePoint)
- Usage notes and recommendations

## Key Benefits

1. Reduced Overfitting: Spec 2 parameters significantly reduce overfitting in LightGBM/XGBoost while maintaining model complexity.

2. Multi-Target Learning: Spec 1 allows training one model for correlated targets (TTH, MDD, MFE), leveraging shared representation.

3. Robust Ensembles: Spec 3 uses proper stacking with CV, preventing data leakage and improving generalization.

4. Backward Compatibility: All changes maintain backward compatibility with existing single-target workflows.

## Usage Recommendations

### For Correlated Targets (TTH, MDD, MFE):
- Use MultiTask trainer with multi-head mode (Spec 1)
- Train ONE model with multiple output heads
- Configure loss weights if certain targets are more important

### For Single Targets:
- Use LightGBM or XGBoost with Spec 2 parameters (high regularization)
- Train separate models for each target
- Use early stopping (50 rounds) to prevent overfitting

### For Robust Predictions:
- Use Ensemble trainer with StackingRegressor (Spec 3)
- Enable `use_stacking=True` and set `stacking_cv=5` or `10`
- Tune `final_estimator_alpha` (1.0-10.0) using cross-validation

## Testing

To verify the implementations:

1. LightGBM/XGBoost: Check that models use the new regularization parameters and early stopping works correctly.

2. MultiTask: Test with both single-target (backward compatibility) and multi-target data (new feature).

3. Ensemble: Test with both `use_stacking=True` (Spec 3) and `use_stacking=False` (legacy mode).

## Configuration Loading

The configuration file can be loaded and merged with existing configs:

```python
import yaml
from pathlib import Path

# Load first batch specs
with open("TRAINING/config/first_batch_specs.yaml") as f:
    first_batch_config = yaml.safe_load(f)

# Merge with existing config
config = {
    **existing_config,
    **first_batch_config.get("lightgbm", {}),
    # ... etc
}
```

## Next Steps

1. QuantileLightGBM: Train two models (alpha=0.05 and alpha=0.95) to get prediction ranges.

2. NGBoost: Configure distribution (Normal/LogNormal) and base learner depth (3-5).

3. Regime Models: Use GMMRegime and ChangePoint for feature engineering (regime indicators, changepoint features).

4. Tuning: Use cross-validation to tune hyperparameters (especially `final_estimator_alpha` in Ensemble).

## Files Modified

- `TRAINING/model_fun/lightgbm_trainer.py`
- `TRAINING/model_fun/xgboost_trainer.py`
- `TRAINING/model_fun/multi_task_trainer.py`
- `TRAINING/model_fun/ensemble_trainer.py`
- `TRAINING/config/first_batch_specs.yaml` (new)

## References

- Spec 1: Multitask Learning (MTL) with multiple output heads
- Spec 2: High Regularization for Gradient Boosted Trees
- Spec 3: Stacking Regressor with Cross-Validation
