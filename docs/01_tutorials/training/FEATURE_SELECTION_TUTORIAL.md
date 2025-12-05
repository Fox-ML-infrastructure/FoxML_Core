# Feature Selection Tutorial

Select the most important features for your models.

## Overview

Feature selection reduces dimensionality and improves model performance by:
1. Training models to get feature importance
2. Ranking features by importance
3. Selecting top N features
4. Retraining with selected features

## Quick Start

### Single-Target Selection

```python
from TRAINING.strategies.single_task import SingleTaskStrategy
from scripts.feature_selection import select_top_features

# Train on all features
config = load_model_config("lightgbm", variant="conservative")
strategy = SingleTaskStrategy(config)
strategy.train(X, {'fwd_ret_5m': y}, feature_names)

# Select top 50 features
selected_features = select_top_features(strategy, n_features=50)

# Retrain with selected features
X_selected = X[selected_features]
strategy.train(X_selected, {'fwd_ret_5m': y}, selected_features)
```

### Multi-Target Selection

```python
from TRAINING.strategies.multi_task import MultiTaskStrategy

# Train on all features with multiple targets
targets = {
    'fwd_ret_5m': y_5m,
    'fwd_ret_15m': y_15m,
    'fwd_ret_30m': y_30m
}
strategy = MultiTaskStrategy(config)
strategy.train(X, targets, feature_names)

# Get aggregated importance
selected_features = select_top_features(strategy, n_features=50)
```

## Feature Selection Methods

### Importance-Based

Uses model feature importance (LightGBM/XGBoost):

```python
from scripts.feature_selection import importance_based_selection

selected = importance_based_selection(
    X, y,
    n_features=50,
    model_type='lightgbm'
)
```

### Recursive Feature Elimination

Iteratively removes least important features:

```python
from scripts.feature_selection import rfe_selection

selected = rfe_selection(
    X, y,
    n_features=50,
    model_type='lightgbm'
)
```

## Comprehensive Ranking

Rank features by both predictive power and data quality:

```python
python scripts/rank_features_comprehensive.py \
    --data data/labeled/AAPL_labeled.parquet \
    --target target_fwd_ret_5m \
    --output results/feature_ranking.csv
```

## Best Practices

1. **Start with all features**: Train on full feature set first
2. **Use multiple methods**: Combine importance and RFE
3. **Validate selection**: Check performance with selected features
4. **Monitor stability**: Feature importance should be consistent

## Reducing from 421 to 50 Features

```python
# Step 1: Train on all 421 features
strategy = SingleTaskStrategy(config)
strategy.train(X_all, y, feature_names_all)

# Step 2: Get importance
importances = strategy.get_feature_importance()

# Step 3: Select top 50
top_50 = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:50]
selected_features = [f[0] for f in top_50]

# Step 4: Retrain
X_selected = X_all[selected_features]
strategy.train(X_selected, y, selected_features)
```

## Next Steps

- [Feature Selection Guide](../../../TRAINING/FEATURE_SELECTION_GUIDE.md) - Detailed guide
- [Comprehensive Feature Ranking](../../COMPREHENSIVE_FEATURE_RANKING.md) - Advanced ranking
- [Multi-Model Feature Selection](../../../INFORMATION/MULTI_MODEL_FEATURE_SELECTION.md) - Multi-model approach

