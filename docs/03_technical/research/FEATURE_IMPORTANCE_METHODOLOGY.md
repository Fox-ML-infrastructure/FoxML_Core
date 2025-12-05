# Feature Importance Methodology

Methods for calculating and interpreting feature importance.

## Overview

Feature importance measures how much each feature contributes to model predictions. Multiple methods are used to get robust importance estimates.

## Methods

### 1. Tree-Based Importance

**LightGBM/XGBoost**: Uses built-in feature importance (gain, split, or permutation).

```python
model = lgb.train(...)
importance = model.feature_importance(importance_type='gain')
```

**Interpretation**: Higher values indicate more important features.

### 2. Permutation Importance

**Method**: Shuffle feature values and measure performance drop.

```python
from sklearn.inspection import permutation_importance

perm_importance = permutation_importance(model, X_test, y_test)
```

**Interpretation**: Larger drops indicate more important features.

### 3. SHAP Values

**Method**: Shapley Additive Explanations for feature contributions.

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
```

**Interpretation**: Average absolute SHAP value indicates importance.

## Aggregation Methods

### Multi-Model Consensus

Combine importance from multiple models:

```python
# Get importance from multiple models
lightgbm_importance = lightgbm_model.feature_importance()
xgboost_importance = xgboost_model.feature_importance()
rf_importance = rf_model.feature_importance()

# Aggregate (mean or weighted)
aggregated = (lightgbm_importance + xgboost_importance + rf_importance) / 3
```

### Target-Specific Ranking

Rank features for each target separately:

```python
for target in targets:
    importance = get_importance_for_target(X, y[target])
    rank_features(importance, target=target)
```

## Comprehensive Ranking

Combine predictive power and data quality:

```python
python scripts/rank_features_comprehensive.py \
    --target y_will_peak_60m_0.8 \
    --output-dir results/ranking
```

**Outputs:**
- Predictive importance (IC, R²)
- Data quality metrics (variance, missing data)
- Composite edge score

## Interpreting Results

### High Importance Features

- Strong predictive power
- Low redundancy
- Good data quality

### Low Importance Features

- Weak predictive power
- High redundancy
- Poor data quality

## Best Practices

1. **Use multiple methods**: Don't rely on single importance metric
2. **Validate selection**: Check performance with selected features
3. **Monitor stability**: Importance should be consistent across folds
4. **Consider costs**: Balance importance with computation costs

## See Also

- [Feature Selection Tutorial](../../01_tutorials/training/FEATURE_SELECTION_TUTORIAL.md) - Selection guide
- [Comprehensive Feature Ranking](../../COMPREHENSIVE_FEATURE_RANKING.md) - Ranking methodology

