# Training Parameters

Complete reference for training parameters and settings.

## Common Training Parameters

### Early Stopping

Prevents overfitting by stopping when validation performance stops improving.

```python
config = {
    "early_stopping_rounds": 50,  # Patience
    "n_estimators": 1000          # Maximum iterations
}
```

### Validation Split

```python
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### Regularization

**L1 (Lasso)**: `reg_alpha` - Encourages sparsity  
**L2 (Ridge)**: `reg_lambda` - Prevents large weights

```python
config = {
    "reg_alpha": 0.1,   # L1 regularization
    "reg_lambda": 0.1   # L2 regularization
}
```

## Model-Specific Parameters

### LightGBM/XGBoost

```python
{
    "max_depth": 8,
    "learning_rate": 0.03,
    "n_estimators": 1000,
    "subsample": 0.75,
    "colsample_bytree": 0.75,
    "reg_alpha": 0.05,
    "reg_lambda": 0.05,
    "early_stopping_rounds": 50
}
```

### Neural Networks

```python
{
    "epochs": 100,
    "learning_rate": 0.0001,
    "batch_size": 64,
    "dropout": 0.3,
    "hidden_size": 128,
    "patience": 10  # Early stopping patience
}
```

## Training Workflow Parameters

### 3-Phase Workflow

**Phase 1**: Feature Engineering & Selection
- Feature selection with LightGBM
- VAE feature engineering
- GMM regime detection

**Phase 2**: Core Model Training
- Train core models (LightGBM, XGBoost, Ensemble)
- Walk-forward validation

**Phase 3**: Sequential Model Training
- Train sequential models (LSTM, Transformer)
- Cascade training

## Best Practices

1. **Start Conservative**: Use conservative variant first
2. **Use Early Stopping**: Always enable early stopping
3. **Validate**: Use walk-forward validation for realistic metrics
4. **Monitor**: Track train vs validation performance

## See Also

- [Model Catalog](MODEL_CATALOG.md) - All models
- [Model Config Reference](MODEL_CONFIG_REFERENCE.md) - Config details
- [Training Optimization Guide](../../03_technical/implementation/TRAINING_OPTIMIZATION_GUIDE.md) - Optimization tips

