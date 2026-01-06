# Model Configuration Reference

Complete reference for model configuration files.

## Configuration Location

All model configs are in `CONFIG/model_config/`:

```
CONFIG/model_config/
├── lightgbm.yaml
├── xgboost.yaml
├── ensemble.yaml
├── multi_task.yaml
├── mlp.yaml
├── transformer.yaml
├── lstm.yaml
├── cnn1d.yaml
├── vae.yaml
├── gan.yaml
├── gmm_regime.yaml
├── ngboost.yaml
├── quantile_lightgbm.yaml
├── change_point.yaml
├── ftrl_proximal.yaml
├── reward_based.yaml
└── meta_learning.yaml
```

## Configuration Structure

Each config file has:

```yaml
default:
  # Default parameters (balanced variant)

variants:
  conservative:
    # High regularization, less overfitting
  balanced:
    # Default settings
  aggressive:
    # Lower regularization, faster training
```

## Loading Configs

```python
from CONFIG.config_loader import load_model_config

# Load default (balanced)
config = load_model_config("lightgbm")

# Load variant
config = load_model_config("xgboost", variant="conservative")

# Load with overrides
config = load_model_config("mlp", overrides={"epochs": 100})
```

## Common Parameters

### Tree Models (LightGBM, XGBoost)

- `max_depth`: Maximum tree depth (6-10)
- `learning_rate`: Learning rate (0.01-0.05)
- `n_estimators`: Number of trees (1000+)
- `subsample`: Row sampling (0.75)
- `colsample_bytree`: Feature sampling (0.75)
- `reg_alpha`: L1 regularization (0.01-0.1)
- `reg_lambda`: L2 regularization (0.01-0.1)
- `early_stopping_rounds`: Early stopping patience (50)

### Neural Networks (MLP, LSTM, Transformer)

- `epochs`: Training epochs (50-200)
- `learning_rate`: Learning rate (0.0001-0.001)
- `batch_size`: Batch size (32-128)
- `dropout`: Dropout rate (0.1-0.5)
- `hidden_size`: Hidden layer size (64-512)

## Variant Differences

### Conservative

- Higher regularization
- Lower learning rate
- Smaller model capacity
- **Use for**: Production, avoiding overfitting

### Balanced

- Default settings
- Moderate regularization
- **Use for**: General use, starting point

### Aggressive

- Lower regularization
- Higher learning rate
- Larger model capacity
- **Use for**: Fast iteration, exploration

## See Also

- [Model Catalog](MODEL_CATALOG.md) - All available models
- [Training Parameters](TRAINING_PARAMETERS.md) - Training settings
- [Config Basics](../../01_tutorials/configuration/CONFIG_BASICS.md) - Configuration tutorial

