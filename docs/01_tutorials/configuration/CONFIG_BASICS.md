# Configuration Basics

Learn the fundamentals of Fox-v1-infra configuration.

## Overview

Fox-v1-infra uses centralized YAML configuration files for all models and training workflows. All 17 production trainers auto-load configs from `CONFIG/model_config/`.

## Basic Usage

### Auto-Load Configuration

```python
from TRAINING.model_fun import LightGBMTrainer

# Automatically loads CONFIG/model_config/lightgbm.yaml
trainer = LightGBMTrainer()
trainer.train(X, y)
```

### Load Specific Variant

```python
from CONFIG.config_loader import load_model_config

config = load_model_config("xgboost", variant="conservative")
trainer = XGBoostTrainer(config)
```

### Override Parameters

```python
config = load_model_config("mlp", overrides={
    "epochs": 100,
    "learning_rate": 0.0001
})
trainer = MLPTrainer(config)
```

## Available Models

**Core:** `lightgbm`, `xgboost`, `ensemble`, `multi_task`  
**Deep Learning:** `mlp`, `transformer`, `lstm`, `cnn1d`  
**Feature Engineering:** `vae`, `gan`, `gmm_regime`  
**Probabilistic:** `ngboost`, `quantile_lightgbm`  
**Advanced:** `change_point`, `ftrl_proximal`, `reward_based`, `meta_learning`

## Configuration Variants

Each model has 3 variants:

- **conservative**: Highest regularization, least overfitting
- **balanced**: Default settings
- **aggressive**: Faster training, lower regularization

```python
config = load_model_config("lightgbm", variant="conservative")
```

## Configuration Structure

```yaml
# CONFIG/model_config/lightgbm.yaml
default:
  max_depth: 8
  learning_rate: 0.03
  n_estimators: 1000

variants:
  conservative:
    max_depth: 6
    learning_rate: 0.01
    reg_alpha: 0.1
    reg_lambda: 0.1
  
  balanced:
    max_depth: 8
    learning_rate: 0.03
  
  aggressive:
    max_depth: 10
    learning_rate: 0.05
    reg_alpha: 0.01
    reg_lambda: 0.01
```

## Common Overrides

```python
# Change learning rate
config = load_model_config("lightgbm", overrides={"learning_rate": 0.01})

# Change regularization
config = load_model_config("xgboost", overrides={
    "reg_alpha": 0.2,
    "reg_lambda": 0.2
})

# Change training iterations
config = load_model_config("mlp", overrides={"epochs": 200})
```

## Next Steps

- [Config Examples](CONFIG_EXAMPLES.md) - Example configurations
- [Advanced Config](ADVANCED_CONFIG.md) - Advanced configuration
- [Config Reference](../../02_reference/configuration/CONFIG_LOADER_API.md) - Complete API reference

