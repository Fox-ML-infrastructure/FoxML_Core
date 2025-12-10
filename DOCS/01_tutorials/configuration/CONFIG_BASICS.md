# Configuration Basics

Learn the fundamentals of FoxML Core configuration.

## Overview

FoxML Core uses centralized YAML configuration files for all models and training workflows. **All 52+ model trainers and specialized models** auto-load configs from the `CONFIG/model_config/` directory.

**✅ Complete Single Source of Truth (SST)**: As of 2025-12-10, **ALL** hardcoded configuration values have been removed from the TRAINING pipeline. Every hyperparameter, test split size, and random seed now loads from centralized config files, ensuring complete reproducibility: same config → same results across all pipeline stages.

> **Note:** These SST improvements were internal changes. Your existing code and configs continue to work unchanged - no migration required. The system automatically uses config-driven parameters and deterministic seeds.

**NEW: Modular Configuration System** - For the intelligent training pipeline, we recommend using **experiment configs** which group all settings in one file and prevent config "crossing" between modules. See [Modular Config System](../../02_reference/configuration/MODULAR_CONFIG_SYSTEM.md) for details.

> **📚 For complete configuration documentation, see the [Configuration Reference](../../02_reference/configuration/README.md).**

## Basic Usage

### Auto-Load Configuration

```python
from TRAINING.model_fun import LightGBMTrainer

# Automatically loads config from CONFIG/model_config/lightgbm.yaml
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

### Model Configs

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

### Logging Config (NEW)

```yaml
# CONFIG/logging_config.yaml
logging:
  global_level: INFO
  
  modules:
    rank_target_predictability:
      level: INFO
      gpu_detail: false      # GPU confirmations
      cv_detail: false        # Fold timestamps
      edu_hints: false        # Educational hints
  
  backends:
    lightgbm:
      native_verbosity: -1    # -1=silent, 0=info, >0=more spam
  
  profiles:
    debug_run:
      global_level: DEBUG
      modules:
        rank_target_predictability:
          gpu_detail: true
          cv_detail: true
```

**Usage:**
```python
from CONFIG.logging_config_utils import get_module_logging_config

log_cfg = get_module_logging_config('rank_target_predictability')
if log_cfg.gpu_detail:
    logger.info("🚀 Training on GPU...")
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
- **[Modular Config System](../../02_reference/configuration/MODULAR_CONFIG_SYSTEM.md)** - Complete guide to modular configs (includes `logging_config.yaml`)
- [Configuration Reference](../../02_reference/configuration/README.md) - Complete configuration guide (includes `logging_config.yaml` documentation)
- [Config Loader API](../../02_reference/configuration/CONFIG_LOADER_API.md) - Programmatic config loading (includes logging config utilities)
- [Usage Examples](../../02_reference/configuration/USAGE_EXAMPLES.md) - Practical configuration examples (includes interval config and CatBoost examples)
- [Ranking and Selection Consistency](../training/RANKING_SELECTION_CONSISTENCY.md) - Unified pipeline behavior guide

