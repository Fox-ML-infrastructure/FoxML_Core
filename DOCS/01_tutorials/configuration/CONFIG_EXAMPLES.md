# Configuration Examples

Example configurations for common use cases.

> **✅ Complete Single Source of Truth (SST)**: All model trainers use config-driven hyperparameters. Full reproducibility: same config → same results.

> **📚 For comprehensive configuration documentation, see the [Configuration Reference](../../02_reference/configuration/README.md).**

## Using Experiment Configs (Recommended for Intelligent Training)

The **preferred way** to configure the intelligent training pipeline is with experiment configs. This keeps all settings in one file:

**1. Create experiment config** (`CONFIG/experiments/my_experiment.yaml`):
```yaml
experiment:
  name: my_experiment
  description: "Test run for fwd_ret_60m"

data:
  data_dir: data/data_labeled/interval=5m
  symbols: [AAPL, MSFT]
  interval: 5m
  max_samples_per_symbol: 3000

targets:
  primary: fwd_ret_60m
  candidate_targets:
    - fwd_ret_60m
    - fwd_ret_30m

feature_selection:
  top_n_features: 30
  model_families: [lightgbm, xgboost]

training:
  model_families: [lightgbm, xgboost]
  cv_folds: 5
```

**2. Use in CLI:**
```bash
python TRAINING/train.py --experiment-config my_experiment --auto-targets
```

**3. Or use programmatically:**
```python
from CONFIG.config_builder import load_experiment_config

exp_cfg = load_experiment_config("my_experiment")
# exp_cfg is a typed ExperimentConfig object with validation
```

See [Modular Config System](../../02_reference/configuration/MODULAR_CONFIG_SYSTEM.md) for complete details.

---

## Conservative Training

High regularization to prevent overfitting:

```python
from CONFIG.config_loader import load_model_config

config = load_model_config("lightgbm", variant="conservative")
# Uses: max_depth=6, learning_rate=0.01, reg_alpha=0.1, reg_lambda=0.1
```

## Fast Training

Lower regularization for faster iteration:

```python
config = load_model_config("xgboost", variant="aggressive")
# Uses: max_depth=10, learning_rate=0.05, minimal regularization
```

## Custom Overrides

Override specific parameters:

```python
config = load_model_config("mlp", overrides={
    "epochs": 100,
    "learning_rate": 0.0001,
    "batch_size": 64,
    "dropout": 0.3
})
```

## Multi-Model Training

Train multiple models with different configs:

```python
models = {
    "lightgbm": load_model_config("lightgbm", variant="conservative"),
    "xgboost": load_model_config("xgboost", variant="balanced"),
    "ensemble": load_model_config("ensemble")
}

for name, config in models.items():
    trainer = get_trainer(name)(config)
    trainer.train(X_train, y_train)
```

## Feature Selection Config

```yaml
# CONFIG/feature_selection_config.yaml
lightgbm:
  device: "cpu"  # or "gpu"
  max_depth: 8
  learning_rate: 0.03
  n_estimators: 1000
  early_stopping_rounds: 50

defaults:
  n_features: 50
  min_importance: 0.001
```

## Training Workflow Config

```yaml
# Example: training_config/first_batch_specs.yaml
data:
  train_test_split: 0.2
  validation_split: 0.2

models:
  lightgbm:
    variant: "conservative"
  xgboost:
    variant: "balanced"

training:
  early_stopping: true
  early_stopping_rounds: 50
```

  max_position_size: 1000

risk:
  max_portfolio_risk: 0.02
  stop_loss: 0.05
```

## Next Steps

- [Config Basics](CONFIG_BASICS.md) - Configuration fundamentals (includes `logging_config.yaml` example)
- [Advanced Config](ADVANCED_CONFIG.md) - Advanced configuration
- **[Modular Config System](../../02_reference/configuration/MODULAR_CONFIG_SYSTEM.md)** - Complete guide to modular configs (includes `logging_config.yaml`)
- [Configuration Reference](../../02_reference/configuration/README.md) - Complete configuration guide (includes `logging_config.yaml` documentation)
- [Config Loader API](../../02_reference/configuration/CONFIG_LOADER_API.md) - Programmatic config loading (includes logging config utilities)
- [Usage Examples](../../02_reference/configuration/USAGE_EXAMPLES.md) - Practical examples (includes interval config and CatBoost examples)
- [Ranking and Selection Consistency](../training/RANKING_SELECTION_CONSISTENCY.md) - Unified pipeline behavior guide
- [Usage Examples](../../02_reference/configuration/USAGE_EXAMPLES.md) - Practical examples

