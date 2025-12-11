# Configuration Examples

Example configurations for common use cases.

> **✅ Complete Single Source of Truth (SST)**: All model trainers use config-driven hyperparameters. Full reproducibility: same config → same results.

> **📚 For comprehensive configuration documentation, see the [Configuration Reference](../../02_reference/configuration/README.md).**

## Intelligent Training Configuration

### Method 1: Test Config Auto-Detection (Easiest for Testing)

The system automatically detects test mode when your `--output-dir` contains "test" (case-insensitive) and uses test-friendly settings.

**1. Edit test config** (`CONFIG/training_config/pipeline_config.yaml`):
```yaml
# Test Configuration (for E2E testing)
test:
  intelligent_training:
    top_n_targets: 23
    max_targets_to_evaluate: 23
    top_m_features: 50
    min_cs: 3
    max_rows_per_symbol: 5000
    max_rows_train: 10000
```

**2. Run with test output directory:**
```bash
python -m TRAINING.orchestration.intelligent_trainer \
    --data-dir "data/data_labeled/interval=5m" \
    --symbols AAPL MSFT GOOGL TSLA NVDA \
    --output-dir "test_e2e_ranking_unified" \
    --families lightgbm xgboost random_forest
```

**What happens:**
- System detects "test" in output-dir → automatically uses `test.intelligent_training` config
- No CLI arguments needed for these settings - all from config!

### Method 2: Default Config (Production)

Edit the default config for production settings:

**1. Edit default config** (`CONFIG/training_config/pipeline_config.yaml`):
```yaml
intelligent_training:
  auto_targets: true
  top_n_targets: 5
  top_m_features: 100
  min_cs: 10
  strategy: single_task
  # ... other settings
```

**2. Run with minimal CLI:**
```bash
python -m TRAINING.orchestration.intelligent_trainer \
    --data-dir "data/data_labeled/interval=5m" \
    --symbols AAPL MSFT GOOGL
```

### Method 3: Experiment Configs (For Complex Experiments)

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
  primary: fwd_ret_60m  # Required for experiment configs

feature_selection:
  top_n: 30
  model_families: [lightgbm, xgboost]

training:
  model_families: [lightgbm, xgboost]
  cv_folds: 5
```

**2. Use in CLI:**
```bash
python -m TRAINING.orchestration.intelligent_trainer \
    --experiment-config my_experiment \
    --output-dir "my_experiment_results"
```

**Note**: Experiment configs require a `targets.primary` field. For auto-target selection, use Method 1 or 2 instead.

**3. Or use programmatically:**
```python
from CONFIG.config_builder import load_experiment_config

exp_cfg = load_experiment_config("my_experiment")
# exp_cfg is a typed ExperimentConfig object with validation
```

See [Modular Config System](../../02_reference/configuration/MODULAR_CONFIG_SYSTEM.md) and [CLI vs Config Separation](../../03_technical/design/CLI_CONFIG_SEPARATION.md) for complete details.

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

