# Advanced Configuration

Advanced configuration techniques and patterns.

> **📚 For comprehensive configuration documentation, see the [Configuration Reference](../../02_reference/configuration/README.md).**

## Configuration Overlays

Combine multiple config files:

```python
import yaml
from pathlib import Path

# Load base config (example - adjust path as needed)
with open("CONFIG/training_config/pipeline_config.yaml") as f:
    base_config = yaml.safe_load(f)

# Load overlay (example - adjust path as needed)
with open("CONFIG/training_config/gpu_config.yaml") as f:
    overlay_config = yaml.safe_load(f)

# Merge (overlay takes precedence)
config = {**base_config, **overlay_config}
```

> **Note**: `load_config` function does not exist in `CONFIG.config_loader`. Use `yaml.safe_load()` directly or use `load_model_config()` / `load_training_config()` for specific config types.

## Environment-Based Configuration

Use environment variables for different environments:

```python
import os

env = os.getenv("ENVIRONMENT", "development")

if env == "production":
    config = load_model_config("lightgbm", variant="conservative")
elif env == "development":
    config = load_model_config("lightgbm", variant="aggressive")
```

## Dynamic Configuration

Generate configs programmatically:

```python
def create_custom_config(base_model, learning_rate, max_depth):
    base = load_model_config(base_model)
    return {
        **base,
        "learning_rate": learning_rate,
        "max_depth": max_depth
    }

config = create_custom_config("lightgbm", 0.02, 7)
```

## Configuration Validation

> **Note**: `validate_config` function does not exist in `CONFIG.config_loader`. Validate configs manually or use try/except when loading.

```python
from CONFIG.config_loader import load_model_config
from TRAINING.model_fun import LightGBMTrainer

config = load_model_config("lightgbm")

# Validate by attempting to create trainer
try:
    trainer = LightGBMTrainer(config)
    print("Config is valid")
except Exception as e:
    print(f"Config error: {e}")
```

## Multi-Target Configuration

Configure for multiple targets:

```yaml
# Example: target_configs.yaml
targets:
  fwd_ret_5m:
    horizon: "5m"
    barrier: 0.001
  fwd_ret_15m:
    horizon: "15m"
    barrier: 0.002
  fwd_ret_30m:
    horizon: "30m"
    barrier: 0.003
```

## Feature Group Configuration

Define feature groups for concept-based selection:

```yaml
# Example: feature_groups.yaml
feature_groups:
  price_features:
    - "return_1m"
    - "return_5m"
    - "volatility_5m"
  volume_features:
    - "volume_ratio"
    - "vwap"
  technical_indicators:
    - "rsi"
    - "macd"
```

## Runtime Configuration Overrides

Override configs at runtime:

```python
# From command line
import sys
overrides = {}
if "--fast" in sys.argv:
    overrides = {"learning_rate": 0.1, "n_estimators": 100}

config = load_model_config("lightgbm", overrides=overrides)
```

## Configuration Templates

Create reusable templates:

```python
# config_templates.py
CONSERVATIVE_TEMPLATE = {
    "max_depth": 6,
    "learning_rate": 0.01,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1
}

def apply_template(model_name, template):
    base = load_model_config(model_name)
    return {**base, **template}

config = apply_template("lightgbm", CONSERVATIVE_TEMPLATE)
```

## Next Steps

- [Config Basics](CONFIG_BASICS.md) - Configuration fundamentals (includes `logging_config.yaml` example)
- [Config Examples](CONFIG_EXAMPLES.md) - Example configurations
- **[Modular Config System](../../02_reference/configuration/MODULAR_CONFIG_SYSTEM.md)** - Complete guide to modular configs (includes `logging_config.yaml`)
- [Configuration Reference](../../02_reference/configuration/README.md) - Complete configuration guide (includes `logging_config.yaml` documentation)
- [Config Loader API](../../02_reference/configuration/CONFIG_LOADER_API.md) - Programmatic config loading (includes logging config utilities)
- [Usage Examples](../../02_reference/configuration/USAGE_EXAMPLES.md) - Practical examples
- [Usage Examples](../../02_reference/configuration/USAGE_EXAMPLES.md) - Practical examples

