# Advanced Configuration

Advanced configuration techniques and patterns.

## Configuration Overlays

Combine multiple config files:

```python
from CONFIG.config_loader import load_config

# Load base config
base_config = load_config("CONFIG/base.yaml")

# Load overlay
overlay_config = load_config("CONFIG/overlays/custom.yaml")

# Merge (overlay takes precedence)
config = {**base_config, **overlay_config}
```

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

Validate configs before use:

```python
from CONFIG.config_loader import validate_config

config = load_model_config("lightgbm")
errors = validate_config(config)

if errors:
    print(f"Config errors: {errors}")
else:
    trainer = LightGBMTrainer(config)
```

## Multi-Target Configuration

Configure for multiple targets:

```yaml
# CONFIG/target_configs.yaml
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
# CONFIG/feature_groups.yaml
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

- [Config Basics](CONFIG_BASICS.md) - Configuration fundamentals
- [Config Examples](CONFIG_EXAMPLES.md) - Example configurations
- [Config Loader API](../../02_reference/configuration/CONFIG_LOADER_API.md) - Complete API

