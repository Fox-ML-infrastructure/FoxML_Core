# Configuration Overlays

Use configuration overlays to combine multiple config files.

## Overview

Overlays allow you to:
- Start with a base configuration
- Apply environment-specific overrides
- Merge multiple config sources
- Maintain separate configs for different use cases

## Basic Overlay

```python
from CONFIG.config_loader import load_config

# Load base config
base = load_config("CONFIG/base.yaml")

# Load overlay
overlay = load_config("CONFIG/overlays/production.yaml")

# Merge (overlay takes precedence)
config = {**base, **overlay}
```

## Environment-Based Overlays

```python
import os

env = os.getenv("ENVIRONMENT", "development")

# Load base
base = load_config("CONFIG/base.yaml")

# Load environment overlay
if env == "production":
    overlay = load_config("CONFIG/overlays/production.yaml")
elif env == "development":
    overlay = load_config("CONFIG/overlays/development.yaml")

config = {**base, **overlay}
```

## Deep Merging

For nested dictionaries, use deep merge:

```python
def deep_merge(base, overlay):
    result = base.copy()
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result

config = deep_merge(base, overlay)
```

## Overlay Structure

Create overlay files in `CONFIG/overlays/`:

```yaml
# CONFIG/overlays/production.yaml
models:
  lightgbm:
    variant: "conservative"
  xgboost:
    variant: "conservative"

training:
  early_stopping_rounds: 100
```

## Use Cases

### Production vs Development

```yaml
# development.yaml
training:
  early_stopping_rounds: 20
  n_estimators: 100

# production.yaml
training:
  early_stopping_rounds: 100
  n_estimators: 1000
```

### Model-Specific Overlays

```yaml
# lightgbm_overlay.yaml
lightgbm:
  max_depth: 10
  learning_rate: 0.05
```

## See Also

- [Config Loader API](CONFIG_LOADER_API.md) - Loader API
- [Advanced Config](../../01_tutorials/configuration/ADVANCED_CONFIG.md) - Advanced techniques

