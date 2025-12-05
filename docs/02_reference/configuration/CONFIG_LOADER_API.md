# Config Loader API

Complete API reference for configuration loading.

## Functions

### load_model_config

Load model configuration with optional variant and overrides.

```python
from CONFIG.config_loader import load_model_config

# Load default (balanced variant)
config = load_model_config("lightgbm")

# Load specific variant
config = load_model_config("xgboost", variant="conservative")

# Load with overrides
config = load_model_config("mlp", overrides={"epochs": 100})
```

**Parameters:**
- `model_name` (str): Model name (e.g., "lightgbm", "xgboost")
- `variant` (str, optional): Variant name ("conservative", "balanced", "aggressive")
- `overrides` (dict, optional): Parameter overrides

**Returns:** dict - Configuration dictionary

### load_training_config

Load training workflow configuration.

```python
from CONFIG.config_loader import load_training_config

config = load_training_config("first_batch_specs")
```

**Parameters:**
- `config_name` (str): Config name
- `overrides` (dict, optional): Parameter overrides

**Returns:** dict - Configuration dictionary

### get_available_model_configs

List all available model configurations.

```python
from CONFIG.config_loader import get_available_model_configs

configs = get_available_model_configs()
# Returns: {"lightgbm": ["conservative", "balanced", "aggressive"], ...}
```

**Returns:** dict - Model names and their variants

### get_config_variants

Get available variants for a model.

```python
from CONFIG.config_loader import get_config_variants

variants = get_config_variants("lightgbm")
# Returns: ["conservative", "balanced", "aggressive"]
```

**Parameters:**
- `model_name` (str): Model name

**Returns:** list - Available variant names

## Environment Variables

### MODEL_VARIANT

Set default variant for all models:

```bash
export MODEL_VARIANT=conservative
```

### MODEL_CONFIG_DIR

Override config directory:

```bash
export MODEL_CONFIG_DIR=/custom/path/to/configs
```

## Examples

### Basic Usage

```python
from CONFIG.config_loader import load_model_config
from TRAINING.model_fun import LightGBMTrainer

config = load_model_config("lightgbm", variant="conservative")
trainer = LightGBMTrainer(config)
trainer.train(X_train, y_train)
```

### With Overrides

```python
config = load_model_config("mlp", overrides={
    "epochs": 200,
    "learning_rate": 0.0001,
    "dropout": 0.3
})
```

### List Available Models

```python
from CONFIG.config_loader import get_available_model_configs

all_configs = get_available_model_configs()
for model, variants in all_configs.items():
    print(f"{model}: {variants}")
```

## See Also

- [Config Basics](../../01_tutorials/configuration/CONFIG_BASICS.md) - Configuration tutorial
- [Config Schema](../api/CONFIG_SCHEMA.md) - Schema reference
- [Config Overlays](CONFIG_OVERLAYS.md) - Overlay system

