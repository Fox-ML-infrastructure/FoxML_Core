# Config Loader API

Complete API reference for configuration loading.

## Logging Configuration

### get_module_logging_config

Get module-specific logging configuration.

```python
from CONFIG.logging_config_utils import get_module_logging_config

# Get config for a specific module
log_cfg = get_module_logging_config('rank_target_predictability')

# Use in code
if log_cfg.gpu_detail:
    logger.info("🚀 Training on GPU...")
if log_cfg.cv_detail:
    logger.info("Using PurgedTimeSeriesSplit: %s folds", n_folds)
if log_cfg.edu_hints:
    logger.info("💡 Note: GPU is most efficient for large datasets")
```

**Parameters:**
- `module_name` (str): Module name (e.g., "rank_target_predictability", "feature_selection")

**Returns:** `ModuleLoggingConfig` - Module-specific logging config object

### get_backend_logging_config

Get backend library logging configuration.

```python
from CONFIG.logging_config_utils import get_backend_logging_config

# Get config for a backend library
lgbm_cfg = get_backend_logging_config('lightgbm')

# Use in model parameters
lgbm_params['verbose'] = lgbm_cfg.native_verbosity
```

**Parameters:**
- `backend_name` (str): Backend name (e.g., "lightgbm", "xgboost", "tensorflow")

**Returns:** `BackendLoggingConfig` - Backend-specific logging config object

**Configuration File:** `CONFIG/logging_config.yaml`

See [Structured Logging Configuration](../../../CHANGELOG.md#structured-logging-configuration-system) for complete details.

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

### list_available_configs

List all available configuration files.

```python
from CONFIG.config_loader import list_available_configs

configs = list_available_configs()
# Returns: {
#     "model_configs": ["lightgbm", "xgboost", "mlp", ...],
#     "training_configs": ["first_batch_specs", ...]
# }
```

**Returns:** dict - Dictionary with "model_configs" and "training_configs" lists

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
from CONFIG.config_loader import list_available_configs, get_config_variants

all_configs = list_available_configs()
print("Model configs:", all_configs["model_configs"])
print("Training configs:", all_configs["training_configs"])

# Get variants for a specific model
variants = get_config_variants("lightgbm")
print(f"LightGBM variants: {variants}")
```

## Config Cleaner Utility

When passing configs to model constructors, use the config cleaner to prevent parameter passing errors:

```python
from TRAINING.utils.config_cleaner import clean_config_for_estimator
from lightgbm import LGBMRegressor

# Load config (may have duplicates or invalid params from inject_defaults)
config = load_model_config("lightgbm")
config = inject_defaults(config, model_family="lightgbm")

# Clean before passing to constructor
extra = {'random_seed': 42}
clean_config = clean_config_for_estimator(LGBMRegressor, config, extra, 'lightgbm')

# Safe to instantiate
model = LGBMRegressor(**clean_config, **extra)
```

See [Config Cleaner API](CONFIG_CLEANER_API.md) for complete documentation.

## See Also

- [Config Basics](../../01_tutorials/configuration/CONFIG_BASICS.md) - Configuration tutorial
- [Config Schema](../api/CONFIG_SCHEMA.md) - Schema reference
- [Config Overlays](CONFIG_OVERLAYS.md) - Overlay system
- [Config Cleaner API](CONFIG_CLEANER_API.md) - Parameter validation utility

