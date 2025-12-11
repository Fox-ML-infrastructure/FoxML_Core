# Config Cleaner API

Systematic parameter validation utility to prevent duplicate argument errors and unknown parameter errors when passing configs to model constructors.

**Location:** `TRAINING/utils/config_cleaner.py`

## Overview

The `clean_config_for_estimator()` function validates configuration dictionaries before passing them to model constructors. It:

1. **Removes duplicate parameters** - If a parameter is passed both in `raw_config` and `extra_kwargs`, it's removed from `raw_config` to avoid "got multiple values for keyword argument" errors
2. **Removes unknown parameters** - Uses `inspect.signature()` to check which parameters the estimator actually accepts, removing any that aren't in the `__init__` signature
3. **Logs what was stripped** - Provides visibility into what parameters were removed (DEBUG level)

This maintains **Single Source of Truth (SST)** - values still come from config/defaults, but only valid, non-duplicated keys are passed to the constructor.

## Quick Start

```python
from TRAINING.utils.config_cleaner import clean_config_for_estimator
from lightgbm import LGBMRegressor

# Raw config from inject_defaults might have duplicates or invalid params
raw_config = {
    'n_estimators': 100,
    'random_seed': 42,  # Will be removed (duplicate)
    'invalid_param': 123  # Will be removed (unknown)
}

# Parameters you want to pass explicitly
extra = {'random_seed': 42}

# Clean the config
clean_config = clean_config_for_estimator(
    LGBMRegressor,
    raw_config,
    extra,
    family_name='lightgbm'
)

# Now safe to instantiate
model = LGBMRegressor(**clean_config, **extra)
```

## API Reference

### clean_config_for_estimator

```python
def clean_config_for_estimator(
    estimator_cls: Type,
    raw_config: Dict[str, Any],
    extra_kwargs: Dict[str, Any] = None,
    family_name: str = None
) -> Dict[str, Any]:
```

**Parameters:**

- `estimator_cls` (Type): The estimator class to instantiate (e.g., `LGBMRegressor`, `RandomForestClassifier`)
- `raw_config` (Dict[str, Any]): Raw config dictionary that may contain invalid/duplicate keys
- `extra_kwargs` (Dict[str, Any], optional): Dictionary of parameters to pass explicitly. Any keys in this dict will be removed from `raw_config` to avoid duplicates. Default: `None`
- `family_name` (str, optional): Model family name for logging (e.g., "lightgbm", "random_forest"). If `None`, uses estimator class name. Default: `None`

**Returns:**

- `Dict[str, Any]`: Cleaned config dictionary safe to pass to estimator constructor

**Raises:**

- No exceptions raised - always returns a dict (empty dict if `raw_config` is None or invalid)

## Examples

### Basic Usage

```python
from TRAINING.utils.config_cleaner import clean_config_for_estimator
from sklearn.ensemble import RandomForestRegressor

# Config from inject_defaults might have sklearn-incompatible params
raw_config = {
    'n_estimators': 100,
    'random_seed': 42,  # sklearn uses 'random_state', not 'random_seed'
    'num_threads': 4,   # sklearn uses 'n_jobs', not 'num_threads'
    'max_depth': 10
}

# You want to pass random_state explicitly
extra = {'random_state': 42, 'n_jobs': 4}

# Clean removes duplicates and unknown params
clean_config = clean_config_for_estimator(
    RandomForestRegressor,
    raw_config,
    extra,
    'random_forest'
)

# Now safe - no 'random_seed' or 'num_threads' in clean_config
model = RandomForestRegressor(**clean_config, **extra)
```

### Without Extra Kwargs

If you don't have explicit parameters to pass, the function still removes unknown parameters:

```python
from TRAINING.utils.config_cleaner import clean_config_for_estimator
from sklearn.linear_model import Lasso

raw_config = {
    'alpha': 0.1,
    'n_jobs': 4,  # Lasso doesn't support n_jobs - will be removed
    'max_iter': 1000
}

clean_config = clean_config_for_estimator(
    Lasso,
    raw_config,
    extra_kwargs=None,  # No explicit params
    family_name='lasso'
)

# clean_config will have 'alpha' and 'max_iter', but not 'n_jobs'
model = Lasso(**clean_config)
```

### Handling None Configs

The function safely handles `None` or invalid configs:

```python
# None config returns empty dict
clean_config = clean_config_for_estimator(LGBMRegressor, None, {}, 'lightgbm')
# Returns: {}

# Non-dict config returns empty dict (with warning)
clean_config = clean_config_for_estimator(LGBMRegressor, "not a dict", {}, 'lightgbm')
# Returns: {} (logs warning)
```

## Integration Patterns

### In Model Constructors

```python
from TRAINING.utils.config_cleaner import clean_config_for_estimator
from lightgbm import LGBMRegressor, LGBMClassifier

def create_lightgbm_model(config, task_type, model_seed):
    # Determine estimator class
    est_cls = LGBMClassifier if task_type == "classification" else LGBMRegressor
    
    # Parameters to pass explicitly
    extra = {'random_seed': model_seed}
    
    # Clean config to remove duplicates and unknown params
    clean_config = clean_config_for_estimator(est_cls, config, extra, 'lightgbm')
    
    # Safe to instantiate
    model = est_cls(**clean_config, **extra)
    return model
```

### In Lambda Closures

When creating lambda functions that will be called later, ensure you capture explicit copies:

```python
from TRAINING.utils.config_cleaner import clean_config_for_estimator
from lightgbm import LGBMRegressor

def create_model_factory(config, model_seed):
    est_cls = LGBMRegressor
    extra = {'random_seed': model_seed}
    
    # Clean config
    clean_config = clean_config_for_estimator(est_cls, config, extra, 'lightgbm')
    
    # Capture explicit copies in closure (prevents reference issues)
    config_final = clean_config.copy()
    extra_final = extra.copy()
    
    # Return lambda that captures values, not references
    return lambda **kwargs: est_cls(**config_final, **extra_final)
```

## Why This Exists

### The Problem

When using `inject_defaults()` to inject global defaults into model configs, you can end up with:

1. **Duplicate parameters**: A parameter like `random_seed` might be in both the config (from `inject_defaults`) and passed explicitly, causing `TypeError: got multiple values for keyword argument 'random_seed'`
2. **Unknown parameters**: A parameter like `num_threads` might be in the config but the estimator uses `n_jobs` instead, causing `TypeError: __init__() got an unexpected keyword argument 'num_threads'`
3. **Parameter name mismatches**: Different libraries use different names (e.g., `random_seed` vs `random_state`, `n_jobs` vs `thread_count`)

### The Solution

`clean_config_for_estimator()` uses Python's `inspect.signature()` to:
- Check what parameters the estimator actually accepts
- Remove parameters that aren't in the signature
- Remove parameters that will be passed explicitly (avoiding duplicates)

This makes the codebase **resilient to config drift** - if `inject_defaults` starts adding new parameters, or if model libraries change their APIs, the cleaner automatically filters out incompatible parameters.

## Logging

The function logs what was stripped at DEBUG level:

```
DEBUG - [lightgbm] stripped unknown=['invalid_param'] duplicate=['random_seed'] before estimator init
```

To see these messages, set logging level to DEBUG:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Best Practices

1. **Always use before estimator instantiation** - Clean configs before passing to constructors
2. **Pass explicit parameters in `extra_kwargs`** - If you need to override config values or pass parameters that might be duplicated, use `extra_kwargs`
3. **Use descriptive `family_name`** - Helps with debugging when logs show what was stripped
4. **Handle None configs gracefully** - The function returns `{}` for None/invalid configs, but your code should handle empty configs appropriately
5. **Capture copies in closures** - When creating lambdas or closures, explicitly copy configs to avoid reference issues

## See Also

- [Config Loader API](CONFIG_LOADER_API.md) - How configs are loaded and injected
- [Reproducibility Tracking Guide](../../03_technical/implementation/REPRODUCIBILITY_TRACKING.md) - Tracking run reproducibility
- [Feature Selection Tutorial](../../01_tutorials/training/FEATURE_SELECTION_TUTORIAL.md) - How feature selection uses config cleaner
