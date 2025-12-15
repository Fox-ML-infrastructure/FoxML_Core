# Centralized Defaults Configuration

## Overview

This system centralizes common configuration settings to a **Single Source of Truth (SST)**, eliminating the need to update 20+ different config files when changing common settings like `random_state`, `n_jobs`, `learning_rate`, etc.

## How It Works

1. **`CONFIG/defaults.yaml`** - Contains all common default values organized by category
2. **Automatic Injection** - The config loader automatically injects these defaults into model configs
3. **Override Priority** - Explicit values in model configs override defaults (highest priority)

## Priority Order (Highest to Lowest)

1. **Explicit overrides** (passed to `load_model_config()`)
2. **Variant configs** (e.g., "conservative", "balanced")
3. **Model-specific config** (from `model_config/*.yaml`)
4. **Global defaults** (from `defaults.yaml`) ← **Single Source of Truth**

## Common Settings Centralized

### Randomness
- `random_state` / `random_seed` - Loaded from `pipeline.determinism.base_seed` by default

### Performance
- `n_jobs` - Parallel jobs (default: 1)
- `num_threads` / `threads` - Thread count (default: 4)
- `device` - CPU/GPU selection (default: "cpu")
- `verbose` / `verbosity` - Logging level (default: -1)

### Tree Models
- `n_estimators` - Number of trees (default: 1000)
- `learning_rate` - Step size (default: 0.03)
- `max_depth` - Tree depth (default: 8)
- `subsample` / `colsample_bytree` - Sampling (default: 0.75)
- Regularization parameters

### Neural Networks
- `learning_rate` - Default: 0.001
- `max_iter` - Default: 300
- `batch_size` - Default: "auto"

### Linear Models
- `max_iter` - Default: 1000
- `alpha` - Default: 0.1

## Usage

### Changing Global Defaults

Edit `CONFIG/defaults.yaml`:

```yaml
randomness:
  random_state: 1337  # Override default seed

performance:
  n_jobs: 4  # Use 4 parallel jobs globally
```

### Overriding for Specific Models

In `model_config/lightgbm.yaml`:

```yaml
hyperparameters:
  learning_rate: 0.05  # Override default 0.03 for LightGBM only
  # random_state: auto-injected from defaults
```

### Runtime Overrides

```python
from CONFIG.config_loader import load_model_config

# Override at runtime
config = load_model_config("lightgbm", overrides={"n_jobs": 8})
```

## Migration Notes

- **Old configs still work** - If a model config explicitly sets a value, it takes precedence
- **Gradual migration** - You can remove explicit values from model configs over time
- **Backward compatible** - No breaking changes

## Benefits

✅ **Single Source of Truth** - Change `random_state` in one place, applies everywhere  
✅ **Consistency** - All models use the same defaults unless explicitly overridden  
✅ **Maintainability** - No more hunting through 20+ files to change common settings  
✅ **Flexibility** - Still allows per-model overrides when needed  
