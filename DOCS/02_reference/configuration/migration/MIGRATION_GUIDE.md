# Config Structure Migration Guide

## Overview

The CONFIG directory has been reorganized into a more human-usable structure. All old locations are still supported via symlinks, so existing code will continue to work.

## New Structure Benefits

1. **Clear organization**: Related configs grouped together
2. **Easier navigation**: Logical directory structure
3. **Better discoverability**: Easy to find what you need
4. **Reduced confusion**: No scattered root-level files

## Quick Reference

### Where to Find Configs

| What You Need | New Location | Old Location (still works) |
|---------------|-------------|---------------------------|
| **Experiment configs** | `experiments/*.yaml` | Same |
| **Model configs** | `models/*.yaml` | `model_config/*.yaml` |
| **Training configs** | `pipeline/training/*.yaml` | `training_config/*.yaml` |
| **Feature registry** | `data/feature_registry.yaml` | Root level |
| **Target ranking** | `ranking/targets/*.yaml` | `target_ranking/*.yaml` |
| **Feature selection** | `ranking/features/*.yaml` | `feature_selection/*.yaml` |
| **System configs** | `core/*.yaml` | `training_config/system_config.yaml` |

## Common Tasks

### Creating a New Experiment

Create a file in `CONFIG/experiments/`:

```yaml
experiment:
  name: my_experiment
  description: "My experiment"

data:
  data_dir: data/data_labeled/interval=5m
  symbols: [AAPL, MSFT]
  min_cs: 10
  max_cs_samples: 1000
```

### Editing Model Hyperparameters

Edit files in `CONFIG/models/` (e.g., `models/lightgbm.yaml`)

### Configuring Training Pipeline

Edit files in `CONFIG/pipeline/training/`:
- `intelligent.yaml` - Main intelligent training config
- `safety.yaml` - Safety & temporal settings
- `preprocessing.yaml` - Data preprocessing

### Configuring System Resources

Edit files in `CONFIG/pipeline/`:
- `gpu.yaml` - GPU settings
- `memory.yaml` - Memory management
- `threading.yaml` - Threading policy

## Migration Status

✅ **Phase 1 Complete**: Files moved to new locations with symlinks  
✅ **Phase 2 Complete**: Loaders updated to check new locations first  
⏳ **Phase 3**: Remove old symlinks after migration period (2-3 releases)

## Backward Compatibility

All old paths are still supported:
- Symlinks maintain compatibility
- Loaders check both old and new locations
- No code changes required

## Questions?

See `CONFIG_README.md` for full documentation.

