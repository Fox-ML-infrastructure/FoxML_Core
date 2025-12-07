# Configuration System Update — Phase 2 Centralized Configuration

**Date:** December 2025  
**Status:** ✅ Complete — Testing Underway  
**Impact:** All training modules, model trainers, and system components

## Overview

The configuration system has been completely refactored to use centralized YAML-based configuration files. This eliminates hardcoded values throughout the codebase and provides a single source of truth for all training parameters, system settings, and model hyperparameters.

## What Changed

### New Configuration Files Created

**9 New Training Configuration Files** (`CONFIG/training_config/`):

1. **`pipeline_config.yaml`** — Main training pipeline orchestration
   - Isolation timeouts (default: 7200s, family-specific overrides)
   - Data processing limits (max_samples_per_symbol, max_rows_train, cross-sectional limits)
   - Sequential model settings (lookback windows, backend selection)
   - Determinism settings (seeds, TensorFlow deterministic ops)
   - Test vs. production defaults

2. **`gpu_config.yaml`** — GPU and CUDA configuration
   - VRAM caps per model family (default: 4096MB, family-specific overrides)
   - CUDA device visibility and selection
   - TensorFlow GPU settings (allocator, allow_growth, thread configuration)
   - PyTorch GPU settings
   - Mixed precision configuration
   - XGBoost GPU settings

3. **`memory_config.yaml`** — Memory management
   - Memory thresholds (warning at 80%, high warning at 90%)
   - Chunking settings (default: 1M rows per chunk)
   - Memory caps for child processes
   - Cleanup aggressiveness settings
   - System-level memory monitoring

4. **`preprocessing_config.yaml`** — Data preprocessing
   - Imputation strategies (default: median)
   - Scaling methods (StandardScaler, RobustScaler)
   - Feature selection thresholds
   - Validation splits (test_size: 0.2, val_ratio: 0.15)
   - Random state seeds
   - NaN handling thresholds

5. **`threading_config.yaml`** — Threading and resource management
   - Default thread counts (calculated from CPU count)
   - Thread policies (omp_heavy, cpu_blas_only, tf_gpu, torch_gpu, tf_cpu)
   - OpenMP settings (dynamic, proc_bind, blocktime)
   - MKL threading layer (GNU vs. Intel OpenMP)
   - Per-family thread allocation overrides

6. **`safety_config.yaml`** — Numerical stability guards
   - Feature clipping bounds (default: [-1000, 1000])
   - Target capping (default: 15 MAD)
   - Safe exponential bounds (default: [-40, 40])
   - Gradient clipping (clipnorm: 1.0, max_norm: 1.0)
   - NumPy error handling (overflow, invalid, divide, underflow)
   - Model output validation settings

7. **`callbacks_config.yaml`** — Training callbacks
   - Early stopping configuration (patience: 10 default, family-specific overrides)
   - Learning rate reduction (patience: 5, factor: 0.5, min_lr: 1e-6)
   - Model checkpointing (disabled by default)
   - TensorBoard/CSV logging (disabled by default)
   - Progress bar settings

8. **`optimizer_config.yaml`** — Optimizer defaults
   - Adam optimizer (lr: 1e-3, clipnorm: 1.0)
   - AdamW optimizer (lr: 1e-3, weight_decay: 0.0)
   - SGD and RMSprop defaults
   - Per-model learning rate overrides

9. **`system_config.yaml`** — System-level settings
   - Paths (data_dir, output_dir, temp_dir, joblib_temp)
   - Environment variables (shell, term, inputrc, pythonpath)
   - Logging configuration (levels, component-specific)
   - Isolation runner settings
   - Security settings (readline suppression, MKL guard)

### Enhanced Config Loader

**`CONFIG/config_loader.py`** has been enhanced with:

- **Nested access via dot notation**: `get_cfg("pipeline.isolation_timeout_seconds")`
- **Family-specific overrides**: `get_family_timeout(family, default)`
- **Convenience functions**: `get_pipeline_config()`, `get_gpu_config()`, `get_memory_config()`, etc.
- **Backward compatibility**: Graceful fallback to hardcoded defaults if config unavailable
- **Type safety**: Automatic type conversion with validation

### Code Integration

**All model trainers updated** to use centralized configs:

- ✅ `base_trainer.py` — Preprocessing, callbacks, safety guards, imputation
- ✅ `lightgbm_trainer.py` — Test splits, random state
- ✅ `xgboost_trainer.py` — Test splits, random state, optimizer clipnorm
- ✅ `mlp_trainer.py` — Test splits, optimizer clipnorm
- ✅ `cnn1d_trainer.py` — Test splits, optimizer clipnorm, callbacks
- ✅ `lstm_trainer.py` — Test splits, optimizer clipnorm, callbacks, dynamic batch/epoch scaling
- ✅ `transformer_trainer.py` — Test splits, optimizer clipnorm, callbacks, dynamic batch scaling
- ✅ `vae_trainer.py` — Test splits, optimizer clipnorm, callbacks
- ✅ `gan_trainer.py` — Test splits, optimizer clipnorm, callbacks
- ✅ `meta_learning_trainer.py` — Test splits, optimizer clipnorm
- ✅ `multi_task_trainer.py` — Test splits, optimizer clipnorm
- ✅ `ngboost_trainer.py` — Test splits, random state (val_ratio: 0.15)
- ✅ `seq_torch_base.py` — Gradient clipping max_norm, optimizer settings

**System components updated**:

- ✅ `train_with_strategies.py` — Pipeline settings, timeouts, data limits, determinism, paths
- ✅ `common/threads.py` — Default thread counts, threading policies
- ✅ `common/runtime_policy.py` — VRAM caps per family
- ✅ `common/safety.py` — Feature clipping, target capping, safe exp bounds, NumPy error handling
- ✅ `memory/memory_manager.py` — Memory thresholds, chunk sizes, cleanup settings

## Why This Change

### Problems Solved

1. **Hardcoded values scattered across codebase** — Difficult to find and modify
2. **Inconsistent defaults** — Same setting had different values in different files
3. **No single source of truth** — Changes required editing multiple files
4. **Difficult experimentation** — Changing hyperparameters required code changes
5. **Poor maintainability** — Hard to track what settings affect what behavior

### Benefits

1. **Single source of truth** — All settings in YAML files
2. **Easy experimentation** — Change YAML files without code changes
3. **Environment-specific configs** — Support for variants (conservative, aggressive, default)
4. **Better documentation** — Config files are self-documenting
5. **Version control friendly** — Track config changes in git
6. **Backward compatible** — Hardcoded defaults still work if config unavailable

## Migration Details

### Backward Compatibility

The system maintains **full backward compatibility**:

- If config files are missing, code falls back to hardcoded defaults
- No breaking changes to existing functionality
- Config loading is optional (graceful degradation)

### Configuration Loading Pattern

All components follow this pattern:

```python
# Add CONFIG directory to path
_REPO_ROOT = Path(__file__).resolve().parents[N]
_CONFIG_DIR = _REPO_ROOT / "CONFIG"
if str(_CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(_CONFIG_DIR))

# Try to import config loader
_CONFIG_AVAILABLE = False
try:
    from config_loader import get_cfg, get_*_config
    _CONFIG_AVAILABLE = True
except ImportError:
    logger.debug("Config loader not available; using hardcoded defaults")

# Use config with fallback
if _CONFIG_AVAILABLE:
    try:
        value = get_cfg("path.to.setting", default=hardcoded_default, config_name="config_file")
    except Exception:
        value = hardcoded_default
else:
    value = hardcoded_default
```

## Usage Examples

### Loading Model Configs

```python
from CONFIG.config_loader import load_model_config

# Load with variant
config = load_model_config("lightgbm", variant="aggressive")

# Load default
config = load_model_config("mlp")
```

### Loading Training Configs

```python
from CONFIG.config_loader import (
    get_pipeline_config,
    get_gpu_config,
    get_cfg
)

# Load entire config
pipeline = get_pipeline_config()
gpu = get_gpu_config()

# Access nested values
timeout = get_cfg("pipeline.isolation_timeout_seconds", default=7200)
vram_cap = get_cfg("gpu.vram_cap_mb", default=4096, config_name="gpu_config")
family_timeout = get_family_timeout("LSTM", default=7200)
```

### Environment Variable Overrides

Most settings can be overridden via environment variables:

```bash
# Override GPU device
export CUDA_VISIBLE_DEVICES=0

# Override thread count
export OMP_NUM_THREADS=8

# Override timeout
export TRAINER_ISOLATION_TIMEOUT=10800
```

## Testing Status

✅ **Integration Complete** — All code updated to use centralized configs  
🔄 **Testing Underway** — Validating all integrations work correctly  
⏳ **Validation Pending** — Comprehensive testing of all model families

### Known Issues Fixed

- ✅ Fixed `NameError: name '_CONFIG_AVAILABLE' is not defined` in `threads.py`
- ✅ Removed duplicate config loader setup in `seq_torch_base.py`
- ✅ Removed redundant imports in `ngboost_trainer.py`
- ✅ All config imports verified working correctly

## Next Steps

1. **Complete testing** — Validate all model families work with new configs
2. **Performance validation** — Ensure no performance regressions
3. **Documentation** — Update user guides with config examples
4. **Validation layer** — Add schema validation for config files
5. **Logging modernization** — Integrate with centralized logging system

## Files Modified

### Configuration Files
- `CONFIG/training_config/pipeline_config.yaml` (new)
- `CONFIG/training_config/gpu_config.yaml` (new)
- `CONFIG/training_config/memory_config.yaml` (new)
- `CONFIG/training_config/preprocessing_config.yaml` (new)
- `CONFIG/training_config/threading_config.yaml` (new)
- `CONFIG/training_config/safety_config.yaml` (new)
- `CONFIG/training_config/callbacks_config.yaml` (new)
- `CONFIG/training_config/optimizer_config.yaml` (new)
- `CONFIG/training_config/system_config.yaml` (new)
- `CONFIG/config_loader.py` (enhanced)
- `CONFIG/README.md` (created)

### Code Files Updated
- `TRAINING/train_with_strategies.py`
- `TRAINING/common/threads.py`
- `TRAINING/common/runtime_policy.py`
- `TRAINING/common/safety.py`
- `TRAINING/memory/memory_manager.py`
- `TRAINING/model_fun/base_trainer.py`
- `TRAINING/model_fun/lightgbm_trainer.py`
- `TRAINING/model_fun/xgboost_trainer.py`
- `TRAINING/model_fun/mlp_trainer.py`
- `TRAINING/model_fun/cnn1d_trainer.py`
- `TRAINING/model_fun/lstm_trainer.py`
- `TRAINING/model_fun/transformer_trainer.py`
- `TRAINING/model_fun/vae_trainer.py`
- `TRAINING/model_fun/gan_trainer.py`
- `TRAINING/model_fun/meta_learning_trainer.py`
- `TRAINING/model_fun/multi_task_trainer.py`
- `TRAINING/model_fun/ngboost_trainer.py`
- `TRAINING/model_fun/seq_torch_base.py`

## Documentation

- [Configuration System Overview](../CONFIG/README.md) — Complete guide to the configuration system
- [Config Loader API](../docs/02_reference/configuration/CONFIG_LOADER_API.md) — API reference
- [Config Schema](../docs/02_reference/api/CONFIG_SCHEMA.md) — Configuration schema documentation
- [Config Tutorials](../docs/01_tutorials/configuration/) — Step-by-step guides

## Questions or Issues?

If you encounter any issues with the new configuration system:

1. Check that config files exist in `CONFIG/training_config/`
2. Verify `config_loader.py` is importable
3. Check logs for "Config loader not available" messages
4. Ensure backward compatibility fallbacks are working
5. Review the [Configuration System Overview](../CONFIG/README.md) for usage examples

---

**This update represents a major milestone in making FoxML Core more maintainable, configurable, and user-friendly. All changes maintain backward compatibility while providing a foundation for future enhancements.**

