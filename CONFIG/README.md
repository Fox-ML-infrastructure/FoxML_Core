# Configuration System

Centralized configuration management for FoxML Core training pipeline and model families.

## Overview

The configuration system provides a single source of truth for all training parameters, system settings, and model hyperparameters. All configurations are stored as YAML files and loaded programmatically via `config_loader.py`.

## Directory Structure

```
CONFIG/
├── model_config/          # Model-specific hyperparameters
│   ├── lightgbm.yaml
│   ├── xgboost.yaml
│   ├── mlp.yaml
│   └── ... (17 model configs)
│
└── training_config/       # Training pipeline and system settings
    ├── pipeline_config.yaml      # Main pipeline settings
    ├── gpu_config.yaml            # GPU/CUDA configuration
    ├── memory_config.yaml         # Memory management
    ├── preprocessing_config.yaml  # Data preprocessing
    ├── threading_config.yaml      # Threading policies
    ├── safety_config.yaml         # Numerical stability guards
    ├── callbacks_config.yaml      # Training callbacks
    ├── optimizer_config.yaml      # Optimizer defaults
    ├── system_config.yaml         # System-level settings
    ├── family_config.yaml         # Model family policies
    ├── sequential_config.yaml     # Sequential model settings
    └── first_batch_specs.yaml     # First batch specifications
```

## Usage

### Loading Configurations

```python
from CONFIG.config_loader import (
    load_model_config,
    get_pipeline_config,
    get_gpu_config,
    get_cfg
)

# Load model-specific config
lightgbm_config = load_model_config("lightgbm", variant="aggressive")

# Load training configs
pipeline = get_pipeline_config()
gpu = get_gpu_config()

# Access nested values with dot notation
timeout = get_cfg("pipeline.isolation_timeout_seconds", default=7200)
vram_cap = get_cfg("gpu.vram_cap_mb", default=4096, config_name="gpu_config")
```

### Configuration Hierarchy

1. **Model Configs** (`model_config/`) - Hyperparameters for specific model families
2. **Training Configs** (`training_config/`) - Pipeline, system, and resource settings

### Key Configuration Files

#### Pipeline Configuration
- **File:** `training_config/pipeline_config.yaml`
- **Purpose:** Main training pipeline orchestration
- **Key Settings:** Timeouts, data limits, sequential model settings, determinism

#### GPU Configuration
- **File:** `training_config/gpu_config.yaml`
- **Purpose:** GPU device management and CUDA settings
- **Key Settings:** VRAM caps, device visibility, TensorFlow/PyTorch GPU options

#### Threading Configuration
- **File:** `training_config/threading_config.yaml`
- **Purpose:** Thread allocation and OpenMP/MKL policies
- **Key Settings:** Default threads, per-family policies, thread planning

#### Memory Configuration
- **File:** `training_config/memory_config.yaml`
- **Purpose:** Memory thresholds and cleanup policies
- **Key Settings:** Memory caps, chunk sizes, cleanup aggressiveness

#### Safety Configuration
- **File:** `training_config/safety_config.yaml`
- **Purpose:** Numerical stability guards and leakage detection
- **Key Settings:** Feature clipping, target capping, gradient clipping, **leakage detection thresholds**
- **Leakage Detection:** Configurable thresholds for auto-fixer (CV scores, training accuracy, R², correlation)

## Environment Variable Overrides

Most configuration values can be overridden via environment variables:

```bash
# Override GPU device
export CUDA_VISIBLE_DEVICES=0

# Override thread count
export OMP_NUM_THREADS=8

# Override timeout
export TRAINER_ISOLATION_TIMEOUT=10800
```

## Configuration Variants

Model configs support variants for different use cases:

- `conservative` - Lower risk, more stable settings
- `aggressive` - Higher performance, more experimental
- `default` - Balanced settings (used if variant not specified)

## Best Practices

1. **Never hardcode values** - Always load from config files
2. **Use defaults** - Provide sensible fallbacks when config unavailable
3. **Validate inputs** - Check config values before use
4. **Document changes** - Update configs with clear comments
5. **Test variants** - Verify all config variants work correctly

## Migration Status

✅ **Phase 2 Complete** - All hardcoded configurations have been migrated to YAML files. The system maintains backward compatibility with hardcoded defaults during the transition period.

## Support

For configuration questions or issues, refer to:
- `config_loader.py` - Implementation details
- Individual config files - Inline documentation
- Training pipeline code - Usage examples

## Related Documentation

- [Config Basics](../docs/01_tutorials/configuration/CONFIG_BASICS.md) - Configuration fundamentals tutorial
- [Config Examples](../docs/01_tutorials/configuration/CONFIG_EXAMPLES.md) - Example configurations
- [Advanced Config](../docs/01_tutorials/configuration/ADVANCED_CONFIG.md) - Advanced configuration guide
- [Config Loader API](../docs/02_reference/configuration/CONFIG_LOADER_API.md) - Complete API reference
- [Config Schema](../docs/02_reference/api/CONFIG_SCHEMA.md) - Configuration schema documentation
- [Environment Variables](../docs/02_reference/configuration/ENVIRONMENT_VARIABLES.md) - Environment variable overrides
- [Model Config Reference](../docs/02_reference/models/MODEL_CONFIG_REFERENCE.md) - Model-specific configurations
- [Adding Proprietary Models](../docs/03_technical/implementation/ADDING_PROPRIETARY_MODELS.md) - Using BaseModelTrainer with custom configs
