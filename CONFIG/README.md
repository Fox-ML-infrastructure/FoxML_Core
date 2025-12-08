# CONFIG Directory

This directory contains all configuration YAML files for FoxML Core.

## Documentation

**All configuration documentation has been moved to the `docs/` folder for better organization.**

See the [Configuration Reference](../../docs/02_reference/configuration/README.md) for complete documentation.

## Quick Links

- [Configuration System Overview](../../docs/02_reference/configuration/README.md)
- [Feature & Target Configs](../../docs/02_reference/configuration/FEATURE_TARGET_CONFIGS.md)
- [Training Pipeline Configs](../../docs/02_reference/configuration/TRAINING_PIPELINE_CONFIGS.md)
- [Safety & Leakage Configs](../../docs/02_reference/configuration/SAFETY_LEAKAGE_CONFIGS.md)
- [Model Configuration](../../docs/02_reference/configuration/MODEL_CONFIGURATION.md)
- [Usage Examples](../../docs/02_reference/configuration/USAGE_EXAMPLES.md)

## Directory Structure

```
CONFIG/
├── backups/                       # Automatic config backups (auto-fixer)
├── model_config/                  # Model-specific hyperparameters (17 models)
├── training_config/               # Training pipeline and system settings (11 configs)
├── excluded_features.yaml         # Patterns for always-excluded features
├── feature_registry.yaml          # Feature metadata (lag_bars, allowed_horizons)
├── feature_target_schema.yaml     # Explicit schema (metadata/targets/features)
├── target_configs.yaml           # Target definitions (63 targets)
└── ... (other config files)
```

For detailed documentation on each config file, see the [Configuration Reference](../../docs/02_reference/configuration/README.md).
