# CLI to Config Migration Status

## Overview

This document tracks the migration of CLI arguments to config files to enforce Single Source of Truth (SST) compliance.

## Policy

See [CLI vs Config Separation](CLI_CONFIG_SEPARATION.md) for the complete policy.

**Key Principle**: CLI should NOT override SST. All configuration values should come from config files.

## Migration Status

### ✅ Completed

#### `intelligent_trainer.py` (TRAINING/orchestration/)
- **Status**: ✅ Migrated
- **Changes**:
  - Removed 15+ config-related CLI arguments
  - Added `intelligent_training` section to `pipeline_config.yaml`
  - CLI now only has: inputs, config overrides, operational flags
  - All settings load from config (SST compliant)
  - Testing overrides available with warnings
- **Config Location**: `CONFIG/training_config/pipeline_config.yaml` → `intelligent_training.*`

#### `rank_target_predictability.py` (SCRIPTS/)
- **Status**: ✅ Partially Migrated
- **Changes**:
  - CLI defaults now load from `pipeline_config.yaml`
  - `--min-cs`, `--max-cs-samples`, `--max-rows-per-symbol` load from config
  - Replaced `random_state=42` with `BASE_SEED`
  - `load_sample_data()` loads `max_samples` from config
- **Remaining**: Some CLI arguments still have defaults (should be removed or marked as overrides)

### 🔄 In Progress

#### `training_strategies/main.py` (TRAINING/training_strategies/)
- **Status**: 🔄 Needs Migration
- **Issues**:
  - 30+ CLI arguments with hardcoded defaults
  - Data limits: `--max-samples-per-symbol`, `--max-rows-train`, `--max-rows-val`
  - Cross-sectional: `--min-cs`, `--cs-normalize`, `--cs-block`, `--cs-winsor-p`, `--cs-ddof`
  - Model config: `--quantile-alpha`, `--epochs`, `--seq-lookback`
  - Performance: `--threads`, `--batch-size`
  - Strategy: `--strategy`, `--model-types`, `--train-order`
- **Action Needed**:
  - Move all defaults to `pipeline_config.yaml`
  - Remove CLI arguments (or mark as override-only)
  - Load from config in code

### 📋 Pending

#### `multi_model_feature_selection.py` (TRAINING/ranking/)
- **Status**: 📋 Needs Review
- **Action**: Check for CLI arguments that should be in config

#### `barrier_pipeline.py` (DATA_PROCESSING/pipeline/)
- **Status**: 📋 Needs Review
- **Issues**:
  - `--horizons`, `--barrier-sizes` with defaults
  - `--n-workers`, `--batch-size`, `--throttle-delay` with defaults
- **Action**: Move to config or mark as operational

#### Other Scripts
- `comprehensive_builder.py` (DATA_PROCESSING/features/)
- `streaming_builder.py` (DATA_PROCESSING/features/)
- `hft_forward.py` (DATA_PROCESSING/targets/)

## Migration Checklist Template

For each CLI script:

- [ ] Identify all CLI arguments with defaults
- [ ] Categorize: Input/Config Override/Operational/Config Value
- [ ] Move config values to appropriate config file
- [ ] Update code to load from config
- [ ] Remove CLI argument (or mark as override-only with warning)
- [ ] Update documentation
- [ ] Test with config file
- [ ] Verify SST compliance

## Config File Locations

### Pipeline Settings
- **File**: `CONFIG/training_config/pipeline_config.yaml`
- **Sections**:
  - `pipeline.data_limits.*` - Data sampling limits
  - `pipeline.sequential.*` - Sequential model settings
  - `intelligent_training.*` - Intelligent trainer settings

### Preprocessing Settings
- **File**: `CONFIG/training_config/preprocessing_config.yaml`
- **Sections**:
  - `preprocessing.cross_sectional.*` - Cross-sectional normalization

### Performance Settings
- **File**: `CONFIG/defaults.yaml`
- **Sections**:
  - `performance.*` - Threading, parallelism

### Experiment-Specific
- **File**: `CONFIG/experiments/*.yaml`
- **Sections**: All settings can be overridden per experiment

## Examples

### Before (Bad - CLI Overrides SST)

```python
parser.add_argument('--max-samples-per-symbol', type=int, default=10000)
parser.add_argument('--top-n-targets', type=int, default=5)

# Later
max_samples = args.max_samples_per_symbol  # ❌ CLI overrides config
```

### After (Good - Config is SST)

```python
# CLI only for override (testing)
parser.add_argument('--override-max-samples', type=int,
                   help='OVERRIDE: Max samples (testing only)')

# Later
if args.override_max_samples:
    logger.warning("⚠️  Using CLI override (testing only)")
    max_samples = args.override_max_samples
else:
    max_samples = get_cfg("pipeline.data_limits.max_samples_per_symbol", default=10000)  # ✅ From config
```

## Next Steps

1. **Complete `training_strategies/main.py` migration** (highest priority - most arguments)
2. **Review and migrate data processing scripts** (barrier_pipeline, feature builders)
3. **Update all CLI documentation** to reflect config-driven approach
4. **Add SST compliance test** to prevent new CLI arguments with defaults

## See Also

- [CLI vs Config Separation](CLI_CONFIG_SEPARATION.md) - Complete policy
- [Config Basics](../../01_tutorials/configuration/CONFIG_BASICS.md) - Config system guide
- [Modular Config System](../../02_reference/configuration/MODULAR_CONFIG_SYSTEM.md) - Config architecture
