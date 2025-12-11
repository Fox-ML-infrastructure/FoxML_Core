# CLI vs Config Separation Policy

## Principle

**CLI arguments should NOT override SST (Single Source of Truth).** All configuration values should come from config files. CLI should only provide:

1. **Required inputs** (data paths, symbols, output paths)
2. **Config file overrides** (which config to use)
3. **Operational flags** (resume, dry-run, verbose, force-refresh)

## CLI Argument Categories

### ✅ Allowed in CLI

#### Required Inputs
- `--data-dir` - Input data directory
- `--symbols` - Symbols to process
- `--output-dir` - Output directory
- `--targets` - Specific targets (optional, can come from config)

#### Config File Overrides
- `--experiment-config` - **PREFERRED**: Use experiment config from `CONFIG/experiments/`
- `--target-ranking-config` - Override target ranking config path
- `--multi-model-config` - Override feature selection config path
- `--strategy-config` - Override strategy config path

#### Operational Flags
- `--resume` - Resume from checkpoint
- `--dry-run` - Show what would be done
- `--force-refresh` - Force refresh cache
- `--no-cache` - Disable caching
- `--verbose` / `--log-level` - Logging verbosity
- `--validate-targets` - Run validation checks
- `--strict-exit` - Exit on any failure

### ❌ Should NOT be in CLI (Move to Config)

#### Data Limits & Sampling
- `--max-samples-per-symbol` → `pipeline.data_limits.max_samples_per_symbol`
- `--max-rows-per-symbol` → `pipeline.data_limits.default_max_rows_per_symbol_ranking`
- `--max-rows-train` → `pipeline.data_limits.max_rows_train`
- `--max-rows-val` → `pipeline.data_limits.max_rows_val`
- `--min-cs` → `pipeline.data_limits.min_cs`
- `--max-cs-samples` → `pipeline.data_limits.max_cs_samples`

#### Model Configuration
- `--top-n-targets` → `intelligent_training.top_n_targets`
- `--top-m-features` → `intelligent_training.top_m_features`
- `--families` → `model_families` in config
- `--strategy` → `training.strategy` in config
- `--seq-backend` → `training.sequential.backend`
- `--seq-lookback` → `pipeline.sequential.default_lookback`
- `--epochs` → Model-specific config
- `--quantile-alpha` → Model-specific config

#### Cross-Sectional Parameters
- `--cs-normalize` → `preprocessing.cross_sectional.normalize`
- `--cs-block` → `preprocessing.cross_sectional.block_size`
- `--cs-winsor-p` → `preprocessing.cross_sectional.winsor_p`
- `--cs-ddof` → `preprocessing.cross_sectional.ddof`

#### Performance Settings
- `--threads` → `performance.num_threads` (or use environment)
- `--n-workers` → `performance.n_workers`
- `--batch-size` → `processing.batch_size`
- `--cpu-only` → `performance.device: "cpu"`

#### Feature Selection
- `--auto-targets` → `intelligent_training.auto_targets`
- `--auto-features` → `intelligent_training.auto_features`

## Migration Pattern

### Before (Bad - CLI Overrides SST)

```python
parser.add_argument('--max-samples-per-symbol', type=int, default=10000,
                   help='Maximum samples per symbol')
parser.add_argument('--top-n-targets', type=int, default=5,
                   help='Number of top targets to select')

# Later in code
max_samples = args.max_samples_per_symbol  # ❌ CLI overrides config
top_n = args.top_n_targets  # ❌ CLI overrides config
```

### After (Good - Config is SST, CLI only for overrides)

```python
# CLI only has operational flags
parser.add_argument('--experiment-config', type=str,
                   help='Experiment config name (preferred)')
parser.add_argument('--force-refresh', action='store_true',
                   help='Force refresh cache')

# Later in code
if args.experiment_config:
    config = load_experiment_config(args.experiment_config)
    max_samples = config.data_limits.max_samples_per_symbol  # ✅ From config
    top_n = config.intelligent_training.top_n_targets  # ✅ From config
else:
    # Load from default config
    max_samples = get_cfg("pipeline.data_limits.max_samples_per_symbol", default=10000)
    top_n = get_cfg("intelligent_training.top_n_targets", default=5)
```

## Implementation Guidelines

### 1. Load Config First

```python
# Load experiment config if provided (PREFERRED)
experiment_config = None
if args.experiment_config:
    experiment_config = load_experiment_config(args.experiment_config)
    # All settings come from config
    max_samples = experiment_config.data_limits.max_samples_per_symbol
    top_n = experiment_config.intelligent_training.top_n_targets
else:
    # Fall back to default configs
    max_samples = get_cfg("pipeline.data_limits.max_samples_per_symbol", default=10000)
    top_n = get_cfg("intelligent_training.top_n_targets", default=5)
```

### 2. CLI Arguments Only for Overrides

If you absolutely need a CLI override (for testing/debugging), make it clear it's an override:

```python
# Only for testing/debugging - overrides config
parser.add_argument('--override-max-samples', type=int,
                   help='OVERRIDE: Max samples (for testing only, overrides config)')

# Later
if args.override_max_samples:
    logger.warning("⚠️  Using CLI override for max_samples (testing only)")
    max_samples = args.override_max_samples
else:
    max_samples = config.data_limits.max_samples_per_symbol
```

### 3. Document Config Locations

All settings should be documented in config files with clear paths:

```yaml
# CONFIG/training_config/pipeline_config.yaml
data_limits:
  max_samples_per_symbol: 50000  # Used by ranking pipeline
  default_max_rows_per_symbol_ranking: 50000
  min_cs: 10
  max_cs_samples: 1000

# CONFIG/experiments/my_experiment.yaml
intelligent_training:
  top_n_targets: 5
  top_m_features: 100
  auto_targets: true
  auto_features: true
```

## Benefits

1. **SST Compliance**: Single source of truth for all settings
2. **Reproducibility**: Same config = same results
3. **Simpler CLI**: Fewer arguments, easier to use
4. **Better Documentation**: All settings in config files with schemas
5. **Version Control**: Config changes tracked in git
6. **Testing**: Easy to test different configs without changing CLI

## Migration Checklist

For each CLI script:

- [ ] Identify all CLI arguments with defaults
- [ ] Move defaults to appropriate config file
- [ ] Remove CLI argument (or mark as override-only)
- [ ] Update code to load from config
- [ ] Update documentation
- [ ] Test with config file
- [ ] Verify SST compliance

## Examples

### Intelligent Trainer

**Before**: 20+ CLI arguments with defaults
**After**: 5-7 CLI arguments (data-dir, symbols, output-dir, experiment-config, operational flags)

### Training Strategies

**Before**: 30+ CLI arguments with defaults
**After**: 5-7 CLI arguments (data-dir, symbols, output-dir, experiment-config, operational flags)

### Target Ranking

**Before**: 10+ CLI arguments with defaults
**After**: 3-5 CLI arguments (data-dir, symbols, output-dir, experiment-config, operational flags)

## See Also

- [Config Basics](../../01_tutorials/configuration/CONFIG_BASICS.md)
- [Modular Config System](../../02_reference/configuration/MODULAR_CONFIG_SYSTEM.md)
- [Modular Config System](../../02_reference/configuration/MODULAR_CONFIG_SYSTEM.md) - Includes experiment configs
