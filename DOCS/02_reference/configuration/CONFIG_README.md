# Configuration System

This directory contains all configuration files for the FoxML Core pipeline.

## Directory Structure

The configuration system uses a clean, human-usable structure organized by purpose:

```
CONFIG/
├── README.md                 # This file
├── defaults.yaml             # Global defaults (SST)
├── config_loader.py         # Configuration loader
├── config_builder.py         # Config builder utilities
├── config_schemas.py         # Type definitions
│
├── core/                     # Core system configs
│   ├── logging.yaml          # Logging configuration
│   └── system.yaml           # System resources & paths
│
├── data/                     # Data-related configs
│   ├── feature_registry.yaml      # Feature registry (allowed/excluded)
│   ├── excluded_features.yaml     # Always-excluded features
│   ├── feature_target_schema.yaml # Feature-target schema
│   └── feature_groups.yaml        # Feature groups
│
├── experiments/              # Experiment configs (user-created)
│   ├── README.md
│   ├── _template.yaml
│   └── *.yaml                # Individual experiments
│
├── models/                   # Model hyperparameters
│   ├── lightgbm.yaml
│   ├── xgboost.yaml
│   └── ... (all model families)
│
├── pipeline/                 # Pipeline execution configs
│   ├── training/             # Training pipeline
│   │   ├── intelligent.yaml  # Intelligent training (main)
│   │   ├── safety.yaml       # Safety & temporal
│   │   ├── preprocessing.yaml
│   │   ├── optimizer.yaml
│   │   ├── callbacks.yaml
│   │   ├── routing.yaml
│   │   ├── stability.yaml
│   │   ├── decisions.yaml
│   │   ├── families.yaml
│   │   ├── sequential.yaml
│   │   └── first_batch.yaml
│   ├── gpu.yaml              # GPU settings
│   ├── memory.yaml           # Memory management
│   ├── threading.yaml        # Threading policy
│   └── pipeline.yaml         # Main pipeline config
│
├── ranking/                  # Ranking & selection configs
│   ├── targets/              # Target ranking
│   │   ├── multi_model.yaml
│   │   └── configs.yaml
│   └── features/             # Feature selection
│       ├── multi_model.yaml
│       └── config.yaml
│
└── archive/                  # Archived/deprecated files
    └── *.yaml                # Legacy configs
```

**Note:** Old locations are still supported via symlinks for backward compatibility. See `MIGRATION_GUIDE.md` for details.

## Quick Start

### Using Experiment Configs (Recommended)

Create an experiment config in `CONFIG/experiments/`:

```yaml
experiment:
  name: my_experiment
  description: "Test run"

data:
  data_dir: data/data_labeled/interval=5m
  symbols: [AAPL, MSFT]
  interval: 5m
  max_samples_per_symbol: 3000
  min_cs: 10
  max_cs_samples: 1000

intelligent_training:
  auto_targets: true
  top_n_targets: 5
  auto_features: true
  top_m_features: 70
```

Then run:

```bash
python -m TRAINING.orchestration.intelligent_trainer --experiment-config my_experiment
```

### Editing Configs

- **Model configs**: Edit files in `models/` (e.g., `models/lightgbm.yaml`)
- **Training configs**: Edit files in `pipeline/training/` (e.g., `pipeline/training/intelligent.yaml`)
- **System configs**: Edit files in `pipeline/` (e.g., `pipeline/gpu.yaml`)
- **Feature registry**: Edit `data/feature_registry.yaml`

## Config Files Status

### ✅ Active Config Files

All files in the new structure are actively used:
- `core/` - Core system configs
- `data/` - Data-related configs
- `experiments/` - Experiment configs
- `models/` - Model hyperparameters
- `pipeline/` - Pipeline execution configs
- `ranking/` - Ranking & selection configs

### 🗑️ Archived Files

Unused files have been moved to `archive/`:
- `comprehensive_feature_ranking.yaml` - Legacy feature ranking
- `fast_target_ranking.yaml` - Legacy fast ranking
- `multi_model_feature_selection.yaml.deprecated` - Deprecated feature selection

## Migration

The config structure has been reorganized for better usability. All old locations are still supported via symlinks.

- **Migration complete**: Files moved to new locations
- **Backward compatible**: Old paths still work
- **See**: `MIGRATION_GUIDE.md` for details

## Documentation

- **Configuration Reference:** See `DOCS/02_reference/configuration/`
- **Experiment Configs:** See `CONFIG/experiments/README.md`
- **Migration Guide:** See `DOCS/02_reference/configuration/migration/MIGRATION_GUIDE.md`
- **Cleanup Plan:** See `DOCS/02_reference/configuration/migration/CLEANUP_PLAN.md`

## Backward Compatibility

All legacy config locations are still supported:
1. Check new location first
2. Fall back to legacy location if new doesn't exist
3. Show debug message when using legacy location

This ensures existing code continues to work while encouraging migration to the new structure.
