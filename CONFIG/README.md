# Configuration System

This directory contains all configuration files for the FoxML Core pipeline.

## New Modular Structure (Recommended)

The configuration system has been refactored into a modular structure to prevent config "crossing" between pipeline components:

```
CONFIG/
├── experiments/              # Experiment-level configs (what are we running?)
│   └── *.yaml
├── feature_selection/        # Feature selection module configs
│   └── multi_model.yaml
├── target_ranking/           # Target ranking module configs
│   └── multi_model.yaml
├── training/                 # Training module configs
│   └── models.yaml
├── logging_config.yaml       # Structured logging configuration (NEW)
│                             # Global, module-level, and backend verbosity controls
├── leakage/                  # Leakage detection configs
├── system/                   # System-level configs (paths, logging)
└── training_config/          # Legacy training configs (still used)
```

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

targets:
  primary: fwd_ret_60m

feature_selection:
  top_n: 30
  model_families: [lightgbm, xgboost]

training:
  model_families: [lightgbm, xgboost]
  cv_folds: 5
```

Then run:

```bash
python TRAINING/train.py --experiment-config my_experiment
```

### Legacy Usage (Still Supported)

You can still use individual config files:

```bash
python TRAINING/train.py \
    --data-dir data/data_labeled/interval=5m \
    --symbols AAPL MSFT \
    --targets fwd_ret_60m
```

## Migration Guide

### Feature Selection Config

**Old location:** `CONFIG/multi_model_feature_selection.yaml`  
**New location:** `CONFIG/feature_selection/multi_model.yaml`

The config loader automatically checks the new location first, then falls back to legacy. You'll see a deprecation warning if using the old location.

### Target Ranking Config

**Old location:** Uses feature selection config  
**New location:** `CONFIG/target_ranking/multi_model.yaml`

### Training Config

**Old location:** Various files in `CONFIG/training_config/`  
**New location:** `CONFIG/training/models.yaml` (for model families)

Training still uses `CONFIG/training_config/` for pipeline, GPU, memory, etc. settings.

## Documentation

- **Configuration Reference:** See `DOCS/02_reference/configuration/`
- **Experiment Configs:** See `CONFIG/experiments/README.md`
- **Feature Selection:** See `CONFIG/feature_selection/README.md`

## Backward Compatibility

All legacy config locations are still supported with deprecation warnings. The system will:
1. Check new location first
2. Fall back to legacy location if new doesn't exist
3. Show deprecation warning when using legacy location

This ensures existing code continues to work while encouraging migration to the new structure.
