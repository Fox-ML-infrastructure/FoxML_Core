# CONFIG Directory Structure

## Overview

This directory contains all configuration files for the trading system. The structure has been organized to eliminate duplication and provide clear separation of concerns.

## Directory Structure

```
CONFIG/
├── core/              # Core system configs (logging, system settings)
├── data/              # Data-related configs (features, exclusions, schemas)
├── experiments/       # Experiment-specific configs (user-defined runs)
├── models/            # Model-specific hyperparameters
├── pipeline/          # Pipeline and training workflow configs
│   ├── training/      # Training-specific configs
│   └── [gpu, memory, threading, pipeline].yaml
├── ranking/           # Ranking configs (targets and features)
│   ├── targets/       # Target ranking configs
│   └── features/      # Feature selection configs
└── defaults.yaml      # Single Source of Truth for common defaults
```

## What Controls What

### 1. Experiment Configs (`experiments/*.yaml`)
**PRIMARY CONFIG FOR RUNS** - Highest priority when using `--experiment-config`

Controls:
- Data sources and limits (`data.*`)
- Target selection (`intelligent_training.top_n_targets`, `max_targets_to_evaluate`)
- Feature selection (`intelligent_training.top_m_features`)
- Training strategy and model families
- Parallel execution settings

**Example:** `experiments/e2e_full_targets_test.yaml`

### 2. Intelligent Training Config (`pipeline/training/intelligent.yaml`)
**BASE CONFIG** - Used when NOT using experiment config

Controls:
- Default data directory and symbols
- Default target/feature selection settings
- Training defaults

**Note:** `training_config/intelligent_training_config.yaml` is a symlink to this file.

### 3. Pipeline Configs (`pipeline/training/*.yaml`)
Training workflow configs:
- `safety.yaml` - Safety checks and reproducibility
- `routing.yaml` - Target routing decisions
- `preprocessing.yaml` - Data preprocessing
- `optimizer.yaml` - Optimization settings
- `stability.yaml` - Stability analysis
- `decisions.yaml` - Decision policies
- `families.yaml` - Model family configs
- `callbacks.yaml` - Training callbacks
- `sequential.yaml` - Sequential training
- `first_batch.yaml` - First batch specs

### 4. Ranking Configs (`ranking/*/`)
- `ranking/targets/configs.yaml` - Target definitions
- `ranking/targets/multi_model.yaml` - Multi-model target ranking
- `ranking/features/config.yaml` - Feature selection config
- `ranking/features/multi_model.yaml` - Multi-model feature selection

### 5. Data Configs (`data/*.yaml`)
- `excluded_features.yaml` - Features to exclude globally
- `feature_registry.yaml` - Feature registry
- `feature_target_schema.yaml` - Feature-target compatibility
- `feature_groups.yaml` - Feature groupings

### 6. Model Configs (`models/*.yaml`)
Model-specific hyperparameters (LightGBM, XGBoost, etc.)

### 7. Defaults (`defaults.yaml`)
**Single Source of Truth** for common settings:
- Random seeds
- Performance settings (n_jobs, threads)
- Common hyperparameters (learning_rate, n_estimators, etc.)

## Config Precedence (Highest to Lowest)

1. **CLI arguments** (highest priority)
2. **Experiment config** (`experiments/*.yaml`) - when using `--experiment-config`
3. **Intelligent training config** (`pipeline/training/intelligent.yaml`)
4. **Pipeline configs** (`pipeline/training/*.yaml`)
5. **Defaults** (`defaults.yaml`) - lowest priority

## Symlinks (Legacy Compatibility)

The following symlinks exist for backward compatibility but point to the organized structure:

- `training_config/*` → `pipeline/training/*` or `pipeline/*`
- `excluded_features.yaml` → `data/excluded_features.yaml`
- `feature_target_schema.yaml` → `data/feature_target_schema.yaml`
- `target_configs.yaml` → `ranking/targets/configs.yaml`
- `feature_selection_config.yaml` → `ranking/features/config.yaml`

**Recommendation:** Use the organized paths directly in new code.

## Quick Reference

### To change data limits:
- **With experiment config:** Edit `experiments/your_experiment.yaml` → `data.*`
- **Without experiment config:** Edit `pipeline/training/intelligent.yaml` → `data.*`

### To change target selection:
- **With experiment config:** Edit `experiments/your_experiment.yaml` → `intelligent_training.top_n_targets`
- **Without experiment config:** Edit `pipeline/training/intelligent.yaml` → `targets.top_n_targets`

### To change feature selection:
- **With experiment config:** Edit `experiments/your_experiment.yaml` → `intelligent_training.top_m_features`
- **Without experiment config:** Edit `pipeline/training/intelligent.yaml` → `features.top_m_features`

### To change model hyperparameters:
- Edit `models/lightgbm.yaml` (or specific model file)

### To change global defaults:
- Edit `defaults.yaml`

## Migration Notes

Old paths that still work (via symlinks):
- `training_config/intelligent_training_config.yaml` → `pipeline/training/intelligent.yaml`
- `training_config/routing_config.yaml` → `pipeline/training/routing.yaml`
- etc.

But prefer using the new organized paths in `pipeline/training/` directly.

