# CONFIG Directory Structure

## Overview

This directory contains all configuration files for the trading system. The structure has been organized to eliminate duplication and provide clear separation of concerns.

## Recent Changes (2025-12-18)

### Config Cleanup
- ✅ **Removed duplicate file**: `multi_model_feature_selection.yaml` (duplicate of `ranking/features/multi_model.yaml`)
- ✅ **Symlink audit**: All symlinks documented and verified for backward compatibility
- ✅ **Path migration**: All hardcoded config paths in TRAINING replaced with centralized config loader API

### New Config Loader Functions
- ✅ `get_experiment_config_path(exp_name)` - Get path to experiment config file
- ✅ `load_experiment_config(exp_name)` - Load experiment config by name (with proper precedence)
- ✅ Enhanced `get_config_path()` to handle experiment configs automatically

### Validation Tools
- ✅ `tools/validate_config_paths.py` - Scans for remaining hardcoded paths and validates config loader access

### SST Compliance
- ✅ All config access now goes through centralized loader
- ✅ Defaults automatically injected from `defaults.yaml`
- ✅ Experiment configs properly override defaults (top-level config)

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
   - **Overrides** intelligent_training_config and defaults
   - **Top-level config** - only need to specify values that differ
   - **Fallback behavior**: Missing values fall back to intelligent_training_config, then defaults
   - **File existence**: If experiment config file doesn't exist, raises error (no fallback)
3. **Intelligent training config** (`pipeline/training/intelligent.yaml`)
   - Used when experiment config is not specified
   - Missing values fall back to defaults
4. **Pipeline configs** (`pipeline/training/*.yaml`)
5. **Defaults** (`defaults.yaml`) - lowest priority, injected automatically

## Symlinks (Legacy Compatibility)

The following symlinks exist for backward compatibility but point to the organized structure:

### Root-Level Symlinks
- `excluded_features.yaml` → `data/excluded_features.yaml`
- `feature_registry.yaml` → `data/feature_registry.yaml`
- `feature_groups.yaml` → `data/feature_groups.yaml`
- `feature_target_schema.yaml` → `data/feature_target_schema.yaml`
- `logging_config.yaml` → `core/logging.yaml`
- `target_configs.yaml` → `ranking/targets/configs.yaml`
- `feature_selection_config.yaml` → `ranking/features/config.yaml`

### Legacy Directory Symlinks
- `feature_selection/multi_model.yaml` → `ranking/features/multi_model.yaml`
- `target_ranking/multi_model.yaml` → `ranking/targets/multi_model.yaml`

### Training Config Symlinks (`training_config/`)
All files in `training_config/` are symlinks to the organized structure:
- `intelligent_training_config.yaml` → `pipeline/training/intelligent.yaml`
- `safety_config.yaml` → `pipeline/training/safety.yaml`
- `preprocessing_config.yaml` → `pipeline/training/preprocessing.yaml`
- `optimizer_config.yaml` → `pipeline/training/optimizer.yaml`
- `callbacks_config.yaml` → `pipeline/training/callbacks.yaml`
- `routing_config.yaml` → `pipeline/training/routing.yaml`
- `stability_config.yaml` → `pipeline/training/stability.yaml`
- `decision_policies.yaml` → `pipeline/training/decisions.yaml`
- `family_config.yaml` → `pipeline/training/families.yaml`
- `sequential_config.yaml` → `pipeline/training/sequential.yaml`
- `first_batch_specs.yaml` → `pipeline/training/first_batch.yaml`
- `gpu_config.yaml` → `pipeline/gpu.yaml`
- `memory_config.yaml` → `pipeline/memory.yaml`
- `threading_config.yaml` → `pipeline/threading.yaml`
- `pipeline_config.yaml` → `pipeline/pipeline.yaml`
- `system_config.yaml` → `core/system.yaml`

**Recommendation:** Use the organized paths directly in new code. The config loader API (`CONFIG.config_loader`) automatically resolves these paths, so you should use the loader functions instead of hardcoded paths.

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

## Changelog

### 2025-12-18: Config Cleanup and Path Migration

**Config Cleanup:**
- Removed duplicate `multi_model_feature_selection.yaml` (now only in `ranking/features/`)
- Documented all symlinks for backward compatibility
- Verified all symlinks are valid

**Path Migration:**
- Replaced all hardcoded `Path("CONFIG/...")` patterns in TRAINING with config loader API
- Updated files:
  - `TRAINING/orchestration/intelligent_trainer.py` (13 instances)
  - `TRAINING/ranking/predictability/model_evaluation.py` (2 instances)
  - `TRAINING/ranking/feature_selector.py` (2 instances)
  - `TRAINING/ranking/target_ranker.py` (5 instances)
  - `TRAINING/ranking/multi_model_feature_selection.py`
  - `TRAINING/ranking/utils/leakage_filtering.py`

**New Functions:**
- `get_experiment_config_path(exp_name)` - Get path to experiment config
- `load_experiment_config(exp_name)` - Load experiment config with proper precedence
- Enhanced `get_config_path()` to handle experiment configs

**Validation:**
- Created `tools/validate_config_paths.py` to scan for remaining hardcoded paths
- All active code paths now use config loader API
- Remaining hardcoded paths are only in fallback code (when loader unavailable)

**SST Compliance:**
- All config access goes through centralized loader
- Defaults automatically injected from `defaults.yaml`
- Experiment configs properly override defaults (top-level config)

## Migration Guide

### For Developers: Using Config Loader API

**❌ DON'T:** Use hardcoded paths
```python
exp_file = Path("CONFIG/experiments") / f"{exp_name}.yaml"
intel_config = Path("CONFIG/pipeline/training/intelligent.yaml")
```

**✅ DO:** Use config loader API
```python
from CONFIG.config_loader import get_experiment_config_path, load_experiment_config, load_training_config

# Get experiment config path
exp_path = get_experiment_config_path("my_experiment")

# Load experiment config
exp_config = load_experiment_config("my_experiment")

# Load training config
intel_config = load_training_config("intelligent_training_config")
```

### Legacy Path Compatibility

Old paths that still work (via symlinks):
- `training_config/intelligent_training_config.yaml` → `pipeline/training/intelligent.yaml`
- `training_config/routing_config.yaml` → `pipeline/training/routing.yaml`
- `excluded_features.yaml` → `data/excluded_features.yaml`
- `feature_selection/multi_model.yaml` → `ranking/features/multi_model.yaml`
- etc.

**But prefer using the new organized paths in `pipeline/training/`, `data/`, `ranking/` directly, or better yet, use the config loader API.**

