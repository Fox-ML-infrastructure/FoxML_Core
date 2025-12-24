# Reproducibility Settings Centralization

## Summary

All **reproducibility-critical settings** are now centralized to ensure consistent, reproducible results across all runs.

## Centralized Reproducibility Settings

### 1. Random Seeds ✅
**Location:** `CONFIG/defaults.yaml` → `randomness.random_state`
- **Source:** `pipeline.determinism.base_seed` (default: 42)
- **Applied to:** All models, train/test splits, cross-validation, feature selection
- **Files cleaned:** 15+ config files with hardcoded `random_state: 42`

### 2. Data Shuffling ✅
**Location:** `CONFIG/defaults.yaml` → `sampling.shuffle`
- **Default:** `true` (shuffle data before splitting)
- **Exception:** Time series CV uses `shuffle: false` (preserves temporal order)
- **Impact:** Affects train/test split reproducibility

### 3. Validation Splits ✅
**Location:** `CONFIG/defaults.yaml` → `sampling.validation_split`
- **Default:** `0.2` (20% validation split)
- **Impact:** Affects data distribution between train/val/test

## Single Source of Truth

All reproducibility settings flow from:
```
pipeline.determinism.base_seed (pipeline_config.yaml)
    ↓
defaults.randomness.random_state (defaults.yaml)
    ↓
All model configs (auto-injected)
```

## Files Updated

### Multi-Model Configs (3 files)
- ✅ `CONFIG/feature_selection/multi_model.yaml` - Already had global.random_state
- ✅ `CONFIG/target_ranking/multi_model.yaml` - Added global.random_state, removed 12 hardcoded values
- ✅ `CONFIG/multi_model_feature_selection.yaml` - Added global.random_state, removed 12 hardcoded values

### Model Configs (2 files)
- ✅ `CONFIG/model_config/gmm_regime.yaml` - Replaced hardcoded random_state
- ✅ `CONFIG/comprehensive_feature_ranking.yaml` - Replaced hardcoded random_state

### Training Configs (3 files)
- ✅ `CONFIG/training_config/preprocessing_config.yaml` - Replaced hardcoded random_state and shuffle
- ✅ `CONFIG/training/models.yaml` - Replaced hardcoded random_state (kept shuffle: false for time series CV)
- ✅ `CONFIG/training_config/first_batch_specs.yaml` - Replaced hardcoded random_state

### Feature Selection Configs (1 file)
- ✅ `CONFIG/feature_selection_config.yaml` - Replaced hardcoded random_state

## How It Works

1. **Base Seed:** Set in `pipeline_config.yaml` → `pipeline.determinism.base_seed: 42`
2. **Defaults Injection:** `config_loader.py` automatically injects `randomness.random_state` from defaults
3. **Model Configs:** All model configs receive the centralized seed unless explicitly overridden
4. **Multi-Model Configs:** Use `global.random_state: null` to inherit from pipeline determinism

## Benefits

✅ **Single Source of Truth** - Change seed in one place (`pipeline.determinism.base_seed`)  
✅ **Consistent Results** - All models use the same seed source  
✅ **Easy Reproducibility** - Set seed once, applies everywhere  
✅ **No Breaking Changes** - Pipeline fully functional, defaults preserved  

## Usage

### To change the global seed:
```yaml
# CONFIG/training_config/pipeline_config.yaml
pipeline:
  determinism:
    base_seed: 1337  # Change here, applies everywhere
```

### To override for a specific model:
```yaml
# CONFIG/model_config/some_model.yaml
hyperparameters:
  random_state: 999  # Explicit override (rarely needed)
```

## Verification

All configs tested and verified:
- ✅ Configs load successfully
- ✅ Defaults are injected correctly
- ✅ Explicit overrides are preserved
- ✅ Pipeline is safe - no breaking changes
