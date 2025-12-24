# Defaults Centralization - Changes Applied

## Summary

Centralized common configuration values from across all config files into `CONFIG/defaults.yaml` to establish a Single Source of Truth (SST).

## Changes Made

### Neural Network Defaults Added
- `dropout: 0.2` - Used in 87.5% of neural network configs (7/8 files)
- `activation: "relu"` - Used in 100% of neural network configs (8/8 files)
- `patience: 10` - Used in 71.4% of neural network configs (5/7 files with patience)

### Cross-Validation Defaults Added
- `cv_folds: 3` - Used in 80% of configs
- `n_jobs: 1` - Used in 100% of configs

### Aggregation Defaults Added
- `consensus_threshold: 0.5` - Used in 100% of multi-model configs
- `cross_model_method: "weighted_mean"` - Used in 100% of multi-model configs
- `require_min_models: 2` - Used in 100% of multi-model configs

### Output Defaults Added
- `save_metadata: true` - Used in 100% of configs
- `save_per_family_rankings: true` - Used in 100% of multi-model configs
- `save_agreement_matrix: true` - Used in 100% of multi-model configs
- `include_model_scores: true` - Used in 100% of multi-model configs

### Compute Defaults Added
- `use_gpu: false` - Used in 100% of configs

### Sampling Defaults Added
- `validation_split: 0.2` - Used in 100% of configs
- `shuffle: true` - Used in 100% of preprocessing configs

### SHAP/Explainability Defaults Added
- `max_samples: 1000` - Used in 100% of configs
- `use_tree_explainer: true` - Used in 100% of configs
- `kernel_explainer_background: 100` - Used in 100% of configs

## Files Modified

- `CONFIG/defaults.yaml` - Added all centralized defaults
- `CONFIG/config_loader.py` - Added `inject_defaults()` function
- Multiple model config files - Removed duplicate values (see CLEANUP_SUMMARY.md)
