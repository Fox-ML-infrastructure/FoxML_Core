# Experiment Configuration Guide

Complete guide to creating and using experiment configuration files for the intelligent training pipeline.

## Overview

Experiment configs (`CONFIG/experiments/*.yaml`) define what targets to evaluate, which features to use, and how to train models. They provide a clean, reusable way to configure experiments without command-line arguments.

## Quick Start

### 1. Copy the Template

```bash
cp CONFIG/experiments/_template.yaml CONFIG/experiments/my_experiment.yaml
```

### 2. Edit Your Config

Open `CONFIG/experiments/my_experiment.yaml` and customize:

```yaml
experiment:
  name: my_experiment
  description: "My first experiment"

data:
  data_dir: data/data_labeled_v2/interval=5m
  symbols: [AAPL, MSFT, GOOGL, TSLA, NVDA]
  interval: 5m
  max_samples_per_symbol: 10000

intelligent_training:
  auto_targets: true      # Auto-discover targets
  top_n_targets: 5        # Select top 5 after ranking
  auto_features: true
  top_m_features: 50
```

### 3. Run It

```bash
python -m TRAINING.orchestration.intelligent_trainer \
  --output-dir "my_experiment_run" \
  --experiment-config my_experiment
```

## Configuration Structure

### Required Sections

#### `experiment`
Basic experiment metadata:
```yaml
experiment:
  name: my_experiment          # Unique experiment name
  description: "Description"   # Optional description
```

#### `data`
Data source configuration:
```yaml
data:
  data_dir: data/data_labeled_v2/interval=5m  # Required: data directory
  symbols: [AAPL, MSFT, GOOGL]                 # Required: list of symbols
  interval: 5m                                 # Bar interval (5m, 15m, 1h, etc.)
  max_samples_per_symbol: 10000               # Cap per symbol (optional)
  max_rows_per_symbol: 10000                   # Additional cap (optional)
  max_rows_train: 20000                        # Training data cap (optional)
  min_cs: 3                                    # Min cross-sectional samples (optional)
```

#### `intelligent_training`
Core training configuration:
```yaml
intelligent_training:
  auto_targets: true           # Auto-discover targets (true) or use manual_targets (false)
  top_n_targets: 5             # Number of top targets to select after ranking
  max_targets_to_evaluate: 100 # Max targets to evaluate (set high to get all)
  auto_features: true          # Auto-discover features (true) or use manual_features (false)
  top_m_features: 50          # Number of top features per target
  strategy: single_task        # Training strategy: single_task, multi_task, cascade
  run_leakage_diagnostics: false  # Run detailed leakage checks (slower)
  
  # Optional: Exclude target patterns (filters discovered targets by substring match)
  exclude_target_patterns:
    - "will_peak"    # Excludes y_will_peak_60m_0.8, y_will_peak_15m_0.5, etc.
    - "will_valley"  # Excludes y_will_valley_60m_0.8, etc.
  
  # Optional: Manual target list (used when auto_targets=false)
  manual_targets:
    - fwd_ret_60m
    - fwd_ret_120m
```

### Optional Sections

#### `targets` (Legacy/Reference)
```yaml
targets:
  primary: fwd_ret_60m  # Fallback if auto_targets=false and no manual_targets
```
**Note:** When `auto_targets=true`, this section is ignored. Only used as fallback when `auto_targets=false`.

#### `feature_selection`
Feature selection model families:
```yaml
feature_selection:
  model_families:
    - lightgbm
    - xgboost
    - random_forest
    - catboost
    - neural_network
    - lasso
    - mutual_information
    - univariate_selection
```

#### `training`
Training model families (usually same as feature_selection):
```yaml
training:
  model_families:
    - lightgbm
    - xgboost
    - random_forest
```

#### `decisions`
Auto-config application settings:
```yaml
decisions:
  apply_mode: "dry_run"        # "off", "dry_run", or "apply"
  min_level_to_apply: 2        # Minimum decision level (0-3)
  use_bayesian: true           # Enable Bayesian patch policy
  bayesian:
    min_runs_for_learning: 5
    p_improve_threshold: 0.8
    min_expected_gain: 0.01
```

#### `multi_target`
Parallel target evaluation:
```yaml
multi_target:
  parallel_targets: true       # Enable parallel evaluation
  skip_on_error: true          # Continue if one target fails
  save_summary: true           # Save summary JSON
```

#### `multi_model_feature_selection`
Parallel feature selection:
```yaml
multi_model_feature_selection:
  parallel_symbols: true       # Enable parallel symbol processing
```

#### `threading`
Worker configuration:
```yaml
threading:
  parallel:
    max_workers_process: 8     # CPU-bound tasks (null = auto-detect)
    max_workers_thread: 8     # I/O-bound tasks (null = auto-detect)
    enabled: true              # Master switch
```

## Common Patterns

### Pattern 1: Auto-Discover All Targets, Select Top 5

```yaml
intelligent_training:
  auto_targets: true
  top_n_targets: 5
  max_targets_to_evaluate: 100  # Evaluate all available
```

**Use case:** Find the most predictable targets in your dataset.

### Pattern 2: Test Specific Targets Only

```yaml
intelligent_training:
  auto_targets: false
  manual_targets:
    - fwd_ret_60m
    - fwd_ret_120m
    - y_will_peak_60m_0.8
```

**Use case:** Test specific targets you're interested in.

### Pattern 3: Fast Testing (Reduced Data)

```yaml
data:
  max_samples_per_symbol: 5000   # Reduced for speed
  max_rows_per_symbol: 5000
  max_rows_train: 10000
  min_cs: 3                       # Lower threshold

intelligent_training:
  top_m_features: 30              # Fewer features = faster
```

**Use case:** Quick iteration during development.

### Pattern 4: Comprehensive Evaluation (All Targets)

```yaml
intelligent_training:
  auto_targets: true
  top_n_targets: 10
  max_targets_to_evaluate: 100

multi_target:
  parallel_targets: true          # Enable parallel for speed

threading:
  parallel:
    max_workers_process: 8
    enabled: true
```

**Use case:** Complete evaluation of all targets in dataset.

## Configuration Precedence

1. **Experiment config** (`--experiment-config`) - Highest priority
2. **Intelligent training config** (`intelligent_training_config.yaml`) - Defaults
3. **Command-line arguments** - Override config when specified

## Validation Rules

### When `auto_targets=true`:
- ✅ `targets.primary` is **optional** (can be omitted or commented out)
- ✅ `intelligent_training.manual_targets` is **optional** (ignored if specified)
- ✅ Targets are auto-discovered from data

### When `auto_targets=false`:
- ✅ Either `intelligent_training.manual_targets` **OR** `targets.primary` must be specified
- ✅ If neither is specified, validation will fail

## Troubleshooting

### Error: "ExperimentConfig.target cannot be empty"
**Cause:** `auto_targets=false` but no `targets.primary` or `manual_targets` specified.

**Fix:** Either:
- Set `auto_targets: true`, OR
- Add `targets.primary: your_target`, OR
- Add `intelligent_training.manual_targets: [your_target]`

### Error: "Experiment config missing required field: targets.primary"
**Cause:** `auto_targets=false` but `targets.primary` is missing.

**Fix:** Either:
- Set `auto_targets: true`, OR
- Add `targets.primary: your_target`

### Only 1 target is being evaluated
**Cause:** `targets.primary` fallback is being used even when `auto_targets=true`.

**Fix:** 
- Ensure `auto_targets: true` in `intelligent_training` section
- Comment out or remove `targets.primary` (it's not needed when `auto_targets=true`)

### Config not found
**Cause:** Path resolution issue.

**Fix:** Use just the name (e.g., `my_experiment`) or relative path (e.g., `experiments/my_experiment`). The system will find it in `CONFIG/experiments/`.

## Example Configs

See these example configs for reference:

- `CONFIG/experiments/_template.yaml` - Complete template with all options
- `CONFIG/experiments/e2e_full_targets_test.yaml` - Comprehensive test (all targets)
- `CONFIG/experiments/e2e_ranking_test.yaml` - Quick ranking test
- `CONFIG/experiments/honest_baseline_test.yaml` - Non-repainting target test
- `CONFIG/experiments/non_repainting_targets_test.yaml` - Multiple forward returns

## Related Documentation

- [Auto Target Ranking Tutorial](../training/AUTO_TARGET_RANKING.md) - How target ranking works
- [Intelligent Training Tutorial](../training/INTELLIGENT_TRAINING_TUTORIAL.md) - Full pipeline overview
- [Config Basics](CONFIG_BASICS.md) - Configuration fundamentals
- [Config Examples](CONFIG_EXAMPLES.md) - More examples
