# Experiment Configs - Self-Contained Configuration

## Overview

**Experiment configs are SELF-CONTAINED** - everything you need to control is in one file!

You don't need to edit 20 different files. Just edit your experiment config file and everything else is automatically overridden.

## Quick Start

1. **Copy the template:**
   ```bash
   cp CONFIG/experiments/_template.yaml CONFIG/experiments/my_experiment.yaml
   ```

2. **Edit your experiment config:**
   - Change `experiment.name` to your experiment name
   - Adjust settings in the file (all documented inline)
   - Everything you need is in this one file!

3. **Run with your experiment:**
   ```bash
   python -m TRAINING.orchestration.intelligent_trainer --experiment-config my_experiment
   ```

## What You Can Control

### Data Settings (`data.*`)
- `data_dir` - Where to load data from
- `symbols` - Which symbols to use
- `max_samples_per_symbol` - Limit data per symbol
- `max_rows_train` - Limit training data
- `min_cs`, `max_cs_samples` - Cross-sectional sampling limits

### Target Selection (`intelligent_training.*`)
- `auto_targets` - Auto-discover or manual list
- `top_n_targets` - How many top targets to SELECT
- `max_targets_to_evaluate` - How many targets to RANK (for speed)
- `exclude_target_patterns` - Patterns to exclude
- `manual_targets` - Specific targets to use (if auto_targets=false)

### Feature Selection (`intelligent_training.*`)
- `auto_features` - Auto-discover or manual list
- `top_m_features` - How many features per target
- `manual_features` - Specific features to use (if auto_features=false)

### Model Families (`feature_selection.*`, `training.*`)
- `model_families` - Which models to use (lightgbm, xgboost, etc.)

### Parallel Execution (`multi_target.*`, `threading.*`)
- `parallel_targets` - Parallel target evaluation
- `parallel_symbols` - Parallel symbol processing
- `max_workers_process` - CPU workers
- `max_workers_thread` - I/O workers

### Decisions (`decisions.*`)
- `apply_mode` - off, dry_run, or apply
- `use_bayesian` - Enable automatic optimization

## Precedence

**Experiment configs have HIGHEST PRIORITY** - they override:
1. ✅ `pipeline/training/intelligent.yaml`
2. ✅ `pipeline/training/*.yaml` (all pipeline configs)
3. ✅ `defaults.yaml`
4. ✅ Everything else!

**Only CLI arguments override experiment configs.**

## Examples

### Fast Test Run
```yaml
intelligent_training:
  top_n_targets: 5
  max_targets_to_evaluate: 10
  top_m_features: 30

data:
  max_samples_per_symbol: 1000
  min_cs: 3
```

### Production Run
```yaml
intelligent_training:
  top_n_targets: 20
  max_targets_to_evaluate: 100
  top_m_features: 100

data:
  max_samples_per_symbol: 50000
  min_cs: 10
```

### Specific Target Test
```yaml
intelligent_training:
  auto_targets: false
  manual_targets: [fwd_ret_60m, fwd_ret_120m]
  top_m_features: 50
```

## Need Help?

Run the config hierarchy tool:
```bash
python3 CONFIG/tools/show_config_hierarchy.py your_experiment_name
```

This shows exactly what your experiment config controls!
