# Experiment Configs

This directory contains experiment-level configuration files. Each experiment config defines:
- What data to use (symbols, data_dir, interval)
- What target to predict
- Which model families to use
- Overrides for specific modules

## Available Configs

- **`_template.yaml`** - Template for creating new experiment configs
- **`e2e_full_targets_test.yaml`** - Comprehensive test (all targets)
- **`e2e_ranking_test.yaml`** - Quick ranking test
- **`honest_baseline_test.yaml`** - Non-repainting target test
- **`non_repainting_targets_test.yaml`** - Multiple forward returns
- **`leakage_canary_test.yaml`** - Leakage canary test (validate pipeline integrity)

## Leakage Canary Test

The `leakage_canary_test.yaml` config uses known-leaky targets as canaries to validate pipeline integrity:

- **Purpose**: Test guardrails (purge/embargo, split logic, feature alignment)
- **Expected**: Canary targets should either hard-stop or be flagged as SUSPICIOUS
- **Negative controls**: Includes permutation/shuffle tests
- **See**: [Leakage Canary Test Guide](../../DOCS/03_technical/testing/LEAKAGE_CANARY_TEST_GUIDE.md)

## Usage

```python
from CONFIG.config_builder import load_experiment_config, build_feature_selection_config

exp_cfg = load_experiment_config("fwd_ret_60m_test")
fs_cfg = build_feature_selection_config(exp_cfg)
```

Or from command line:

```bash
python -m TRAINING.orchestration.intelligent_trainer \
  --output-dir "my_experiment_results" \
  --experiment-config my_experiment
```

