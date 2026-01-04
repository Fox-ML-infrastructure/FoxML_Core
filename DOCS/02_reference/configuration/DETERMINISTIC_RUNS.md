# Deterministic Runs

This document explains how to run the training pipeline with **bitwise reproducible results** for financial audit compliance and research reproducibility.

## Quick Start

### Run with Strict Determinism
```bash
bin/run_deterministic.sh python TRAINING/orchestration/intelligent_trainer.py --experiment-config determinism_test
```

### Run Normal (Best Effort)
```bash
python TRAINING/orchestration/intelligent_trainer.py --experiment-config your_config
```

## Modes

| Mode | Command | Reproducibility | Performance |
|------|---------|-----------------|-------------|
| **Strict** | `bin/run_deterministic.sh python ...` | Bitwise identical | CPU only, single-threaded |
| **Best Effort** | `python ...` | Seeded but may vary | Full GPU/parallelism |

## What Strict Mode Does

The launcher script (`bin/run_deterministic.sh`) sets critical environment variables **before Python starts**:

```bash
PYTHONHASHSEED=42        # Deterministic Python hash
REPRO_MODE=strict        # Enable strict mode
OMP_NUM_THREADS=1        # Single-threaded OpenMP
MKL_NUM_THREADS=1        # Single-threaded MKL
CUBLAS_WORKSPACE_CONFIG=:4096:8  # CUDA determinism
```

The training pipeline then:
- Forces all models to use `n_jobs=1` (single-threaded)
- Forces tree models to use `device_type=cpu` (no GPU)
- Sets `deterministic=True` for LightGBM
- Uses SHA256-based seed derivation for stability

## Verification

### Run Twice and Compare
```bash
# Run 1
bin/run_deterministic.sh python TRAINING/orchestration/intelligent_trainer.py \
    --experiment-config determinism_test 2>&1 | tee run1.log

# Run 2
bin/run_deterministic.sh python TRAINING/orchestration/intelligent_trainer.py \
    --experiment-config determinism_test 2>&1 | tee run2.log

# Compare fingerprints (should be identical)
diff <(grep fingerprint run1.log) <(grep fingerprint run2.log)
```

### Verify Using Snapshots

Each run produces `snapshot.json` files with determinism-relevant signatures:

```bash
# Find snapshots from your run
find RESULTS/runs/*/targets/*/reproducibility -name "snapshot.json"

# Compare comparison_group fields between runs
jq '.comparison_group' run1/targets/fwd_ret_10m/reproducibility/CROSS_SECTIONAL/cohort=*/snapshot.json
```

**Key fields to compare:**

| Field | Purpose |
|-------|---------|
| `dataset_signature` | Hash of data_dir + symbols |
| `task_signature` | Hash of target config |
| `routing_signature` | Hash of view + symbol routing |
| `hyperparameters_signature` | Hash of model hyperparameters |
| `train_seed` | Random seed used |
| `metrics_sha256` | Hash of output metrics |

If all signatures match between runs, the configuration is identical.

### What to Expect
- **Identical fingerprints** = determinism working
- **Different fingerprints** = something is non-deterministic

## Config Requirements

For deterministic runs, your experiment config MUST disable parallelism:

```yaml
multi_target:
  parallel_targets: false  # Sequential execution

multi_model_feature_selection:
  parallel_symbols: false  # Sequential execution

threading:
  parallel:
    max_workers_process: 1  # Single worker
    max_workers_thread: 1   # Single worker
    enabled: false          # Disable parallelism
```

See `CONFIG/experiments/determinism_test.yaml` for a complete example.

## Architecture

```
bin/run_deterministic.sh
    │
    ├── Sets PYTHONHASHSEED=42 (before Python starts)
    ├── Sets REPRO_MODE=strict
    ├── Sets OMP_NUM_THREADS=1, MKL_NUM_THREADS=1
    │
    ▼
intelligent_trainer.py
    │
    ├── import repro_bootstrap (FIRST - sets thread env vars)
    ├── Validates PYTHONHASHSEED is set
    ├── Checks no numeric libs imported before bootstrap
    │
    ▼
determinism.py
    │
    ├── load_reproducibility_config() - ENV overrides YAML
    ├── seed_all() - Sets Python/NumPy/Torch seeds
    ├── create_estimator() - Single choke point for model creation
    │   ├── Applies n_jobs=1 in strict mode
    │   ├── Applies device_type=cpu in strict mode
    │   └── Uses normalize_seed() to prevent edge cases
    │
    ▼
Bitwise Identical Results
```

## Output Directory Structure

Reproducibility artifacts are organized under each target:

```
RESULTS/runs/<run_name>/
└── targets/<target>/
    └── reproducibility/
        ├── CROSS_SECTIONAL/
        │   ├── universe=<sig>/           # Identifies symbol set
        │   │   └── feature_importances/  # Per-model CSV files
        │   └── cohort=cs_<id>/
        │       ├── snapshot.json         # Full snapshot with signatures
        │       ├── metrics.json          # Output metrics
        │       ├── metadata.json         # Run metadata
        │       └── diff_prev.json        # Comparison to previous run
        └── SYMBOL_SPECIFIC/
            └── symbol=AAPL/              # Per-symbol directory
                ├── feature_importances/  # Per-model CSV files
                └── cohort=sy_<id>/
                    ├── snapshot.json
                    ├── metrics.json
                    ├── metadata.json
                    └── diff_prev.json
```

**Key directories:**
- `CROSS_SECTIONAL/` - Multi-symbol (panel) runs
- `SYMBOL_SPECIFIC/symbol=XXX/` - Per-symbol runs

## Configuration Files

| File | Purpose |
|------|---------|
| `CONFIG/pipeline/training/reproducibility.yaml` | Default determinism settings |
| `CONFIG/experiments/determinism_test.yaml` | Test config with parallelism disabled |
| `bin/run_deterministic.sh` | Launcher script |

### reproducibility.yaml
```yaml
reproducibility:
  mode: best_effort  # Default (strict via launcher)
  seed: 42
  version: v1
  
  strict:
    require_env_vars: true
    disable_gpu_tree_models: true
    force_single_thread: true
    enforce_stable_ordering: true
```

## For Production Financial Use

### Audit Trail
1. **Always use strict mode** for compliance-critical runs
2. **Store fingerprints** in your audit database
3. **Compare fingerprints** to detect code/environment drift

### Environment Pinning
Strict determinism only guarantees reproducibility within the same environment:
- Same Python version
- Same library versions (lightgbm, xgboost, numpy, etc.)
- Same CPU architecture

Pin versions in `requirements.txt` or use Docker for full reproducibility.

### Regulatory Compliance
- Run determinism test before production deployments
- Store prediction hashes alongside model artifacts
- Document any fingerprint changes in change log

## Troubleshooting

### "PYTHONHASHSEED not set"
Use the launcher script:
```bash
bin/run_deterministic.sh python your_script.py
```

### "Bootstrap imported too late"
Ensure `repro_bootstrap` is imported FIRST in your entrypoint:
```python
import TRAINING.common.repro_bootstrap  # MUST be first
import numpy as np  # Now safe
```

### Different fingerprints across runs
Check:
1. Parallelism is disabled in config
2. Using the launcher script
3. Same library versions
4. No external randomness (e.g., shuffled data loading)

## API Reference

### Key Functions

```python
from TRAINING.common.determinism import (
    create_estimator,      # Single choke point for model creation
    seed_all,              # Set all random seeds
    resolve_seed,          # SHA256-based seed derivation
    is_strict_mode,        # Check if strict mode enabled
    stable_sort,           # Deterministic ordering
    load_reproducibility_config,  # Load SST config
)
```

### Example Usage
```python
import TRAINING.common.repro_bootstrap  # FIRST!

from TRAINING.common.determinism import create_estimator, seed_all, resolve_seed

seed_all(42)

# Create model with determinism params automatically applied
model = create_estimator(
    library="lightgbm",
    base_config={"n_estimators": 100},
    seed=resolve_seed(42, "training", target="fwd_ret_10m"),
    problem_kind="regression"
)
```
