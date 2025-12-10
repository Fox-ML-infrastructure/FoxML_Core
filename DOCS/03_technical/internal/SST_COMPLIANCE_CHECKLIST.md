# SST Compliance Checklist

**Use this checklist before committing changes that touch trainers, safety files, or data processing.**

---

## Pre-Commit Checklist

### 1. Hyperparameters
- [ ] All model hyperparameters (`n_estimators`, `max_depth`, `learning_rate`, etc.) loaded from `CONFIG/model_config/*.yaml`?
- [ ] No hardcoded fallbacks unless they're truly last-resort defaults?
- [ ] Fallback values match config defaults?

### 2. Randomness & Seeds
- [ ] All `random_state` / `seed` values use `BASE_SEED` from `TRAINING/common/determinism`?
- [ ] Per-target/fold seeds use `seed_for(target, fold)`?
- [ ] No hardcoded `random_state=42` or `seed=42`?

### 3. Data Splits
- [ ] `test_size`, `cv_folds`, `n_splits` loaded from `preprocessing_config.yaml`?
- [ ] Shuffle flags controlled by config?

### 4. Safety Thresholds
- [ ] Leakage thresholds (`cv_score`, `training_accuracy`, etc.) from `safety_config.yaml`?
- [ ] Warning thresholds (`high`, `very_high`) from config?
- [ ] Auto-fix confidence thresholds from config?

### 5. Resource Use
- [ ] `batch_size`, `max_rows`, `max_epochs` from config?
- [ ] GPU flags from config (when user-configurable)?

### 6. Feature Selection
- [ ] `top_k` values from config?
- [ ] Correlation thresholds from config?
- [ ] Importance cutoffs from config?

### 7. Hardcoded Values
- [ ] Any remaining hardcoded values have `# FALLBACK_DEFAULT_OK` or `# DESIGN_CONSTANT_OK` marker?
- [ ] Design constants are truly invariant (not behavioral knobs)?

### 8. Testing
- [ ] Run `pytest TRAINING/tests/test_no_hardcoded_hparams.py`?
- [ ] Test passes?

---

## Quick Reference: Common Patterns

### ✅ Good (SST-Compliant)

```python
# Load from config
from CONFIG.config_loader import load_model_config, get_cfg
from TRAINING.common.determinism import BASE_SEED

config = load_model_config("lightgbm")
n_estimators = config.get("n_estimators", 100)

test_size = get_cfg("preprocessing.validation.test_size", default=0.2)
random_state = BASE_SEED if BASE_SEED is not None else 42
```

### ❌ Bad (Hardcoded)

```python
# Hardcoded values
n_estimators = 100
test_size = 0.2
random_state = 42
```

### ✅ Acceptable (Design Constant)

```python
# DESIGN_CONSTANT_OK: max 3 auto-reruns for safety (not exposed as user config)
MAX_AUTO_RERUNS_INTERNAL = 3

# DESIGN_CONSTANT_OK: numerical epsilon for stability
EPSILON = 1e-9
```

---

## File-Specific Checks

### Trainers (`TRAINING/model_fun/*.py`)
- [ ] All hyperparameters from `load_model_config()`?
- [ ] Seeds from determinism system?
- [ ] Batch sizes from config?

### Safety Files (`TRAINING/common/leakage_*.py`)
- [ ] All thresholds from `get_safety_config()`?
- [ ] Confidence values from config?
- [ ] Test sizes from config?

### Feature Selection (`TRAINING/utils/feature_*.py`)
- [ ] `top_k` from config?
- [ ] Correlation thresholds from config?
- [ ] Pruning parameters from config?

### Data Processing (`TRAINING/utils/data_*.py`)
- [ ] `test_size` from config?
- [ ] Shuffle flags from config?
- [ ] Seeds from determinism system?

---

## Automated Enforcement

Run before committing:

```bash
# SST enforcement test
pytest TRAINING/tests/test_no_hardcoded_hparams.py -v

# Or as part of full test suite
pytest TRAINING/tests/ -v
```

---

## When to Skip

You can skip SST checks for:
- Test files (`test_*.py`, `*_test.py`)
- Debug scripts (`tools/smoke_test*.py`, `EXPERIMENTS/`)
- Files explicitly excluded in test configuration

But still prefer config even in tests when testing config-driven behavior.

---

## Questions?

See `DOCS/03_technical/internal/SST_DETERMINISM_GUARANTEES.md` for detailed guidelines.
