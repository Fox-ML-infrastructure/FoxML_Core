# SST & Determinism Guarantees

**Single Source of Truth (SST) for Configuration**

This document defines the guarantees and policies for configuration management in the trading system. The core principle:

> **Same config → same behavior → same results.**

---

## 1. What Must Be Config (SST-Controlled)

Any value that affects **model behavior**, **data processing**, **safety thresholds**, or **resource use** must be loaded from YAML configuration files.

### Model Behavior
- Hyperparameters: `n_estimators`, `max_depth`, `learning_rate`, `alpha`, `num_leaves`, etc.
- Model architecture: hidden dimensions, layers, activation functions
- Training parameters: epochs, early stopping, callbacks

### Data Processing
- Data splits: `test_size`, `cv_folds`, `n_splits`, `train_size`
- Shuffling: `shuffle`, `random_state` (must use determinism system)
- Feature selection: `top_k`, correlation thresholds, importance cutoffs

### Randomness & Determinism
- **All seeds must use the determinism system** (`TRAINING/common/determinism.py`)
- Never hardcode `random_state=42` or `seed=42`
- Use `seed_for(target, fold)` or `BASE_SEED` from determinism module
- Seeds are derived deterministically from base seed + target/fold

### Safety Thresholds
- Leakage detection: `cv_score`, `training_accuracy`, `training_r2`, `perfect_correlation`
- Warning thresholds: `high`, `very_high` for classification/regression
- Model alerts: `suspicious_score`, importance thresholds
- Auto-fix confidence: `min_confidence`, `auto_fix_min_confidence`

### Resource Use
- `batch_size`, `max_rows`, `max_epochs`
- GPU flags, thread counts (when user-configurable)
- Memory limits, timeout values

### Routing & Confidence
- HIGH/MED/LOW confidence cutoffs
- Feature ranking thresholds
- Model selection criteria

---

## 2. What Can Stay Hardcoded

### Numerical Constants
- **Epsilon values**: `1e-9`, `np.finfo(float).eps` (for numerical stability)
- **Mathematical constants**: `math.pi`, `math.e` (truly invariant)
- **Architectural limits**: e.g., "if n <= 0: raise ValueError" (sanity checks, not behavioral knobs)

### Design Constants (with annotation)
If you intentionally keep a constant hardcoded, annotate it:

```python
# DESIGN_CONSTANT_OK: max 3 auto-reruns for safety (not exposed as user config)
MAX_AUTO_RERUNS_INTERNAL = 3
```

### Debug-Only Flags
- One-off debug flags in dev-only scripts (`EXPERIMENTS/`, `tools/`)
- Test fixtures and mock data generators

### Cosmetic Strings
- Logging format strings
- Error messages (unless they affect behavior)

---

## 3. Configuration Loading Patterns

### Model Hyperparameters

```python
from CONFIG.config_loader import load_model_config

# Load model config (supports variants and overrides)
config = load_model_config("lightgbm", variant="conservative")
n_estimators = config.get("n_estimators", 100)  # Default fallback
max_depth = config.get("max_depth", 6)
learning_rate = config.get("learning_rate", 0.1)
```

### Training Config (Nested Paths)

```python
from CONFIG.config_loader import get_cfg

# Get nested config values
test_size = get_cfg("preprocessing.validation.test_size", default=0.2)
batch_size = get_cfg("training.batch_size", default=32)
```

### Safety Thresholds

```python
from CONFIG.config_loader import get_safety_config

safety_cfg = get_safety_config()
leakage_cfg = safety_cfg.get("leakage_detection", {})
cv_threshold = float(leakage_cfg.get("auto_fix_thresholds", {}).get("cv_score", 0.99))
```

### Determinism (Seeds)

```python
from TRAINING.common.determinism import BASE_SEED, seed_for

# Use BASE_SEED (set globally at startup)
random_state = BASE_SEED if BASE_SEED is not None else 42

# Or derive per-target/fold seeds
seed = seed_for(target_name, fold_idx, "all_symbols")
```

---

## 4. Configuration File Structure

```
CONFIG/
├── model_config/
│   ├── lightgbm.yaml
│   ├── xgboost.yaml
│   └── ...
├── training_config/
│   ├── safety_config.yaml      # Leakage thresholds, safety guards
│   ├── preprocessing_config.yaml  # Data splits, feature selection
│   ├── pipeline_config.yaml    # Pipeline settings
│   └── ...
└── config_loader.py            # Centralized loader
```

### Example: `safety_config.yaml`

```yaml
safety:
  leakage_detection:
    auto_fix_thresholds:
      cv_score: 0.99
      training_accuracy: 0.999
      training_r2: 0.999
      perfect_correlation: 0.999
    warning_thresholds:
      classification:
        high: 0.90
        very_high: 0.95
```

### Example: `model_config/lightgbm.yaml`

```yaml
hyperparameters:
  n_estimators: 100
  max_depth: 6
  learning_rate: 0.1
  num_leaves: 31

variants:
  conservative:
    n_estimators: 200
    learning_rate: 0.05
  aggressive:
    n_estimators: 50
    learning_rate: 0.2
```

---

## 5. Determinism Guarantees

### Global Determinism Setup

```python
from TRAINING.common.determinism import set_global_determinism

# Call this FIRST (before importing torch/tf/lightgbm/xgboost)
BASE_SEED = set_global_determinism(
    base_seed=1234,  # From config or env var
    threads=1,  # For strict determinism
    deterministic_algorithms=True
)
```

### Per-Target/Fold Seeds

```python
from TRAINING.common.determinism import seed_for

# Deterministic seed for each target/fold combination
seed = seed_for("fwd_ret_5m", fold_idx=0, symbol_group="all_symbols")
```

### Guarantees

1. **Same base seed + same target + same fold → same seed**
2. **Same seed + same data + same config → same model**
3. **Same config → same hyperparameters → same behavior**

---

## 6. Testing & Enforcement

### SST Enforcement Test

Run the automated test:

```bash
pytest TRAINING/tests/test_no_hardcoded_hparams.py -v
```

This test:
- Scans all Python files in `TRAINING/`
- Flags hardcoded hyperparameters, thresholds, seeds
- Allows exceptions for DESIGN CONSTANT annotations
- Excludes test files and debug scripts

### Manual Checklist

Before committing changes that touch trainers or safety files:

- [ ] All hyperparameters loaded from config?
- [ ] All seeds use determinism system?
- [ ] All thresholds loaded from safety_config.yaml?
- [ ] Any hardcoded values have `# DESIGN CONSTANT` comment?
- [ ] SST test passes?

See `DOCS/03_technical/internal/SST_COMPLIANCE_CHECKLIST.md` for detailed checklist.

---

## 7. Migration Guide

### Before (Hardcoded)

```python
model = lgb.LGBMRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
```

### After (SST-Compliant)

```python
from CONFIG.config_loader import load_model_config
from TRAINING.common.determinism import BASE_SEED

config = load_model_config("lightgbm")
model = lgb.LGBMRegressor(
    n_estimators=config.get("n_estimators", 100),
    max_depth=config.get("max_depth", 6),
    learning_rate=config.get("learning_rate", 0.1),
    random_state=BASE_SEED if BASE_SEED is not None else 42
)
```

---

## 8. Why This Matters

### For Enterprise Buyers
- **Reproducibility**: Same config → same results
- **Auditability**: All behavior controlled via config files
- **Compliance**: Clear separation of code vs. configuration

### For Quants
- **Experimentation**: Easy to sweep hyperparameters via config
- **Backtesting**: Deterministic runs for comparison
- **Debugging**: Isolate config vs. code issues

### For Operations
- **Deployment**: Change behavior without code changes
- **A/B Testing**: Swap configs for different strategies
- **Monitoring**: Track config changes in version control

---

## 9. Related Documentation

- `CONFIG/README.md` - Configuration system overview
- `TRAINING/common/determinism.py` - Determinism system
- `DOCS/03_technical/internal/SST_COMPLIANCE_CHECKLIST.md` - Pre-commit checklist (internal)
- `TRAINING/tests/test_no_hardcoded_hparams.py` - Enforcement test

---

## 10. Questions?

If you're unsure whether something should be config or hardcoded:

1. **Ask**: "Would a user or future-me reasonably want to change this without touching code?"
2. **If yes** → It should be config
3. **If no** → Can stay hardcoded (with `# DESIGN_CONSTANT_OK` marker if non-obvious)

**When in doubt, make it config.** It's easier to ignore a config value than to retrofit config loading later.
