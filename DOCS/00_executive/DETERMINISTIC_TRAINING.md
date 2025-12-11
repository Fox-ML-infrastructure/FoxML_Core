# Deterministic Training & Reproducibility

**Enterprise-Grade ML Infrastructure**

This ML research infrastructure system guarantees **deterministic, reproducible training** through a Single Source of Truth (SST) configuration architecture.

---

## Core Guarantee

> **Same config → same behavior → same results.**

Every training run with identical configuration files produces identical model outputs, enabling:

- **Reproducible backtests** - Compare strategies with confidence
- **Auditable decisions** - All behavior controlled via versioned config files
- **Deterministic debugging** - Isolate issues by comparing config vs. code
- **Compliance-ready** - Clear separation of code (immutable) vs. configuration (tunable)

---

## For Existing Users: No Action Required

**✅ Your existing code and configuration files continue to work unchanged.**

The SST and determinism improvements (completed 2025-12-10) were **internal changes** that enhance reproducibility without requiring any user migration:

- **Same API** - All function calls and CLI commands work exactly as before
- **Same configs** - Your existing YAML configuration files are fully compatible
- **Automatic** - SST enforcement and deterministic seeds are applied automatically
- **Backward compatible** - Legacy config locations and patterns still work (with deprecation warnings)

**What changed internally:**
- Removed hardcoded hyperparameters from source code (now all load from config)
- Replaced hardcoded `random_state=42` with centralized `BASE_SEED` system
- Added automated enforcement test to prevent future hardcoded values

**What didn't change:**
- Your code - no modifications needed
- Your configs - existing YAML files work as-is
- Your workflow - same commands, same results (just more reproducible)

If you want to take advantage of new features (like experiment configs or modular configs), see the [Modular Config System](../02_reference/configuration/MODULAR_CONFIG_SYSTEM.md), but this is **optional** - your current setup works fine.

---

## How It Works

### 1. Centralized Configuration

All hyperparameters, thresholds, and behavioral knobs are defined in YAML configuration files:

```
CONFIG/
├── model_config/          # Model hyperparameters (LightGBM, XGBoost, etc.)
├── training_config/       # Training workflows, safety thresholds, data splits
└── feature_selection/     # Feature selection and ranking parameters
```

**Example**: LightGBM hyperparameters are defined in `CONFIG/model_config/lightgbm.yaml`, not hardcoded in source code.

### 2. Deterministic Seeds

All randomness is controlled through a centralized determinism system:

- Base seed set globally at startup
- Per-target/fold seeds derived deterministically
- Same target + same fold + same config → same seed → same model

### 3. Automated Enforcement

An automated test (`TRAINING/tests/test_no_hardcoded_hparams.py`) enforces SST compliance:

- Scans all training code for hardcoded hyperparameters
- Flags violations that bypass configuration
- Ensures new code follows SST principles

---

## Benefits

### For Quants & Researchers

- **Easy experimentation**: Sweep hyperparameters by editing YAML files
- **Reproducible results**: Share config files to reproduce exact results
- **Version control**: Track config changes alongside code changes

### For Operations

- **Deployment flexibility**: Change model behavior without code changes
- **A/B testing**: Swap configs for different strategies
- **Rollback safety**: Revert to previous configs if needed

### For Enterprise Buyers

- **Auditability**: All behavior controlled via config files (not hidden in code)
- **Compliance**: Clear separation of code vs. configuration
- **Transparency**: Config files are human-readable and reviewable

---

## Usage Example

### Running with Default Config

```bash
python TRAINING/training_strategies/main.py --target fwd_ret_5m
```

Uses default hyperparameters from `CONFIG/model_config/lightgbm.yaml`.

### Running with Custom Config

```bash
# Create custom config overlay
cp CONFIG/model_config/lightgbm.yaml CONFIG/model_config/lightgbm_custom.yaml
# Edit hyperparameters in custom config
python TRAINING/training_strategies/main.py --target fwd_ret_5m --config lightgbm_custom
```

### Training Profiles

Switch between debug, default, and production profiles:

```bash
# Fast debug run (small batch size, few epochs)
python TRAINING/training_strategies/main.py --profile debug

# Production run (optimized batch size, full epochs)
python TRAINING/training_strategies/main.py --profile throughput_optimized
```

---

## Configuration Structure

### Model Hyperparameters

```yaml
# CONFIG/model_config/lightgbm.yaml
hyperparameters:
  n_estimators: 1000
  max_depth: 8
  learning_rate: 0.03
  num_leaves: 96

variants:
  conservative:
    learning_rate: 0.01
    n_estimators: 2000
  aggressive:
    learning_rate: 0.05
    n_estimators: 500
```

### Safety Thresholds

```yaml
# CONFIG/training_config/safety_config.yaml
safety:
  leakage_detection:
    auto_fix_thresholds:
      cv_score: 0.99
      training_accuracy: 0.999
```

### Training Profiles

```yaml
# CONFIG/training_config/optimizer_config.yaml
training_profiles:
  default:
    batch_size: 256
    max_epochs: 50
  debug:
    batch_size: 32
    max_epochs: 5
```

---

## Verification

Verify your setup is deterministic:

```bash
# Run same target twice with same config
python TRAINING/training_strategies/main.py --target fwd_ret_5m
python TRAINING/training_strategies/main.py --target fwd_ret_5m

# Models should be identical (same predictions, same metrics)
```

---

## Related Documentation

- **Internal Technical Details**: `DOCS/03_technical/internal/SST_DETERMINISM_GUARANTEES.md`
- **Configuration Reference**: `DOCS/02_reference/configuration/`
- **Model Configuration**: `DOCS/02_reference/configuration/MODEL_CONFIGURATION.md`

---

## Questions?

If you need to change model behavior:

1. **Check config files first** - Most parameters are already configurable
2. **Use config overlays** - Create variant configs without modifying base files
3. **Contact support** - For advanced configuration needs

**Never modify source code to change hyperparameters.** Use configuration files instead.
