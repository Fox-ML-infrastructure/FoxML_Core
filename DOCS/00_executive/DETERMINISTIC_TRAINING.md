# Reproducible Training & Auditability

**ML Infrastructure**

This ML research infrastructure system ensures **reproducible, auditable training** through a Single Source of Truth (SST) configuration architecture. The system provides comprehensive tracking and reproducibility verification, enabling full auditability of training runs.

> **Note**: True bitwise determinism (identical outputs at the binary level) requires lower-level language implementations and strict control over floating-point operations. This system focuses on reproducibility (consistent results within expected variance) and auditability (full tracking of inputs, configs, and outputs).

---

## Core Guarantee

> **Same config → same behavior → reproducible results.**

Every training run with identical configuration files produces reproducible results within expected variance, enabling:

- **Reproducible backtests** - Compare strategies with confidence
- **Auditable decisions** - All behavior controlled via versioned config files with full tracking
- **Reproducible debugging** - Isolate issues by comparing config vs. code with complete audit trails
- **Compliance-ready** - Clear separation of code (immutable) vs. configuration (tunable) with full provenance

---

## For Existing Users: No Action Required

**✅ Your existing code and configuration files continue to work unchanged.**

The SST and reproducibility improvements (completed 2025-12-10) were **internal changes** that enhance reproducibility and auditability without requiring any user migration:

- **Same API** - All function calls and CLI commands work exactly as before
- **Same configs** - Your existing YAML configuration files are fully compatible
- **Automatic** - SST enforcement and reproducible seeds are applied automatically
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

### 2. Reproducible Seeds

All randomness is controlled through a centralized reproducibility system:

- Base seed set globally at startup
- Per-target/fold seeds derived reproducibly
- Same target + same fold + same config → same seed → reproducible model

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

## Run Identity System

The run identity system (added 2026-01-03) provides cryptographically robust identity keys for comparing runs:

### Identity Keys

| Key | Purpose |
|-----|---------|
| `replicate_key` | Cross-seed stability analysis (excludes seed) |
| `strict_key` | Diff telemetry (includes seed) |

Same config + different seeds → same `replicate_key` → grouped for stability analysis.

### Configuration

```yaml
# CONFIG/identity_config.yaml
identity:
  mode: strict  # strict | relaxed | legacy

stability:
  filter_mode: replicate
  allow_legacy_snapshots: false
```

**Modes**:
- `strict` (default): Fail if identity signatures missing
- `relaxed`: Log error, continue anyway
- `legacy`: Backward compatibility

### What Gets Hashed

- Dataset (symbols, data_dir, sample counts)
- Target (column, task type, horizon)
- Features (names + registry metadata)
- Hyperparameters (per model family)
- Splits (CV method, fold boundaries)
- Routing (view, symbol if SS)

For full details, see [Run Identity Reference](../02_reference/configuration/RUN_IDENTITY.md).

---

## Verification

Verify your setup is reproducible:

```bash
# Run same target twice with same config
python TRAINING/training_strategies/main.py --target fwd_ret_5m
python TRAINING/training_strategies/main.py --target fwd_ret_5m

# Results should be reproducible (consistent predictions and metrics within expected variance)
# Use the reproducibility tracking system to compare runs and verify consistency
```

---

## Related Documentation

- **Run Identity System**: [DOCS/02_reference/configuration/RUN_IDENTITY.md](../02_reference/configuration/RUN_IDENTITY.md)
- **Internal Technical Details**: `DOCS/03_technical/internal/SST_DETERMINISM_GUARANTEES.md` (note: file name preserved for historical reference)
- **Configuration Reference**: `DOCS/02_reference/configuration/`
- **Model Configuration**: `DOCS/02_reference/configuration/MODEL_CONFIGURATION.md`

---

## Questions?

If you need to change model behavior:

1. **Check config files first** - Most parameters are already configurable
2. **Use config overlays** - Create variant configs without modifying base files
3. **Contact support** - For advanced configuration needs

**Never modify source code to change hyperparameters.** Use configuration files instead.
