# Configuration Usage Examples

Practical examples for common configuration tasks in FoxML Core.

## Quick Reference

### Loading Configs

```python
from CONFIG.config_loader import (
    load_model_config,
    get_pipeline_config,
    get_gpu_config,
    get_safety_config,
    get_system_config,
    get_cfg
)

# Model configs
lightgbm = load_model_config("lightgbm", variant="aggressive")

# Training configs
pipeline = get_pipeline_config()
gpu = get_gpu_config()
safety = get_safety_config()

# Access nested values
timeout = get_cfg("pipeline.isolation_timeout_seconds", default=7200)
```

### Loading Feature/Target Configs

```python
import yaml

# Excluded features
with open("CONFIG/excluded_features.yaml") as f:
    excluded = yaml.safe_load(f)

# Feature registry
with open("CONFIG/feature_registry.yaml") as f:
    registry = yaml.safe_load(f)

# Target configs
with open("CONFIG/target_configs.yaml") as f:
    targets = yaml.safe_load(f)
```

## Example 0: Setting Data Interval in Experiment Config

**Goal:** Prevent interval auto-detection warnings and ensure consistent behavior across ranking and selection.

**Steps:**

1. **Add `bar_interval` to experiment config:**
```yaml
# CONFIG/experiments/my_experiment.yaml
experiment:
  name: my_experiment

data:
  data_dir: data/data_labeled/interval=5m
  symbols: [AAPL, MSFT, GOOGL]
  bar_interval: "5m"  # Explicit interval (prevents warnings)
  max_samples_per_symbol: 5000
```

2. **Run with experiment config:**
```bash
python TRAINING/train.py \
    --experiment-config my_experiment \
    --auto-targets \
    --top-n-targets 5 \
    --auto-features
```

**Benefits:**
- No interval auto-detection warnings in logs
- Consistent interval handling across ranking and selection
- Proper horizon conversion for leakage filtering
- Supports formats: `"5m"`, `"15m"`, `"1h"`, `"1d"`, or integer minutes

**Note:** If `bar_interval` is not specified, the pipeline will auto-detect from timestamps, but may log warnings if detection is unclear.

---

## Example 1: Adding a New Feature

**Goal:** Add a custom feature and make it available for training.

**Steps:**

1. **Add to `feature_registry.yaml`:**
```yaml
features:
  my_custom_momentum:
    source: price
    lag_bars: 5
    allowed_horizons: [12, 24, 60]
    description: "5-bar momentum indicator"
```

2. **Verify not excluded:**
   - Check `excluded_features.yaml` - ensure no patterns match `my_custom_momentum`
   - Check `feature_target_schema.yaml` - ensure not in metadata/target patterns

3. **Feature is now available** for targets with horizons 12, 24, or 60 bars

**Verification:**
```python
# Check if feature is available for horizon=12
with open("CONFIG/feature_registry.yaml") as f:
    registry = yaml.safe_load(f)
    feature = registry["features"]["my_custom_momentum"]
    is_allowed = 12 in feature["allowed_horizons"]  # Should be True
```

---

## Example 2: Excluding a Leaky Feature

**Goal:** Permanently exclude a feature that causes leakage.

**Steps:**

1. **Add to `excluded_features.yaml`:**
```yaml
always_exclude:
  exact_patterns:
    - future_price  # Exact feature name
```

2. **Or add pattern if multiple features:**
```yaml
always_exclude:
  regex_patterns:
    - "^future_"  # All features starting with "future_"
```

**Verification:**
```python
# Check if feature is excluded
with open("CONFIG/excluded_features.yaml") as f:
    excluded = yaml.safe_load(f)
    exact = excluded["always_exclude"]["exact_patterns"]
    is_excluded = "future_price" in exact  # Should be True
```

---

## Example 3: Adjusting Leakage Detection Sensitivity

**Goal:** Make leakage detection more or less sensitive.

**Steps:**

1. **Edit `training_config/safety_config.yaml`:**
```yaml
leakage_detection:
  # More sensitive (detects at lower thresholds)
  auto_fix_thresholds:
    cv_score: 0.95  # Lower from 0.99
    training_accuracy: 0.98  # Lower from 0.999
  
  # Less aggressive auto-fixer
  auto_fix_min_confidence: 0.9  # Higher from 0.8
```

**Usage:**
```python
# Config is automatically loaded by training pipeline
# No code changes needed
```

---

## Example 4: Configuring CatBoost Loss Function

**Goal:** Override CatBoost's auto-detected loss function (if needed).

**Context:** By default, CatBoost auto-detects target type and sets:
- `Logloss` for binary classification
- `MultiClass` for multiclass classification  
- `RMSE` for regression

**Steps:**

1. **Override in model config** (`CONFIG/training_config/multi_model_feature_selection.yaml` or experiment config):
```yaml
model_families:
  catboost:
    enabled: true
    loss_function: "CrossEntropy"  # Override auto-detection
    # ... other params
```

2. **Or let auto-detection handle it** (recommended):
```yaml
model_families:
  catboost:
    enabled: true
    # loss_function not specified - will be auto-detected from target type
    # ... other params
```

**Usage:**
```python
# Auto-detection happens automatically in ranking and selection
# No code changes needed
```

**When to override:**
- Custom loss functions for specific use cases
- Multi-class with specific class weights
- Regression with custom objectives

---

## Example 5: Configuring Multi-GPU Setup

**Goal:** Use multiple GPUs for training.

**Steps:**

1. **Edit `training_config/gpu_config.yaml`:**
```yaml
gpu:
  device_visibility: [0, 1, 2, 3]  # Use all 4 GPUs
  vram_cap_mb: 8192  # Per-GPU limit
```

2. **Set environment variable:**
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

**Verification:**
```python
gpu_config = get_gpu_config()
print(gpu_config["gpu"]["device_visibility"])  # Should show [0, 1, 2, 3]
```

---

## Example 6: Enabling More Targets

**Goal:** Enable additional targets for training.

**Steps:**

1. **Edit `target_configs.yaml`:**
```yaml
targets:
  swing_high_15m:
    enabled: true  # Change from false
    top_n: 50
    method: "mean"
```

2. **Targets are automatically discovered** in intelligent training pipeline

**Verification:**
```python
with open("CONFIG/target_configs.yaml") as f:
    targets = yaml.safe_load(f)
    enabled = {
        name: cfg for name, cfg in targets["targets"].items()
        if cfg.get("enabled", False)
    }
    print(f"Enabled targets: {len(enabled)}")
```

---

## Example 7: Customizing Model Hyperparameters

**Goal:** Create a custom model variant.

**Steps:**

1. **Edit `model_config/lightgbm.yaml`:**
```yaml
my_custom_variant:
  n_estimators: 1000
  learning_rate: 0.01
  num_leaves: 255
  max_depth: 10
  min_child_samples: 30
  subsample: 0.8
  colsample_bytree: 0.8
  reg_alpha: 0.1
  reg_lambda: 0.1
```

2. **Use in code:**
```python
config = load_model_config("lightgbm", variant="my_custom_variant")
trainer = LightGBMTrainer(config)
trainer.train(X_train, y_train)
```

---

## Example 8: Adjusting Backup Retention

**Goal:** Keep more or fewer config backups.

**Steps:**

1. **Edit `training_config/system_config.yaml`:**
```yaml
system:
  backup:
    max_backups_per_target: 50  # Keep last 50 backups (default: 20)
    enable_retention: true
```

**Verification:**
```python
system_config = get_system_config()
max_backups = system_config["system"]["backup"]["max_backups_per_target"]
print(f"Max backups per target: {max_backups}")
```

---

## Example 9: Changing Default Paths

**Goal:** Use custom data/output directories.

**Steps:**

1. **Edit `training_config/system_config.yaml`:**
```yaml
system:
  paths:
    data_dir: "/custom/data/path"
    output_dir: "/custom/output/path"
```

2. **Or override via environment:**
```bash
export FOXML_DATA_DIR=/custom/data/path
```

**Usage:**
```python
system_config = get_system_config()
data_dir = system_config["system"]["paths"]["data_dir"]
print(f"Data directory: {data_dir}")
```

---

## Example 9: Adjusting Memory Limits

**Goal:** Configure memory usage for large datasets.

**Steps:**

1. **Edit `training_config/memory_config.yaml`:**
```yaml
memory:
  memory_cap_mb: 65536  # 64GB limit
  chunk_size: 5000  # Smaller chunks
  cleanup_aggressiveness: "aggressive"
```

**Usage:**
```python
from CONFIG.config_loader import get_memory_config

memory_config = get_memory_config()
print(f"Memory cap: {memory_config['memory']['memory_cap_mb']} MB")
```

---

## Example 10: Configuring Auto-Rerun

**Goal:** Adjust automatic re-evaluation after leakage fixes.

**Steps:**

1. **Edit `training_config/safety_config.yaml`:**
```yaml
leakage_detection:
  auto_rerun:
    enabled: true
    max_reruns: 5  # Increase from 3
    rerun_on_perfect_train_acc: true
    rerun_on_high_auc_only: false
```

**Usage:**
```python
safety_config = get_safety_config()
auto_rerun = safety_config["leakage_detection"]["auto_rerun"]
print(f"Auto-rerun enabled: {auto_rerun['enabled']}")
print(f"Max reruns: {auto_rerun['max_reruns']}")
```

---

## Example 11: Adjusting Thread Allocation

**Goal:** Configure thread usage per model family.

**Steps:**

1. **Edit `training_config/threading_config.yaml`:**
```yaml
threading:
  default_threads: 16
  per_family_policies:
    lightgbm:
      threads: 8
    xgboost:
      threads: 8
    neural_network:
      threads: 2
```

**Usage:**
```python
from CONFIG.config_loader import get_threading_config

threading_config = get_threading_config()
default_threads = threading_config["threading"]["default_threads"]
print(f"Default threads: {default_threads}")
```

---

## Example 12: Customizing Feature Selection

**Goal:** Adjust multi-model feature selection weights.

**Steps:**

1. **Edit `multi_model_feature_selection.yaml`:**
```yaml
model_families:
  lightgbm:
    enabled: true
    weight: 1.5  # Increase weight
  random_forest:
    enabled: true
    weight: 1.0
  neural_network:
    enabled: false  # Disable
```

**Usage:**
```python
import yaml

with open("CONFIG/multi_model_feature_selection.yaml") as f:
    config = yaml.safe_load(f)
    lightgbm_weight = config["model_families"]["lightgbm"]["weight"]
    print(f"LightGBM weight: {lightgbm_weight}")
```

---

## Configuration Decision Tree

**Which config should I edit?**

```
Need to exclude a leaky feature?
  → excluded_features.yaml

Adding a new feature?
  → feature_registry.yaml

Enabling/disabling targets?
  → target_configs.yaml

Adjusting leakage detection?
  → training_config/safety_config.yaml

Changing GPU/memory/threads?
  → training_config/gpu_config.yaml
  → training_config/memory_config.yaml
  → training_config/threading_config.yaml

Tuning model hyperparameters?
  → model_config/{model_name}.yaml

Configuring feature selection?
  → CONFIG/feature_selection/multi_model.yaml (NEW - preferred)
  → CONFIG/multi_model_feature_selection.yaml (LEGACY - deprecated)
  
  **Better:** Use experiment configs (see Example 13 below)

Using experiment configs?
  → CONFIG/experiments/*.yaml (NEW - recommended)

Changing system paths/backups?
  → training_config/system_config.yaml

Adjusting training pipeline?
  → training_config/pipeline_config.yaml
```

---

## Example 13: Using Experiment Configs (NEW - Recommended)

**Goal:** Use the new modular config system with experiment configs. This is the **preferred way** to configure the intelligent training pipeline.

**Steps:**

1. **Create experiment config** (`CONFIG/experiments/my_experiment.yaml`):
```yaml
experiment:
  name: my_experiment
  description: "Test run for fwd_ret_60m"

data:
  data_dir: data/data_labeled/interval=5m
  symbols: [AAPL, MSFT]
  interval: 5m
  max_samples_per_symbol: 3000

targets:
  primary: fwd_ret_60m

feature_selection:
  top_n: 30
  model_families: [lightgbm, xgboost]

training:
  model_families: [lightgbm, xgboost]
  cv_folds: 5
```

2. **Use in CLI:**
```bash
python TRAINING/train.py \
    --experiment-config my_experiment \
    --auto-targets \
    --top-n-targets 5 \
    --max-targets-to-evaluate 23
```

3. **Or use programmatically:**
```python
from CONFIG.config_builder import load_experiment_config
from TRAINING.orchestration.intelligent_trainer import IntelligentTrainer

# Load experiment config
exp_cfg = load_experiment_config("my_experiment")

# Create trainer
trainer = IntelligentTrainer(
    data_dir=exp_cfg.data_dir,
    symbols=exp_cfg.symbols,
    output_dir=Path("output"),
    experiment_config=exp_cfg
)

# Train (configs built automatically)
results = trainer.train_with_intelligence(
    auto_targets=True,
    top_n_targets=5,
    max_targets_to_evaluate=23  # Limit for faster testing
)
```

**Benefits:**
- All settings in one file
- Type-safe configs with validation
- No config "crossing" between modules
- Easy to version and share

---

## Example 14: Faster E2E Testing

**Goal:** Speed up end-to-end testing by limiting target evaluation.

**Steps:**

1. **Use `--max-targets-to-evaluate` option:**
```bash
python TRAINING/train.py \
    --data-dir data/data_labeled/interval=5m \
    --symbols AAPL MSFT \
    --auto-targets \
    --top-n-targets 3 \
    --max-targets-to-evaluate 23 \
    --min-cs 3 \
    --max-rows-per-symbol 5000
```

**What it does:**
- Evaluates only first 23 targets (instead of all 63)
- Still returns top 3 targets after ranking
- Significantly faster for E2E testing

**Use case:** Quick validation of pipeline functionality without waiting for all 63 targets.

---

## Related Documentation

- **[Modular Config System](MODULAR_CONFIG_SYSTEM.md)** - Complete guide to modular configs (includes `logging_config.yaml`)
- [Configuration System Overview](README.md) - Main configuration overview (includes `logging_config.yaml` documentation)
- [Config Loader API](CONFIG_LOADER_API.md) - Programmatic config loading (includes logging config utilities)
- [Ranking and Selection Consistency](../../01_tutorials/training/RANKING_SELECTION_CONSISTENCY.md) - Unified pipeline behavior guide
- [Feature & Target Configs](FEATURE_TARGET_CONFIGS.md) - Feature configuration guide
- [Training Pipeline Configs](TRAINING_PIPELINE_CONFIGS.md) - Training configuration guide
- [Intelligent Training Tutorial](../../01_tutorials/training/INTELLIGENT_TRAINING_TUTORIAL.md) - Complete pipeline guide
- [Safety & Leakage Configs](SAFETY_LEAKAGE_CONFIGS.md) - Leakage detection guide
- [Model Configuration](MODEL_CONFIGURATION.md) - Model hyperparameters guide

