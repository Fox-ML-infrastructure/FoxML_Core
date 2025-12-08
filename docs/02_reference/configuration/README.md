# Configuration System Overview

Centralized configuration management for FoxML Core training pipeline, model families, feature selection, and leakage detection.

> **📚 Complete Configuration Guides:**
> - [Feature & Target Configs](FEATURE_TARGET_CONFIGS.md) - Complete feature/target configuration guide
> - [Training Pipeline Configs](TRAINING_PIPELINE_CONFIGS.md) - System resources and training behavior
> - [Safety & Leakage Configs](SAFETY_LEAKAGE_CONFIGS.md) - Leakage detection and numerical stability
> - [Model Configuration](MODEL_CONFIGURATION.md) - Model hyperparameters and variants
> - [Usage Examples](USAGE_EXAMPLES.md) - 12 practical examples with code snippets
> - [Config Loader API](CONFIG_LOADER_API.md) - Programmatic config loading
> - [Config Overlays](CONFIG_OVERLAYS.md) - Overlay system for environment-specific configs
> - [Environment Variables](ENVIRONMENT_VARIABLES.md) - Environment-based configuration

## Overview

The configuration system provides a single source of truth for all training parameters, system settings, model hyperparameters, feature management, and safety controls. All configurations are stored as YAML files and loaded programmatically via `config_loader.py`.

## Complete Directory Structure

```
CONFIG/
├── config_loader.py              # Configuration loader implementation
├── README.md                      # This file
│
├── backups/                       # Automatic config backups (auto-fixer)
│   └── {target_name}/
│       └── {timestamp}/
│           ├── excluded_features.yaml
│           ├── feature_registry.yaml
│           └── manifest.json
│
├── model_config/                  # Model-specific hyperparameters (17 models)
│   ├── lightgbm.yaml
│   ├── xgboost.yaml
│   ├── mlp.yaml
│   ├── transformer.yaml
│   ├── lstm.yaml
│   ├── cnn1d.yaml
│   ├── ensemble.yaml
│   ├── multi_task.yaml
│   ├── vae.yaml
│   ├── gan.yaml
│   ├── gmm_regime.yaml
│   ├── ngboost.yaml
│   ├── quantile_lightgbm.yaml
│   ├── change_point.yaml
│   ├── ftrl_proximal.yaml
│   ├── reward_based.yaml
│   └── meta_learning.yaml
│
├── training_config/               # Training pipeline and system settings (11 configs)
│   ├── pipeline_config.yaml      # Main pipeline orchestration
│   ├── gpu_config.yaml            # GPU/CUDA configuration
│   ├── memory_config.yaml         # Memory management
│   ├── preprocessing_config.yaml # Data preprocessing
│   ├── threading_config.yaml     # Threading policies
│   ├── safety_config.yaml         # Numerical stability + leakage detection
│   ├── callbacks_config.yaml      # Training callbacks
│   ├── optimizer_config.yaml     # Optimizer defaults
│   ├── system_config.yaml         # System-level settings (paths, backups, env)
│   ├── family_config.yaml         # Model family policies
│   ├── sequential_config.yaml    # Sequential model settings
│   └── first_batch_specs.yaml    # First batch specifications
│
├── excluded_features.yaml         # Patterns for always-excluded features
├── feature_registry.yaml          # Feature metadata (lag_bars, allowed_horizons)
├── feature_target_schema.yaml     # Explicit schema (metadata/targets/features)
├── feature_groups.yaml            # Feature grouping definitions
├── feature_selection_config.yaml  # Feature selection settings
├── multi_model_feature_selection.yaml  # Multi-model consensus config
├── comprehensive_feature_ranking.yaml  # Comprehensive ranking config
├── fast_target_ranking.yaml       # Fast ranking config
└── target_configs.yaml            # Target definitions (63 targets)
```

## Configuration Categories

### 1. Root-Level Feature & Target Configs

#### `excluded_features.yaml`
**Purpose:** Defines patterns for features that are always excluded from training.

**Structure:**
- `always_exclude.regex_patterns` - Regex patterns matching leaky features
- `always_exclude.prefix_patterns` - Prefix patterns (e.g., `y_*`, `fwd_ret_*`)
- `always_exclude.exact_patterns` - Exact feature names to exclude

**Key Patterns:**
- `^y_*` - All target columns
- `^fwd_ret_*` - Forward returns (future information)
- `^barrier_*` - Barrier-related features
- `^mfe_*`, `^mdd_*` - Maximum favorable/adverse excursion

**Auto-Fixer Integration:** Auto-fixer can add patterns here when leakage is detected.

#### `feature_registry.yaml`
**Purpose:** Defines temporal metadata for features to prevent leakage.

**Structure:**
- `features.{feature_name}`:
  - `source` - Data source (price, volume, etc.)
  - `lag_bars` - Number of bars lagged (must be ≥ 0)
  - `allowed_horizons` - List of target horizons this feature is safe for
  - `rejected` - If true, feature is rejected (with reason)
  - `description` - Human-readable description

**Example:**
```yaml
features:
  ret_1:
    source: price
    lag_bars: 1
    allowed_horizons: [1, 2, 3, 5, 12, 24, 60]
    description: 1-bar lagged return
```

**Usage:** Features are filtered based on target horizon. Only features with `allowed_horizons` including the target's horizon are used.

#### `feature_target_schema.yaml`
**Purpose:** Explicitly defines which columns are metadata, targets, or features.

**Structure:**
- `metadata_columns` - Always excluded (symbol, ts, date, etc.)
- `target_patterns` - Regex patterns for target columns
- `feature_families` - Feature family definitions with mode-specific rules
  - `ranking_mode` - More permissive rules for target ranking
  - `training_mode` - Strict rules for actual training

**Modes:**
- **Ranking Mode:** Allows basic OHLCV/TA features even if in `always_exclude`
- **Training Mode:** Enforces all leakage filters strictly

#### `target_configs.yaml`
**Purpose:** Defines all available targets (63 total).

**Structure:**
- `targets.{target_name}`:
  - `target_column` - Column name in dataset
  - `description` - What the target predicts
  - `use_case` - Trading use case
  - `top_n` - Number of top features to select
  - `method` - Feature selection method
  - `enabled` - Enable/disable flag

**Categories:**
- Triple Barrier (peak/valley/first_touch)
- Swing High/Low
- MFE (Maximum Favorable Excursion)
- MDD (Maximum Drawdown)

#### `multi_model_feature_selection.yaml`
**Purpose:** Configures multi-model consensus for feature selection.

**Structure:**
- `model_families.{family}`:
  - `enabled` - Enable/disable this model family
  - `importance_method` - native/SHAP/permutation
  - `weight` - Weight in consensus aggregation
  - `config` - Model-specific hyperparameters

**Model Families:**
- Tree-based: LightGBM, XGBoost, Random Forest
- Neural: MLP, Transformer, LSTM, CNN1D
- Ensemble: Ensemble, MultiTask

#### `feature_selection_config.yaml`
**Purpose:** General feature selection settings.

**Settings:**
- Feature importance aggregation methods
- Selection criteria
- Minimum feature requirements

#### `feature_groups.yaml`
**Purpose:** Defines feature groups for organization and analysis.

#### `comprehensive_feature_ranking.yaml` & `fast_target_ranking.yaml`
**Purpose:** Alternative ranking configurations for different use cases.

### 2. Training Configuration (`training_config/`)

**See [Training Pipeline Configs](TRAINING_PIPELINE_CONFIGS.md) for complete guide.**

- `pipeline_config.yaml` - Main pipeline orchestration
- `gpu_config.yaml` - GPU/CUDA configuration
- `memory_config.yaml` - Memory management
- `preprocessing_config.yaml` - Data preprocessing
- `threading_config.yaml` - Thread allocation
- `callbacks_config.yaml` - Training callbacks
- `optimizer_config.yaml` - Optimizer defaults
- `system_config.yaml` - System-level settings (paths, backups, env)
- `family_config.yaml` - Model family policies
- `sequential_config.yaml` - Sequential model settings
- `first_batch_specs.yaml` - First batch specifications

### 3. Safety & Leakage Detection

**See [Safety & Leakage Configs](SAFETY_LEAKAGE_CONFIGS.md) for complete guide.**

- `safety_config.yaml` - Numerical stability + leakage detection
  - Feature clipping, target capping, gradient clipping
  - Pre-training leak scan thresholds
  - Auto-fixer thresholds and settings
  - Auto-rerun configuration

### 4. Model Configuration (`model_config/`)

**See [Model Configuration](MODEL_CONFIGURATION.md) for complete guide.**

Each model has its own YAML file with hyperparameters.

**Supported Models (17 total):**
- **Tree-based:** LightGBM, XGBoost
- **Neural Networks:** MLP, Transformer, LSTM, CNN1D
- **Ensemble:** Ensemble, MultiTask
- **Feature Engineering:** VAE, GAN, GMMRegime
- **Probabilistic:** NGBoost, QuantileLightGBM
- **Advanced:** ChangePoint, FTRL, RewardBased, MetaLearning

**Variants:** Each model supports `default`, `conservative`, and `aggressive` variants.

### 5. Backup System (`backups/`)

**Purpose:** Automatic backups of config files before auto-fixer modifications.

**Structure:**
```
CONFIG/backups/
└── {target_name}/              # Per-target organization
    └── {timestamp}/            # Timestamped snapshots (YYYYMMDD_HHMMSS_microseconds)
        ├── excluded_features.yaml
        ├── feature_registry.yaml
        └── manifest.json       # Backup metadata
```

**Manifest Contents:**
- `backup_version` - Schema version
- `source` - What created the backup (auto_fix_leakage)
- `target_name` - Target being evaluated
- `timestamp` - Backup timestamp
- `git_commit` - Git commit hash at backup time
- `backup_files` - List of backed-up files
- `excluded_features_path` - Original config path
- `feature_registry_path` - Original config path

**Retention Policy:**
- Keeps last N backups per target (configurable, default: 20)
- Automatic pruning of old backups
- Configurable via `system_config.yaml`

**Restoration:**
Use `LeakageAutoFixer.list_backups()` and `LeakageAutoFixer.restore_backup()` methods.

## Documentation Structure

This README provides an overview. For detailed guides, see:

- **[Feature & Target Configs](FEATURE_TARGET_CONFIGS.md)** - Complete guide to feature and target configuration
- **[Training Pipeline Configs](TRAINING_PIPELINE_CONFIGS.md)** - System resources and training behavior
- **[Safety & Leakage Configs](SAFETY_LEAKAGE_CONFIGS.md)** - Leakage detection and numerical stability
- **[Model Configuration](MODEL_CONFIGURATION.md)** - Model hyperparameters and variants
- **[Usage Examples](USAGE_EXAMPLES.md)** - Practical examples for common tasks

## Quick Start

### When to Use Each Config Type

#### 1. Feature & Target Management Configs

**Use these when:**
- Adding new features to your dataset
- Excluding leaky features
- Defining which features are safe for which target horizons
- Enabling/disabling targets

**Files:**
- `excluded_features.yaml` - Exclude leaky features by pattern
- `feature_registry.yaml` - Define feature temporal metadata
- `feature_target_schema.yaml` - Define explicit schema rules
- `target_configs.yaml` - Enable/disable and configure targets

**Example: Adding a New Feature**

1. Add to `feature_registry.yaml`:
```yaml
features:
  my_new_feature:
    source: price
    lag_bars: 5
    allowed_horizons: [12, 24, 60]  # Safe for 12, 24, 60-bar horizons
    description: "My custom feature"
```

2. If feature is leaky, add to `excluded_features.yaml`:
```yaml
always_exclude:
  exact_patterns:
    - my_leaky_feature
```

**Example: Enabling a Target**

Edit `target_configs.yaml`:
```yaml
targets:
  my_target:
    target_column: "y_will_peak_30m_0.8"
    enabled: true  # Change from false to true
    top_n: 50
    method: "mean"
```

#### 2. Training Pipeline Configs

**Use these when:**
- Adjusting system resources (GPU, memory, threads)
- Changing training behavior (timeouts, data limits)
- Configuring preprocessing steps
- Setting up callbacks and optimizers

**Files:**
- `training_config/pipeline_config.yaml` - Main pipeline settings
- `training_config/gpu_config.yaml` - GPU/CUDA settings
- `training_config/memory_config.yaml` - Memory management
- `training_config/threading_config.yaml` - Thread allocation
- `training_config/preprocessing_config.yaml` - Data preprocessing
- `training_config/callbacks_config.yaml` - Training callbacks
- `training_config/optimizer_config.yaml` - Optimizer defaults

**Example: Configuring GPU Usage**

Edit `training_config/gpu_config.yaml`:
```yaml
gpu:
  vram_cap_mb: 8192  # Limit to 8GB VRAM
  device_visibility: [0, 1]  # Use GPUs 0 and 1
  tensorflow:
    allow_growth: true
```

**Example: Adjusting Memory Limits**

Edit `training_config/memory_config.yaml`:
```yaml
memory:
  memory_cap_mb: 32768  # 32GB RAM limit
  chunk_size: 10000  # Process 10k rows at a time
  cleanup_aggressiveness: "moderate"
```

#### 3. Safety & Leakage Detection Configs

**Use these when:**
- Adjusting leakage detection sensitivity
- Configuring auto-fixer behavior
- Setting numerical stability guards
- Enabling/disabling auto-rerun

**Files:**
- `training_config/safety_config.yaml` - All safety and leakage settings

**Example: Adjusting Leakage Detection**

Edit `training_config/safety_config.yaml`:
```yaml
leakage_detection:
  # Make detection more sensitive
  auto_fix_thresholds:
    cv_score: 0.95  # Lower from 0.99 (detect at 95% instead of 99%)
    training_accuracy: 0.98  # Lower from 0.999
  
  # Make auto-fixer more aggressive
  auto_fix_min_confidence: 0.7  # Lower from 0.8
  auto_fix_max_features_per_run: 30  # Increase from 20
  
  # Enable auto-rerun
  auto_rerun:
    enabled: true
    max_reruns: 5  # Increase from 3
```

**Example: Adjusting Numerical Stability**

Edit `training_config/safety_config.yaml`:
```yaml
numerical_stability:
  feature_clipping:
    enabled: true
    min_value: -10.0
    max_value: 10.0
  
  target_capping:
    enabled: true
    max_abs_value: 1.0
```

#### 4. Model Hyperparameter Configs

**Use these when:**
- Tuning model hyperparameters
- Creating custom model variants
- Adjusting model-specific settings

**Files:**
- `model_config/{model_name}.yaml` - One file per model

**Example: Customizing LightGBM**

Edit `model_config/lightgbm.yaml`:
```yaml
default:
  n_estimators: 200  # Increase from 100
  learning_rate: 0.05  # Decrease from 0.1
  num_leaves: 63  # Increase from 31
  max_depth: 7  # Set explicit depth

# Or create a new variant
my_custom_variant:
  n_estimators: 500
  learning_rate: 0.01
  num_leaves: 127
  # ... other params
```

Then use it:
```python
config = load_model_config("lightgbm", variant="my_custom_variant")
```

#### 5. Feature Selection Configs

**Use these when:**
- Configuring multi-model feature selection
- Adjusting feature ranking methods
- Changing consensus aggregation

**Files:**
- `multi_model_feature_selection.yaml` - Multi-model consensus
- `feature_selection_config.yaml` - General selection settings
- `comprehensive_feature_ranking.yaml` - Comprehensive ranking
- `fast_target_ranking.yaml` - Fast ranking

**Example: Adjusting Multi-Model Selection**

Edit `multi_model_feature_selection.yaml`:
```yaml
model_families:
  lightgbm:
    enabled: true
    weight: 1.5  # Increase weight (more influence)
  
  random_forest:
    enabled: true
    weight: 1.0
  
  neural_network:
    enabled: false  # Disable this model family
```

#### 6. System Configs

**Use these when:**
- Changing default paths
- Adjusting backup retention
- Configuring environment variables
- Setting logging levels

**Files:**
- `training_config/system_config.yaml` - System-level settings

**Example: Customizing Paths**

Edit `training_config/system_config.yaml`:
```yaml
system:
  paths:
    data_dir: "/custom/path/to/data"
    output_dir: "/custom/path/to/output"
    config_backup_dir: "/custom/path/to/backups"
  
  backup:
    max_backups_per_target: 50  # Keep more backups
    enable_retention: true
```

### Loading Configurations

#### Loading Model Configs

```python
from CONFIG.config_loader import load_model_config

# Load with default variant
lightgbm_config = load_model_config("lightgbm")

# Load with specific variant
lightgbm_aggressive = load_model_config("lightgbm", variant="aggressive")

# Load with overrides
lightgbm_custom = load_model_config(
    "lightgbm",
    variant="conservative",
    overrides={"n_estimators": 500, "learning_rate": 0.01}
)
```

#### Loading Training Configs

```python
from CONFIG.config_loader import (
    get_pipeline_config,
    get_gpu_config,
    get_safety_config,
    get_system_config,
    get_memory_config,
    get_threading_config,
    get_cfg
)

# Load full configs
pipeline = get_pipeline_config()
gpu = get_gpu_config()
safety = get_safety_config()
system = get_system_config()

# Access nested values with dot notation
timeout = get_cfg("pipeline.isolation_timeout_seconds", default=7200)
vram_cap = get_cfg("gpu.vram_cap_mb", default=4096, config_name="gpu_config")
max_backups = get_cfg("system.backup.max_backups_per_target", default=20)
leakage_threshold = get_cfg(
    "leakage_detection.auto_fix_thresholds.cv_score",
    default=0.99,
    config_name="safety_config"
)
```

#### Loading Feature/Target Configs

```python
import yaml
from pathlib import Path

# Load excluded features
with open("CONFIG/excluded_features.yaml") as f:
    excluded = yaml.safe_load(f)
    patterns = excluded["always_exclude"]["regex_patterns"]

# Load feature registry
with open("CONFIG/feature_registry.yaml") as f:
    registry = yaml.safe_load(f)
    features = registry["features"]
    # Check if feature is allowed for horizon
    feature_allowed = 12 in features["ret_1"]["allowed_horizons"]

# Load target configs
with open("CONFIG/target_configs.yaml") as f:
    targets = yaml.safe_load(f)
    enabled_targets = {
        name: cfg for name, cfg in targets["targets"].items()
        if cfg.get("enabled", False)
    }

# Load feature/target schema
with open("CONFIG/feature_target_schema.yaml") as f:
    schema = yaml.safe_load(f)
    metadata_cols = schema["metadata_columns"]
    target_patterns = schema["target_patterns"]
```

### Common Configuration Scenarios

#### Scenario 1: Adding a New Feature

**Goal:** Add a new feature and make it available for training.

**Steps:**
1. Add feature to `feature_registry.yaml`:
```yaml
features:
  my_new_feature:
    source: price
    lag_bars: 3
    allowed_horizons: [5, 12, 24]
    description: "3-bar momentum"
```

2. Ensure it's not in `excluded_features.yaml` (should not match any patterns)

3. Feature will automatically be available for targets with horizons 5, 12, or 24 bars

#### Scenario 2: Excluding a Leaky Feature

**Goal:** Permanently exclude a feature that causes leakage.

**Steps:**
1. Add to `excluded_features.yaml`:
```yaml
always_exclude:
  exact_patterns:
    - my_leaky_feature
```

2. Or if it matches a pattern, add regex:
```yaml
always_exclude:
  regex_patterns:
    - "^future_"
```

#### Scenario 3: Adjusting Leakage Detection Sensitivity

**Goal:** Make leakage detection more or less sensitive.

**Steps:**
1. Edit `training_config/safety_config.yaml`:
```yaml
leakage_detection:
  auto_fix_thresholds:
    cv_score: 0.95  # Lower = more sensitive (detects at 95% instead of 99%)
    training_accuracy: 0.98  # Lower = more sensitive
```

2. Adjust auto-fixer confidence:
```yaml
  auto_fix_min_confidence: 0.7  # Lower = more aggressive (fixes with 70% confidence)
```

#### Scenario 4: Configuring for Multi-GPU Setup

**Goal:** Use multiple GPUs for training.

**Steps:**
1. Edit `training_config/gpu_config.yaml`:
```yaml
gpu:
  device_visibility: [0, 1, 2, 3]  # Use all 4 GPUs
  vram_cap_mb: 8192  # Per-GPU limit
```

2. Set environment variable:
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

#### Scenario 5: Enabling More Targets

**Goal:** Enable additional targets for training.

**Steps:**
1. Edit `target_configs.yaml`:
```yaml
targets:
  swing_high_15m:
    enabled: true  # Change from false
    top_n: 50
    method: "mean"
```

2. Targets will be automatically discovered and ranked

#### Scenario 6: Customizing Model Hyperparameters

**Goal:** Create a custom model variant for specific use case.

**Steps:**
1. Edit `model_config/lightgbm.yaml`:
```yaml
my_custom_variant:
  n_estimators: 1000
  learning_rate: 0.01
  num_leaves: 255
  max_depth: 10
  # ... other params
```

2. Use in code:
```python
config = load_model_config("lightgbm", variant="my_custom_variant")
```

#### Scenario 7: Adjusting Backup Retention

**Goal:** Keep more or fewer config backups.

**Steps:**
1. Edit `training_config/system_config.yaml`:
```yaml
system:
  backup:
    max_backups_per_target: 50  # Keep last 50 backups (default: 20)
    enable_retention: true
```

#### Scenario 8: Changing Default Data Paths

**Goal:** Use custom data/output directories.

**Steps:**
1. Edit `training_config/system_config.yaml`:
```yaml
system:
  paths:
    data_dir: "/custom/data/path"
    output_dir: "/custom/output/path"
```

2. Or override via environment:
```bash
export FOXML_DATA_DIR=/custom/data/path
```

## Configuration Hierarchy

1. **Model Configs** (`model_config/`) - Hyperparameters for specific model families
2. **Training Configs** (`training_config/`) - Pipeline, system, and resource settings
3. **Feature Configs** (root) - Feature filtering, registry, and schema
4. **Target Configs** (root) - Target definitions and settings
5. **Backup System** (`backups/`) - Automatic config backups

## Environment Variable Overrides

Most configuration values can be overridden via environment variables:

```bash
# Override GPU device
export CUDA_VISIBLE_DEVICES=0

# Override thread count
export OMP_NUM_THREADS=8

# Override timeout
export TRAINER_ISOLATION_TIMEOUT=10800

# Override data directory
export FOXML_DATA_DIR=/path/to/data
```

## Configuration Workflow

### 1. Feature Filtering Flow

```
Dataset Columns
    ↓
[feature_target_schema.yaml] → Classify as metadata/target/feature
    ↓
[excluded_features.yaml] → Remove always-excluded patterns
    ↓
[feature_registry.yaml] → Filter by allowed_horizons for target
    ↓
Pre-training leak scan (safety_config.yaml thresholds)
    ↓
Final Feature Set
```

### 2. Target Ranking Flow

```
[target_configs.yaml] → Discover available targets
    ↓
For each target:
    ↓
Filter features (using schema + excluded + registry)
    ↓
Pre-training leak scan
    ↓
Train models (multi_model_feature_selection.yaml)
    ↓
Detect leakage (safety_config.yaml thresholds)
    ↓
Auto-fix if needed (creates backup in backups/)
    ↓
Auto-rerun if enabled (safety_config.yaml)
    ↓
Rank targets by predictability score
```

### 3. Auto-Fixer Flow

```
Leakage Detected
    ↓
Create backup (backups/{target}/{timestamp}/)
    ↓
Detect leaking features
    ↓
Update excluded_features.yaml or feature_registry.yaml
    ↓
Reload configs
    ↓
Re-evaluate target (if auto-rerun enabled)
```

## Best Practices

1. **Never hardcode values** - Always load from config files
2. **Use defaults** - Provide sensible fallbacks when config unavailable
3. **Validate inputs** - Check config values before use
4. **Document changes** - Update configs with clear comments
5. **Test variants** - Verify all config variants work correctly
6. **Backup before manual edits** - The auto-fixer creates backups automatically, but manual edits should be backed up too
7. **Use feature registry** - Always specify `lag_bars` and `allowed_horizons` for new features
8. **Review backups** - Check `CONFIG/backups/` to understand what auto-fixer changed

## Migration Status

✅ **Complete** - All hardcoded configurations have been migrated to YAML files. The system maintains backward compatibility with hardcoded defaults during the transition period.

## Support

For configuration questions or issues, refer to:
- `config_loader.py` - Implementation details
- Individual config files - Inline documentation
- Training pipeline code - Usage examples
- [Config Loader API](../docs/02_reference/configuration/CONFIG_LOADER_API.md) - Complete API reference

## Related Documentation

- [Config Basics](../docs/01_tutorials/configuration/CONFIG_BASICS.md) - Configuration fundamentals tutorial
- [Config Examples](../docs/01_tutorials/configuration/CONFIG_EXAMPLES.md) - Example configurations
- [Advanced Config](../docs/01_tutorials/configuration/ADVANCED_CONFIG.md) - Advanced configuration guide
- [Config Loader API](../docs/02_reference/configuration/CONFIG_LOADER_API.md) - Complete API reference
- [Config Schema](../docs/02_reference/api/CONFIG_SCHEMA.md) - Configuration schema documentation
- [Environment Variables](../docs/02_reference/configuration/ENVIRONMENT_VARIABLES.md) - Environment variable overrides
- [Model Config Reference](../docs/02_reference/models/MODEL_CONFIG_REFERENCE.md) - Model-specific configurations
- [Intelligence Layer Overview](../docs/03_technical/research/INTELLIGENCE_LAYER.md) - How configs are used in intelligent training
- [Leakage Analysis](../docs/03_technical/research/LEAKAGE_ANALYSIS.md) - Leakage detection configuration details
