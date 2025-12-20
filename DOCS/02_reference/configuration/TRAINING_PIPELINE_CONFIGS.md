# Training Pipeline Configuration Guide

Complete guide to configuring the training pipeline, system resources, and model training behavior.

## Overview

Training pipeline configs control system resources (GPU, memory, threads), training behavior (timeouts, data limits), preprocessing, callbacks, and optimizers.

## Configuration Files

### `training_config/pipeline_config.yaml`

**Purpose:** Main training pipeline orchestration.

**When to use:** When adjusting timeouts, data limits, or pipeline behavior.

**Key Settings:**
- `isolation_timeout_seconds` - Maximum time per training job
- `max_rows_per_symbol` - Data loading limits
- `deterministic` - Reproducibility settings
- Sequential model configuration

**Example: Adjusting Timeout**

```yaml
pipeline:
  isolation_timeout_seconds: 10800  # 3 hours (default: 7200 = 2 hours)
```

**Example: Limiting Data Size**

```yaml
pipeline:
  max_rows_per_symbol: 10000  # Limit to 10k rows per symbol
```

---

### `training_config/gpu_config.yaml`

**Purpose:** GPU device management, CUDA settings, and GPU acceleration for target ranking and feature selection.

**When to use:** When configuring GPU usage, VRAM limits, multi-GPU setups, or GPU acceleration for ranking/selection.

**Key Settings:**
- `cuda_visible_devices` - GPU IDs to use (comma-separated)
- `vram.caps` - Per-family VRAM limits (in MB)
- TensorFlow/PyTorch GPU options
- **LightGBM/XGBoost/CatBoost GPU settings** (NEW 2025-12-12) - GPU acceleration for target ranking and feature selection

**Example: Enable GPU for Target Ranking and Feature Selection**

```yaml
gpu:
  cuda_visible_devices: "0"  # Use GPU 0
  
  # LightGBM GPU Settings
  lightgbm:
    device: "cuda"  # "cuda" (CUDA) or "gpu" (OpenCL) or "cpu"
    gpu_device_id: 0
    test_enabled: true  # Test GPU before use
    try_cuda_first: true  # Try CUDA before OpenCL
  
  # XGBoost GPU Settings
  xgboost:
    device: "cuda"  # "cuda" or "cpu"
    tree_method: "hist"  # "hist" (new API) or "gpu_hist" (legacy)
    gpu_id: 0
    test_enabled: true
  
  # CatBoost GPU Settings
  catboost:
    task_type: "GPU"  # "GPU" or "CPU"
    devices: "0"  # GPU device IDs
    test_enabled: true
  
  # TensorFlow GPU Settings
  tensorflow:
    allocator: "cuda_malloc_async"
    force_gpu_allow_growth: true
  
  # VRAM Management
  vram:
    caps:
      MLP: 4096
      default: 4096
```

**Example: Disable GPU Test (Faster Startup)**

```yaml
gpu:
  lightgbm:
    test_enabled: false  # Skip GPU test, use config directly
  xgboost:
    test_enabled: false
  catboost:
    test_enabled: false
```

**Environment Variable Override:**
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

**NEW (2025-12-12)**: GPU acceleration is now automatically enabled for target ranking and feature selection when GPU is available. All settings are config-driven from this file (SST compliance).

**CatBoost Thread Limiting (2025-12-12):**
```yaml
gpu:
  catboost:
    thread_count: 8  # Limit CPU threads during GPU training (prevents CPU bottleneck)
    task_type: "GPU"  # Required for GPU usage
    devices: "0"  # GPU device IDs
```

**Note:** For GPU training, set `cv_n_jobs=1` in `intelligent_training_config.yaml` to avoid outer parallelism conflicts (see `intelligent_training_config.yaml` section below).

---

### `training_config/memory_config.yaml`

**Purpose:** Memory thresholds and cleanup policies.

**When to use:** When adjusting memory limits or cleanup behavior.

**Key Settings:**
- `memory_cap_mb` - Maximum memory usage
- `chunk_size` - Data chunking for large datasets
- `cleanup_aggressiveness` - How aggressively to free memory

**Example: Adjusting Memory Limits**

```yaml
memory:
  memory_cap_mb: 32768  # 32GB RAM limit
  chunk_size: 10000  # Process 10k rows at a time
  cleanup_aggressiveness: "moderate"  # moderate/aggressive/conservative
```

**Example: More Aggressive Cleanup**

```yaml
memory:
  cleanup_aggressiveness: "aggressive"  # Free memory more aggressively
```

---

### `pipeline/threading.yaml`

**Purpose:** Thread allocation and OpenMP/MKL policies. **Shared by feature selection, target ranking, and model training.**

**When to use:** When adjusting thread counts or thread allocation per model family.

**Key Settings:**
- `defaults.default_threads` - Default thread count (null = auto-detect)
- `planning.reserve_threads` - Reserve threads for system
- `family_allocation` - Per-family thread allocation (optional)
- OpenMP/MKL thread planning

**Note:** The threading utilities (`TRAINING/common/threads.py`) used by feature selection and target ranking automatically read from this config. Models use `plan_for_family()` to determine optimal OMP/MKL thread allocation and `thread_guard()` for GPU-aware thread limiting.

**Example: Setting Default Threads**

```yaml
threading:
  default_threads: 8  # Use 8 threads by default
```

**Example: Per-Family Thread Allocation**

```yaml
threading:
  per_family_policies:
    lightgbm:
      threads: 4
    xgboost:
      threads: 4
    neural_network:
      threads: 2
```

**Environment Variable Override:**
```bash
export OMP_NUM_THREADS=8
```

---

### `training_config/preprocessing_config.yaml`

**Purpose:** Data preprocessing settings.

**When to use:** When adjusting normalization, missing value handling, or feature scaling.

**Key Settings:**
- Normalization methods
- Missing value handling
- Feature scaling
- Data validation

**Example: Adjusting Normalization**

```yaml
preprocessing:
  normalization:
    method: "standard"  # standard/minmax/robust
    enabled: true
```

---

### `training_config/callbacks_config.yaml`

**Purpose:** Training callback configuration.

**When to use:** When adjusting early stopping, learning rate scheduling, or checkpointing.

**Key Settings:**
- Early stopping criteria
- Learning rate scheduling
- Model checkpointing
- Progress monitoring

**Example: Adjusting Early Stopping**

```yaml
callbacks:
  early_stopping:
    patience: 10  # Stop after 10 epochs without improvement
    min_delta: 0.001
```

---

### `training_config/optimizer_config.yaml`

**Purpose:** Optimizer default settings.

**When to use:** When adjusting optimizer parameters or learning rate schedules.

**Key Settings:**
- Default optimizer parameters
- Learning rate schedules
- Weight decay
- Momentum settings

**Example: Adjusting Learning Rate**

```yaml
optimizer:
  learning_rate: 0.001  # Default learning rate
  schedule:
    type: "exponential_decay"
    decay_rate: 0.95
```

---

### `training_config/system_config.yaml`

**Purpose:** System-level settings (paths, backups, environment, logging).

**When to use:** When changing default paths, backup retention, or environment settings.

**Key Settings:**

**Paths:**
```yaml
system:
  paths:
    data_dir: "data/data_labeled/interval=5m"
    output_dir: null  # null = auto-generated
    config_backup_dir: null  # null = CONFIG/backups/ (legacy) or RESULTS/{cohort_id}/{run_name}/backups/ (when output_dir provided)
```

**Backup System:**
```yaml
system:
  backup:
    max_backups_per_target: 20  # Keep last 20 backups
    enable_retention: true
```

**Environment:**
```yaml
system:
  environment:
    pythonhashseed: "42"
    joblib_start_method: "spawn"
```

**Logging:**
```yaml
system:
  logging:
    level: "INFO"  # DEBUG, INFO, WARNING, ERROR
```

**Example: Customizing Paths**

```yaml
system:
  paths:
    data_dir: "/custom/data/path"
    output_dir: "/custom/output/path"
```

**Example: Adjusting Backup Retention**

```yaml
system:
  backup:
    max_backups_per_target: 50  # Keep more backups
    enable_retention: true
```

---

### `training_config/family_config.yaml`

**Purpose:** Model family policies and defaults.

**When to use:** When adjusting family-specific defaults or enabling/disabling families.

**Key Settings:**
- Family-specific defaults
- Enabled/disabled families
- Family-specific overrides

---

### `training_config/sequential_config.yaml`

**Purpose:** Sequential model (LSTM, Transformer) settings.

**When to use:** When adjusting sequence length, batch size, or attention mechanisms.

**Key Settings:**
- Sequence length
- Batch size
- Padding strategies
- Attention mechanisms

**Example: Adjusting Sequence Length**

```yaml
sequential:
  sequence_length: 60  # Use 60-bar sequences
  batch_size: 32
```

---

### `training_config/first_batch_specs.yaml`

**Purpose:** First batch specifications for training.

**When to use:** When adjusting first batch behavior.

---

### `training_config/decision_policies.yaml` ⭐ NEW (2025-12-12)

**Purpose:** Decision policy thresholds for automated decision-making system.

**When to use:** When adjusting thresholds for feature instability, route instability, feature explosion decline, or class balance drift detection.

**Key Settings:**
- Feature instability thresholds (jaccard similarity)
- Route instability thresholds (entropy, change frequency)
- Feature explosion decline thresholds (AUC decline, feature increase)
- Class balance drift thresholds (pos_rate drift)

**Example: Adjusting Feature Instability Thresholds**

```yaml
feature_instability:
  jaccard_threshold: 0.5  # Trigger if jaccard_topK < 0.5
  jaccard_collapse_ratio: 0.8  # Trigger if jaccard drops by 20%
  min_runs: 3  # Minimum runs needed to evaluate
  min_recent_runs: 2  # Minimum recent runs for comparison
```

**Example: Adjusting Route Instability Thresholds**

```yaml
route_instability:
  entropy_threshold: 1.5  # Trigger if route_entropy > 1.5
  change_threshold: 3  # Trigger if 3+ route changes in last N runs
  change_window: 5  # Number of recent runs to check
  min_runs: 3
```

**How It Works:**
- Policies are evaluated after each run using historical cohort data
- When thresholds are exceeded, actions are triggered (e.g., `freeze_features`, `tighten_routing`)
- All thresholds are config-driven (SST: Single Source of Truth)
- See [Decision Engine Documentation](../../../TRAINING/decisioning/README.md) for details

---

### `training_config/stability_config.yaml` ⭐ NEW (2025-12-12)

**Purpose:** Stability analysis thresholds for feature importance comparison.

**When to use:** When adjusting thresholds for detecting importance differences between full and safe feature sets.

**Key Settings:**
- Absolute difference threshold
- Relative difference threshold
- Minimum importance threshold
- Top N features to analyze

**Example: Adjusting Importance Difference Thresholds**

```yaml
importance_diff:
  diff_threshold: 0.1  # Minimum absolute difference to flag
  relative_diff_threshold: 0.5  # Minimum relative difference (50%) to flag
  min_importance_full: 0.01  # Minimum importance in full set to consider
  top_n: 10  # Number of top features to compare
```

**How It Works:**
- Compares feature importance between models trained with all features vs safe features only
- Flags features with high importance in full set but low in safe set (potential leaks)
- All thresholds are config-driven (SST: Single Source of Truth)
- Used by `ImportanceDiffDetector` in stability analysis

---

### `training_config/intelligent_training_config.yaml` ⭐ NEW (2025-12-12)

**Purpose:** Main configuration for the intelligent training pipeline, including cross-validation settings and CatBoost-specific parameters.

**When to use:** When adjusting CV folds, parallel jobs, or CatBoost training parameters.

**Key Settings:**
- `training.cv_folds` - Number of cross-validation folds (default: 3)
- `training.cv_n_jobs` - Parallel jobs for CV (1 = sequential, -1 = all cores). **Set to 1 for GPU training** to avoid outer parallelism conflicts
- `training.catboost.metric_period` - CatBoost metric calculation frequency (default: 50). Reduces evaluation overhead

**Example: Configuring CV Settings**
```yaml
training:
  cv_folds: 3  # Number of CV folds
  cv_n_jobs: 1  # Sequential CV (required for GPU training to avoid CPU bottleneck)
  
  catboost:
    metric_period: 50  # Calculate metrics every 50 trees (reduces evaluation overhead)
```

**Example: CPU Training (Parallel CV)**
```yaml
training:
  cv_folds: 5  # More folds for better validation
  cv_n_jobs: -1  # Use all CPU cores for parallel CV
```

**SST Compliance:**
- All CV and CatBoost settings are pulled from this config file via `get_cfg()`
- Fallback chain: `intelligent_training_config.yaml` → model config → defaults
- No hardcoded values in code (except as final fallback)

**How It Works:**
- `cv_folds` and `cv_n_jobs` control cross-validation behavior in target ranking and feature selection
- `metric_period` is automatically injected into CatBoost params if not specified in model config
- For GPU training, `cv_n_jobs=1` prevents outer parallelism from causing CPU bottlenecks

---

### `training_config/safety_config.yaml` (Updated 2025-12-12)

**Purpose:** Safety and temporal configuration, including default purge/embargo settings.

**When to use:** When adjusting safety thresholds, temporal safety, or default purge minutes.

**Key Settings:**
- Feature clipping thresholds
- Target capping settings
- Numerical stability bounds
- Gradient clipping
- **Temporal safety** (NEW):
  - `temporal.default_purge_minutes: 85.0` - Default purge if horizon cannot be determined (SST)

**Example: Adjusting Default Purge Minutes**

```yaml
safety:
  temporal:
    default_purge_minutes: 85.0  # Default purge if horizon cannot be determined
    purge_include_feature_lookback: true  # Include feature lookback in purge calculation
```

**How It Works:**
- `default_purge_minutes` is used when horizon cannot be determined from data
- All temporal safety parameters are config-driven (SST: Single Source of Truth)
- Used by `derive_purge_embargo()` in `resolved_config.py`

---

## Common Scenarios

### Scenario 1: Configuring for Multi-GPU Setup

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

### Scenario 2: Adjusting Memory for Large Datasets

1. **Edit `training_config/memory_config.yaml`:**
```yaml
memory:
  memory_cap_mb: 65536  # 64GB limit
  chunk_size: 5000  # Smaller chunks for large datasets
  cleanup_aggressiveness: "aggressive"
```

### Scenario 3: Changing Default Paths

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

### Scenario 4: Adjusting Thread Allocation

1. **Edit `CONFIG/pipeline/threading.yaml`:**
```yaml
threading:
  defaults:
    default_threads: 16  # Override auto-detection
  family_allocation:
    QuantileLightGBM:
      thread_clamp: [4, 8]  # Clamp threads to 4-8 range
    lightgbm:
      threads: 8
```

---

## Best Practices

1. **Set reasonable limits** - Don't set memory/GPU limits too high (causes OOM)
2. **Use environment variables** - Override paths via env vars for different environments
3. **Monitor resource usage** - Adjust limits based on actual usage
4. **Test timeout settings** - Ensure timeouts are long enough for your datasets
5. **Configure backups** - Set appropriate `max_backups_per_target` based on disk space

---

## Related Documentation

- [Configuration System Overview](README.md) - Main configuration overview
- [Feature & Target Configs](FEATURE_TARGET_CONFIGS.md) - Feature configuration
- [Safety & Leakage Configs](SAFETY_LEAKAGE_CONFIGS.md) - Leakage detection settings
- [Model Configuration](MODEL_CONFIGURATION.md) - Model hyperparameters
- [Usage Examples](USAGE_EXAMPLES.md) - Practical configuration examples

