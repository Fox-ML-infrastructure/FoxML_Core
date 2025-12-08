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

**Purpose:** GPU device management and CUDA settings.

**When to use:** When configuring GPU usage, VRAM limits, or multi-GPU setups.

**Key Settings:**
- `vram_cap_mb` - Maximum VRAM usage per GPU
- `device_visibility` - Which GPUs to use
- TensorFlow/PyTorch GPU options
- CUDA device selection

**Example: Single GPU Setup**

```yaml
gpu:
  vram_cap_mb: 8192  # 8GB VRAM limit
  device_visibility: [0]  # Use GPU 0 only
  tensorflow:
    allow_growth: true
```

**Example: Multi-GPU Setup**

```yaml
gpu:
  device_visibility: [0, 1, 2, 3]  # Use all 4 GPUs
  vram_cap_mb: 8192  # Per-GPU limit
```

**Environment Variable Override:**
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

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

### `training_config/threading_config.yaml`

**Purpose:** Thread allocation and OpenMP/MKL policies.

**When to use:** When adjusting thread counts or thread allocation per model family.

**Key Settings:**
- `default_threads` - Default thread count
- `per_family_policies` - Thread allocation per model family
- OpenMP/MKL thread planning

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
    config_backup_dir: null  # null = CONFIG/backups/
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

1. **Edit `training_config/threading_config.yaml`:**
```yaml
threading:
  default_threads: 16  # Increase from default
  per_family_policies:
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

