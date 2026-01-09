# Quick Start

Installation and basic usage.

> **📊 Testing & Development:** All testing, validation, and development work is performed using **5-minute interval data**. The software supports various data intervals, but all tests, benchmarks, and development workflows use 5-minute bars as the standard reference.

## Prerequisites

**⚠️ Important**: FoxML Core requires significant computational resources. This is **not suitable for laptop deployment** for production workloads.

### Minimum Requirements (Development/Testing Only)

- **OS**: Linux (Arch Linux, Ubuntu 22.04+, or similar)
  - Tested on: Arch Linux (kernel 6.17+)
- **Python**: 3.10 (as specified in `environment.yml`)
- **16 GB RAM minimum** (32 GB recommended for development)
- GPU optional (CUDA 12.9 if using GPU) - **Recommended for target ranking and feature selection** (10-50x speedup)

**Note**: Minimum configuration is suitable for small-scale testing and development only.

### Production Requirements

- **OS**: Linux (Arch Linux, Ubuntu 22.04+, or similar)
  - Tested on: Arch Linux (kernel 6.17+)
- **Python**: 3.10 (as specified in `environment.yml`)
- **64 GB RAM minimum** (128 GB+ recommended)
- Multi-core processor (16+ cores)
- GPU recommended (11 GB+ VRAM, CUDA 12.9 for optimal performance)

**Verified Stable Range**: Up to 100 GB RAM (tested and verified)  
**Targeted Capacity**: 1 TB+ RAM (enterprise deployment)

See [System Requirements](SYSTEM_REQUIREMENTS.md) for complete specifications.

## Installation

### Quick Install (Recommended)

Use the automated install script for the fastest setup:

```bash
git clone <repository-url>
cd trader
bash bin/install.sh
conda activate <env-name>  # Name from environment.yml (typically foxml_env)
```

The install script will:
- Create the conda environment from `environment.yml`
- Display the correct environment name to activate
- Handle existing environments gracefully

### Verify Installation

Test that everything is set up correctly:

```bash
bash bin/test_install.sh
```

This script verifies:
- Python version
- Critical library imports (numpy, pandas, polars, sklearn)
- Model libraries (LightGBM)
- Pipeline imports
- Config system

### Manual Install (Alternative)

If you prefer manual setup or the script fails:

```bash
git clone <repository-url>
cd trader

# Create environment
conda env create -f environment.yml
conda activate <env-name>  # Check environment.yml for the name (typically foxml_env)

# Or use pip (not recommended - use environment.yml with conda)
pip install -r requirements.txt
```

**Note**: The environment name is specified in `environment.yml` (currently `foxml_env`). Always use the name from that file when activating.

## Data Pipeline

```bash
# Verify data exists
ls data/data_labeled/interval=5m/

# Run feature engineering
python DATA_PROCESSING/features/comprehensive_builder.py \
    --config config/features.yaml \
    --output-dir DATA_PROCESSING/data/processed/

# Generate targets
python DATA_PROCESSING/pipeline/barrier_pipeline.py \
    --input-dir data/data_labeled/interval=5m/ \
    --output-dir DATA_PROCESSING/data/labeled/
```

## GPU Acceleration (Optional but Recommended)

GPU acceleration is automatically enabled for target ranking and feature selection when available:

- **Automatic**: System detects and uses GPU automatically
- **Configuration**: All settings in `CONFIG/training_config/gpu_config.yaml`
- **Performance**: 10-50x speedup on large datasets (>100k samples)
- **Supported**: LightGBM, XGBoost, CatBoost

See [GPU Setup Guide](../01_tutorials/setup/GPU_SETUP.md) for detailed configuration.

## Model Training

### Intelligent Training Pipeline (Recommended)

The intelligent training pipeline automates the complete workflow: target ranking → feature selection → training plan generation → model training:

```bash
# One-command end-to-end pipeline
python -m TRAINING.orchestration.intelligent_trainer \
    --data-dir "data/data_labeled/interval=5m" \
    --symbols AAPL MSFT GOOGL \
    --auto-targets \
    --auto-features \
    --output-dir "my_training_run"
```

**What it does:**
1. Ranks all targets → selects top N
2. Selects features per target
3. Generates routing plan → training plan
4. Trains models using plan (2-stage: CPU → GPU)

**For testing - auto-detects test config when output-dir contains "test":**
```bash
python -m TRAINING.orchestration.intelligent_trainer \
    --data-dir "data/data_labeled/interval=5m" \
    --symbols AAPL MSFT GOOGL TSLA NVDA \
    --output-dir "test_e2e_run" \
    --auto-targets \
    --auto-features
```

**Configuration:**
- All settings come from `CONFIG/training_config/pipeline_config.yaml`
- Test config auto-detected when output-dir contains "test"
- Training plan automatically generated and used for filtering
- See [Intelligent Training Tutorial](../01_tutorials/training/INTELLIGENT_TRAINING_TUTORIAL.md) for details
- See [Training Routing Guide](../02_reference/training_routing/README.md) for routing system details

### Manual Training (Programmatic)

```python
from TRAINING.model_fun import LightGBMTrainer
from CONFIG.config_loader import load_model_config
import polars as pl
import numpy as np

# Load data
df = pl.read_parquet("DATA_PROCESSING/data/labeled/AAPL_labeled.parquet")

# Prepare features and targets
feature_cols = [col for col in df.columns 
                if not col.startswith('y_') and col not in ['ts', 'symbol']]
X = df[feature_cols].to_pandas()
y = df['y_will_peak'].to_pandas()

# Load config (all hyperparameters load from config - Single Source of Truth)
config = load_model_config("lightgbm", variant="conservative")

# Train (fully reproducible: same config → same results)
trainer = LightGBMTrainer(config)
trainer.train(X, y)

# Feature importance (returns numpy array)
importance = trainer.get_feature_importance()
if importance is not None:
    # Get top 20 features
    top_indices = np.argsort(importance)[-20:][::-1]
    print("Top 20 features:")
    for idx in top_indices:
        print(f"  Feature {idx}: {importance[idx]:.4f}")
```

## Troubleshooting

**Import errors**: Ensure conda environment is activated.

**GPU not detected**: Install CUDA toolkit and PyTorch with CUDA support.

**Out of memory**: Use streaming builder or reduce batch size.

## Related Documentation

- [Getting Started](GETTING_STARTED.md)
- [Architecture Overview](ARCHITECTURE_OVERVIEW.md)
- [Documentation Index](../INDEX.md)
