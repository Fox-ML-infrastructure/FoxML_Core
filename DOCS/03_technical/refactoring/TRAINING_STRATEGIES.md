# Training Strategies Module

This document describes the `TRAINING/training_strategies/` module structure, split from the original 2,523-line `train_with_strategies.py` file for better maintainability.

## Purpose

Provides multiple training strategies (single-task, multi-task, cascade) with support for all 20 model families, GPU acceleration, memory management, and cross-sectional training.

## Module Structure

### Core Modules

- **`setup.py`** (156 lines): Bootstrap and setup code
  - Path configuration and environment setup
  - CUDA library path configuration
  - Joblib/Loky cleanup setup
  - Family classifications (TF_FAMS, TORCH_FAMS, CPU_FAMS)
  - Model family capabilities (FAMILY_CAPS)

- **`family_runners.py`** (427 lines): Family execution functions
  - `_run_family_inproc`: Run family trainer in main process with unified threading
  - `_run_family_isolated`: Run family trainer in isolated process (for stability)
  - Handles thread pool management, TensorFlow setup, memory cleanup

- **`utils.py`** (442 lines): Utility functions
  - `setup_logging`: Configure logging with journald support
  - `_now`: Get current time (performance counter)
  - `safe_duration`: Calculate safe duration between timestamps
  - `_pkg_ver`: Get package version
  - `_env_guard`: Set environment variables for thread control
  - `build_sequences_from_features`: Build sequences from features
  - `tf_available`: Check if TensorFlow is available
  - `ngboost_available`: Check if NGBoost is available
  - `pick_tf_device`: Select TensorFlow device (GPU/CPU)

- **`data_preparation.py`** (593 lines): Data preparation functions
  - `prepare_training_data_cross_sectional`: Prepare cross-sectional training data
  - `_prepare_training_data_polars`: Polars-based data preparation (fast)
  - `_prepare_training_data_pandas`: Pandas-based data preparation (fallback)
  - `_process_combined_data_pandas`: Process combined pandas data

- **`training.py`** (989 lines): Core training functions
  - `train_models_for_interval_comprehensive`: Train models for a specific interval
  - `train_model_comprehensive`: Comprehensive model training with all features
  - `_legacy_train_fallback`: Fallback to legacy training if needed
  - Handles model family routing, validation, error handling

- **`strategies.py`** (443 lines): Strategy implementations
  - `load_mtf_data`: Load multi-timeframe data
  - `discover_targets`: Discover available targets from data
  - `prepare_training_data`: Prepare training data for strategies
  - `create_strategy_config`: Create configuration for a strategy
  - `train_with_strategy`: Train models using specified strategy
  - `compare_strategies`: Compare different training strategies

- **`main.py`** (429 lines): Main entry point
  - `main`: CLI entry point
  - Handles argument parsing, data loading, strategy execution, output generation

## Usage

### Direct Import (Recommended)

```python
from TRAINING.training_strategies.execution.training import train_models_for_interval_comprehensive
from TRAINING.training_strategies.strategies import train_with_strategy
from TRAINING.training_strategies.data_preparation import load_mtf_data
```

### Backward Compatible Import

```python
# Still works - imports from training_strategies/ modules
from TRAINING.train_with_strategies import (
    train_models_for_interval_comprehensive,
    load_mtf_data,
    ALL_FAMILIES
)
```

### CLI Usage

```bash
# Train with default strategy
python -m TRAINING.training_strategies.main --data-dir data/ --symbols AAPL,MSFT

# Train with specific strategy
python -m TRAINING.training_strategies.main --strategy multi_task --families LightGBM,XGBoost
```

## Training Strategies

1. **Single-Task**: Train one model per target independently
2. **Multi-Task**: Train models that predict multiple targets simultaneously
3. **Cascade**: Train models in a cascade where later models use earlier predictions

## Model Families Supported

All 20 model families are supported:
- **Tree-based**: LightGBM, XGBoost, QuantileLightGBM, NGBoost
- **Neural Networks**: MLP, CNN1D, LSTM, Transformer, TabCNN, TabLSTM, TabTransformer
- **Specialized**: VAE, GAN, MetaLearning, MultiTask, RewardBased, GMMRegime, ChangePoint, FTRLProximal, Ensemble

## Configuration

- **Pipeline config**: `CONFIG/training_config/pipeline_config.yaml`
- **Model configs**: `CONFIG/model_config/`
- **Family config**: `CONFIG/training_config/family_config.yaml`

## Features

- **GPU Acceleration**: Automatic GPU detection and usage
- **Memory Management**: Efficient memory usage for large datasets
- **Thread Management**: Unified thread pool control across model families
- **Isolation**: Optional process isolation for stability
- **Cross-Sectional Training**: Support for cross-sectional data preparation
- **Target Discovery**: Automatic target discovery from data

## Notes

- All modules maintain backward compatibility with the original `train_with_strategies.py` interface
- The original file is now a thin wrapper (66 lines) that re-exports everything
- Original file archived at `TRAINING/archive/original_large_files/train_with_strategies.py.original`
- Large modules (`training.py`, `data_preparation.py`) are cohesive subsystems with clear responsibilities

## Related Documentation

- **[Refactoring & Wrappers Guide](../../01_tutorials/REFACTORING_AND_WRAPPERS.md)** - Complete explanation of how wrappers work and import patterns
