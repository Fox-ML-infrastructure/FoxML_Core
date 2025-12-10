# Specialized Models Module

This document describes the `TRAINING/models/specialized/` module structure, split from the original 4,518-line `specialized_models.py` file for better maintainability.

## Module Structure

### Core Modules

- **`wrappers.py`** (135 lines): Model wrapper classes
  - `TFSeriesRegressor`: TensorFlow model wrapper with preprocessing
  - `GMMRegimeRegressor`: GMM-based regime detection with regime-specific models
  - `OnlineChangeHeuristic`: Online change point detection heuristic

- **`predictors.py`** (99 lines): Predictor classes
  - `GANPredictor`: GAN-based predictor with synthetic feature generation
  - `ChangePointPredictor`: Change point predictor with feature engineering

- **`trainers.py`** (817 lines): Core training functions
  - `train_changepoint_heuristic`: Train change point detection model
  - `train_ftrl_proximal`: Train FTRL-Proximal model
  - `train_vae`: Train Variational Autoencoder
  - `train_gan`: Train Generative Adversarial Network
  - `train_ensemble`: Train ensemble models
  - `train_meta_learning`: Train meta-learning models
  - `train_multitask_temporal`: Train temporal multi-task models
  - `train_multi_task`: Train multi-task models
  - `train_lightgbm_ranker`: Train LightGBM ranking model
  - `train_xgboost_ranker`: Train XGBoost ranking model
  - `safe_predict`: Safe prediction with proper preprocessing

- **`trainers_extended.py`** (1,204 lines): Extended training functions
  - `train_lightgbm`: Train LightGBM regression model
  - `train_xgboost`: Train XGBoost regression model
  - `train_mlp`: Train Multi-Layer Perceptron
  - `train_cnn1d_temporal`: Train 1D CNN for temporal data
  - `train_tabcnn`: Train tabular CNN
  - `train_lstm_temporal`: Train LSTM for temporal data
  - `train_tablstm`: Train tabular LSTM
  - `train_transformer_temporal`: Train Transformer for temporal data
  - `train_tabtransformer`: Train tabular Transformer
  - `train_reward_based`: Train reward-based model
  - `train_quantile_lightgbm`: Train quantile LightGBM
  - `train_ngboost`: Train NGBoost model
  - `train_gmm_regime`: Train GMM regime model

- **`metrics.py`** (104 lines): Metrics functions
  - `cs_metrics_by_time`: Calculate cross-sectional metrics per timestamp

- **`data_utils.py`** (989 lines): Data loading and preparation utilities
  - `load_mtf_data`: Load multi-timeframe data
  - `get_common_feature_columns`: Get common feature columns across symbols
  - `load_global_feature_list`: Load global feature list
  - `save_global_feature_list`: Save global feature list
  - `targets_for_interval`: Get targets for a specific interval
  - `cs_transform_live`: Transform data for cross-sectional analysis
  - `prepare_sequence_cs`: Prepare sequence data for cross-sectional training
  - `prepare_training_data_cross_sectional`: Prepare cross-sectional training data

- **`core.py`** (1,391 lines): Main orchestration and entry point
  - `train_model`: Main model training function (routes to appropriate trainer)
  - `save_model`: Save trained models with metadata
  - `train_with_strategy`: Train models using specified strategy
  - `normalize_symbols`: Normalize symbol list from CLI args
  - `setup_tf`: Initialize TensorFlow with GPU/CPU configuration
  - `main`: Main entry point for CLI execution

- **`constants.py`**: Shared constants and global variables
  - `FAMILY_CAPS`: Model family capabilities map
  - `USE_POLARS`: Polars usage flag
  - `TF_DEVICE`: TensorFlow device configuration
  - Helper functions: `assert_no_nan`, `tf_available`, `ngboost_available`

## Usage

### Direct Import (Recommended)

```python
from TRAINING.models.specialized.trainers import train_lightgbm
from TRAINING.models.specialized.core import train_model
from TRAINING.models.specialized.data_utils import load_mtf_data
```

### Backward Compatible Import

```python
# Still works - imports from specialized/ modules
from TRAINING.models.specialized_models import train_model, TFSeriesRegressor
```

## Dependencies

- **TensorFlow**: Required for neural network models (MLP, VAE, GAN, etc.)
- **LightGBM**: Required for LightGBM models
- **XGBoost**: Required for XGBoost models
- **scikit-learn**: Required for preprocessing and some models
- **Polars** (optional): For faster data loading (falls back to pandas)

## Configuration

Model configurations are loaded from `CONFIG/model_config/` via the config loader:

```python
from CONFIG.config_loader import get_cfg
config = get_cfg("model_config.lightgbm")
```

## Notes

- All modules maintain backward compatibility with the original `specialized_models.py` interface
- The original file is now a thin wrapper (82 lines) that re-exports everything
- Original file archived at `TRAINING/archive/original_large_files/specialized_models.py.original`
