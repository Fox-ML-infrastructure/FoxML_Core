# Model Catalog

Complete catalog of all available machine learning models.

## Core Models

### LightGBM

Gradient boosting with highly regularized settings.

**Config**: `CONFIG/model_config/lightgbm.yaml`  
**Trainer**: `TRAINING.model_fun.LightGBMTrainer`  
**Variants**: conservative, balanced, aggressive

**Best For**: General-purpose regression, feature selection

### XGBoost

Gradient boosting with alternative implementation.

**Config**: `CONFIG/model_config/xgboost.yaml`  
**Trainer**: `TRAINING.model_fun.XGBoostTrainer`  
**Variants**: conservative, balanced, aggressive

**Best For**: Alternative to LightGBM, ensemble diversity

### Ensemble

Stacking ensemble (HGB + RF + Ridge).

**Config**: `CONFIG/model_config/ensemble.yaml`  
**Trainer**: `TRAINING.model_fun.EnsembleTrainer`

**Best For**: Robust predictions, reducing overfitting

### MultiTask

Multi-task neural network for multiple targets.

**Config**: `CONFIG/model_config/multi_task.yaml`  
**Trainer**: `TRAINING.model_fun.MultiTaskTrainer`

**Best For**: Predicting multiple targets simultaneously (TTH, MDD, MFE)

## Deep Learning Models

### MLP

Multi-layer perceptron (feedforward neural network).

**Config**: `CONFIG/model_config/mlp.yaml`  
**Trainer**: `TRAINING.model_fun.MLPTrainer`  
**Variants**: conservative, balanced, aggressive

**Best For**: Non-linear relationships, large feature sets

### Transformer

Attention-based transformer model.

**Config**: `CONFIG/model_config/transformer.yaml`  
**Trainer**: `TRAINING.model_fun.TransformerTrainer`

**Best For**: Sequential patterns, attention mechanisms

### LSTM

Long short-term memory network.

**Config**: `CONFIG/model_config/lstm.yaml`  
**Trainer**: `TRAINING.model_fun.LSTMTrainer`

**Best For**: Time series patterns, sequential dependencies

### CNN1D

1D convolutional neural network.

**Config**: `CONFIG/model_config/cnn1d.yaml`  
**Trainer**: `TRAINING.model_fun.CNN1DTrainer`

**Best For**: Local pattern detection, feature extraction

## Feature Engineering Models

### VAE

Variational Autoencoder for feature engineering.

**Config**: `CONFIG/model_config/vae.yaml`  
**Trainer**: `TRAINING.model_fun.VAETrainer`

**Best For**: Dimensionality reduction, feature extraction

### GAN

Generative Adversarial Network.

**Config**: `CONFIG/model_config/gan.yaml`  
**Trainer**: `TRAINING.model_fun.GANTrainer`

**Best For**: Data augmentation, synthetic data generation

### GMM Regime

Gaussian Mixture Model for regime detection.

**Config**: `CONFIG/model_config/gmm_regime.yaml`  
**Trainer**: `TRAINING.model_fun.GMMRegimeTrainer`

**Best For**: Market regime identification, regime-aware strategies

## Probabilistic Models

### NGBoost

Natural Gradient Boosting for probabilistic predictions.

**Config**: `CONFIG/model_config/ngboost.yaml`  
**Trainer**: `TRAINING.model_fun.NGBoostTrainer`

**Best For**: Uncertainty quantification, probabilistic forecasts

### Quantile LightGBM

LightGBM with quantile regression.

**Config**: `CONFIG/model_config/quantile_lightgbm.yaml`  
**Trainer**: `TRAINING.model_fun.QuantileLightGBMTrainer`

**Best For**: Quantile predictions, risk estimation

## Advanced Models

### Change Point

Change point detection model.

**Config**: `CONFIG/model_config/change_point.yaml`  
**Trainer**: `TRAINING.model_fun.ChangePointTrainer`

**Best For**: Detecting regime changes, structural breaks

### FTRL Proximal

Follow-the-Regularized-Leader Proximal.

**Config**: `CONFIG/model_config/ftrl_proximal.yaml`  
**Trainer**: `TRAINING.model_fun.FTRLProximalTrainer`

**Best For**: Online learning, streaming data

### Reward Based

Reward-based learning model.

**Config**: `CONFIG/model_config/reward_based.yaml`  
**Trainer**: `TRAINING.model_fun.RewardBasedTrainer`

**Best For**: Reinforcement learning approaches

### Meta Learning

Meta-learning model.

**Config**: `CONFIG/model_config/meta_learning.yaml`  
**Trainer**: `TRAINING.model_fun.MetaLearningTrainer`

**Best For**: Few-shot learning, adaptation

## Model Selection Guide

| Use Case | Recommended Models |
|----------|-------------------|
| General regression | LightGBM, XGBoost |
| Feature selection | LightGBM (importance) |
| Multiple targets | MultiTask |
| Non-linear patterns | MLP, Transformer |
| Time series | LSTM, CNN1D |
| Uncertainty | NGBoost, QuantileLightGBM |
| Regime detection | GMM Regime |
| Feature engineering | VAE |

## See Also

- [Model Config Reference](MODEL_CONFIG_REFERENCE.md) - Configuration details
- [Training Parameters](TRAINING_PARAMETERS.md) - Training settings
- [Model Training Guide](../../01_tutorials/training/MODEL_TRAINING_GUIDE.md) - Training tutorial

