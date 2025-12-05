# Model Training Guide

Complete guide to training machine learning models on labeled data.

## Overview

```
Labeled Data
    ↓
[1] Data Preparation → Feature selection, train/test split
    ↓
[2] Model Selection → Choose model family
    ↓
[3] Configuration → Load/override hyperparameters
    ↓
[4] Training → Fit model with early stopping
    ↓
[5] Evaluation → Metrics, feature importance, diagnostics
    ↓
Trained Model
```

## Available Models

### Core Models

**LightGBM** - Gradient boosting (highly regularized)
```python
from TRAINING.model_fun import LightGBMTrainer
from CONFIG.config_loader import load_model_config

config = load_model_config("lightgbm", variant="conservative")
trainer = LightGBMTrainer(config)
trainer.train(X_train, y_train)
```

**XGBoost** - Gradient boosting
```python
from TRAINING.model_fun import XGBoostTrainer

config = load_model_config("xgboost", variant="balanced")
trainer = XGBoostTrainer(config)
trainer.train(X_train, y_train)
```

**Ensemble** - Stacking (HGB + RF + Ridge)
```python
from TRAINING.model_fun import EnsembleTrainer

config = load_model_config("ensemble")
trainer = EnsembleTrainer(config)
trainer.train(X_train, y_train)
```

**MultiTask** - Multi-task neural network
```python
from TRAINING.model_fun import MultiTaskTrainer

config = load_model_config("multi_task")
trainer = MultiTaskTrainer(config)
trainer.train(X_train, y_train)
```

### Deep Learning Models

**MLP** - Multi-layer perceptron
```python
from TRAINING.model_fun import MLPTrainer

config = load_model_config("mlp", overrides={"epochs": 100})
trainer = MLPTrainer(config)
trainer.train(X_train, y_train)
```

**Transformer** - Attention-based model
```python
from TRAINING.model_fun import TransformerTrainer

config = load_model_config("transformer")
trainer = TransformerTrainer(config)
trainer.train(X_train, y_train)
```

**LSTM** - Long short-term memory
```python
from TRAINING.model_fun import LSTMTrainer

config = load_model_config("lstm")
trainer = LSTMTrainer(config)
trainer.train(X_train, y_train)
```

## Training Workflow

### 1. Load Data

```python
import pandas as pd

labeled_data = pd.read_parquet("data/labeled/AAPL_labeled.parquet")

# Separate features and targets
target_cols = [col for col in labeled_data.columns if col.startswith("target_")]
feature_cols = [col for col in labeled_data.columns if col not in target_cols]

X = labeled_data[feature_cols]
y = labeled_data["target_fwd_ret_5m"]
```

### 2. Split Data

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 3. Train Model

```python
from TRAINING.model_fun import LightGBMTrainer
from CONFIG.config_loader import load_model_config

config = load_model_config("lightgbm", variant="conservative")
trainer = LightGBMTrainer(config)
trainer.train(X_train, y_train)

# Evaluate
metrics = trainer.evaluate(X_test, y_test)
print(metrics)
```

### 4. Save Model

```python
trainer.save("models/lightgbm_AAPL.pkl")
```

## Configuration Variants

Each model has three variants:

- **conservative**: Highest regularization, least overfitting
- **balanced**: Default settings
- **aggressive**: Faster training, lower regularization

```python
config = load_model_config("lightgbm", variant="conservative")
```

## Early Stopping

All models support early stopping to prevent overfitting:

```python
config = load_model_config("lightgbm", overrides={
    "early_stopping_rounds": 50,
    "n_estimators": 1000
})
```

## Next Steps

- [Walk-Forward Validation](WALKFORWARD_VALIDATION.md) - Realistic validation
- [Feature Selection Tutorial](FEATURE_SELECTION_TUTORIAL.md) - Select features
- [Model Catalog](../../02_reference/models/MODEL_CATALOG.md) - All available models

