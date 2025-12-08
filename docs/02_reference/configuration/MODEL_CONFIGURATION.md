# Model Configuration Guide

Complete guide to configuring model hyperparameters in FoxML Core.

## Overview

Each model family has its own YAML configuration file in `model_config/` with hyperparameters organized by variant (default, conservative, aggressive).

## Supported Models (17 total)

### Tree-Based Models
- **LightGBM** - Gradient boosting (highly optimized)
- **XGBoost** - Gradient boosting

### Neural Networks
- **MLP** - Multi-layer perceptron
- **Transformer** - Transformer architecture
- **LSTM** - Long short-term memory
- **CNN1D** - 1D convolutional neural network

### Ensemble Models
- **Ensemble** - Stacking (HGB + RF + Ridge)
- **MultiTask** - Multi-task neural network

### Feature Engineering Models
- **VAE** - Variational autoencoder
- **GAN** - Generative adversarial network
- **GMMRegime** - Gaussian mixture model for regime detection

### Probabilistic Models
- **NGBoost** - Natural gradient boosting
- **QuantileLightGBM** - Quantile regression with LightGBM

### Advanced Models
- **ChangePoint** - Change point detection
- **FTRL** - Follow-the-regularized-leader
- **RewardBased** - Reward-based learning
- **MetaLearning** - Meta-learning

## Configuration Structure

Each model config file follows this structure:

```yaml
default:
  # Default/balanced hyperparameters
  n_estimators: 100
  learning_rate: 0.1
  # ... more params

conservative:
  # More stable, lower risk settings
  n_estimators: 200
  learning_rate: 0.05
  # ... more stable params

aggressive:
  # Higher performance, more experimental
  n_estimators: 50
  learning_rate: 0.2
  # ... more experimental params
```

## Variants

### `default`
- Balanced settings
- Good starting point
- Used when variant not specified

### `conservative`
- Lower risk, more stable
- More regularization
- Slower training, more robust

### `aggressive`
- Higher performance potential
- Less regularization
- Faster training, may overfit

## Usage

### Loading Model Configs

```python
from CONFIG.config_loader import load_model_config

# Load with default variant
config = load_model_config("lightgbm")

# Load with specific variant
config = load_model_config("lightgbm", variant="aggressive")

# Load with overrides
config = load_model_config(
    "lightgbm",
    variant="conservative",
    overrides={"n_estimators": 500, "learning_rate": 0.01}
)
```

### Creating Custom Variants

1. **Edit model config file:**
```yaml
# In model_config/lightgbm.yaml
my_custom_variant:
  n_estimators: 1000
  learning_rate: 0.01
  num_leaves: 255
  max_depth: 10
  # ... other params
```

2. **Use in code:**
```python
config = load_model_config("lightgbm", variant="my_custom_variant")
```

## Model-Specific Examples

### LightGBM

**File:** `model_config/lightgbm.yaml`

**Key Parameters:**
- `n_estimators` - Number of boosting rounds
- `learning_rate` - Learning rate
- `num_leaves` - Maximum tree leaves
- `max_depth` - Maximum tree depth
- `min_child_samples` - Minimum samples in leaf
- `subsample` - Row sampling ratio
- `colsample_bytree` - Column sampling ratio
- `reg_alpha`, `reg_lambda` - L1/L2 regularization

**Example: Custom LightGBM Config**

```yaml
my_lightgbm_variant:
  n_estimators: 500
  learning_rate: 0.05
  num_leaves: 127
  max_depth: 8
  min_child_samples: 30
  subsample: 0.8
  colsample_bytree: 0.8
  reg_alpha: 0.1
  reg_lambda: 0.1
```

### XGBoost

**File:** `model_config/xgboost.yaml`

**Key Parameters:**
- `n_estimators` - Number of boosting rounds
- `learning_rate` - Learning rate
- `max_depth` - Maximum tree depth
- `min_child_weight` - Minimum sum of instance weight
- `subsample` - Row sampling ratio
- `colsample_bytree` - Column sampling ratio
- `reg_alpha`, `reg_lambda` - L1/L2 regularization

### Neural Networks (MLP, LSTM, Transformer, CNN1D)

**Key Parameters:**
- `hidden_layers` - Number and size of hidden layers
- `dropout` - Dropout rate
- `batch_size` - Batch size
- `learning_rate` - Learning rate
- `optimizer` - Optimizer type (adam, sgd, etc.)
- `epochs` - Number of training epochs

**Example: Custom MLP Config**

```yaml
my_mlp_variant:
  hidden_layers: [256, 128, 64]
  dropout: 0.3
  batch_size: 64
  learning_rate: 0.001
  optimizer: "adam"
  epochs: 100
```

### Sequential Models (LSTM, Transformer)

**Additional Parameters:**
- `sequence_length` - Input sequence length
- `attention_heads` - Number of attention heads (Transformer)
- `embedding_dim` - Embedding dimension

**Example: Custom LSTM Config**

```yaml
my_lstm_variant:
  sequence_length: 60
  hidden_layers: [128, 64]
  dropout: 0.2
  batch_size: 32
  learning_rate: 0.001
```

## Common Scenarios

### Scenario 1: Creating a Custom Model Variant

**Goal:** Create a variant optimized for your specific use case.

**Steps:**
1. Edit model config file (e.g., `model_config/lightgbm.yaml`):
```yaml
my_custom_variant:
  n_estimators: 1000
  learning_rate: 0.01
  num_leaves: 255
  # ... other params
```

2. Use in code:
```python
config = load_model_config("lightgbm", variant="my_custom_variant")
```

### Scenario 2: Adjusting Model Hyperparameters

**Goal:** Fine-tune existing variant.

**Steps:**
1. Edit existing variant in model config:
```yaml
default:
  n_estimators: 200  # Increase from 100
  learning_rate: 0.05  # Decrease from 0.1
```

2. Or use overrides:
```python
config = load_model_config(
    "lightgbm",
    variant="default",
    overrides={"n_estimators": 200, "learning_rate": 0.05}
)
```

### Scenario 3: Comparing Variants

**Goal:** Test different variants to find best performance.

**Steps:**
1. Test each variant:
```python
for variant in ["conservative", "default", "aggressive"]:
    config = load_model_config("lightgbm", variant=variant)
    # Train and evaluate
```

2. Compare results and select best variant

## Best Practices

1. **Start with default variant** - Good baseline for most cases
2. **Test variants** - Verify all variants work correctly
3. **Document custom variants** - Add comments explaining why custom variants exist
4. **Use overrides sparingly** - Prefer config files over runtime overrides
5. **Validate parameters** - Ensure hyperparameters are within valid ranges
6. **Monitor training** - Watch for overfitting/underfitting with different variants

## Model-Specific Tips

### Tree-Based Models (LightGBM, XGBoost)
- `num_leaves` should be ≤ 2^max_depth
- Lower `learning_rate` with more `n_estimators` for better generalization
- Use `subsample` and `colsample_bytree` for regularization

### Neural Networks
- Start with smaller networks, increase if underfitting
- Use dropout for regularization
- Batch normalization can help with training stability

### Sequential Models
- `sequence_length` should match your data's temporal structure
- Longer sequences = more memory, but may capture more patterns
- Attention mechanisms (Transformer) help with long sequences

## Related Documentation

- [Configuration System Overview](README.md) - Main configuration overview
- [Training Pipeline Configs](TRAINING_PIPELINE_CONFIGS.md) - Training configuration
- [Usage Examples](USAGE_EXAMPLES.md) - Practical configuration examples
- [Model Config Reference](../models/MODEL_CONFIG_REFERENCE.md) - Detailed model documentation

