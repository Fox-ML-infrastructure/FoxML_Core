# Model Comparisons

Comparison of different model types and their performance.

## Model Performance

### Tree Models

**LightGBM**:
- Training: Fast (2-5 min)
- Prediction: Very fast (< 1ms)
- Accuracy: High
- Best for: General-purpose regression

**XGBoost**:
- Training: Fast (3-6 min)
- Prediction: Very fast (< 1ms)
- Accuracy: High
- Best for: Alternative to LightGBM

### Neural Networks

**MLP**:
- Training: Medium (10-20 min)
- Prediction: Fast (1-5ms)
- Accuracy: High (non-linear patterns)
- Best for: Complex relationships

**LSTM**:
- Training: Slow (20-40 min)
- Prediction: Medium (5-10ms)
- Accuracy: High (sequential patterns)
- Best for: Time series

**Transformer**:
- Training: Slow (30-60 min)
- Prediction: Medium (5-15ms)
- Accuracy: Very high
- Best for: Attention-based patterns

## Selection Guide

| Use Case | Recommended Model |
|----------|------------------|
| Fast iteration | LightGBM, XGBoost |
| High accuracy | Transformer, LSTM |
| Time series | LSTM, CNN1D |
| Feature selection | LightGBM (importance) |
| Multiple targets | MultiTask |

## See Also

- [Model Catalog](../../02_reference/models/MODEL_CATALOG.md) - All models
- [Model Training Guide](../../01_tutorials/training/MODEL_TRAINING_GUIDE.md) - Training guide

