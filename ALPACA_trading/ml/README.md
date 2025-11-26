# Machine Learning Integration

This directory contains the ML model interface, registry, and runtime for integrating trained models into the trading system.

## Components

### `model_interface.py` - Model Interface
Abstract interface that all ML models must implement for use in trading.

**Interface Methods:**
- `predict()` - Generate predictions
- `predict_proba()` - Generate probability predictions (for classification)
- `get_feature_importance()` - Get feature importance scores
- `get_model_info()` - Get model metadata

**Purpose:** Provides a consistent interface regardless of model type (LightGBM, XGBoost, PyTorch, ONNX, etc.)

### `registry.py` - Model Registry
Manages model loading, caching, and versioning.

**Features:**
- **Model Loading**: Loads models from various formats (pickle, PyTorch, ONNX)
- **SHA256 Verification**: Verifies model file integrity
- **Caching**: Caches loaded models for performance
- **Versioning**: Tracks model versions and metadata

**Supported Formats:**
- `pickle`: Python pickle files (LightGBM, XGBoost, sklearn)
- `torch`: PyTorch model files
- `onnx`: ONNX Runtime models

**Usage:**
```python
from ml.registry import load_model
from ml.model_interface import ModelSpec

spec = ModelSpec(
    path="models/my_model.pkl",
    kind="pickle"
)

model, checksum = load_model(spec)
predictions = model.predict(features)
```

### `runtime.py` - Model Runtime
Runtime environment for executing models in production.

**Features:**
- Model inference execution
- Batch prediction support
- Performance monitoring
- Error handling and fallbacks

**Integration:** Connects models from the registry to the trading engine.

## Model Integration Flow

1. **Model Training** → Models trained in `TRAINING/` directory
2. **Model Registration** → Models registered in `config/models.yaml`
3. **Model Loading** → Registry loads models on startup
4. **Feature Preparation** → Trading engine prepares features
5. **Model Inference** → Runtime executes predictions
6. **Signal Generation** → Predictions converted to trading signals

## Configuration

Models are configured in `config/models.yaml`:
```yaml
models:
  - name: "my_model"
    path: "models/my_model.pkl"
    kind: "pickle"
    features: ["feature1", "feature2", ...]
    target: "target_column"
    task_type: "regression"  # or "classification"
```

## Model Requirements

Models must:
- Accept features as numpy array or pandas DataFrame
- Return predictions in expected format
- Handle missing values gracefully
- Provide feature importance (optional but recommended)

## Supported Model Types

- **Gradient Boosting**: LightGBM, XGBoost, CatBoost
- **Neural Networks**: PyTorch models, ONNX models
- **Traditional ML**: scikit-learn models
- **Custom Models**: Any model implementing the interface

## Performance Considerations

- Models are loaded once and cached
- Batch predictions for efficiency
- Lazy loading for large models
- GPU support for PyTorch models (if available)

## Error Handling

- Model loading failures are logged and handled gracefully
- Missing models trigger fallback behavior
- Prediction errors are caught and logged
- Invalid inputs are validated before inference

