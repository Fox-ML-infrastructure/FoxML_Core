# Module Reference

Python API reference for FoxML Core modules.

## Configuration

### Config Loader

```python
from CONFIG.config_loader import (
    load_model_config,
    load_training_config,
    list_available_configs
)

# Load model config
config = load_model_config("lightgbm", variant="conservative")

# Load training config
training_cfg = load_training_config("first_batch_specs")

# List available configs
configs = list_available_configs()
# Returns: {"model_configs": [...], "training_configs": [...]}
```

## Data Processing

### Pipeline

```python
from DATA_PROCESSING.pipeline import normalize_interval, assert_bars_per_day

# Normalize data
df_clean = normalize_interval(df, interval="5m")

# Verify bar count
assert_bars_per_day(df_clean, interval="5m", min_full_day_frac=0.90)
```

### Feature Builders

```python
from DATA_PROCESSING.features import (
    SimpleFeatureComputer,
    ComprehensiveFeatureBuilder
)

# Simple features
computer = SimpleFeatureComputer()
features = computer.compute(df)

# Comprehensive features (200+ features)
builder = ComprehensiveFeatureBuilder(config_path="config/features.yaml")
features = builder.build_features(input_paths, output_dir, universe_config)
```

### Target Functions

```python
from DATA_PROCESSING.targets import (
    add_barrier_targets_to_dataframe,
    compute_neutral_band,
    classify_excess_return
)

# Barrier targets (functions, not classes)
df = add_barrier_targets_to_dataframe(
    df, horizon_minutes=15, barrier_size=0.5
)

# Excess returns
df = compute_neutral_band(df, horizon="5m")
df = classify_excess_return(df, horizon="5m")
```

## Training Utilities

### Target Type Detection

**Module:** `TRAINING/ranking/utils/target_utils.py` (also available via `TRAINING/utils/target_utils.py` for backward compatibility)

Provides reusable helpers for detecting target types consistently:

```python
from TRAINING.utils.target_utils import (  # Backward-compatible import
    is_classification_target,
    is_binary_classification_target,
    is_multiclass_target
)
# Or use the new path:
# from TRAINING.ranking.utils.target_utils import ...

# Detect target type
if is_classification_target(y):
    if is_binary_classification_target(y):
        # Binary classification (0/1)
    elif is_multiclass_target(y):
        # Multiclass (3+ classes)
else:
    # Regression
```

**Functions:**
- `is_classification_target(y, max_classes=20)` - Detects classification vs regression
- `is_binary_classification_target(y)` - Detects binary classification
- `is_multiclass_target(y, max_classes=10)` - Detects multiclass classification

**Used by:** CatBoost model builder (ranking and selection)

### Sklearn Preprocessing

**Module:** `TRAINING/utils/sklearn_safe.py`

Provides consistent preprocessing for sklearn-based models:

```python
from TRAINING.utils.sklearn_safe import make_sklearn_dense_X

# Convert to dense float32 array with consistent preprocessing
X_dense, feature_names_dense = make_sklearn_dense_X(X, feature_names)
# Returns: dense float32, median-imputed, inf values handled
```

**Function:**
- `make_sklearn_dense_X(X, feature_names=None)` - Converts tabular data to dense float32 numpy array

**Used by:** Lasso, Mutual Information, Univariate Selection, Boruta, Stability Selection (ranking and selection)

**Note:** Boruta uses this for preprocessing but acts as a statistical gatekeeper (not just another importance scorer). See [Ranking and Selection Consistency](../../01_tutorials/training/RANKING_SELECTION_CONSISTENCY.md) for details.

## Training

### Model Trainers

```python
from TRAINING.model_fun import (
    LightGBMTrainer,
    XGBoostTrainer,
    EnsembleTrainer,
    MultiTaskTrainer,
    MLPTrainer,
    TransformerTrainer,
    LSTMTrainer
)

trainer = LightGBMTrainer(config)
trainer.train(X_train, y_train)
predictions = trainer.predict(X_test)
# Note: evaluate() method doesn't exist. Use sklearn metrics or compute manually:
# from sklearn.metrics import mean_squared_error, r2_score
# metrics = {'mse': mean_squared_error(y_test, predictions), 'r2': r2_score(y_test, predictions)}
```

### Training Strategies

```python
from TRAINING.strategies.single_task import SingleTaskStrategy
from TRAINING.strategies.multi_task import MultiTaskStrategy

# Single target
strategy = SingleTaskStrategy(config)
strategy.train(X, {'fwd_ret_5m': y}, feature_names)

# Multiple targets
strategy = MultiTaskStrategy(config)
strategy.train(X, {
    'fwd_ret_5m': y_5m,
    'fwd_ret_15m': y_15m
}, feature_names)
```

## Model Integration

**Note:** Trading integration modules have been removed from the core repository. The system focuses on ML research infrastructure and model training. Trained models can be integrated with external trading systems through standard interfaces.

## See Also

- [Ranking and Selection Consistency](../../01_tutorials/training/RANKING_SELECTION_CONSISTENCY.md) - Guide to unified pipeline behavior (includes utility usage)
- [Intelligent Trainer API](INTELLIGENT_TRAINER_API.md) - Intelligent training pipeline API
- [Config Loader API](../configuration/CONFIG_LOADER_API.md) - Configuration loading (includes logging config utilities)
- [Module Reference](MODULE_REFERENCE.md) - This file
- [CLI Reference](CLI_REFERENCE.md) - Command-line tools
- [Config Schema](CONFIG_SCHEMA.md) - Configuration schema

