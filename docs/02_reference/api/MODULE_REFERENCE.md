# Module Reference

Python API reference for Fox-v1-infra modules.

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

## Trading

### Alpaca

```python
from ALPACA_trading.core.paper import PaperTradingEngine
from ALPACA_trading.brokers.paper import PaperBroker

broker = PaperBroker()
engine = PaperTradingEngine(broker)
engine.run()
```

### IBKR

```python
from IBKR_trading.live_trading import LiveTradingSystem

system = LiveTradingSystem(config)
system.run()
```

## See Also

- [CLI Reference](CLI_REFERENCE.md) - Command-line tools
- [Config Schema](CONFIG_SCHEMA.md) - Configuration schema

