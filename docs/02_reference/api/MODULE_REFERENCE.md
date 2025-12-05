# Module Reference

Python API reference for Fox-v1-infra modules.

## Configuration

### Config Loader

```python
from CONFIG.config_loader import (
    load_model_config,
    load_training_config,
    get_available_model_configs
)

# Load model config
config = load_model_config("lightgbm", variant="conservative")

# Load training config
training_cfg = load_training_config("first_batch_specs")

# List available configs
configs = get_available_model_configs()
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
    SimpleFeatureBuilder,
    ComprehensiveFeatureBuilder,
    StreamingFeatureBuilder
)

builder = ComprehensiveFeatureBuilder()
features = builder.build(df)
```

### Target Builders

```python
from DATA_PROCESSING.targets import (
    BarrierTargetBuilder,
    ExcessReturnsBuilder,
    HFTForwardReturnsBuilder
)

builder = BarrierTargetBuilder()
targets = builder.build(df, horizon="5m", barrier=0.001)
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
metrics = trainer.evaluate(X_test, y_test)
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
from ALPACA_trading.core.engine.paper import PaperTradingEngine
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

