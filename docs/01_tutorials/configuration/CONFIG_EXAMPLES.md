# Configuration Examples

Example configurations for common use cases.

## Conservative Training

High regularization to prevent overfitting:

```python
from CONFIG.config_loader import load_model_config

config = load_model_config("lightgbm", variant="conservative")
# Uses: max_depth=6, learning_rate=0.01, reg_alpha=0.1, reg_lambda=0.1
```

## Fast Training

Lower regularization for faster iteration:

```python
config = load_model_config("xgboost", variant="aggressive")
# Uses: max_depth=10, learning_rate=0.05, minimal regularization
```

## Custom Overrides

Override specific parameters:

```python
config = load_model_config("mlp", overrides={
    "epochs": 100,
    "learning_rate": 0.0001,
    "batch_size": 64,
    "dropout": 0.3
})
```

## Multi-Model Training

Train multiple models with different configs:

```python
models = {
    "lightgbm": load_model_config("lightgbm", variant="conservative"),
    "xgboost": load_model_config("xgboost", variant="balanced"),
    "ensemble": load_model_config("ensemble")
}

for name, config in models.items():
    trainer = get_trainer(name)(config)
    trainer.train(X_train, y_train)
```

## Feature Selection Config

```yaml
# CONFIG/feature_selection_config.yaml
lightgbm:
  device: "cpu"  # or "gpu"
  max_depth: 8
  learning_rate: 0.03
  n_estimators: 1000
  early_stopping_rounds: 50

defaults:
  n_features: 50
  min_importance: 0.001
```

## Training Workflow Config

```yaml
# CONFIG/training_config/first_batch_specs.yaml
data:
  train_test_split: 0.2
  validation_split: 0.2

models:
  lightgbm:
    variant: "conservative"
  xgboost:
    variant: "balanced"

training:
  early_stopping: true
  early_stopping_rounds: 50
```

## IBKR Trading Config

```yaml
# IBKR_trading/config/ibkr_enhanced.yaml
ibkr:
  host: "127.0.0.1"
  port: 7497
  client_id: 1

trading:
  symbols: ["AAPL", "MSFT", "GOOGL"]
  horizons: ["5m", "10m", "15m", "30m", "60m"]
  max_position_size: 100

safety:
  max_portfolio_risk: 0.02
  stop_loss: 0.05
  take_profit: 0.10
```

## Alpaca Paper Trading Config

```yaml
# ALPACA_trading/config/base.yaml
alpaca:
  api_key: "${ALPACA_API_KEY}"
  secret_key: "${ALPACA_SECRET_KEY}"
  base_url: "https://paper-api.alpaca.markets"

trading:
  symbols: ["AAPL", "MSFT", "GOOGL"]
  max_position_size: 1000

risk:
  max_portfolio_risk: 0.02
  stop_loss: 0.05
```

## Next Steps

- [Config Basics](CONFIG_BASICS.md) - Configuration fundamentals
- [Advanced Config](ADVANCED_CONFIG.md) - Advanced configuration
- [Config Schema](../../02_reference/api/CONFIG_SCHEMA.md) - Complete schema

