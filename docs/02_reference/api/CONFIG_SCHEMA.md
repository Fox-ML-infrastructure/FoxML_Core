# Configuration Schema

Complete configuration schema reference.

## Model Configuration Schema

All model configs follow this structure:

```yaml
default:
  param1: value1
  param2: value2

variants:
  conservative:
    param1: value1_conservative
    param2: value2_conservative
  
  balanced:
    param1: value1_balanced
    param2: value2_balanced
  
  aggressive:
    param1: value1_aggressive
    param2: value2_aggressive
```

## Available Models

17 model types with configs in `CONFIG/model_config/`:

- `lightgbm.yaml`
- `xgboost.yaml`
- `ensemble.yaml`
- `multi_task.yaml`
- `mlp.yaml`
- `transformer.yaml`
- `lstm.yaml`
- `cnn1d.yaml`
- `vae.yaml`
- `gan.yaml`
- `gmm_regime.yaml`
- `ngboost.yaml`
- `quantile_lightgbm.yaml`
- `change_point.yaml`
- `ftrl_proximal.yaml`
- `reward_based.yaml`
- `meta_learning.yaml`

## LightGBM Schema

```yaml
default:
  max_depth: 8
  learning_rate: 0.03
  n_estimators: 1000
  subsample: 0.75
  colsample_bytree: 0.75
  reg_alpha: 0.05
  reg_lambda: 0.05
  early_stopping_rounds: 50

variants:
  conservative:
    max_depth: 6
    learning_rate: 0.01
    reg_alpha: 0.1
    reg_lambda: 0.1
  
  balanced:
    max_depth: 8
    learning_rate: 0.03
  
  aggressive:
    max_depth: 10
    learning_rate: 0.05
    reg_alpha: 0.01
    reg_lambda: 0.01
```

## Training Configuration Schema

```yaml
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

## Feature Selection Schema

```yaml
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

## IBKR Trading Schema

```yaml
ibkr:
  host: "127.0.0.1"
  port: 7497
  client_id: 1

trading:
  symbols: ["AAPL", "MSFT"]
  horizons: ["5m", "10m", "15m", "30m", "60m"]
  max_position_size: 100

safety:
  max_portfolio_risk: 0.02
  stop_loss: 0.05
  take_profit: 0.10
```

## See Also

- [Config Reference](../configuration/CONFIG_LOADER_API.md) - Config loader API
- [Config Basics](../../01_tutorials/configuration/CONFIG_BASICS.md) - Configuration tutorial

