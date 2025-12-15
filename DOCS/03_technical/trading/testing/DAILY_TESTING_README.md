# IBKR Daily Model Testing

This setup allows you to test the IBKR trading stack with your current daily models before switching to intraday models.

## Purpose

- **Test the entire IBKR stack** with your existing daily models
- **Validate all components** (rebalancer, rotation engine, execution, safety guards)
- **Easy switching** to intraday models when they're trained
- **No changes needed** to your existing daily models

## Files Created

- `config/ibkr_daily_test.yaml` - Configuration for daily model testing
- `test_daily_models.py` - Main testing script
- `run_daily_test.sh` - Simple test runner
- `switch_to_intraday.py` - Model switching utility

## Quick Start

### 1. Test with Daily Models

```bash
# Run the daily model test
cd /home/Jennifer/secure/trader
./IBKR_trading/run_daily_test.sh
```

### 2. Check Results

```bash
# View test results
tail -f logs/daily_model_test.log
```

### 3. Switch to Intraday (When Ready)

```bash
# Switch to intraday models
python IBKR_trading/switch_to_intraday.py switch

# Check current mode
python IBKR_trading/switch_to_intraday.py status
```

## ️ Configuration

### Daily Model Settings

```yaml
models:
  daily:
    enabled: true
    horizons: [1]  # 1-day horizon
    families: ["momentum", "mean_reversion", "volatility", "volume"]
    model_path: "models/daily_models/"
```

### Portfolio Settings

```yaml
portfolio:
  max_positions: 20
  position_size_method: "vol_target"
  vol_target: 0.15  # 15% annual vol
  per_name_cap: 0.05  # 5% max per name
```

### Rebalancing Schedule

```yaml
rebalancing:
  schedule: ["09:35", "15:45"]  # Open and close
  no_trade_threshold: 0.01  # 1% NAV threshold
```

## Model Switching

### Current Mode: Daily

- **Horizons**: [1] (daily)
- **Rebalancing**: 2x daily (09:35, 15:45)
- **Position caps**: 5% max per name
- **Features**: returns, volatility, volume, momentum, mean_reversion

### After Switch: Intraday

- **Horizons**: [5, 10, 15, 30, 60] (5m, 10m, 15m, 30m, 60m)
- **Rebalancing**: 5x intraday (09:35, 10:30, 12:00, 14:30, 15:50)
- **Position caps**: 2% max per name
- **Features**: + microstructure, barrier_targets

## Testing Process

### 1. Daily Model Test

```python
# Load your daily models
models = load_daily_models()

# Generate signals
signals = generate_daily_signals(models, data, symbols)

# Run rebalancing
target_weights = integration.run_rebalancing_cycle(symbols, horizons=[1])
```

### 2. Validation

- **Model loading** - Your daily models load correctly
- **Signal generation** - Signals are generated properly
- **Rebalancing** - Portfolio rebalancing works
- **Rotation** - Position rotation logic works
- **Execution** - Order execution simulation works
- **Safety guards** - Risk management works

### 3. Performance Metrics

- **Total return** - Portfolio performance
- **Turnover** - Trading activity
- **Risk metrics** - Volatility, drawdown
- **Cost analysis** - Trading costs

## Customization

### Update Model Loading

Edit `test_daily_models.py`:

```python
def load_daily_models(self) -> dict:
    # Replace with your actual model loading
    models = {
        "momentum": load_your_momentum_model(),
        "mean_reversion": load_your_mean_reversion_model(),
        # ... etc
    }
    return models
```

### Update Data Loading

```python
def get_daily_data(self, symbols: list, lookback_days: int = 252) -> pd.DataFrame:
    # Replace with your actual data loading
    data = load_your_daily_data(symbols, lookback_days)
    return data
```

### Update Signal Generation

```python
def generate_daily_signals(self, models: dict, data: pd.DataFrame, symbols: list) -> dict:
    # Replace with your actual signal generation
    signals = {}
    for symbol in symbols:
        # Your signal generation logic here
        signals[symbol] = your_signal_generation(symbol, models, data)
    return signals
```

## Expected Results

### Daily Model Test

- **Portfolio return**: Varies based on your models
- **Turnover**: Should be reasonable (not too high)
- **Risk metrics**: Within acceptable bounds
- **Execution**: Orders should be placed correctly

### Switching to Intraday

- **More frequent rebalancing**: 5x vs 2x daily
- **Smaller position sizes**: 2% vs 5% max per name
- **More features**: microstructure, barrier_targets
- **Higher frequency**: 5m vs 1d data

## Troubleshooting

### Common Issues

1. **Model loading errors**: Update `load_daily_models()` with your actual model loading
2. **Data loading errors**: Update `get_daily_data()` with your actual data loading
3. **Signal generation errors**: Update `generate_daily_signals()` with your actual signal generation
4. **Configuration errors**: Check `config/ibkr_daily_test.yaml` for correct paths

### Debug Mode

```python
# Enable debug logging
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## Next Steps

1. **Run daily test** - Validate with your current models
2. **Fix any issues** - Update model/data loading as needed
3. **Train intraday models** - When ready, train your intraday models
4. **Switch to intraday** - Use `switch_to_intraday.py`
5. **Run intraday test** - Validate with intraday models

## Benefits

- **Test everything** before intraday models are ready
- **Validate the stack** with known working models
- **Easy switching** when intraday models are trained
- **No changes** to your existing daily models
- **Full IBKR integration** ready to go

---

**Ready to test? Run `./IBKR_trading/run_daily_test.sh`!**
