# Trading Strategies

This directory contains trading strategy implementations.

## Components

### `factory.py` - Strategy Factory
Factory pattern for creating strategy instances based on configuration.

**Supported Strategies:**
- `regime_aware_ensemble` - Regime-aware ensemble strategy
- `trend_following` - Trend-following strategy
- `mean_reversion` - Mean-reversion strategy
- Custom strategies (extensible)

**Usage:**
```python
from strategies.factory import strategy_factory

strategy = strategy_factory.create(
    name="regime_aware_ensemble",
    params=params_dict
)
```

### `regime_aware_ensemble.py` - Regime-Aware Ensemble Strategy
Advanced strategy that blends multiple signals with regime-specific weighting.

**Key Features:**
- **Regime Detection**: Adapts to trending, choppy, or volatile markets
- **Signal Blending**: Combines trend-following and mean-reversion signals
- **Dynamic Weighting**: Adjusts weights based on recent performance (IC, Sharpe)
- **Lookback Adaptation**: Adjusts feature lookback periods by regime

**Parameters:**
- `combination_method`: How to combine signals (rolling_ic, sharpe, ridge, voting)
- `confidence_threshold`: Minimum confidence to trade
- `use_regime_switching`: Enable/disable regime-based adaptation
- `trend_following_weight`: Weight for trend signals
- `mean_reversion_weight`: Weight for mean-reversion signals

**Regime Adjustments:**
- **Trending Markets**: Longer lookback, higher trend weight
- **Choppy Markets**: Shorter lookback, higher mean-reversion weight
- **Volatile Markets**: Shortest lookback, balanced weights

**Signal Combination Methods:**
1. **rolling_ic**: Weight by rolling Information Coefficient
2. **sharpe**: Weight by rolling Sharpe ratio
3. **ridge**: Ridge regression-based combination
4. **voting**: Simple voting mechanism

## Strategy Interface

All strategies implement the `BaseStrategy` interface:
- `generate_signals()` - Generate trading signals
- `calculate_position_size()` - Calculate position sizes
- `should_enter()` - Entry logic
- `should_exit()` - Exit logic

## Integration

Strategies are integrated into the trading engine via:
1. **Strategy Factory** - Creates strategy instances
2. **Strategy Selector** - Selects strategy based on regime
3. **Paper Trading Engine** - Executes strategy signals

## Configuration

Strategies are configured in:
- `config/base.yaml` - Base strategy parameters
- `config/paper_trading_config.json` - Strategy-specific settings

## Performance

Strategies are evaluated on:
- **Information Coefficient (IC)**: Signal quality
- **Sharpe Ratio**: Risk-adjusted returns
- **Win Rate**: Percentage of profitable trades
- **Maximum Drawdown**: Worst peak-to-trough decline

## Extending Strategies

To add a new strategy:
1. Create a class inheriting from `BaseStrategy`
2. Implement required methods
3. Register in `factory.py`
4. Add configuration parameters

