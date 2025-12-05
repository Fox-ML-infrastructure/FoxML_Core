# Alpaca System Reference

Complete reference for Alpaca paper trading system.

## Architecture

### Core Components

- **paper.py**: Core paper trading engine
- **regime_detector.py**: Market regime detection
- **strategy_selector.py**: Strategy selection
- **performance.py**: Performance tracking
- **risk/guardrails.py**: Risk management

### Broker Integration

- **paper.py**: Alpaca paper broker
- **data_provider.py**: Market data provider
- **interface.py**: Broker interface

### ML Integration

- **model_interface.py**: Model prediction interface
- **registry.py**: Model registry
- **runtime.py**: Model runtime

## Configuration

### Alpaca API

```yaml
alpaca:
  api_key: "${ALPACA_API_KEY}"
  secret_key: "${ALPACA_SECRET_KEY}"
  base_url: "https://paper-api.alpaca.markets"
```

### Trading Settings

```yaml
trading:
  symbols: ["AAPL", "MSFT", "GOOGL"]
  max_position_size: 1000
```

### Risk Settings

```yaml
risk:
  max_portfolio_risk: 0.02
  stop_loss: 0.05
  take_profit: 0.10
```

## Usage

### Run Paper Trading

```bash
python ALPACA_trading/scripts/paper_runner.py
```

### CLI Commands

```bash
# Status
python ALPACA_trading/cli/paper.py status

# Positions
python ALPACA_trading/cli/paper.py positions

# Performance
python ALPACA_trading/cli/paper.py performance
```

## Strategies

### Regime-Aware Ensemble

Selects strategies based on market regime:

```python
from ALPACA_trading.strategies.regime_aware_ensemble import RegimeAwareEnsemble

strategy = RegimeAwareEnsemble(config)
```

## See Also

- [Alpaca Integration Guide](../../01_tutorials/trading/ALPACA_INTEGRATION.md) - Setup tutorial
- [Paper Trading Setup](../../01_tutorials/trading/PAPER_TRADING_SETUP.md) - Setup guide
- [Alpaca README](../../../ALPACA_trading/README.md) - System overview

