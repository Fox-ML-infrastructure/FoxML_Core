# Alpaca Integration Guide

Complete guide to integrating with Alpaca paper trading.

## Overview

Alpaca integration provides paper trading capabilities with:
- Regime-aware ensemble strategies
- Risk management and guardrails
- Performance tracking
- Model integration

## Architecture

### Core Components

- **paper.py**: Core paper trading engine
- **regime_detector.py**: Market regime detection
- **strategy_selector.py**: Strategy selection based on regime
- **performance.py**: Performance tracking
- **risk/guardrails.py**: Risk management

### Broker Integration

- **paper.py**: Alpaca paper broker implementation
- **data_provider.py**: Market data provider
- **interface.py**: Broker interface abstraction

### ML Integration

- **model_interface.py**: Model prediction interface
- **registry.py**: Model registry
- **runtime.py**: Model runtime execution

## Setup

### 1. Install Dependencies

```bash
pip install alpaca-trade-api pandas numpy
```

### 2. Configure API Keys

Set environment variables:

```bash
export ALPACA_API_KEY="your_key"
export ALPACA_SECRET_KEY="your_secret"
```

### 3. Configure System

Edit `ALPACA_trading/config/base.yaml`:

```yaml
alpaca:
  api_key: "${ALPACA_API_KEY}"
  secret_key: "${ALPACA_SECRET_KEY}"
  base_url: "https://paper-api.alpaca.markets"

trading:
  symbols: ["AAPL", "MSFT", "GOOGL"]
  max_position_size: 1000
```

### 4. Run Paper Trading

```bash
python ALPACA_trading/scripts/paper_runner.py
```

## Usage

### Basic Trading

```python
from ALPACA_trading.core.engine.paper import PaperTradingEngine
from ALPACA_trading.brokers.paper import PaperBroker

broker = PaperBroker()
engine = PaperTradingEngine(broker)

engine.run()
```

### With Custom Strategy

```python
from ALPACA_trading.strategies.regime_aware_ensemble import RegimeAwareEnsemble

strategy = RegimeAwareEnsemble(config)
engine = PaperTradingEngine(broker, strategy=strategy)
engine.run()
```

## Configuration

### Risk Settings

```yaml
risk:
  max_position_size: 1000
  max_portfolio_risk: 0.02
  stop_loss: 0.05
  take_profit: 0.10
```

### Strategy Settings

```yaml
strategy:
  type: "regime_aware_ensemble"
  models: ["lightgbm", "xgboost"]
  regime_detector: "basic_kpis"
```

## Monitoring

### CLI Commands

```bash
# Check status
python ALPACA_trading/cli/paper.py status

# View positions
python ALPACA_trading/cli/paper.py positions

# View performance
python ALPACA_trading/cli/paper.py performance
```

## Next Steps

- [Alpaca System Reference](../../02_reference/systems/ALPACA_SYSTEM_REFERENCE.md) - Complete reference
- [Paper Trading Setup](PAPER_TRADING_SETUP.md) - Setup guide

