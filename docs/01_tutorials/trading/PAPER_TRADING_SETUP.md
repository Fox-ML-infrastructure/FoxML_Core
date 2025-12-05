# Paper Trading Setup

Set up Alpaca paper trading for testing strategies without real money.

## Overview

Alpaca paper trading provides a risk-free environment to test trading strategies using simulated market data and orders.

## Prerequisites

- Alpaca account (free paper trading account)
- API keys (paper trading keys)
- Python environment with dependencies

## Setup Steps

### 1. Get Alpaca API Keys

1. Sign up at [Alpaca Markets](https://alpaca.markets)
2. Navigate to Paper Trading section
3. Generate API keys (API Key ID and Secret Key)

### 2. Configure Environment

Create `.env` file or set environment variables:

```bash
export ALPACA_API_KEY="your_api_key_id"
export ALPACA_SECRET_KEY="your_secret_key"
export ALPACA_BASE_URL="https://paper-api.alpaca.markets"
```

### 3. Configure System

Edit `ALPACA_trading/config/paper_trading_config.json`:

```json
{
  "alpaca": {
    "api_key": "${ALPACA_API_KEY}",
    "secret_key": "${ALPACA_SECRET_KEY}",
    "base_url": "https://paper-api.alpaca.markets"
  },
  "trading": {
    "symbols": ["AAPL", "MSFT", "GOOGL"],
    "max_position_size": 1000,
    "risk_limit": 0.02
  }
}
```

### 4. Run Paper Trading

```bash
cd ALPACA_trading
python scripts/paper_runner.py
```

## Configuration Options

### Risk Management

```json
{
  "risk": {
    "max_position_size": 1000,
    "max_portfolio_risk": 0.02,
    "stop_loss": 0.05,
    "take_profit": 0.10
  }
}
```

### Strategy Selection

```json
{
  "strategy": {
    "type": "regime_aware_ensemble",
    "models": ["lightgbm", "xgboost", "ensemble"],
    "regime_detector": "basic_kpis"
  }
}
```

## Monitoring

### Check Status

```bash
python ALPACA_trading/cli/paper.py status
```

### View Positions

```bash
python ALPACA_trading/cli/paper.py positions
```

### View Performance

```bash
python ALPACA_trading/cli/paper.py performance
```

## Best Practices

1. **Start Small**: Test with small position sizes first
2. **Monitor Closely**: Watch for unexpected behavior
3. **Log Everything**: Review logs for debugging
4. **Validate Models**: Ensure models are trained and loaded correctly

## Next Steps

- [Alpaca Integration](ALPACA_INTEGRATION.md) - Detailed integration guide
- [Alpaca System Reference](../../02_reference/systems/ALPACA_SYSTEM_REFERENCE.md) - Complete reference

