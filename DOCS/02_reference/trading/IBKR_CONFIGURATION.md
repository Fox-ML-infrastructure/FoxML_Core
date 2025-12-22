# IBKR Trading Configuration Guide

Complete guide to configuring the IBKR trading module.

## ⚠️ Status Warning

**Module Status**: ⚠️ **UNTESTED** - This module requires comprehensive testing before production use. Not recommended for live trading.

**Recommendation**: Focus on the TRAINING pipeline first. This module should be thoroughly tested before use.

## Overview

The IBKR trading module uses YAML configuration files in `IBKR_trading/config/` for different trading modes and scenarios.

## Configuration Files

### `ibkr_live.yaml` - Live Trading Configuration

Main configuration for live trading with Interactive Brokers:
- Broker connection settings
- Account information
- Trading parameters
- Risk limits
- Model paths and settings

**⚠️ Warning**: Live trading configuration. Test thoroughly before use.

**Location:** `IBKR_trading/config/ibkr_live.yaml`

### `ibkr_enhanced.yaml` - Enhanced Trading Configuration

Enhanced configuration with additional features:
- Advanced risk management
- Multi-horizon settings
- Performance optimization
- Monitoring and alerting

**Location:** `IBKR_trading/config/ibkr_enhanced.yaml`

### `ibkr_daily_test.yaml` - Daily Testing Configuration

Configuration for daily testing and validation:
- Test symbols
- Reduced position sizes
- Enhanced logging
- Performance tracking

**Location:** `IBKR_trading/config/ibkr_daily_test.yaml`

### `live_trading_config.yaml` - Live Trading Settings

Alternative live trading configuration format.

**Location:** `IBKR_trading/config/live_trading_config.yaml`

## Configuration Structure

All IBKR configuration files follow this structure:

```yaml
broker:
  host: "127.0.0.1"
  port: 7497
  client_id: 1

account:
  account_id: "YOUR_ACCOUNT_ID"

trading:
  symbols: ["SPY", "QQQ", "IWM"]
  horizons: [5, 10, 15, 30, 60]  # minutes
  position_sizing:
    max_position_size: 0.1
    max_portfolio_risk: 0.02

models:
  - name: "model_5m"
    path: "models/model_5m.pkl"
    horizon: 5
    features: ["feature1", "feature2"]
    target: "fwd_ret_5m"

risk_management:
  max_drawdown: 0.10
  stop_loss: 0.02
  take_profit: 0.05
```

## Multi-Horizon Trading

IBKR module supports trading across multiple time horizons:
- 5-minute bars
- 10-minute bars
- 15-minute bars
- 30-minute bars
- 60-minute bars

Each horizon can have:
- Separate model
- Separate feature set
- Separate risk parameters

## Safety Guards

All configurations include safety guards:
- Maximum position size limits
- Portfolio risk limits
- Drawdown protection
- Stop loss and take profit levels
- Circuit breakers

## Testing

**⚠️ Important**: Always test configurations in paper trading mode before live trading.

Use `ibkr_daily_test.yaml` for daily validation and testing.

## Related Documentation

- [IBKR Scripts Guide](IBKR_SCRIPTS.md) - Script usage
- [Trading Modules Overview](TRADING_MODULES.md) - Complete guide
- [Live Trading Integration](../03_technical/trading/implementation/LIVE_TRADING_INTEGRATION.md) - Integration guide
- [IBKR Technical Docs](../03_technical/trading/) - Deep technical documentation

