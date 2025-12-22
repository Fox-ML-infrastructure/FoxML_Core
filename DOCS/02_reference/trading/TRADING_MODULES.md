# Trading Modules Overview

Complete guide to the ALPACA and IBKR trading modules in FoxML Core.

## ⚠️ Important Status Notice

**Current Focus**: The TRAINING pipeline is the primary focus of development. Trading modules are provided for reference but are not actively maintained.

**Module Status**:
- **ALPACA_trading**: ⚠️ **BROKEN** - Has issues and is not functional. Needs fixes before use.
- **IBKR_trading**: ⚠️ **UNTESTED** - Requires comprehensive testing before production use. Not recommended for live trading.

**Recommendation**: Focus on the TRAINING pipeline first. Trading modules should only be used after thorough testing and fixes.

## Overview

FoxML Core includes two trading execution modules for different brokerages:

1. **ALPACA_trading** - Paper trading and backtesting framework for Alpaca Markets
2. **IBKR_trading** - Production live trading system for Interactive Brokers

Both modules integrate with the TRAINING pipeline to execute trades based on trained models.

## ALPACA Trading Module

**Location**: `ALPACA_trading/`

**Status**: ⚠️ **BROKEN** - Has issues and is not functional. Needs fixes before use.

**Purpose**: Paper trading and backtesting framework

**⚠️ Warning**: This module is currently broken and should not be used until fixed.

**Features**:
- Alpaca Markets API integration
- yfinance data source support
- Paper trading engine with simulated execution
- Risk management and position sizing
- Multiple risk profiles (balanced, low, strict)
- Configuration-driven trading strategies

**Documentation**:
- [Configuration Guide](ALPACA_CONFIGURATION.md) - Configuration files and settings
- [Scripts Guide](ALPACA_SCRIPTS.md) - Script usage and execution
- [ALPACA Module README](../../../ALPACA_trading/config/README.md) - Module-level documentation

**Related**:
- [Trading Technical Docs](../03_technical/trading/README.md) - Deep technical documentation
- [Broker Integration Compliance](../../../LEGAL/BROKER_INTEGRATION_COMPLIANCE.md) - Legal framework

## IBKR Trading Module

**Location**: `IBKR_trading/`

**Status**: ⚠️ **UNTESTED** - Requires comprehensive testing before production use. Not recommended for live trading.

**Purpose**: Production live trading system for Interactive Brokers

**⚠️ Warning**: This module is untested and should not be used for live trading until thoroughly tested.

**Features**:
- Interactive Brokers API integration
- Multi-horizon trading system (5m, 10m, 15m, 30m, 60m)
- Comprehensive safety guards and risk management
- C++ optimization components for performance
- Live trading execution
- Health monitoring and performance tracking

**Documentation**:
- [IBKR Configuration Guide](IBKR_CONFIGURATION.md) - Configuration files and settings
- [IBKR Scripts Guide](IBKR_SCRIPTS.md) - Script usage and execution
- [IBKR Technical Docs](../03_technical/trading/) - Architecture, implementation, testing, operations

**Related**:
- [Mathematical Foundations](../03_technical/trading/architecture/MATHEMATICAL_FOUNDATIONS.md) - Mathematical equations for cost-aware ensemble trading
- [Optimization Architecture](../03_technical/trading/architecture/OPTIMIZATION_ARCHITECTURE.md) - C++ optimization system
- [Live Trading Integration](../03_technical/trading/implementation/LIVE_TRADING_INTEGRATION.md) - Integration guide
- [Broker Integration Compliance](../../../LEGAL/BROKER_INTEGRATION_COMPLIANCE.md) - Legal framework

## Integration with TRAINING Pipeline

Both trading modules integrate with the TRAINING pipeline:

1. **Model Loading**: Load trained models from `RESULTS/runs/` or `models/` directory
2. **Feature Extraction**: Use same feature engineering pipeline as training
3. **Prediction**: Generate predictions using trained models
4. **Execution**: Execute trades based on predictions and risk management rules

**Configuration**:
- Model paths specified in `config/models.yaml` (ALPACA) or `config/models.yaml` (IBKR)
- Feature lists must match training configuration
- Target columns must match training targets

## Compliance and Legal

**Important**: Both modules include comprehensive compliance framework:

- Non-advisory disclaimers (we do not provide investment advice)
- Non-custodial statements (user-owned accounts, user-provided API keys)
- User responsibility statements (brokerage, compliance, trading decisions)
- Regulatory compliance notices
- Risk warnings

See [Broker Integration Compliance](../../../LEGAL/BROKER_INTEGRATION_COMPLIANCE.md) for complete legal framework.

## Quick Start

**⚠️ Important**: Both trading modules have status issues. Focus on the TRAINING pipeline first.

### ALPACA Paper Trading

**⚠️ Status**: BROKEN - Module is not functional. Do not use until fixed.

```bash
# ⚠️ NOT RECOMMENDED - Module is broken
# Set up environment variables
cp ALPACA_trading/config/paper-trading.env .env
# Edit .env with your API keys

# Run paper trading (will likely fail - module is broken)
python ALPACA_trading/scripts/paper_runner.py \
    --symbols SPY,TSLA,AAPL \
    --profile risk_balanced \
    --config ALPACA_trading/config/paper_trading_config.json
```

### IBKR Live Trading

**⚠️ Status**: UNTESTED - Requires comprehensive testing before use. Not recommended for live trading.

```bash
# ⚠️ NOT RECOMMENDED - Module is untested
# Configure IBKR credentials in config/live_trading_config.yaml
# Run health check
python IBKR_trading/scripts/health_check.sh

# Start live trading (NOT RECOMMENDED - untested)
python IBKR_trading/scripts/start_live_trading.py
```

## Development Priority

**Current Focus**: TRAINING pipeline is the primary development priority.

- ✅ **TRAINING Pipeline**: Actively maintained and tested
- ⚠️ **Trading Modules**: Provided for reference, not actively maintained
- 🔧 **Recommendation**: Use TRAINING pipeline to train models. Trading modules should be fixed and tested separately before use.

## Related Documentation

- [Trading Technical Documentation](../03_technical/trading/README.md) - Deep technical docs
- [Broker Integration Compliance](../../../LEGAL/BROKER_INTEGRATION_COMPLIANCE.md) - Legal framework
- [ALPACA Configuration](ALPACA_CONFIGURATION.md) - ALPACA config guide
- [IBKR Configuration](IBKR_CONFIGURATION.md) - IBKR config guide
- [Trading Operations](../03_technical/trading/operations/) - Deployment and operations

