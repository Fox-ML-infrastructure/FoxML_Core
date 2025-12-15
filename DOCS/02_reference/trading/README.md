# Trading Modules Reference

This directory contains reference documentation for the trading modules in FoxML Core.

## Overview

FoxML Core includes two trading modules:
- **ALPACA_trading**: Paper trading and backtesting framework
- **IBKR_trading**: Production live trading system for Interactive Brokers

For a comprehensive overview and comparison, see [TRADING_MODULES.md](TRADING_MODULES.md).

## Directory Structure

```
trading/
├── alpaca/          # ALPACA trading module reference docs
└── ibkr/           # IBKR trading module reference docs
```

## ALPACA Trading

- Module location: `ALPACA_trading/`
- Primary use: Paper trading, backtesting, prototyping
- See: [ALPACA_trading/README.md](../../../ALPACA_trading/README.md)

## IBKR Trading

- Module location: `IBKR_trading/`
- Primary use: Live trading with real capital
- Reference docs:
  - [Live Trading Components](ibkr/LIVE_TRADING_README.md) - Core live trading components

For detailed technical documentation, see:
- Architecture: [`03_technical/trading/architecture/`](../../03_technical/trading/architecture/)
- Implementation: [`03_technical/trading/implementation/`](../../03_technical/trading/implementation/)
- Testing: [`03_technical/trading/testing/`](../../03_technical/trading/testing/)
- Operations: [`03_technical/trading/operations/`](../../03_technical/trading/operations/)

## Related Documentation

- [Trading Modules Overview](TRADING_MODULES.md) - Complete guide to both modules
- [IBKR Trading README](../../../IBKR_trading/README.md) - Module-specific README
- [ALPACA Trading README](../../../ALPACA_trading/README.md) - Module-specific README

---

**Copyright (c) 2025-2026 Fox ML Infrastructure LLC**
