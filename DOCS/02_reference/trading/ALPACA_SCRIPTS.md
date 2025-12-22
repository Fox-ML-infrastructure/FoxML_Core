# ALPACA Trading Scripts Guide

Complete guide to using ALPACA trading scripts.

## ⚠️ Status Warning

**Module Status**: ⚠️ **BROKEN** - This module has issues and is not functional. Needs fixes before use.

**Recommendation**: Focus on the TRAINING pipeline first. These scripts should not be used until the module is fixed.

## Overview

The ALPACA trading module includes executable scripts in `ALPACA_trading/scripts/` for running paper trading and data fetching.

## Scripts

### `paper_runner.py` - Main Paper Trading Runner

The main entry point for running paper trading.

**Features:**
- Command-line interface
- Configuration loading
- Trading engine initialization
- Continuous trading loop
- Error handling and recovery

**Usage:**
```bash
python ALPACA_trading/scripts/paper_runner.py \
    --symbols SPY,TSLA,AAPL \
    --profile risk_balanced \
    --config ALPACA_trading/config/paper_trading_config.json
```

**Arguments:**
- `--symbols`: Comma-separated list of symbols to trade
- `--profile`: Risk profile (risk_balanced, risk_low, risk_strict)
- `--config`: Configuration file path
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR)

**Location:** `ALPACA_trading/scripts/paper_runner.py`

## Script Execution

All scripts can be run from the repository root:
```bash
# From repo root
python ALPACA_trading/scripts/paper_runner.py [options]

# Or from ALPACA_trading directory
cd ALPACA_trading
python scripts/paper_runner.py [options]
```

## Environment Setup

Scripts require:
- Python 3.8+
- Required packages (see main README)
- Alpaca API credentials (environment variables)
- Configuration files in `config/` directory

## Logging

Scripts output logs to:
- Console (colored output)
- Log files in `logs/` directory
- Separate files for trades, performance, errors

## Error Handling

All scripts include:
- Graceful error handling
- Automatic retry logic
- Detailed error logging
- Recovery mechanisms

## Related Documentation

- [ALPACA Configuration Guide](ALPACA_CONFIGURATION.md) - Configuration files
- [Trading Modules Overview](TRADING_MODULES.md) - Complete guide
- [ALPACA Scripts README](../../../ALPACA_trading/scripts/README.md) - Module-level docs

