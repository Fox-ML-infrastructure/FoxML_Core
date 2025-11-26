# Scripts

This directory contains executable scripts for running the ALPACA trading system.

## Components

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
    --config config/paper_trading_config.json
```

**Arguments:**
- `--symbols`: Comma-separated list of symbols to trade
- `--profile`: Risk profile (risk_balanced, risk_low, risk_strict)
- `--config`: Configuration file path
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR)

### `data/alpaca_batch.py` - Batch Data Fetching
Fetches historical data from Alpaca API in batches.

**Features:**
- Batch data retrieval
- Multiple symbols support
- Date range specification
- Data validation

**Usage:**
```python
from scripts.data.alpaca_batch import fetch_batch_data

data = fetch_batch_data(
    symbols=["SPY", "TSLA"],
    start_date="2024-01-01",
    end_date="2024-12-31"
)
```

### `data/alpaca_batch_optimized.py` - Optimized Batch Data Fetching
Optimized version with:
- Parallel data fetching
- Caching support
- Rate limit handling
- Error recovery

**Performance:** Faster than standard batch fetching for large symbol lists.

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

