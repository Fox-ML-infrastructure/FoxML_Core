# IBKR Trading Scripts Guide

Complete guide to using IBKR trading scripts.

## ⚠️ Status Warning

**Module Status**: ⚠️ **UNTESTED** - This module requires comprehensive testing before production use. Not recommended for live trading.

**Recommendation**: Focus on the TRAINING pipeline first. These scripts should be thoroughly tested before use.

## Overview

The IBKR trading module includes scripts in `IBKR_trading/scripts/` for health monitoring, performance tracking, and trading execution.

## Scripts

### `health_check.sh` - Health Check Script

Monitors system health and broker connection status.

**Features:**
- Broker connection verification
- Account status check
- System resource monitoring
- Alert generation

**Usage:**
```bash
./IBKR_trading/scripts/health_check.sh
```

**Location:** `IBKR_trading/scripts/health_check.sh`

### `performance_monitor.sh` - Performance Monitor Script

Monitors trading performance and generates reports.

**Features:**
- Performance metrics tracking
- P&L reporting
- Risk metrics calculation
- Report generation

**Usage:**
```bash
./IBKR_trading/scripts/performance_monitor.sh
```

**Location:** `IBKR_trading/scripts/performance_monitor.sh`

## Test Scripts

Test scripts in `IBKR_trading/tests/`:

### `test_live_integration.py` - Live Integration Tests

Tests live trading integration with Interactive Brokers.

**⚠️ Warning**: These tests interact with live broker. Use with caution.

**Location:** `IBKR_trading/tests/test_live_integration.py`

### `test_optimization_engine.py` - Optimization Engine Tests

Tests the C++ optimization engine integration.

**Location:** `IBKR_trading/tests/test_optimization_engine.py`

### `test_rotation_engine.py` - Rotation Engine Tests

Tests the rotation engine functionality.

**Location:** `IBKR_trading/tests/test_rotation_engine.py`

## Script Execution

All scripts can be run from the repository root:
```bash
# From repo root
./IBKR_trading/scripts/health_check.sh

# Or from IBKR_trading directory
cd IBKR_trading
./scripts/health_check.sh
```

## Environment Setup

Scripts require:
- Python 3.8+
- Interactive Brokers TWS or IB Gateway running
- IBKR API credentials configured
- Configuration files in `config/` directory
- C++ optimization components compiled (for optimization tests)

## Logging

Scripts output logs to:
- Console output
- Log files in `logs/` directory
- Performance reports in `reports/` directory

## Error Handling

All scripts include:
- Graceful error handling
- Connection retry logic
- Detailed error logging
- Recovery mechanisms

## Related Documentation

- [IBKR Configuration Guide](IBKR_CONFIGURATION.md) - Configuration files
- [Trading Modules Overview](TRADING_MODULES.md) - Complete guide
- [Live Trading Integration](../03_technical/trading/implementation/LIVE_TRADING_INTEGRATION.md) - Integration guide
- [IBKR Technical Docs](../03_technical/trading/) - Deep technical documentation

