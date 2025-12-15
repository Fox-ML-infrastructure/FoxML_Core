# Trading Modules Overview

This document explains the two main trading modules in the FoxML Core repository: `ALPACA_trading` and `IBKR_trading`. Both modules provide complete trading infrastructure but are designed for different use cases and broker integrations.

> **⚠️ IMPORTANT LEGAL NOTICE**: FoxML Core provides **client-side trading execution software** that connects to **user-owned brokerage accounts** via **user-provided API keys**. We are **NOT a broker, investment advisor, or custodian**. Users are solely responsible for brokerage relationships, regulatory compliance, API credentials, and all trading decisions. See [`LEGAL/BROKER_INTEGRATION_COMPLIANCE.md`](LEGAL/BROKER_INTEGRATION_COMPLIANCE.md) for complete compliance framework.

## Table of Contents

1. [Overview](#overview)
2. [ALPACA_trading Module](#alpaca_trading-module)
3. [IBKR_trading Module](#ibkr_trading-module)
4. [Key Differences](#key-differences)
5. [When to Use Which Module](#when-to-use-which-module)
6. [Shared Components](#shared-components)
7. [Migration Guide](#migration-guide)

---

## Overview

The FoxML Core trading infrastructure consists of two independent but complementary modules:

- **ALPACA_trading**: A paper trading and backtesting framework optimized for Alpaca Markets API, with support for yfinance data sources. Designed for rapid prototyping, testing, and paper trading.
  - ⚠️ **Status**: Has minor issues — needs testing and fixes before production use

- **IBKR_trading**: A production-ready live trading system for Interactive Brokers (IBKR), featuring comprehensive safety guards, multi-horizon model ensembles, and C++ optimization components. Designed for live trading with real capital.
  - ⚠️ **Status**: Untested — requires comprehensive testing before live trading. Not recommended for production use until tested.

Both modules share the same underlying ML models, feature engineering pipeline, and strategy components from the main `TRAINING/` directory, but differ in their execution environments, broker integrations, and operational focus.

**⚠️ Important**: Both modules require testing before production use. Users should test thoroughly in paper trading environments first.

---

## ALPACA_trading Module

### Purpose

The `ALPACA_trading` module is designed for:
- **Paper trading** and simulation
- **Backtesting** strategies before live deployment
- **Rapid prototyping** of new trading ideas
- **Educational** and research use cases
- **Low-cost testing** without broker account requirements

### Key Features

1. **Paper Trading Engine**
   - Simulated order execution with configurable slippage and fees
   - Position tracking and portfolio management
   - Trade history logging and performance metrics

2. **Data Sources**
   - **Primary**: Alpaca Markets API (paper trading account)
   - **Fallback**: yfinance (free, no API key required)
   - **Optional**: IBKR data provider (for data only, no execution)

3. **Core Components**
   - `core/paper.py`: Main paper trading engine
   - `brokers/paper.py`: Simulated broker implementation
   - `brokers/interface.py`: Broker abstraction layer
   - `core/regime_detector.py`: Market regime detection
   - `strategies/regime_aware_ensemble.py`: Adaptive strategy selection
   - `core/performance.py`: Performance tracking and metrics

4. **Risk Management**
   - Basic position sizing and risk limits
   - Kill switches for maximum loss protection
   - Drawdown monitoring
   - Optional advanced guardrails (`core/risk/guardrails.py`)

5. **Notifications**
   - Discord webhook integration
   - Trade and performance alerts
   - Daily summary reports

### Architecture

```
ALPACA_trading/
├── brokers/          # Broker interfaces and implementations
│   ├── interface.py  # Broker protocol definition
│   ├── paper.py      # Paper trading broker
│   ├── data_provider.py  # Market data provider
│   └── ibkr_broker.py    # IBKR data-only integration
├── core/             # Core trading logic
│   ├── paper.py     # Main paper trading engine
│   ├── regime_detector.py
│   ├── strategy_selector.py
│   ├── performance.py
│   └── risk/guardrails.py
├── strategies/       # Trading strategies
│   ├── factory.py
│   └── regime_aware_ensemble.py
├── ml/              # ML model integration
│   ├── model_interface.py
│   ├── registry.py
│   └── runtime.py
├── cli/             # Command-line interface
│   └── paper.py
└── tools/           # Utility tools
    └── provenance.py
```

### Usage

**Quick Start:**
```bash
# Run paper trading with default configuration
python ALPACA_trading/cli/paper.py --symbols SPY,QQQ --profile risk_balanced

# Or use the main runner
python ALPACA_trading/scripts/paper_runner.py --symbols SPY,TSLA
```

**Configuration:**
- Configuration files in `ALPACA_trading/config/`
- Supports profile-based configuration (risk_low, risk_balanced, risk_strict)
- Environment variables for API keys: `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`

**Dependencies:**
- `alpaca-trade-api` or `alpaca-py` (optional, for Alpaca integration)
- `yfinance` (for free market data)
- `pandas`, `numpy` (data processing)
- Standard ML libraries (for model inference)

### Advantages

✅ **Easy Setup**: No broker account required (can use yfinance)
✅ **Fast Iteration**: Quick backtesting and paper trading cycles
✅ **Low Cost**: Free paper trading accounts available
✅ **Flexible**: Supports multiple data sources
✅ **Educational**: Great for learning and experimentation

### Limitations

⚠️ **Simulation Only**: Not designed for live trading with real money
⚠️ **Simplified Execution**: Paper trading doesn't capture real market microstructure
⚠️ **Limited Safety Features**: Basic risk management compared to IBKR module
⚠️ **No Real Market Impact**: Orders don't affect real markets

---

## IBKR_trading Module

### Purpose

The `IBKR_trading` module is designed for:
- **Live trading** with real capital
- **Production deployment** in institutional environments
- **High-frequency intraday trading** with multiple time horizons
- **Institutional-grade safety** and compliance
- **Advanced optimization** with C++ components

### Key Features

1. **Multi-Horizon Trading System**
   - Supports 5 time horizons: 5m, 10m, 15m, 30m, 60m
   - 16+ machine learning models across horizons
   - Horizon-aware position sizing and risk management

2. **Comprehensive Safety Layer**
   - **PreTradeGuards**: Pre-trade validation and safety checks
   - **MarginGate**: Real-time margin simulation using IBKR's what-if API
   - **ShortSaleGuard**: Short sale compliance (borrow checks, SSR enforcement)
   - **RateLimiter**: API rate limiting and automatic reconnection
   - **Fail-closed design**: No orders if any guard fails

3. **Advanced Decision Making**
   - **ZooBalancer**: Normalize and blend model outputs within horizons
   - **HorizonArbiter**: Cost-aware selection across horizons
   - **BarrierGates**: Microstructure protection using barrier targets
   - **Online Learning**: Exp3-IX bandit for adaptive model weighting

4. **Optimization & Efficiency**
   - **C++ Inference Engine**: High-performance model inference
   - **UniversePrefilter**: Cheap alpha proxies to avoid expensive inference
   - **NettingSuppression**: Prevent self-trading across horizons
   - **DriftHealth**: Online model performance monitoring and quarantine
   - **Feature Caching**: Incremental feature updates

5. **Robustness & Recovery**
   - **OrderReconciler**: Broker-of-record truth reconciliation
   - **Disaster Recovery**: Emergency flatten procedures
   - **State Recovery**: WAL replay and state reconstruction
   - **Mode Controller**: LIVE/DEGRADED/OBSERVE/EMERGENCY state machine

6. **Performance Targets**
   - Decision latency: < 350ms (p99)
   - Order routing: < 2s (p99)
   - Model inference: < 100ms (p95)

### Architecture

```
IBKR_trading/
├── live_trading/          # Core live trading components
│   ├── main_loop.py       # Main trading orchestrator
│   ├── barrier_gate.py    # Entry/exit guards
│   ├── cost_arbitrator.py # Cost-aware decision making
│   ├── horizon_blender.py # Multi-horizon blending
│   ├── model_predictor.py # Model inference
│   └── position_sizer.py  # Dynamic position sizing
├── cpp_engine/            # C++ optimization components
│   ├── src/               # C++ source files
│   └── python_bindings/   # Python bindings
├── brokers/               # IBKR broker integration
│   └── __init__.py
├── optimization/          # Optimization algorithms
├── tests/                 # Comprehensive test suite
└── deprecated/            # Legacy components
```

### Usage

**Prerequisites:**
- IBKR TWS (Trader Workstation) or Gateway running
- Active IBKR account with API access enabled
- Trained models in `TRAINING/models/` directory
- Configuration file: `CONFIG/ibkr_enhanced.yaml`

**Quick Start:**
```bash
# Navigate to IBKR trading directory
cd IBKR_trading

# Run the trading system
python run_trading_system.py
```

**Configuration:**
- All settings in `CONFIG/ibkr_enhanced.yaml`
- Key sections:
  - **Safety**: Market integrity, time windows, risk limits
  - **IBKR Connection**: Host, port, account settings
  - **Models**: Model families, horizons, inference settings
  - **Execution**: Order types, TIF, bracket settings
  - **Risk Management**: Position limits, kill switches

**Emergency Procedures:**
```bash
# Emergency stop (flattens all positions)
touch IBKR_trading/panic.flag

# System automatically:
# 1. Stops accepting new orders
# 2. Cancels all pending orders
# 3. Flattens all positions
# 4. Enters EMERGENCY mode
```

### Advantages

✅ **Production Ready**: Designed for live trading with real money
✅ **Comprehensive Safety**: Multiple layers of guards and checks
✅ **High Performance**: C++ components for low-latency execution
✅ **Institutional Grade**: Broker reconciliation, disaster recovery
✅ **Multi-Horizon**: Sophisticated ensemble across time horizons
✅ **Cost-Aware**: Explicit transaction cost modeling

### Limitations

⚠️ **Complex Setup**: Requires IBKR account and TWS/Gateway
⚠️ **Higher Barrier**: More configuration and operational overhead
⚠️ **Real Money Risk**: Designed for live trading (use with caution)
⚠️ **Resource Intensive**: C++ compilation and optimization required

---

## Key Differences

| Aspect | ALPACA_trading | IBKR_trading |
|--------|----------------|--------------|
| **Primary Use Case** | Paper trading, backtesting | Live trading, production |
| **Broker** | Alpaca Markets (paper) or yfinance | Interactive Brokers |
| **Execution** | Simulated | Real market orders |
| **Safety Features** | Basic risk limits | Comprehensive guards, margin checks |
| **Performance** | Python-only | C++ optimization components |
| **Time Horizons** | Single or simple multi-horizon | 5 horizons (5m-60m) |
| **Model Ensemble** | Basic regime-aware | Advanced multi-horizon blending |
| **Cost Modeling** | Simple slippage/fees | Detailed transaction cost analysis |
| **State Management** | In-memory | Persistent with WAL |
| **Disaster Recovery** | Basic | Comprehensive (panic mode, recovery) |
| **Setup Complexity** | Low (can use free data) | High (requires IBKR account) |
| **Best For** | Learning, prototyping, testing | Production live trading |

---

## When to Use Which Module

### Use ALPACA_trading When:

- ✅ You're learning or prototyping new strategies
- ✅ You want to backtest without broker setup
- ✅ You're testing with paper trading accounts
- ✅ You need quick iteration cycles
- ✅ You don't have an IBKR account
- ✅ You want to use free data sources (yfinance)
- ✅ You're doing research or education

### Use IBKR_trading When:

- ✅ You're ready for live trading with real capital
- ✅ You need institutional-grade safety and compliance
- ✅ You want multi-horizon trading (5m-60m)
- ✅ You require high-performance execution
- ✅ You need comprehensive risk management
- ✅ You have an IBKR account and TWS/Gateway
- ✅ You're deploying to production

### Migration Path

A typical workflow:

1. **Develop & Test** → Use `ALPACA_trading` for backtesting and paper trading
2. **Validate Strategy** → Test with paper trading on Alpaca
3. **Production Deploy** → Migrate to `IBKR_trading` for live trading

---

## Shared Components

Both modules share:

1. **ML Models**: From `TRAINING/models/` directory
2. **Feature Engineering**: From `DATA_PROCESSING/features/`
3. **Data Pipeline**: From `DATA_PROCESSING/pipeline/`
4. **Configuration System**: YAML-based configuration in `CONFIG/`
5. **Core Utilities**: Common utilities and helpers

The modules are designed to be independent but can share:
- Model artifacts
- Configuration profiles
- Feature definitions
- Strategy implementations (with broker-specific adaptations)

---

## Migration Guide

### From ALPACA_trading to IBKR_trading

If you've developed a strategy in `ALPACA_trading` and want to deploy it live:

1. **Validate Strategy Performance**
   - Ensure strategy performs well in paper trading
   - Review risk metrics and drawdowns
   - Test across different market regimes

2. **Prepare IBKR Environment**
   - Set up IBKR TWS or Gateway
   - Configure API access
   - Test connection and data feeds

3. **Adapt Configuration**
   - Copy relevant config from `ALPACA_trading/config/`
   - Adapt to `CONFIG/ibkr_enhanced.yaml` format
   - Set appropriate risk limits and safety parameters

4. **Test in IBKR Paper Trading**
   - Use IBKR's paper trading account first
   - Verify all safety guards work correctly
   - Monitor performance and execution quality

5. **Gradual Live Deployment**
   - Start with small position sizes
   - Monitor closely for first few days
   - Gradually scale up as confidence increases

### Key Configuration Differences

**ALPACA_trading:**
```yaml
# Simple configuration
symbols: [SPY, QQQ]
initial_capital: 100000
max_position_size: 0.1
```

**IBKR_trading:**
```yaml
# Comprehensive configuration
ibkr:
  host: 127.0.0.1
  port: 7497
  account: DU123456
safety:
  max_daily_loss_pct: 2.0
  margin_buffer: 0.15
models:
  horizons: [5m, 10m, 15m, 30m, 60m]
  families: [ensemble_v1, ensemble_v2]
```

---

## Summary

- **ALPACA_trading**: Best for learning, prototyping, and paper trading. Easy setup, flexible, great for rapid iteration.

- **IBKR_trading**: Best for production live trading. Comprehensive safety, high performance, institutional-grade features.

Both modules are part of the FoxML Core infrastructure and share the same underlying ML models and strategies. Choose based on your use case: development and testing → ALPACA_trading; production deployment → IBKR_trading.

---

## Additional Resources

### Module Documentation
- **ALPACA_trading README**: [`ALPACA_trading/README.md`](ALPACA_trading/README.md)
- **IBKR_trading README**: [`IBKR_trading/README.md`](IBKR_trading/README.md)

### Trading Documentation
- **Trading Reference**: [`DOCS/02_reference/trading/README.md`](DOCS/02_reference/trading/README.md)
- **Trading Technical Docs**: [`DOCS/03_technical/trading/README.md`](DOCS/03_technical/trading/README.md)
  - Architecture: [`DOCS/03_technical/trading/architecture/`](DOCS/03_technical/trading/architecture/)
  - Implementation: [`DOCS/03_technical/trading/implementation/`](DOCS/03_technical/trading/implementation/)
  - Testing: [`DOCS/03_technical/trading/testing/`](DOCS/03_technical/trading/testing/)
  - Operations: [`DOCS/03_technical/trading/operations/`](DOCS/03_technical/trading/operations/)

### General Documentation
- **Configuration Guide**: [`DOCS/01_tutorials/configuration/`](DOCS/01_tutorials/configuration/)
- **Training Guide**: [`DOCS/01_tutorials/training/`](DOCS/01_tutorials/training/)
- **Architecture Docs**: [`DOCS/02_reference/`](DOCS/02_reference/)
- **Documentation Index**: [`DOCS/INDEX.md`](DOCS/INDEX.md)

---

**Copyright (c) 2025-2026 Fox ML Infrastructure LLC**

For licensing information, see `LICENSE` and `DUAL_LICENSE.md`.
