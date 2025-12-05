# IBKR Integration Guide

Set up Interactive Brokers (IBKR) for live trading.

## Overview

IBKR integration provides production-ready live trading with:
- Multi-horizon model blending (5m, 10m, 15m, 30m, 60m)
- Safety guards and risk management
- C++ performance optimization
- Real-time order execution

## Prerequisites

- IBKR account with API access enabled
- TWS (Trader Workstation) or IB Gateway running
- Trained models in `models/` directory
- Configuration file: `IBKR_trading/config/ibkr_enhanced.yaml`

## Setup Steps

### 1. Configure IBKR API

1. Enable API in TWS: Configure → API → Settings
2. Set port (default: 7497 for paper, 7496 for live)
3. Enable "Enable ActiveX and Socket Clients"
4. Add trusted IPs if needed

### 2. Configure System

Edit `IBKR_trading/config/ibkr_enhanced.yaml`:

```yaml
ibkr:
  host: "127.0.0.1"
  port: 7497  # Paper trading port
  client_id: 1

trading:
  symbols: ["AAPL", "MSFT", "GOOGL"]
  horizons: ["5m", "10m", "15m", "30m", "60m"]
  max_position_size: 100
  risk_limit: 0.02
```

### 3. Load Models

Ensure trained models are in `models/` directory:

```
models/
├── lightgbm_5m.pkl
├── lightgbm_10m.pkl
├── xgboost_15m.pkl
└── ...
```

### 4. Run Trading System

```bash
cd IBKR_trading
python run_trading_system.py
```

## System Architecture

### Safety Layer
- PreTradeGuards: Pre-trade safety checks
- MarginGate: Broker-truth margin simulation
- ShortSaleGuard: Short-sale compliance
- RateLimiter: API pacing and reconnection

### Decision Layer
- ZooBalancer: Normalize and blend model outputs
- HorizonArbiter: Cost-aware horizon selection
- BarrierGates: Entry/exit guards

### Efficiency Layer
- UniversePrefilter: Alpha proxies and cost filtering
- NettingSuppression: Prevent self-trading
- DriftHealth: Online drift detection

## Testing

### Daily Model Testing

```bash
cd IBKR_trading
python test_daily_models.py
```

### Comprehensive Testing

```bash
./test_all_models_comprehensive.sh
```

### C++ Component Testing

```bash
python test_cpp_components.py
```

## Monitoring

### Check System Status

```bash
python IBKR_trading/scripts/check_status.py
```

### View Positions

Check TWS or use IBKR API to query positions.

### Review Logs

```bash
tail -f logs/ibkr_trading.log
```

## Best Practices

1. **Start with Paper Trading**: Test thoroughly before live
2. **Monitor Closely**: Watch for unexpected behavior
3. **Use Safety Guards**: Never disable safety checks
4. **Validate Models**: Ensure models are recent and accurate

## Next Steps

- [IBKR System Reference](../../02_reference/systems/IBKR_SYSTEM_REFERENCE.md) - Complete reference
- [IBKR Live Trading Integration](../../../IBKR_trading/LIVE_TRADING_INTEGRATION.md) - Detailed integration
- [IBKR Daily Testing](../../../IBKR_trading/DAILY_TESTING_README.md) - Testing procedures

