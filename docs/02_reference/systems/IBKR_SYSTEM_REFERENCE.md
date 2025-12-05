# IBKR System Reference

Complete reference for IBKR trading system.

## Architecture

Multi-layered safety and efficiency architecture:

### Safety & Guard Layer
- **PreTradeGuards**: Pre-trade safety checks
- **MarginGate**: Broker-truth margin simulation
- **ShortSaleGuard**: Short-sale compliance
- **RateLimiter**: API pacing and reconnection

### Decision & Ensemble Layer
- **ZooBalancer**: Normalize and blend model outputs
- **HorizonArbiter**: Cost-aware horizon selection
- **BarrierGates**: Entry/exit guards
- **ShortHorizonExecutionPolicy**: TIF and staged aggression

### Efficiency & Optimization Layer
- **UniversePrefilter**: Alpha proxies and cost filtering
- **NettingSuppression**: Prevent self-trading
- **DriftHealth**: Online drift detection
- **ModeController**: State machine (LIVE/DEGRADED/OBSERVE/EMERGENCY)

### Robustness & Recovery Layer
- **OrderReconciler**: Broker-of-record truth reconciliation
- **Disaster Recovery**: Flatten-all panic path
- **Trade Cost Analytics**: Post-trade analysis

## Configuration

### IBKR Connection

```yaml
ibkr:
  host: "127.0.0.1"
  port: 7497  # Paper: 7497, Live: 7496
  client_id: 1
```

### Trading Settings

```yaml
trading:
  symbols: ["AAPL", "MSFT", "GOOGL"]
  horizons: ["5m", "10m", "15m", "30m", "60m"]
  max_position_size: 100
```

### Safety Settings

```yaml
safety:
  max_portfolio_risk: 0.02
  stop_loss: 0.05
  take_profit: 0.10
  max_drawdown: 0.20
```

## C++ Components

High-performance C++ kernels for hot path operations:

- **InferenceEngine**: Model inference
- **FeaturePipeline**: Feature computation
- **MarketDataParser**: Market data parsing
- **LinearAlgebraEngine**: Linear algebra operations

See [C++ Engine README](../../../IBKR_trading/cpp_engine/README.md) for details.

## Usage

### Run Trading System

```bash
cd IBKR_trading
python run_trading_system.py
```

### Test Models

```bash
python test_daily_models.py
python test_cpp_components.py
```

## See Also

- [IBKR Integration Guide](../../01_tutorials/trading/IBKR_INTEGRATION.md) - Setup tutorial
- [IBKR Live Trading Integration](../../../IBKR_trading/LIVE_TRADING_INTEGRATION.md) - Integration details
- [IBKR Implementation Status](../../../IBKR_trading/IMPLEMENTATION_STATUS.md) - Current status

