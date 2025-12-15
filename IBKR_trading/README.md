# IBKR Trading System

A comprehensive, production-ready framework for intraday trading with 16 machine learning models across 5 horizons (5m, 10m, 15m, 30m, 60m).

> **Note**: For a comprehensive comparison of ALPACA_trading and IBKR_trading modules, see the main [TRADING_MODULES.md](../DOCS/02_reference/trading/TRADING_MODULES.md) documentation.

## ️ Architecture Overview

The system implements a multi-layered safety and efficiency architecture:

### **Safety & Guard Layer**
- **PreTradeGuards**: Comprehensive pre-trade safety checks
- **MarginGate**: Broker-truth margin simulation using IBKR's what-if functionality
- **ShortSaleGuard**: Short-sale compliance including borrow/SSR enforcement
- **RateLimiter**: IB API pacing and automatic reconnection with backoff

### **Decision & Ensemble Layer**
- **ZooBalancer**: Normalize, calibrate, and blend model outputs within horizons
- **HorizonArbiter**: Cost-aware selection and blending across horizons
- **BarrierGates**: Entry/exit guards using barrier targets for microstructure protection
- **ShortHorizonExecutionPolicy**: TIF and staged aggression for short-horizon orders

### **Efficiency & Optimization Layer**
- **UniversePrefilter**: Cheap alpha proxies and cost filtering
- **NettingSuppression**: Intelligent netting across horizons to prevent self-trading
- **DriftHealth**: Online drift detection and automatic model quarantine
- **ModeController**: LIVE/DEGRADED/OBSERVE/EMERGENCY state machine

### **Robustness & Recovery Layer**
- **OrderReconciler**: Broker-of-record truth reconciliation with idempotency keys
- **Disaster Recovery**: Flatten-all panic path and state recovery mechanisms
- **Trade Cost Analytics**: Post-trade analysis and threshold autotuning

## Quick Start

### **Prerequisites**
- Python 3.11+
- IBKR TWS or Gateway running
- Trained models in `models/` directory
- Configuration file: `config/ibkr_enhanced.yaml`

### **Running the System**

```bash
# Navigate to the trading directory
cd IBKR_trading

# Run the trading system
python run_trading_system.py
```

### **Configuration**

The system uses `config/ibkr_enhanced.yaml` for all configuration. Key sections:

- **Safety**: Market integrity, time windows, risk limits
- **IBKR Connection**: Host, port, account settings
- **Models**: Model families, horizons, inference settings
- **Execution**: Order types, TIF, bracket settings
- **Risk Management**: Position limits, kill switches

## System Components

### **Main Execution System**
- `live_trading/ibkr_trading_system.py` - Main orchestrator
- `run_trading_system.py` - Entry point script

### **Safety Components**
- `live_trading/pre_trade_guards.py` - Pre-trade safety checks
- `live_trading/margin_gate.py` - Margin simulation
- `live_trading/short_sale_guard.py` - Short-sale compliance
- `live_trading/rate_limiter.py` - API pacing and reconnection

### **Decision Components**
- `live_trading/zoo_balancer.py` - Model ensemble balancing
- `live_trading/horizon_arbiter.py` - Cross-horizon arbitration
- `live_trading/barrier_gates.py` - Barrier target gating
- `live_trading/short_horizon_execution.py` - Execution policy

### **Optimization Components**
- `live_trading/universe_prefilter.py` - Symbol pre-filtering
- `live_trading/netting_suppression.py` - Churn suppression
- `live_trading/drift_health.py` - Model health monitoring
- `live_trading/mode_controller.py` - System mode management

### **Robustness Components**
- `live_trading/order_reconciler.py` - Order reconciliation
- `config/ibkr_enhanced.yaml` - Comprehensive configuration

## Key Features

### **Safety First**
- **Fail-Closed Design**: No orders if any guard fails
- **Broker-Truth Reconciliation**: Idempotency and state management
- **Margin Simulation**: Pre-trade what-if margin checks
- **Short Sale Compliance**: Borrow/SSR enforcement

### **Cost-Aware Decision Making**
- **Two-Stage Ensemble**: Within-horizon → across-horizon blending
- **Correlation-Penalized Weights**: Ridge risk-parity with temperature compression
- **Barrier-Gated Decisions**: Microstructure protection
- **Online Learning**: Exp3-IX bandit for adaptive weighting

### **Efficiency & Performance**
- **Pre-Filtering**: Cheap alpha proxies prevent expensive inference
- **Netting**: Intelligent netting prevents self-trading
- **Churn Suppression**: Hysteresis prevents position thrashing
- **Feature Caching**: Incremental updates and efficient reuse

### **Robustness & Recovery**
- **Crash Recovery**: WAL replay and state reconstruction
- **Model Health**: Automatic quarantine of underperforming models
- **Mode Transitions**: Adaptive behavior based on system health
- **Disaster Recovery**: Emergency flatten and panic procedures

## Performance Targets

### **Latency SLOs**
- Decision latency: < 350ms (p99)
- Order routing: < 2s (p99)
- Model inference: < 100ms (p95)

### **Safety Targets**
- Zero guard violations
- Zero orphan orders
- Zero margin violations
- Zero SSR violations

### **Efficiency Targets**
- Net P&L improvement after costs
- Reduced slippage vs baseline
- Lower churn ratio
- Higher fill rates

## ️ Development

### **Adding New Components**
1. Create component in appropriate directory
2. Add configuration to `config/ibkr_enhanced.yaml`
3. Wire into `ibkr_trading_system.py`
4. Add tests in `tests/` directory

### **Configuration Management**
- All settings in `config/ibkr_enhanced.yaml`
- Environment-specific overrides supported
- Real-time parameter updates
- Mode-specific configurations

### **Monitoring & Observability**
- Comprehensive logging to `logs/ibkr_trading.log`
- Performance metrics tracking
- Health monitoring and alerts
- Audit trails for compliance

## Emergency Procedures

### **Emergency Stop**
```bash
# Create panic file
touch panic.flag

# System will automatically:
# 1. Stop accepting new orders
# 2. Cancel all pending orders
# 3. Flatten all positions
# 4. Enter EMERGENCY mode
```

### **Manual Recovery**
```bash
# Remove panic file
rm panic.flag

# Restart system
python run_trading_system.py
```

## Documentation

> **Note**: All IBKR trading documentation has been moved to the main `DOCS/` directory. See [TRADING_MODULES.md](../DOCS/02_reference/trading/TRADING_MODULES.md) for an overview of both trading modules.

### Architecture & Design
- **Mathematical Foundations**: [`DOCS/03_technical/trading/architecture/MATHEMATICAL_FOUNDATIONS.md`](../DOCS/03_technical/trading/architecture/MATHEMATICAL_FOUNDATIONS.md)
- **Optimization Architecture**: [`DOCS/03_technical/trading/architecture/OPTIMIZATION_ARCHITECTURE.md`](../DOCS/03_technical/trading/architecture/OPTIMIZATION_ARCHITECTURE.md)
- **Optimization Engine Analysis**: [`DOCS/03_technical/trading/architecture/OPTIMIZATION_ENGINE_ANALYSIS.md`](../DOCS/03_technical/trading/architecture/OPTIMIZATION_ENGINE_ANALYSIS.md)

### Implementation
- **Live Trading Integration**: [`DOCS/03_technical/trading/implementation/LIVE_TRADING_INTEGRATION.md`](../DOCS/03_technical/trading/implementation/LIVE_TRADING_INTEGRATION.md)
- **Implementation Status**: [`DOCS/03_technical/trading/implementation/IMPLEMENTATION_STATUS.md`](../DOCS/03_technical/trading/implementation/IMPLEMENTATION_STATUS.md)
- **C++ Integration Summary**: [`DOCS/03_technical/trading/implementation/C++_INTEGRATION_SUMMARY.md`](../DOCS/03_technical/trading/implementation/C++_INTEGRATION_SUMMARY.md)
- **Live Trading Components**: [`DOCS/02_reference/trading/ibkr/LIVE_TRADING_README.md`](../DOCS/02_reference/trading/ibkr/LIVE_TRADING_README.md)

### Testing
- **Testing Plan**: [`DOCS/03_technical/trading/testing/TESTING_PLAN.md`](../DOCS/03_technical/trading/testing/TESTING_PLAN.md)
- **Testing Summary**: [`DOCS/03_technical/trading/testing/TESTING_SUMMARY.md`](../DOCS/03_technical/trading/testing/TESTING_SUMMARY.md)
- **Daily Testing**: [`DOCS/03_technical/trading/testing/DAILY_TESTING_README.md`](../DOCS/03_technical/trading/testing/DAILY_TESTING_README.md)

### Operations
- **Systemd Deployment**: [`DOCS/03_technical/trading/operations/SYSTEMD_DEPLOYMENT_PLAN.md`](../DOCS/03_technical/trading/operations/SYSTEMD_DEPLOYMENT_PLAN.md)
- **Yahoo Finance Integration**: [`DOCS/03_technical/trading/operations/YAHOO_FINANCE_INTEGRATION.md`](../DOCS/03_technical/trading/operations/YAHOO_FINANCE_INTEGRATION.md)
- **Intraday Trading Analysis**: [`DOCS/03_technical/trading/operations/INTRADAY_TRADING_ANALYSIS.md`](../DOCS/03_technical/trading/operations/INTRADAY_TRADING_ANALYSIS.md)
- **Performance Optimization**: [`DOCS/03_technical/trading/operations/PERFORMANCE_OPTIMIZATION_PLAN.md`](../DOCS/03_technical/trading/operations/PERFORMANCE_OPTIMIZATION_PLAN.md)

## ️ Risk Warnings & Legal Compliance

### Trading Risks

- **Live Trading**: This system is designed for live trading with real money
- **Risk Management**: Always test thoroughly in paper trading first
- **Monitoring**: Continuous monitoring required during live operation
- **Substantial Risk**: Trading involves substantial risk of loss of capital

### Legal & Regulatory Compliance

**IMPORTANT LEGAL NOTICES:**

- **Fox ML Infrastructure LLC is NOT a broker, investment advisor, or custodian**
- **We provide SOFTWARE INFRASTRUCTURE, not brokerage services or investment advice**
- **This is NON-CUSTODIAL execution on USER-OWNED accounts via USER-PROVIDED API keys**

**User Responsibilities:**
- Users are solely responsible for establishing and maintaining their brokerage relationship with IBKR
- Users are solely responsible for providing and securing their own API credentials
- Users are solely responsible for regulatory compliance (SEC, CFTC, state regulations, exchange rules)
- Users are solely responsible for all trading decisions, strategy configuration, and risk management
- Users must comply with IBKR's terms of service and API usage policies

**What We Provide:**
- Execution infrastructure and software
- Risk controls and safety rails (user-configurable)
- Research engines and backtesting tools
- Audit trails and compliance logging

**What We Do NOT Provide:**
- Investment advice or trading recommendations
- Brokerage services or custodial services
- Regulatory compliance services
- Guaranteed returns or performance

**See [`LEGAL/BROKER_INTEGRATION_COMPLIANCE.md`](../LEGAL/BROKER_INTEGRATION_COMPLIANCE.md) for complete compliance framework.**

## Support

For issues or questions:
1. Check logs in `logs/ibkr_trading.log`
2. Review system status and metrics
3. Consult architecture documentation
4. Contact system administrator

For legal/compliance questions:
- See [`LEGAL/BROKER_INTEGRATION_COMPLIANCE.md`](../LEGAL/BROKER_INTEGRATION_COMPLIANCE.md)
- See [`LEGAL/REGULATORY_DISCLAIMERS.md`](../LEGAL/REGULATORY_DISCLAIMERS.md)
- Consult your own legal counsel for regulatory compliance

---

**️ DISCLAIMER**: This system is execution infrastructure software. Users are solely responsible for all trading decisions, regulatory compliance, and risk management. Trading involves substantial risk of loss. Always test thoroughly before live trading.
