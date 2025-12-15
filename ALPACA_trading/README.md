# ALPACA Paper Trading Service Package

This folder contains all files required for the Alpaca paper trading service.

> **Note**: For a comprehensive comparison of ALPACA_trading and IBKR_trading modules, see the main [TRADING_MODULES.md](../TRADING_MODULES.md) documentation.

## Structure

```
ALPACA_trading/
├── scripts/
│   ├── paper_runner.py              # Main paper trading runner
│   └── data/
│       ├── alpaca_batch_optimized.py
│       └── alpaca_batch.py
├── core/
│   ├── engine/
│   │   └── paper.py                 # Core paper trading engine
│   ├── enhanced_logging.py
│   ├── feature_reweighter.py
│   ├── notifications.py
│   ├── performance.py
│   ├── regime_detector.py
│   ├── strategy_selector.py
│   ├── utils.py
│   ├── data_sanity.py
│   ├── risk/
│   │   └── guardrails.py            # Optional
│   └── telemetry/
│       └── snapshot.py              # Optional
├── brokers/
│   ├── paper.py                     # Paper broker implementation
│   ├── interface.py                 # Broker interface
│   ├── data_provider.py
│   └── ibkr_broker.py
├── strategies/
│   ├── factory.py
│   └── regime_aware_ensemble.py
├── ml/
│   ├── model_interface.py
│   ├── registry.py
│   └── runtime.py
├── utils/
│   └── ops_runtime.py
├── tools/
│   └── provenance.py
├── cli/
│   └── paper.py                     # CLI interface
└── config/
    ├── base.yaml
    ├── models.yaml
    ├── paper_trading_config.json
    ├── paper_config.json
    ├── paper-trading.env
    └── [other paper configs]
```

## Main Entry Points

- `scripts/paper_runner.py` - Main paper trading runner
- `core/engine/paper.py` - Core trading engine
- `cli/paper.py` - CLI interface

## Configuration

All configuration files are in the `config/` directory:
- `base.yaml` - Base configuration
- `models.yaml` - Model registry
- `paper_trading_config.json` - Paper trading settings
- `paper-trading.env` - Environment variables

## Dependencies

### Python Packages
- `alpaca-trade-api` or `alpaca-py`
- `pandas`
- `numpy`
- `yfinance`
- `yaml`
- `requests`

### Environment Variables
- `ALPACA_API_KEY`
- `ALPACA_SECRET_KEY`
- `ALPACA_BASE_URL` (optional)

## Usage

From the repo root:
```bash
python ALPACA_trading/scripts/paper_runner.py --symbols SPY,TSLA --profile risk_balanced
```

Or using the CLI:
```bash
python ALPACA_trading/cli/paper.py [options]
```

## Notes

- Some files are optional (marked in structure above)
- Import paths may need adjustment if running from this folder directly
- Original files remain in their original locations; these are copies

## Legal & Regulatory Compliance

**IMPORTANT LEGAL NOTICES:**

- **Fox ML Infrastructure LLC is NOT a broker, investment advisor, or custodian**
- **We provide SOFTWARE INFRASTRUCTURE, not brokerage services or investment advice**
- **This is NON-CUSTODIAL execution on USER-OWNED accounts via USER-PROVIDED API keys**

**User Responsibilities:**
- Users are solely responsible for establishing and maintaining their brokerage relationship with Alpaca (or using yfinance for data-only)
- Users are solely responsible for providing and securing their own API credentials (if using Alpaca)
- Users are solely responsible for regulatory compliance (SEC, CFTC, state regulations, exchange rules)
- Users are solely responsible for all trading decisions, strategy configuration, and risk management
- Users must comply with Alpaca's terms of service and API usage policies (if applicable)

**What We Provide:**
- Paper trading and backtesting infrastructure
- Execution simulation and portfolio management
- Research engines and strategy testing tools
- Performance analytics and reporting

**What We Do NOT Provide:**
- Investment advice or trading recommendations
- Brokerage services or custodial services
- Regulatory compliance services
- Guaranteed returns or performance

**See [`LEGAL/BROKER_INTEGRATION_COMPLIANCE.md`](../LEGAL/BROKER_INTEGRATION_COMPLIANCE.md) for complete compliance framework.**

