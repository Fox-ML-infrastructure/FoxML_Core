# Trading Modules Technical Documentation

This directory contains deep technical documentation for the trading modules in FoxML Core.

## Directory Structure

```
trading/
├── architecture/    # Architecture and design documents
├── implementation/  # Implementation details and status
├── testing/         # Testing plans and procedures
└── operations/      # Deployment and operational guides
```

## Architecture & Design

Mathematical foundations, optimization architecture, and system design:

- [Mathematical Foundations](architecture/MATHEMATICAL_FOUNDATIONS.md) - Mathematical equations and formulas for IBKR cost-aware ensemble trading
- [Optimization Architecture](architecture/OPTIMIZATION_ARCHITECTURE.md) - Clean architecture boundaries for the IBKR optimization system

## Implementation

Implementation details, status, and integration guides:

- [Live Trading Integration](implementation/LIVE_TRADING_INTEGRATION.md) - Integration guide for live trading
- [C++ Integration Summary](implementation/C++_INTEGRATION_SUMMARY.md) - C++ components integration
- [C++ Engine README](implementation/CPP_ENGINE_README.md) - C++ inference engine documentation
- [Enhanced Rebalancing Trading Plan](implementation/ENHANCED_REBALANCING_TRADING_PLAN.md) - Rebalancing strategy
- [Pressure Test Implementation Roadmap](implementation/PRESSURE_TEST_IMPLEMENTATION_ROADMAP.md) - Pressure testing roadmap
- [Pressure Test Upgrades](implementation/PRESSURE_TEST_UPGRADES.md) - Pressure test improvements
- [Deprecated Components](implementation/DEPRECATED_README.md) - Deprecated component documentation
- [Deprecation Notice](implementation/DEPRECATION_NOTICE.md) - Deprecation information

## Testing

Testing plans, procedures, and results:

- [Testing Plan](testing/TESTING_PLAN.md) - Comprehensive testing strategy
- [Testing Summary](testing/TESTING_SUMMARY.md) - Test results and status
- [Daily Testing](testing/DAILY_TESTING_README.md) - Daily testing procedures

## Operations

Deployment, integration, and operational guides:

- [Systemd Deployment Plan](operations/SYSTEMD_DEPLOYMENT_PLAN.md) - Systemd service deployment
- [Yahoo Finance Integration](operations/YAHOO_FINANCE_INTEGRATION.md) - Yahoo Finance data integration
- [Performance Optimization Plan](operations/PERFORMANCE_OPTIMIZATION_PLAN.md) - Performance optimization strategies

## Related Documentation

- Trading technical documentation and architecture guides

---

**Copyright (c) 2025-2026 Fox ML Infrastructure LLC**
