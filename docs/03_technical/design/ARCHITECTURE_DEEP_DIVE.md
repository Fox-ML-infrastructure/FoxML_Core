# Architecture Deep Dive

Detailed system architecture and design decisions.

## System Overview

Fox-v1-infra is a multi-layered trading infrastructure with:

1. **Data Processing Layer**: Raw data → Features → Targets
2. **Model Training Layer**: 17+ model types with centralized configs
3. **Trading Layer**: IBKR (production) and Alpaca (paper) integration
4. **Performance Layer**: C++ kernels for hot path operations

## Data Flow

```
Raw Market Data
    ↓
Normalization (RTH alignment, grid correction)
    ↓
Feature Engineering (200+ technical features)
    ↓
Target Generation (barrier, excess returns, HFT)
    ↓
Labeled Dataset
    ↓
Model Training (17+ models, walk-forward validation)
    ↓
Trained Models
    ↓
Live Trading (multi-horizon blending, cost-aware arbitration)
```

## Component Architecture

### Data Processing

- **Normalization**: Session alignment, grid correction
- **Feature Builders**: Simple, Comprehensive, Streaming
- **Target Builders**: Barrier, Excess Returns, HFT Forward Returns

### Model Training

- **Training Strategies**: Single-task, Multi-task
- **Validation**: Walk-forward validation
- **Configuration**: Centralized YAML configs with variants

### Trading Systems

**IBKR (Production)**:
- Multi-horizon model blending (5m, 10m, 15m, 30m, 60m)
- Safety guards and risk management
- C++ performance optimization

**Alpaca (Paper)**:
- Regime-aware ensemble strategies
- Risk management and guardrails
- Performance tracking

## Design Principles

1. **Configuration-Driven**: All runtime parameters from config
2. **Safety-First**: Multiple layers of guards and checks
3. **Performance-Critical**: C++ for hot paths, Python for orchestration
4. **Modular**: Clear separation of concerns
5. **Extensible**: Easy to add new models, features, strategies

## See Also

- [Architecture Overview](../../00_executive/ARCHITECTURE_OVERVIEW.md) - High-level overview
- [IBKR System Reference](../../02_reference/systems/IBKR_SYSTEM_REFERENCE.md) - IBKR architecture
- [Alpaca System Reference](../../02_reference/systems/ALPACA_SYSTEM_REFERENCE.md) - Alpaca architecture

