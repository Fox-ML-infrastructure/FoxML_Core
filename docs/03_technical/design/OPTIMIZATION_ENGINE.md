# Optimization Engine

Optimization algorithms and techniques used in the trading system.

## Overview

The optimization engine handles:
- Model blending weights (within-horizon)
- Horizon arbitration (across-horizons)
- Position sizing
- Risk parity allocation

## Within-Horizon Blending

### Problem

Blend multiple model predictions within a horizon, accounting for:
- Model correlations
- Cost-to-alpha ratios
- Confidence scores

### Solution: Ridge Risk-Parity

```
w_h ∝ (Σ_h + λI)^{-1} μ_h
```

Where:
- `Σ_h` = correlation matrix
- `λ` = ridge regularization
- `μ_h` = net IC after costs

**Benefits:**
- Prevents overfitting to correlated models
- Ensures risk parity
- Accounts for trading costs

## Cross-Horizon Arbitration

### Problem

Select optimal horizon considering:
- Alpha predictions
- Trading costs (spread, impact)
- Volatility scaling
- Execution constraints

### Solution: Cost-Aware Net Score

```
net_h = α_h - k₁ × spread - k₂ × σ × √(h/5) - k₃ × impact(q)
```

**Benefits:**
- Balances alpha vs costs
- Scales with volatility
- Accounts for market impact

## Position Sizing

### Volatility Targeting

```
size = (vol_target × equity) / (σ × √(h/252))
```

**Benefits:**
- Consistent risk across horizons
- Scales with account size
- Accounts for time horizon

## Implementation

### C++ Kernels

High-performance C++ implementations:
- `risk_parity_ridge`: Ridge risk parity solver
- `horizon_softmax`: Horizon arbitration
- `project_simplex`: Weight projection

### Python Fallback

Python implementations for:
- Development and testing
- Fallback when C++ unavailable
- Prototyping new algorithms

## See Also

- [C++ Integration](C++_INTEGRATION.md) - Performance implementation
- [Mathematical Foundations](MATHEMATICAL_FOUNDATIONS.md) - Equations
- [IBKR System Reference](../../02_reference/systems/IBKR_SYSTEM_REFERENCE.md) - System details

