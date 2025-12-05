# Mathematical Foundations

Mathematical equations and formulas used in the trading system.

## Standardization & Calibration

### Z-Score Standardization

```
s_{m,h} = clip((r̂_{m,h} - μ_{m,h}) / σ_{m,h}, -3, 3)
```

**Where:**
- `s_{m,h}` = standardized score for model m, horizon h
- `r̂_{m,h}` = raw prediction from model m for horizon h
- `μ_{m,h}` = rolling mean of predictions (N≈5-10 trading days)
- `σ_{m,h}` = rolling standard deviation of predictions
- `clip(-3, 3)` = prevents extreme outliers from dominating

**Used in:** Model score standardization  
**Purpose:** Ensures fair comparison across different model types and scales.

### Confidence Calculation

```
c_{m,h} = IC_{m,h} × freshness × capacity × stability
```

**Where:**
- `IC_{m,h}` = Information Coefficient (Spearman correlation)
- `freshness = e^{-Δt/τ_h}` with time constants per horizon
- `capacity = min(1, κ × ADV / planned_dollars)`
- `stability = 1 / (rolling_RMSE_of_calibration)`

**Used in:** Model confidence weighting  
**Purpose:** Adjusts model confidence based on data freshness, capacity, and stability.

## Ensemble Blending

### Ridge Risk-Parity Weights

```
w_h ∝ (Σ_h + λI)^{-1} μ_h
w_h ← clip(w_h, 0, ∞)
∑w_h = 1
```

**Where:**
- `Σ_h` = correlation matrix of standardized scores
- `λ` = ridge regularization parameter (typically 0.15)
- `μ_h` = target vector of net IC after costs
- `I` = identity matrix

**Used in:** Within-horizon model blending  
**Purpose:** Prevents overfitting to correlated models, ensures risk parity.

### Temperature Compression

```
w_h^{(T)} ∝ w_h^{1/T}
T_{5m} = 0.75, T_{10m} = 0.85
```

**Used in:** Short-horizon weighting  
**Purpose:** Reduces extreme weights for short horizons, more conservative blending.

## Cost-Aware Arbitration

### Net Score Calculation

```
net_h = α_h - k₁ × spread_bps - k₂ × σ × √(h/5) - k₃ × impact(q)
```

**Where:**
- `α_h` = horizon alpha (blended prediction)
- `spread_bps` = bid-ask spread in basis points
- `σ` = volatility
- `h` = horizon in minutes
- `impact(q)` = market impact function

**Used in:** Cross-horizon arbitration  
**Purpose:** Selects optimal horizon considering alpha, costs, and risk.

## Position Sizing

### Volatility-Targeted Sizing

```
size = (vol_target × equity) / (σ × √(h/252))
```

**Where:**
- `vol_target` = target volatility (typically 0.15)
- `equity` = account equity
- `σ` = asset volatility
- `h` = horizon in trading days

**Used in:** Position sizing  
**Purpose:** Sizes positions to target volatility across different horizons.

## See Also

- [Mathematical Foundations](../../../IBKR_trading/MATHEMATICAL_FOUNDATIONS.md) - Complete reference
- [C++ Integration](C++_INTEGRATION.md) - Performance implementation

