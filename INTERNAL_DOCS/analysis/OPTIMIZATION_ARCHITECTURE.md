# Optimization Architecture - Clean Boundaries

## Overview

This document outlines the clean architecture boundaries for the IBKR optimization system, implementing the "drop green / buy better / cut low performer" logic as an **optimization/decision engine**, not a predictive model.

## Architecture Boundaries

### 1. **Signals Layer** (Input)
- **Purpose**: Model predictions and risk estimates
- **Inputs**: Raw model scores, volatility estimates, market data
- **Outputs**: Risk-scaled z-scores, correlation matrix, beta exposures
- **Components**: Model inference, feature pipeline, risk model

### 2. **Optimization Layer** (Decision Engine)
- **Purpose**: Convert signals + risk + costs into target weights
- **Inputs**: z-scores, correlation matrix, costs, current weights
- **Outputs**: Target weights, optimization info, reason codes
- **Components**: GreedyRotationEngine, QPOptimizationEngine

### 3. **Policy Layer** (Triggers)
- **Purpose**: Determine when to trade based on triggers/bands
- **Inputs**: Target weights, current weights, market conditions
- **Outputs**: Trading decisions, reason codes
- **Components**: No-trade bands, liquidity filters, time windows

### 4. **Execution Layer** (Output)
- **Purpose**: Execute trades through IBKR
- **Inputs**: Trading decisions, order specifications
- **Outputs**: Order confirmations, position updates
- **Components**: Order router, execution algorithms, reconciliation

## Objective Function

### QP/LP Solver Approach
```
maximize: z^T w - λ w^T Σ w - γ |C(w - w_cur)|_1
subject to:
  - Σ w_i = 1 (weights sum to 1)
  - w_i ≤ w_max (per-name caps)
  - Σ |w_i| ≤ G_max (gross exposure)
  - |Σ w_i| ≤ N_max (net exposure)
  - Σ |w_i - w_cur,i| ≤ T_max (turnover)
  - |β^T w| ≤ β_tol (beta neutrality)
  - Σ w_i * s_i ≤ S_max (sector caps)
```

Where:
- `z`: Risk-scaled scores
- `Σ`: Correlation matrix (shrunk)
- `C`: Per-name cost vector
- `β`: Beta exposures
- `s`: Sector exposures

### Greedy Approach
```
For each symbol i:
  if rank(i) > K_sell and z_i < z_cut:
    w_i = 0  # Cut low performers

  if rank(i) ≤ K_buy and z_i > z_keep:
    w_i = target_weight(i)  # Buy better names

  if utility(i,j) > τ_U:
    rotate(w_i, w_j)  # Rotate green to better
```

## Implementation Patterns

### 1. **Greedy Top-K Rotation** (Recommended for Intraday)
- **Complexity**: O(N log N)
- **Deterministic**: Yes
- **Speed**: Very fast (< 100ms)
- **Use case**: Intraday trading, real-time decisions
- **Pros**: Fast, simple, deterministic
- **Cons**: May not find global optimum

### 2. **QP/LP Optimizer** (For Complex Constraints)
- **Complexity**: O(N^3) for QP
- **Deterministic**: Yes (with fixed solver)
- **Speed**: Fast (< 1s for N < 300)
- **Use case**: Complex constraints, beta hedging
- **Pros**: Global optimum, handles complex constraints
- **Cons**: Requires cvxpy, slower than greedy

## Run Cadence

### **Recompute Every Interval**
```python
# Every bar/heartbeat (use fastest model's cadence)
def run_optimization_cycle():
    # 1. Fetch data → update features/vol/spreads
    market_data = get_market_data(symbols)
    scores = get_model_scores(symbols, horizons)
    sigma = calculate_volatility(scores)

    # 2. Models infer scores (all timeframes)
    z_scores = scale_scores(scores, sigma)

    # 3. Aggregator merges scores → portfolio target
    target_weights = optimization_engine.optimize(
        z_scores, correlation_matrix, costs, current_weights, universe
    )

    # 4. Rebalancer: costs + constraints → final weights
    final_weights = apply_constraints(target_weights)

    # 5. Decision: if triggers hit → schedule/route orders
    if should_trade(final_weights, current_weights):
        execute_orders(final_weights, current_weights)
    else:
        do_nothing()  # Respect no-trade bands
```

### **Trade Only When Triggers Fire**
- **Portfolio drift**: `|w_tar - w_cur|_1 > τ_L1`
- **Per-name drift**: `|w_tar,i - w_cur,i| > τ_i`
- **Scheduled windows**: 09:35, 10:30, 12:00, 14:30, 15:50
- **Risk overrides**: Vol spike, spread blowout, kill-switch

## Component Responsibilities

### **GreedyRotationEngine**
- Implements top-K rotation logic
- Handles "drop green / cut low performer" rules
- Applies no-trade bands and utility thresholds
- O(N log N) complexity, deterministic

### **QPOptimizationEngine**
- Solves full portfolio optimization
- Handles complex constraints (beta, sector, turnover)
- Uses cvxpy + OSQP solver
- O(N^3) complexity, global optimum

### **IBKROptimizationIntegration**
- Wires optimization engine with IBKR stack
- Manages state and history
- Provides clean API for trading system
- Handles engine switching (greedy ↔ QP)

## Configuration

### **Default Thresholds**
```python
config = OptimizationConfig(
    # Risk parameters
    lambda_risk=0.1,          # Risk aversion
    gamma_cost=0.05,         # Cost aversion
    tau_utility=0.0,         # Utility threshold

    # Portfolio constraints
    max_gross_exposure=1.5,   # Max gross exposure
    max_net_exposure=0.3,     # Max net exposure
    per_name_cap=0.1,        # Max weight per name
    sector_cap=0.3,          # Max weight per sector

    # Turnover constraints
    max_turnover=0.2,        # Max turnover per period
    max_turnover_per_name=0.05,  # Max turnover per name

    # Beta neutrality
    beta_tolerance=0.05,     # Beta neutrality tolerance

    # Triggers
    portfolio_drift_threshold=0.01,    # Portfolio L1 drift threshold
    name_drift_threshold=0.005,        # Per-name drift threshold

    # Hysteresis
    K=20, K_buy=18, K_sell=24,        # Top-K portfolio with hysteresis

    # Performance thresholds
    z_keep=0.8, z_cut=0.2, delta_z_min=0.25,  # Alpha gaps
    k_ATR=1.2, Tmax_min=120           # Risk and time limits
)
```

## Usage Examples

### **Greedy Engine (Recommended)**
```python
# Create engine
config = OptimizationConfig(K=20, K_buy=18, K_sell=24)
engine = GreedyRotationEngine(config)

# Optimize
target_weights, info = engine.optimize(
    z_scores, correlation_matrix, costs, current_weights, universe
)

# Check results
print(f"Changes: {len(info['changes'])}")
print(f"Reason codes: {info['reason_codes']}")
```

### **QP Engine (Complex Constraints)**
```python
# Create engine
config = OptimizationConfig(lambda_risk=0.1, gamma_cost=0.05)
engine = QPOptimizationEngine(config)

# Optimize with constraints
target_weights, info = engine.optimize(
    z_scores, correlation_matrix, costs, current_weights, universe,
    beta_exposures=beta_exposures,
    sector_exposures=sector_exposures
)

# Check results
print(f"Status: {info['status']}")
print(f"Objective: {info['objective_value']}")
```

### **Integration with IBKR**
```python
# Create integration
integration = IBKROptimizationIntegration(config)

# Update risk model
integration.update_risk_model(symbols, correlation_matrix, beta_exposures)

# Run optimization cycle
success, summary = integration.run_optimization_cycle(symbols)

# Switch engines if needed
integration.switch_engine_type('qp')  # or 'greedy'
```

## Benefits

### **Clean Architecture**
- **Separation of concerns**: Signals, optimization, policy, execution
- **Testable**: Each layer can be tested independently
- **Flexible**: Easy to switch between greedy and QP approaches
- **Maintainable**: Clear boundaries and responsibilities

### **Performance**
- **Greedy**: O(N log N), deterministic, very fast
- **QP**: O(N^3), global optimum, handles complex constraints
- **Triggers**: Only trade when necessary, respect no-trade bands

### **Robustness**
- **Cost-aware**: Rotations must clear utility threshold
- **Hysteresis**: Prevents churn with buy/sell bands
- **Liquidity-aware**: Blocks trading in illiquid names
- **Time-aware**: Cuts dead money after timeout

## Conclusion

The optimization architecture provides clean boundaries between signals, optimization, policy, and execution. The system computes every interval but only trades when triggers fire, implementing the "drop green / buy better / cut low performer" logic as a decision engine rather than a predictive model.

Choose **GreedyRotationEngine** for intraday trading (fast, deterministic) or **QPOptimizationEngine** for complex constraints (global optimum, beta hedging).
