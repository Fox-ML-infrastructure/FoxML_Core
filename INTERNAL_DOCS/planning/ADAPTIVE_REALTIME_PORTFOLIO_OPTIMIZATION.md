# Adaptive Real-Time Portfolio Optimization with iDiffODE

**Status**: Research & Design Phase  
**Date**: 2025-12-08  
**Classification**: Internal Planning Document

## Executive Summary

This document outlines a framework for **adaptive real-time portfolio optimization** using the Neural ODE-Diffusion-Invertible ResNet (iDiffODE) architecture. The system learns continuously from profit/loss (P&L) metrics, market conditions, and trading outcomes to optimize portfolio allocation, position sizing, and risk management in real time.

**Core Value Proposition**: Transform static portfolio management into a self-adapting system that:
- Learns optimal position sizes from real-time P&L feedback
- Adapts to changing market regimes automatically
- Optimizes portfolio allocation continuously
- Prevents drawdowns through predictive risk modeling
- Improves Sharpe ratio and risk-adjusted returns over time

---

## 1. Current State Analysis

### 1.1 Existing Portfolio Management

Current system provides:
- **Static Position Sizing**: Fixed rules (volatility targeting, fixed dollar amounts)
- **Manual Rebalancing**: Scheduled or threshold-based rebalancing
- **Risk Limits**: Static maximum position sizes, drawdown limits
- **Model Predictions**: Model outputs used for entry/exit decisions
- **No Learning**: Portfolio rules don't adapt based on performance

### 1.2 Current Limitations

- **No Real-Time Adaptation**: Position sizes don't adapt to changing market conditions
- **No P&L Learning**: System doesn't learn from actual trading outcomes
- **Static Risk Management**: Risk limits don't adapt to volatility regimes
- **No Portfolio-Level Optimization**: Individual position decisions, not portfolio-level
- **No Regime Adaptation**: Same rules apply in all market conditions
- **No Continuous Learning**: Doesn't improve over time based on experience

### 1.3 Opportunity

The system generates rich real-time data:
- P&L per position, per symbol, per strategy
- Market regime indicators (volatility, trend, correlation)
- Model prediction accuracy over time
- Risk metrics (drawdown, Sharpe, max exposure)
- Execution quality (slippage, fills, latency)

**This data is currently logged but not used for real-time portfolio optimization.**

---

## 2. System Architecture

### 2.1 High-Level Design

```
┌─────────────────────────────────────────────────────────┐
│    Adaptive Real-Time Portfolio Optimization (ARPO)     │
│              iDiffODE-Powered Intelligence Layer        │
└─────────────────────────────────────────────────────────┘
                            │
                            │ continuously learns
                            ▼
┌─────────────────────────────────────────────────────────┐
│              iDiffODE Portfolio Model                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  Continuous  │  │  Diffusion   │  │  Invertible  │ │
│  │  Portfolio   │  │  Refinement  │  │  Allocation  │ │
│  │  Dynamics    │  │              │  │  Mapping     │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
                            │
                            │ generates
                            ▼
┌─────────────────────────────────────────────────────────┐
│         Real-Time Portfolio Decisions                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Position   │  │   Risk       │  │  Rebalancing │ │
│  │   Sizing     │  │  Management  │  │   Signals    │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
                            │
                            │ executes
                            ▼
┌─────────────────────────────────────────────────────────┐
│              Live Trading System                         │
│         (IBKR / Alpaca Integration)                      │
└─────────────────────────────────────────────────────────┘
                            │
                            │ generates
                            ▼
┌─────────────────────────────────────────────────────────┐
│         Real-Time P&L & Market Data                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   P&L        │  │   Market     │  │   Risk      │ │
│  │   Metrics    │  │   Regime     │  │   Metrics   │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
                            │
                            │ feeds back
                            └──────────────────────────────┐
                                                           │
                            ┌──────────────────────────────┘
                            │
                            ▼
                    (Learning Loop)
```

### 2.2 Core Components

#### 2.2.1 Continuous Portfolio Dynamics Model

**Purpose**: Model portfolio state as continuous function of time using Neural ODE.

**State Representation**:
- Portfolio weights: \(w(t) \in \mathbb{R}^n\) (n = number of positions)
- Portfolio value: \(V(t) \in \mathbb{R}\)
- Risk metrics: \(\sigma(t), \beta(t), \text{DD}(t)\) (volatility, beta, drawdown)
- Market regime: \(r(t) \in \mathbb{R}^m\) (regime indicators)

**Neural ODE Dynamics**:

\[
\frac{d w(t)}{d t} = f_{\theta}\big(w(t), V(t), r(t), t\big)
\]

\[
\frac{d V(t)}{d t} = g_{\theta}\big(w(t), r(t), \text{P\&L}(t), t\big)
\]

Where:
- \(f_\theta\): Position weight dynamics (how weights should evolve)
- \(g_\theta\): Portfolio value dynamics (how value changes)
- \(\text{P\&L}(t)\): Real-time profit/loss signal
- \(r(t)\): Market regime state

**Key Insight**: Portfolio weights evolve continuously, not just at discrete rebalancing times. This captures:
- Optimal rebalancing timing
- Smooth position adjustments
- Regime transition handling

#### 2.2.2 Real-Time P&L Learning

**Purpose**: Learn from actual trading outcomes to improve position sizing and allocation.

**Learning Signal**:

\[
L(t) = \text{P\&L}(t) - \text{Expected P\&L}(t)
\]

Where:
- \(\text{P\&L}(t)\): Actual realized profit/loss
- \(\text{Expected P\&L}(t)\): Predicted P&L from model
- \(L(t)\): Learning signal (positive = better than expected, negative = worse)

**Learning Mechanism**:
1. Track \(L(t)\) for each position over time
2. Learn which position sizes produce best risk-adjusted returns
3. Adapt position sizing rules based on historical \(L(t)\)
4. Learn optimal allocation weights per market regime

**Adaptive Position Sizing**:

\[
w_i(t) = w_i^{\text{base}}(t) \cdot \alpha_i(t) \cdot \beta_r(t)
\]

Where:
- \(w_i^{\text{base}}(t)\): Base position size from model prediction
- \(\alpha_i(t)\): Learned adjustment factor for position \(i\) (from P&L history)
- \(\beta_r(t)\): Regime-specific multiplier (learned from regime performance)

#### 2.2.3 Diffusion-Based Regime Refinement

**Purpose**: Use diffusion model to refine portfolio allocation based on complex market dynamics.

**Process**:
1. **Initial Allocation**: From Neural ODE solution
2. **Diffusion Refinement**: Iteratively refine allocation to capture:
   - Non-linear correlations between positions
   - Regime-dependent optimal allocations
   - Risk-adjusted return optimization
3. **Final Allocation**: Optimized portfolio weights

**Regime Detection**:
- Volatility regimes (low/medium/high vol)
- Trend regimes (trending/ranging)
- Correlation regimes (high/low correlation)
- Liquidity regimes (liquid/illiquid)

#### 2.2.4 Invertible Portfolio Mapping

**Purpose**: Ensure bijective mapping between portfolio state and optimal allocation.

**Forward Mapping**: Portfolio state → Optimal allocation
- Current positions, P&L, risk metrics → Optimal weights

**Inverse Mapping**: Optimal allocation → Required trades
- Target weights → Trade orders (buy/sell quantities)

**Benefits**:
- Interpretability: Can reconstruct portfolio state from allocation
- Validation: Can verify allocation is optimal for current state
- Debugging: Can trace allocation decisions back to portfolio state

#### 2.2.5 Real-Time Risk Controller

**Purpose**: Continuously monitor and adjust risk limits based on learned patterns.

**Adaptive Risk Limits**:

\[
\text{RiskLimit}(t) = \text{RiskLimit}^{\text{base}} \cdot \gamma(t) \cdot \delta_r(t)
\]

Where:
- \(\text{RiskLimit}^{\text{base}}\): Base risk limit (e.g., max drawdown 20%)
- \(\gamma(t)\): Learned adjustment from historical drawdowns
- \(\delta_r(t)\): Regime-specific risk multiplier

**Learning from Drawdowns**:
- Track actual drawdowns vs. predicted
- Learn optimal risk limits per regime
- Adapt limits to prevent future drawdowns

---

## 3. Mathematical Foundation

### 3.1 Continuous Portfolio Dynamics

**Neural ODE for Portfolio Weights**:

\[
\frac{d w(t)}{d t} = f_{\theta}\big(w(t), V(t), r(t), \text{P\&L}(t), t\big)
\]

\[
w(t_0) = w_0 = \Phi(X_{\text{portfolio}})
\]

Where:
- \(X_{\text{portfolio}}\): Current portfolio state (positions, P&L, risk metrics)
- \(\Phi(\cdot)\): Encoder mapping portfolio state to initial weight vector
- \(f_\theta\): Neural ODE vector field learning optimal weight evolution

**ODE Solver**:

\[
w(t) = \text{ODESolve}\big(w_0, f_\theta, \{t_1, \dots, t_N\}\big)
\]

Where \(\{t_1, \dots, t_N\}\) are irregular rebalancing times (determined by model, not fixed schedule).

### 3.2 Portfolio Value Dynamics

**Value Evolution**:

\[
\frac{d V(t)}{d t} = \sum_{i=1}^n w_i(t) \cdot \frac{d P_i(t)}{d t} - \text{Cost}(t)
\]

Where:
- \(P_i(t)\): Price of asset \(i\) at time \(t\)
- \(\text{Cost}(t)\): Trading costs, fees, slippage

**Learned Value Model**:

\[
\frac{d V(t)}{d t} = g_{\theta}\big(w(t), r(t), \text{P\&L}(t), \text{ExpectedReturns}(t), t\big)
\]

Where \(g_\theta\) learns to predict portfolio value changes from:
- Current weights
- Market regime
- Real-time P&L
- Expected returns from models

### 3.3 P&L Learning Signal

**Learning Objective**:

\[
\mathcal{L}_{\text{P\&L}} = \mathbb{E}\left[\left(\text{P\&L}(t) - \hat{\text{P\&L}}(t)\right)^2\right]
\]

Where:
- \(\text{P\&L}(t)\): Actual realized P&L
- \(\hat{\text{P\&L}}(t)\): Predicted P&L from model

**Position Sizing Update**:

\[
\alpha_i(t+1) = \alpha_i(t) + \eta \cdot \frac{\partial \mathcal{L}_{\text{P\&L}}}{\partial \alpha_i(t)}
\]

Where \(\eta\) is learning rate for position sizing adjustments.

### 3.4 Risk-Adjusted Objective

**Sharpe Ratio Optimization**:

\[
\text{Sharpe}(t) = \frac{\mathbb{E}[\text{Returns}(t)]}{\sigma(\text{Returns}(t))}
\]

**Learning Objective**:

\[
\mathcal{L}_{\text{Sharpe}} = -\text{Sharpe}(t) + \lambda \cdot \text{Penalty}(\text{Risk}(t))
\]

Where:
- Maximize Sharpe ratio
- Penalize excessive risk (drawdown, volatility, correlation)

---

## 4. Architecture Specification

### 4.1 Portfolio State Encoder

**Input**: Current portfolio state
- Position weights: \(w_{\text{current}} \in \mathbb{R}^n\)
- Portfolio value: \(V_{\text{current}} \in \mathbb{R}\)
- P&L history: \(\text{P\&L}_{\text{history}} \in \mathbb{R}^T\)
- Risk metrics: \(\sigma, \beta, \text{DD} \in \mathbb{R}\)
- Market regime: \(r \in \mathbb{R}^m\)
- Model predictions: \(\text{predictions} \in \mathbb{R}^n\)

**Output**: Initial latent state \(h_0 \in \mathbb{R}^d\)

**Architecture**: Transformer encoder handling:
- Variable-length P&L history
- Multi-modal inputs (weights, values, metrics, regime)
- Temporal attention to important time points

### 4.2 Neural ODE Portfolio Dynamics

**Input**: Initial latent state \(h_0\)

**Process**:
- Solve ODE: \(\frac{d h(t)}{d t} = f_\theta(h(t), t)\)
- Over irregular time grid (rebalancing times determined by model)
- Output: Latent trajectory \(Z = \{h(t_1), \dots, h(t_N)\}\)

**Vector Field \(f_\theta\)**:
- Learns optimal portfolio weight evolution
- Incorporates P&L feedback
- Adapts to market regime
- Respects risk constraints

### 4.3 Diffusion Refinement

**Input**: Latent trajectory \(Z\) from Neural ODE

**Process**:
- Iteratively denoise/refine portfolio allocation
- Capture non-linear correlations
- Optimize for risk-adjusted returns
- Adapt to regime transitions

**Output**: Refined latent states \(Z_{\text{refined}}\)

### 4.4 Invertible Portfolio Allocator

**Input**: Refined latent states \(Z_{\text{refined}}\)

**Process**:
- Invertible ResNet maps latent → optimal weights
- Ensures bijective mapping
- Enables validation and debugging

**Output**: Optimal portfolio weights \(w_{\text{optimal}}(t)\)

### 4.5 Trade Generator

**Input**: Optimal weights \(w_{\text{optimal}}(t)\), current weights \(w_{\text{current}}\)

**Process**:
- Compute required trades: \(\Delta w = w_{\text{optimal}} - w_{\text{current}}\)
- Apply constraints (min trade size, max position change)
- Generate trade orders

**Output**: Trade orders (symbol, quantity, side, order type)

---

## 5. Implementation Outline

### 5.1 Module Structure

```python
# PORTFOLIO/adaptive_realtime/
├── __init__.py
├── encoder.py              # Portfolio state encoder (Transformer)
├── neural_ode.py           # Neural ODE for portfolio dynamics
├── diffusion.py            # Diffusion model for allocation refinement
├── invertible_allocator.py # Invertible ResNet for weight mapping
├── pnl_learner.py          # P&L learning and position sizing adaptation
├── risk_controller.py      # Adaptive risk management
├── trade_generator.py      # Generate trade orders from optimal weights
├── portfolio_model.py      # Complete iDiffODE portfolio model
└── utils/
    ├── data_loader.py      # Real-time data loading
    ├── metrics.py          # P&L, Sharpe, risk metrics
    └── visualization.py    # Portfolio state visualization
```

### 5.2 Core Implementation Skeleton

```python
import torch
import torch.nn as nn
from torchdiffeq import odeint
from typing import Tuple, Optional, Dict, List
import pandas as pd
import numpy as np

class PortfolioStateEncoder(nn.Module):
    """Transformer encoder for portfolio state embedding."""
    
    def __init__(
        self,
        n_positions: int,           # Number of positions
        d_model: int = 128,         # Transformer dimension
        nhead: int = 8,             # Attention heads
        num_layers: int = 4,        # Transformer layers
        pnl_history_len: int = 100  # P&L history length
    ):
        super().__init__()
        # Implementation:
        # - Embed current portfolio state (weights, value, risk metrics)
        # - Embed P&L history as time-series
        # - Embed market regime indicators
        # - Multi-head attention across all inputs
        # - Output: initial latent state h_0
        
    def forward(
        self,
        current_weights: torch.Tensor,      # [batch, n_positions]
        portfolio_value: torch.Tensor,      # [batch, 1]
        pnl_history: torch.Tensor,          # [batch, pnl_history_len]
        risk_metrics: torch.Tensor,        # [batch, n_risk_metrics]
        market_regime: torch.Tensor,       # [batch, n_regime_features]
        model_predictions: torch.Tensor     # [batch, n_positions]
    ) -> torch.Tensor:
        """
        Encode portfolio state into initial latent state.
        
        Returns:
            h_0: [batch, d_model] - Initial latent state
        """
        # Combine all inputs
        # Apply transformer encoding
        # Return initial latent state
        pass


class PortfolioODEFunc(nn.Module):
    """Neural ODE vector field for portfolio dynamics."""
    
    def __init__(
        self,
        d_model: int = 128,
        hidden_dim: int = 256,
        n_positions: int = None
    ):
        super().__init__()
        # Implementation:
        # - MLP defining continuous portfolio dynamics
        # - Input: latent state h(t), time t, P&L signal, regime
        # - Output: dh/dt (how portfolio state should evolve)
        
    def forward(
        self,
        t: torch.Tensor,
        h: torch.Tensor,
        pnl_signal: Optional[torch.Tensor] = None,
        regime: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute portfolio dynamics: dh/dt = f_θ(h(t), t, P&L, regime)
        
        Args:
            t: Scalar time
            h: [batch, d_model] - Current latent state
            pnl_signal: [batch, 1] - Real-time P&L signal (optional)
            regime: [batch, n_regime] - Market regime (optional)
        
        Returns:
            dh/dt: [batch, d_model] - Time derivative
        """
        # Compute dynamics incorporating P&L feedback and regime
        pass


class PortfolioDiffusionRefiner(nn.Module):
    """Diffusion model for refining portfolio allocation."""
    
    def __init__(
        self,
        d_model: int = 128,
        num_steps: int = 100,
        n_positions: int = None
    ):
        super().__init__()
        # Implementation:
        # - Denoising diffusion process
        # - Refines allocation to optimize risk-adjusted returns
        # - Captures non-linear position correlations
        
    def forward(
        self,
        Z: torch.Tensor,
        risk_constraints: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Refine portfolio allocation via diffusion.
        
        Args:
            Z: [batch, seq_len, d_model] - Latent trajectory from ODE
            risk_constraints: [batch, seq_len, n_constraints] - Risk limits
        
        Returns:
            Z_refined: [batch, seq_len, d_model] - Refined latent states
        """
        # Apply diffusion denoising with risk constraints
        pass


class InvertiblePortfolioAllocator(nn.Module):
    """Invertible ResNet for portfolio weight mapping."""
    
    def __init__(
        self,
        d_model: int = 128,
        n_positions: int = None,
        num_blocks: int = 4
    ):
        super().__init__()
        # Implementation:
        # - Invertible ResNet blocks
        # - Forward: latent → optimal weights
        # - Inverse: weights → latent (for validation)
        
    def forward(
        self,
        Z: torch.Tensor,
        inverse: bool = False,
        current_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Map latent states to optimal portfolio weights.
        
        Args:
            Z: [batch, seq_len, d_model] - Latent states
            inverse: If True, map weights → latent (for validation)
            current_weights: [batch, seq_len, n_positions] - Current weights (if inverse)
        
        Returns:
            optimal_weights: [batch, seq_len, n_positions] - Optimal portfolio weights
            OR
            latent_recon: [batch, seq_len, d_model] - Reconstructed latent (if inverse)
        """
        # Forward: latent → weights
        # Inverse: weights → latent (for validation)
        pass


class PnLLearner(nn.Module):
    """Learn from P&L to adapt position sizing."""
    
    def __init__(
        self,
        n_positions: int,
        learning_rate: float = 0.01,
        momentum: float = 0.9
    ):
        super().__init__()
        # Implementation:
        # - Track P&L per position over time
        # - Learn optimal position sizing adjustments
        # - Adapt based on risk-adjusted returns
        
    def update_position_sizing(
        self,
        position_id: int,
        pnl: float,
        expected_pnl: float,
        risk_metric: float
    ) -> float:
        """
        Update position sizing adjustment factor based on P&L.
        
        Args:
            position_id: Position identifier
            pnl: Actual realized P&L
            expected_pnl: Predicted P&L from model
            risk_metric: Risk metric (e.g., Sharpe ratio)
        
        Returns:
            adjustment_factor: Learned adjustment for position sizing
        """
        # Compute learning signal: L = P&L - Expected P&L
        # Update adjustment factor: α = α + η * ∂L/∂α
        # Return updated adjustment
        pass
    
    def get_position_adjustments(self) -> torch.Tensor:
        """Get current position sizing adjustments."""
        # Return learned adjustment factors per position
        pass


class AdaptiveRiskController(nn.Module):
    """Adaptive risk management based on learned patterns."""
    
    def __init__(
        self,
        base_risk_limit: float = 0.20,  # 20% max drawdown
        learning_rate: float = 0.001
    ):
        super().__init__()
        # Implementation:
        # - Track actual vs. predicted drawdowns
        # - Learn optimal risk limits per regime
        # - Adapt limits to prevent future drawdowns
        
    def update_risk_limits(
        self,
        actual_drawdown: float,
        predicted_drawdown: float,
        regime: str
    ) -> float:
        """
        Update risk limits based on drawdown experience.
        
        Args:
            actual_drawdown: Actual realized drawdown
            predicted_drawdown: Predicted drawdown
            regime: Current market regime
        
        Returns:
            adjusted_risk_limit: Updated risk limit
        """
        # Learn from drawdown experience
        # Adjust limits per regime
        # Return updated limit
        pass
    
    def check_risk_constraints(
        self,
        proposed_weights: torch.Tensor,
        current_weights: torch.Tensor,
        portfolio_value: float
    ) -> Tuple[bool, torch.Tensor]:
        """
        Check if proposed weights violate risk constraints.
        
        Returns:
            is_valid: Whether weights satisfy constraints
            adjusted_weights: Risk-adjusted weights if needed
        """
        # Check drawdown limits
        # Check position size limits
        # Check correlation limits
        # Return validation and adjusted weights
        pass


class iDiffODEPortfolioModel(nn.Module):
    """Complete iDiffODE model for adaptive portfolio optimization."""
    
    def __init__(
        self,
        n_positions: int,
        d_model: int = 128,
        pnl_history_len: int = 100
    ):
        super().__init__()
        self.encoder = PortfolioStateEncoder(n_positions, d_model, pnl_history_len)
        self.ode_func = PortfolioODEFunc(d_model, n_positions=n_positions)
        self.diffusion = PortfolioDiffusionRefiner(d_model, n_positions=n_positions)
        self.allocator = InvertiblePortfolioAllocator(d_model, n_positions)
        self.pnl_learner = PnLLearner(n_positions)
        self.risk_controller = AdaptiveRiskController()
        
    def forward(
        self,
        portfolio_state: Dict[str, torch.Tensor],
        eval_times: Optional[torch.Tensor] = None,
        pnl_signal: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute optimal portfolio allocation.
        
        Args:
            portfolio_state: Dict with keys:
                - current_weights: [batch, n_positions]
                - portfolio_value: [batch, 1]
                - pnl_history: [batch, pnl_history_len]
                - risk_metrics: [batch, n_risk]
                - market_regime: [batch, n_regime]
                - model_predictions: [batch, n_positions]
            eval_times: [batch, eval_len] - Times to evaluate (optional)
            pnl_signal: [batch, 1] - Real-time P&L signal (optional)
        
        Returns:
            Dict with keys:
                - optimal_weights: [batch, eval_len, n_positions]
                - latent_trajectory: [batch, eval_len, d_model]
                - risk_adjusted_weights: [batch, eval_len, n_positions]
                - position_adjustments: [batch, n_positions]
        """
        # 1. Encode portfolio state
        h_0 = self.encoder(**portfolio_state)
        
        # 2. Solve Neural ODE (with P&L signal if available)
        if eval_times is None:
            # Determine optimal rebalancing times from model
            eval_times = self._determine_rebalancing_times(h_0)
        
        Z = odeint(
            lambda t, h: self.ode_func(t, h, pnl_signal, portfolio_state['market_regime']),
            h_0,
            eval_times
        )  # [batch, eval_len, d_model]
        
        # 3. Refine with diffusion
        risk_constraints = self.risk_controller.get_constraints(portfolio_state)
        Z_refined = self.diffusion(Z, risk_constraints)
        
        # 4. Map to optimal weights
        optimal_weights = self.allocator(Z_refined)  # [batch, eval_len, n_positions]
        
        # 5. Apply learned position adjustments
        position_adjustments = self.pnl_learner.get_position_adjustments()
        adjusted_weights = optimal_weights * position_adjustments.unsqueeze(0)
        
        # 6. Apply risk constraints
        is_valid, risk_adjusted_weights = self.risk_controller.check_risk_constraints(
            adjusted_weights[:, -1],  # Latest weights
            portfolio_state['current_weights'],
            portfolio_state['portfolio_value']
        )
        
        return {
            'optimal_weights': optimal_weights,
            'risk_adjusted_weights': risk_adjusted_weights.unsqueeze(0),
            'latent_trajectory': Z_refined,
            'position_adjustments': position_adjustments,
            'rebalancing_times': eval_times
        }
    
    def update_from_pnl(
        self,
        position_id: int,
        actual_pnl: float,
        expected_pnl: float,
        risk_metric: float
    ):
        """Update model from real-time P&L feedback."""
        # Update position sizing adjustments
        adjustment = self.pnl_learner.update_position_sizing(
            position_id, actual_pnl, expected_pnl, risk_metric
        )
        
        # Update risk controller if drawdown occurred
        if risk_metric < 0:  # Negative risk-adjusted return
            self.risk_controller.update_risk_limits(
                actual_drawdown=abs(risk_metric),
                predicted_drawdown=expected_pnl,
                regime='current'  # TODO: get actual regime
            )
        
        return adjustment
    
    def _determine_rebalancing_times(
        self,
        h_0: torch.Tensor
    ) -> torch.Tensor:
        """Determine optimal rebalancing times from portfolio state."""
        # Learn when to rebalance (not fixed schedule)
        # Based on regime changes, drift from optimal, risk events
        # Return: [batch, n_rebalance] tensor of times
        pass


class TradeGenerator:
    """Generate trade orders from optimal portfolio weights."""
    
    def __init__(
        self,
        min_trade_size: float = 0.01,  # 1% minimum position change
        max_position_change: float = 0.10  # 10% max change per rebalance
    ):
        self.min_trade_size = min_trade_size
        self.max_position_change = max_position_change
    
    def generate_trades(
        self,
        current_weights: torch.Tensor,
        optimal_weights: torch.Tensor,
        portfolio_value: float,
        prices: Dict[str, float]
    ) -> List[Dict]:
        """
        Generate trade orders to move from current to optimal weights.
        
        Args:
            current_weights: [n_positions] - Current position weights
            optimal_weights: [n_positions] - Target position weights
            portfolio_value: Total portfolio value
            prices: Dict[symbol -> price] - Current prices
        
        Returns:
            List of trade orders: [{'symbol': str, 'quantity': float, 'side': str, 'order_type': str}, ...]
        """
        # Compute required weight changes
        delta_weights = optimal_weights - current_weights
        
        # Apply constraints (min trade size, max change)
        delta_weights = self._apply_trade_constraints(delta_weights)
        
        # Convert weights to dollar amounts
        dollar_amounts = delta_weights * portfolio_value
        
        # Convert to share quantities
        trades = []
        for i, (symbol, price) in enumerate(prices.items()):
            quantity = dollar_amounts[i] / price
            if abs(quantity) > self.min_trade_size:
                trades.append({
                    'symbol': symbol,
                    'quantity': abs(quantity),
                    'side': 'BUY' if quantity > 0 else 'SELL',
                    'order_type': 'MARKET'  # or 'LIMIT' with price
                })
        
        return trades
    
    def _apply_trade_constraints(
        self,
        delta_weights: torch.Tensor
    ) -> torch.Tensor:
        """Apply min trade size and max change constraints."""
        # Clip to max_position_change
        delta_weights = torch.clamp(
            delta_weights,
            -self.max_position_change,
            self.max_position_change
        )
        
        # Zero out changes below min_trade_size
        delta_weights[torch.abs(delta_weights) < self.min_trade_size] = 0.0
        
        return delta_weights
```

### 5.3 Real-Time Integration

```python
# PORTFOLIO/adaptive_realtime/realtime_optimizer.py

class RealTimePortfolioOptimizer:
    """Real-time portfolio optimization system."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        update_frequency: float = 1.0  # Update every 1 second
    ):
        self.model = iDiffODEPortfolioModel(n_positions=50)  # Example: 50 positions
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        
        self.trade_generator = TradeGenerator()
        self.update_frequency = update_frequency
        self.last_update_time = None
        
        # Real-time data sources
        self.pnl_tracker = PnLTracker()
        self.market_data = MarketDataFeed()
        self.risk_monitor = RiskMonitor()
    
    def run_optimization_loop(self):
        """Main real-time optimization loop."""
        while True:
            # 1. Collect current portfolio state
            portfolio_state = self._collect_portfolio_state()
            
            # 2. Get real-time P&L
            pnl_signal = self.pnl_tracker.get_current_pnl()
            
            # 3. Compute optimal allocation
            result = self.model(
                portfolio_state,
                pnl_signal=pnl_signal
            )
            
            # 4. Generate trades if needed
            if self._should_rebalance(result, portfolio_state):
                trades = self.trade_generator.generate_trades(
                    current_weights=portfolio_state['current_weights'],
                    optimal_weights=result['risk_adjusted_weights'][0],
                    portfolio_value=portfolio_state['portfolio_value'].item(),
                    prices=self.market_data.get_current_prices()
                )
                
                # 5. Execute trades
                self._execute_trades(trades)
            
            # 6. Update model from P&L feedback
            self._update_model_from_pnl()
            
            # 7. Sleep until next update
            time.sleep(self.update_frequency)
    
    def _collect_portfolio_state(self) -> Dict[str, torch.Tensor]:
        """Collect current portfolio state from live system."""
        return {
            'current_weights': self._get_current_weights(),
            'portfolio_value': self._get_portfolio_value(),
            'pnl_history': self.pnl_tracker.get_history(),
            'risk_metrics': self.risk_monitor.get_current_metrics(),
            'market_regime': self.market_data.get_regime_indicators(),
            'model_predictions': self._get_model_predictions()
        }
    
    def _update_model_from_pnl(self):
        """Update model from real-time P&L feedback."""
        for position_id, pnl_data in self.pnl_tracker.get_per_position_pnl().items():
            self.model.update_from_pnl(
                position_id=position_id,
                actual_pnl=pnl_data['actual'],
                expected_pnl=pnl_data['expected'],
                risk_metric=pnl_data['sharpe']
            )
    
    def _should_rebalance(
        self,
        result: Dict[str, torch.Tensor],
        portfolio_state: Dict[str, torch.Tensor]
    ) -> bool:
        """Determine if rebalancing is needed."""
        current = portfolio_state['current_weights']
        optimal = result['risk_adjusted_weights'][0]
        
        # Rebalance if drift exceeds threshold
        drift = torch.norm(optimal - current)
        return drift > 0.05  # 5% drift threshold
```

---

## 6. Learning Mechanisms

### 6.1 P&L-Based Position Sizing

**Learning Signal**:

\[
L_i(t) = \text{P\&L}_i(t) - \hat{\text{P\&L}}_i(t)
\]

**Update Rule**:

\[
\alpha_i(t+1) = \alpha_i(t) + \eta \cdot \text{sign}(L_i(t)) \cdot \min(|L_i(t)|, \text{clip})
\]

Where:
- \(\alpha_i(t)\): Position sizing adjustment for position \(i\)
- \(\eta\): Learning rate
- Clip prevents extreme adjustments

**Adaptation**:
- Positive \(L_i\): Increase position size (performing better than expected)
- Negative \(L_i\): Decrease position size (performing worse than expected)
- Magnitude of adjustment proportional to error magnitude

### 6.2 Regime-Based Learning

**Regime Identification**:
- Volatility: Low/Medium/High
- Trend: Trending/Ranging
- Correlation: High/Low correlation between positions

**Regime-Specific Adjustments**:

\[
\beta_r(t) = \beta_r^{\text{base}} \cdot \exp\left(\sum_{i \in \text{regime } r} L_i(t)\right)
\]

Where \(\beta_r(t)\) adapts based on P&L performance in regime \(r\).

### 6.3 Risk Limit Learning

**Drawdown Learning**:

\[
\gamma(t+1) = \gamma(t) - \eta_{\text{risk}} \cdot \max(0, \text{DD}_{\text{actual}}(t) - \text{DD}_{\text{predicted}}(t))
\]

Where:
- If actual drawdown exceeds predicted, tighten risk limits
- Learn optimal risk limits per regime

### 6.4 Sharpe Ratio Optimization

**Objective**:

\[
\max_{w(t)} \frac{\mathbb{E}[\text{Returns}(t)]}{\sigma(\text{Returns}(t))}
\]

**Learning**:
- Track Sharpe ratio over time
- Adjust allocation to maximize Sharpe
- Learn optimal weights per regime

---

## 7. Integration Points

### 7.1 Live Trading System Integration

**IBKR Integration**:
- Real-time position data
- Real-time P&L tracking
- Trade execution
- Market data feed

**Alpaca Integration**:
- Paper trading for testing
- Real-time market data
- Order execution

### 7.2 Model Prediction Integration

**Input**: Model predictions from training pipeline
- Target predictions per symbol
- Confidence scores
- Feature importance

**Usage**: 
- Inform initial position sizing
- Update expected P&L
- Adjust allocation based on prediction confidence

### 7.3 Risk Management Integration

**Input**: Risk limits, constraints
- Max drawdown limits
- Position size limits
- Correlation limits

**Usage**:
- Enforce constraints in allocation
- Adapt limits based on learned patterns
- Prevent excessive risk

---

## 8. Implementation Roadmap

### Phase 1: Core Framework (Weeks 1-4)
- Implement portfolio state encoder
- Implement Neural ODE for portfolio dynamics
- Implement basic diffusion model
- Implement invertible allocator
- Create end-to-end forward pass
- **Deliverable**: Working iDiffODE portfolio model on simulated data

### Phase 2: P&L Learning (Weeks 5-6)
- Implement P&L tracking system
- Implement position sizing adaptation
- Implement regime-based learning
- Add learning update mechanisms
- **Deliverable**: Model learns from P&L feedback

### Phase 3: Real-Time Integration (Weeks 7-8)
- Integrate with live trading system (IBKR/Alpaca)
- Implement real-time data feeds
- Implement trade generation and execution
- Add monitoring and logging
- **Deliverable**: Real-time optimization system operational

### Phase 4: Risk & Production (Weeks 9-10)
- Implement adaptive risk controller
- Add safety mechanisms and rollback
- Optimize for low-latency execution
- Add comprehensive monitoring
- Production testing and validation
- **Deliverable**: Production-ready adaptive portfolio optimizer

---

## 9. Safety & Risk Management

### 9.1 Safety Mechanisms

- **Conservative Defaults**: Start with conservative position sizes
- **Hard Limits**: Never exceed maximum position sizes or drawdown limits
- **Gradual Adaptation**: Limit learning rate to prevent extreme adjustments
- **Human Override**: Manual position limits always respected
- **Rollback**: Can revert to baseline allocation at any time

### 9.2 Risk Controls

- **Position Size Limits**: Maximum position size per symbol
- **Portfolio Limits**: Maximum total exposure
- **Drawdown Limits**: Hard stop on drawdown
- **Correlation Limits**: Prevent over-concentration
- **Liquidity Checks**: Ensure positions are liquid enough

### 9.3 Monitoring

- **Real-Time Metrics**: P&L, Sharpe, drawdown, risk metrics
- **Learning Metrics**: Track adaptation effectiveness
- **Risk Metrics**: Monitor constraint violations
- **Performance Metrics**: Compare to baseline strategies

---

## 10. Success Metrics

### 10.1 Performance Metrics

- **Sharpe Ratio**: Target 20% improvement over baseline
- **Risk-Adjusted Returns**: Maximize returns per unit risk
- **Drawdown Reduction**: Reduce max drawdown by 30%
- **Consistency**: More consistent returns over time

### 10.2 Learning Effectiveness

- **Position Sizing Accuracy**: Learned sizes outperform static
- **Regime Adaptation**: Better performance in different regimes
- **Risk Limit Optimization**: Optimal risk limits learned
- **P&L Prediction**: Improved P&L prediction accuracy

### 10.3 Operational Metrics

- **Latency**: < 100ms optimization cycle
- **Update Frequency**: Real-time updates (1-10 seconds)
- **Trade Execution**: Efficient trade generation
- **System Stability**: 99.9% uptime

---

## 11. Advantages

### 11.1 Continuous-Time Modeling

- **Natural Handling**: Market data is irregularly sampled, iDiffODE handles naturally
- **Optimal Timing**: Learns when to rebalance, not fixed schedule
- **Smooth Transitions**: Smooth position adjustments, not abrupt changes

### 11.2 Real-Time Learning

- **P&L Feedback**: Learns from actual trading outcomes
- **Adaptive Sizing**: Position sizes adapt to performance
- **Regime Adaptation**: Adapts to changing market conditions

### 11.3 Risk Management

- **Predictive Risk**: Models risk continuously, not just at discrete points
- **Adaptive Limits**: Risk limits adapt based on experience
- **Constraint Satisfaction**: Ensures all risk constraints satisfied

### 11.4 Interpretability

- **Invertible Mapping**: Can reconstruct portfolio state from allocation
- **Visualization**: Can visualize portfolio dynamics over time
- **Debugging**: Can trace allocation decisions to portfolio state

---

## 12. Research Questions

1. **How does iDiffODE compare to discrete rebalancing?**
   - Evaluate Sharpe ratio improvement
   - Compare transaction costs
   - Assess risk-adjusted returns

2. **What is optimal learning rate for P&L adaptation?**
   - Balance between adaptation speed and stability
   - Regime-specific learning rates
   - Position-specific learning rates

3. **How to handle regime transitions?**
   - Detect regime changes quickly
   - Adapt allocation smoothly
   - Prevent whipsaw in transitions

4. **What are computational requirements?**
   - Latency requirements for real-time
   - GPU vs. CPU for inference
   - Scalability to large portfolios

---

## 13. References

- Neural ODEs: Chen et al., "Neural Ordinary Differential Equations" (NeurIPS 2018)
- Diffusion Models: Ho et al., "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)
- Invertible ResNets: Behrmann et al., "Invertible Residual Networks" (ICML 2019)
- Portfolio Optimization: Markowitz, "Portfolio Selection" (1952)
- Reinforcement Learning for Trading: Moody et al., "Reinforcement Learning for Trading" (1998)
- Continuous-Time Finance: Merton, "Continuous-Time Finance" (1990)
- **Integrated Feedback Loop**: `DOCS/internal/planning/INTEGRATED_LEARNING_FEEDBACK_LOOP.md` - Links ARPO with CILS and Training Pipeline for automated scheduling
- **Continuous Learning System**: `DOCS/internal/planning/CONTINUOUS_INTEGRATED_LEARNING_SYSTEM.md` - Training-side learning system

---

**End of Document**

