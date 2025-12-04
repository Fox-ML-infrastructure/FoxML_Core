"""
Copyright (c) 2025 Fox ML Infrastructure

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

"""
Position Sizer - Alpha to Weights Conversion
==========================================

Converts final alpha to target weights with vol scaling, risk management, and execution.
Uses C++ kernels for hot path operations.
"""


import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from scipy.optimize import minimize
import warnings
import sys
import os

# Add C++ engine to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'cpp_engine', 'python_bindings'))

try:
    import ibkr_trading_engine_py as cpp_engine
    CPP_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("C++ engine available for position sizing operations")
except ImportError:
    CPP_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("C++ engine not available, falling back to Python implementations")

logger = logging.getLogger(__name__)

class PositionSizer:
    """Converts alpha to position weights with risk management."""
    
    def __init__(self, z_max: float = 3.0, max_weight: float = 0.05, 
                 target_gross: float = 0.5, no_trade_band: float = 0.008):
        self.z_max = z_max
        self.max_weight = max_weight
        self.target_gross = target_gross
        self.no_trade_band = no_trade_band
        self.vol_estimates = {}
        
    def estimate_volatility(self, returns: np.ndarray, alpha: float = 0.1) -> np.ndarray:
        """Estimate volatility using EWMA."""
        if len(returns) == 0:
            return np.ones(1)
        
        vol = np.zeros_like(returns)
        vol[0] = np.abs(returns[0])
        
        for i in range(1, len(returns)):
            vol[i] = alpha * np.abs(returns[i]) + (1 - alpha) * vol[i-1]
        
        return vol
    
    def vol_scale_alpha(self, alpha: np.ndarray, vol_estimates: np.ndarray) -> np.ndarray:
        """Scale alpha by volatility estimates."""
        # Clip volatility to avoid division by zero
        vol_safe = np.clip(vol_estimates, 1e-8, None)
        z_scores = alpha / vol_safe
        z_scores = np.clip(z_scores, -self.z_max, self.z_max)
        return z_scores
    
    def cross_sectional_standardize(self, z_scores: np.ndarray) -> np.ndarray:
        """Cross-sectional z-score standardization."""
        if len(z_scores) == 0:
            return z_scores
        
        mean_z = np.mean(z_scores)
        std_z = np.std(z_scores)
        
        if std_z > 1e-8:
            return (z_scores - mean_z) / std_z
        else:
            return z_scores - mean_z
    
    def risk_parity_ridge(self, z_scores: np.ndarray, covariance: np.ndarray, 
                         lambda_reg: float = 0.01) -> np.ndarray:
        """
        Solve ridge risk parity optimization.
        
        min_w 0.5 * w^T * Σ * w - λ * z^T * w
        s.t. |w| <= w_max, sum(|w|) = target_gross
        """
        n = len(z_scores)
        if n == 0:
            return np.array([])
        
        if CPP_AVAILABLE and n >= 4:
            # Use C++ SIMD implementation for large matrices
            try:
                return cpp_engine.risk_parity_ridge(
                    z_scores.astype(np.float64),
                    covariance.astype(np.float64),
                    lambda_reg
                )
            except Exception as e:
                logger.warning(f"C++ risk parity failed: {e}, falling back to Python")
        
        # Python fallback implementation
        # Add regularization to covariance
        reg_cov = covariance + lambda_reg * np.eye(n)
        
        # Solve: w = λ * (Σ + εI)^(-1) * z
        try:
            inv_cov = np.linalg.inv(reg_cov)
            w_raw = lambda_reg * inv_cov @ z_scores
        except np.linalg.LinAlgError:
            # Fallback to diagonal
            w_raw = lambda_reg * z_scores / (np.diag(reg_cov) + 1e-8)
        
        return w_raw
    
    def apply_caps_and_renormalize(self, weights: np.ndarray, 
                                 current_weights: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply weight caps and renormalize to target gross."""
        # Apply individual caps
        capped_weights = np.clip(weights, -self.max_weight, self.max_weight)
        
        # Renormalize to target gross
        current_gross = np.sum(np.abs(capped_weights))
        if current_gross > 0:
            scale_factor = self.target_gross / current_gross
            capped_weights = capped_weights * scale_factor
        
        return capped_weights
    
    def apply_no_trade_band(self, target_weights: np.ndarray, 
                           current_weights: np.ndarray) -> np.ndarray:
        """Apply no-trade band to reduce turnover."""
        if current_weights is None:
            return target_weights
        
        # Calculate drift
        drift = np.abs(target_weights - current_weights)
        
        # Only trade symbols with significant drift
        trade_mask = drift > self.no_trade_band
        
        # Keep current weights for symbols below threshold
        final_weights = np.where(trade_mask, target_weights, current_weights)
        
        # Renormalize to maintain target gross
        current_gross = np.sum(np.abs(final_weights))
        if current_gross > 0:
            scale_factor = self.target_gross / current_gross
            final_weights = final_weights * scale_factor
        
        return final_weights
    
    def size_positions(self, alpha: np.ndarray, market_data: Dict, 
                      current_weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Main position sizing method.
        
        Args:
            alpha: Final alpha values [N]
            market_data: Dict with 'vol_short', 'covariance', 'returns' keys
            current_weights: Current portfolio weights [N]
            
        Returns:
            Target weights [N]
        """
        if len(alpha) == 0:
            return np.array([])
        
        # 1. Volatility scaling
        vol_estimates = market_data.get('vol_short', np.ones_like(alpha))
        z_scores = self.vol_scale_alpha(alpha, vol_estimates)
        
        # 2. Cross-sectional standardization
        z_scores = self.cross_sectional_standardize(z_scores)
        
        # 3. Risk parity optimization (if covariance available)
        if 'covariance' in market_data and market_data['covariance'] is not None:
            cov_matrix = market_data['covariance']
            if cov_matrix.shape[0] == len(z_scores):
                weights = self.risk_parity_ridge(z_scores, cov_matrix)
            else:
                weights = z_scores
        else:
            weights = z_scores
        
        # 4. Apply caps and renormalize
        weights = self.apply_caps_and_renormalize(weights, current_weights)
        
        # 5. Apply no-trade band
        if current_weights is not None:
            weights = self.apply_no_trade_band(weights, current_weights)
        
        return weights

class AdvancedPositionSizer(PositionSizer):
    """Advanced position sizer with session-aware and correlation-aware adjustments."""
    
    def __init__(self, z_max: float = 3.0, max_weight: float = 0.05, 
                 target_gross: float = 0.5, no_trade_band: float = 0.008):
        super().__init__(z_max, max_weight, target_gross, no_trade_band)
        self.session_adjustments = {}
        self.correlation_penalties = {}
        
    def get_session_adjustment(self, current_time: datetime) -> float:
        """Get session-based position adjustment."""
        hour = current_time.hour
        
        # Reduce position sizes during volatile periods
        if 9 <= hour <= 10 or 15 <= hour <= 16:
            return 0.8  # Reduce by 20%
        elif 10 <= hour <= 15:
            return 1.0  # Normal sizing
        else:
            return 0.6  # Reduce by 40% outside market hours
        
    def get_correlation_penalty(self, symbol: str, other_symbols: List[str], 
                               correlation_matrix: np.ndarray) -> float:
        """Get correlation-based penalty for position sizing."""
        if symbol not in other_symbols:
            return 1.0
        
        symbol_idx = other_symbols.index(symbol)
        correlations = correlation_matrix[symbol_idx, :]
        
        # Penalty based on average correlation
        avg_correlation = np.mean(np.abs(correlations))
        penalty = 1.0 - 0.3 * avg_correlation  # Reduce size for highly correlated positions
        
        return np.clip(penalty, 0.5, 1.0)
    
    def apply_session_adjustments(self, weights: np.ndarray, current_time: datetime) -> np.ndarray:
        """Apply session-based adjustments to weights."""
        session_adj = self.get_session_adjustment(current_time)
        return weights * session_adj
    
    def apply_correlation_adjustments(self, weights: np.ndarray, symbols: List[str], 
                                    correlation_matrix: np.ndarray) -> np.ndarray:
        """Apply correlation-based adjustments to weights."""
        if correlation_matrix is None or len(symbols) != len(weights):
            return weights
        
        adjusted_weights = weights.copy()
        for i, symbol in enumerate(symbols):
            penalty = self.get_correlation_penalty(symbol, symbols, correlation_matrix)
            adjusted_weights[i] *= penalty
        
        return adjusted_weights

class PositionValidator:
    """Validates position weights for risk management."""
    
    def __init__(self, max_weight: float = 0.05, max_gross: float = 1.0, 
                 max_net: float = 0.3):
        self.max_weight = max_weight
        self.max_gross = max_gross
        self.max_net = max_net
        
    def validate_weights(self, weights: np.ndarray, symbols: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate position weights.
        
        Returns:
            (is_valid, list_of_violations)
        """
        violations = []
        
        # Check individual weight limits
        if np.any(np.abs(weights) > self.max_weight):
            violations.append(f"Individual weight exceeds {self.max_weight}")
        
        # Check gross exposure
        gross_exposure = np.sum(np.abs(weights))
        if gross_exposure > self.max_gross:
            violations.append(f"Gross exposure {gross_exposure:.3f} exceeds {self.max_gross}")
        
        # Check net exposure
        net_exposure = np.sum(weights)
        if abs(net_exposure) > self.max_net:
            violations.append(f"Net exposure {net_exposure:.3f} exceeds {self.max_net}")
        
        # Check for NaN or infinite values
        if np.any(~np.isfinite(weights)):
            violations.append("Non-finite weights detected")
        
        return len(violations) == 0, violations
    
    def fix_violations(self, weights: np.ndarray, symbols: List[str]) -> np.ndarray:
        """Fix common weight violations."""
        fixed_weights = weights.copy()
        
        # Fix non-finite values
        fixed_weights[~np.isfinite(fixed_weights)] = 0.0
        
        # Apply individual caps
        fixed_weights = np.clip(fixed_weights, -self.max_weight, self.max_weight)
        
        # Renormalize to target gross
        current_gross = np.sum(np.abs(fixed_weights))
        if current_gross > self.max_gross:
            scale_factor = self.max_gross / current_gross
            fixed_weights = fixed_weights * scale_factor
        
        return fixed_weights
