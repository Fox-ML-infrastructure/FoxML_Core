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
Cost Arbitrator - Cost Model and Horizon Arbitration
===================================================

Handles cost estimation and horizon arbitration for optimal alpha selection.
Uses C++ kernels for hot path operations.
"""


import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import sys
import os

# Add C++ engine to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'cpp_engine', 'python_bindings'))

try:
    import ibkr_trading_engine_py as cpp_engine
    CPP_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("C++ engine available for cost arbitration operations")
except ImportError:
    CPP_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("C++ engine not available, falling back to Python implementations")

logger = logging.getLogger(__name__)

class CostModel:
    """Cost model for estimating trading costs per symbol/horizon."""
    
    def __init__(self, k1: float = 0.5, k2: float = 0.3, k3: float = 0.2):
        self.k1 = k1  # Spread coefficient
        self.k2 = k2  # Volatility coefficient  
        self.k3 = k3  # Participation coefficient
        self.cost_history = defaultdict(list)
        
    def estimate_costs(self, market_data: Dict, horizon_minutes: int, 
                      participation_rate: float = 0.1) -> np.ndarray:
        """
        Estimate costs per symbol for a given horizon.
        
        Args:
            market_data: Dict with 'spread_bps', 'vol_short', 'volume' keys
            horizon_minutes: Horizon in minutes
            participation_rate: Participation rate (0-1)
            
        Returns:
            Cost estimates in bps [N]
        """
        spread_bps = market_data.get('spread_bps', np.zeros(len(market_data.get('symbols', []))))
        vol_short = market_data.get('vol_short', np.ones(len(spread_bps)))
        volume = market_data.get('volume', np.ones(len(spread_bps)))
        
        # Cost formula: c_h = k1 * spread + k2 * vol * sqrt(h/5) + k3 * participation^0.6
        spread_cost = self.k1 * spread_bps
        vol_cost = self.k2 * vol_short * np.sqrt(horizon_minutes / 5.0)
        participation_cost = self.k3 * (participation_rate ** 0.6)
        
        total_cost = spread_cost + vol_cost + participation_cost
        
        return total_cost
    
    def update_cost_history(self, symbol: str, horizon: str, actual_cost: float):
        """Update cost history for adaptive estimation."""
        key = f"{symbol}_{horizon}"
        self.cost_history[key].append(actual_cost)
        
        # Keep only recent history
        if len(self.cost_history[key]) > 100:
            self.cost_history[key] = self.cost_history[key][-100:]
    
    def get_adaptive_cost(self, symbol: str, horizon: str, base_cost: float) -> float:
        """Get adaptive cost estimate based on history."""
        key = f"{symbol}_{horizon}"
        if key in self.cost_history and self.cost_history[key]:
            # Use EWMA of historical costs
            alpha = 0.1
            adaptive_cost = base_cost
            for cost in self.cost_history[key][-10:]:  # Last 10 observations
                adaptive_cost = alpha * cost + (1 - alpha) * adaptive_cost
            return adaptive_cost
        return base_cost

class CostArbitrator:
    """Cost-aware horizon arbitrator."""
    
    def __init__(self, cost_model: CostModel):
        self.cost_model = cost_model
        self.horizon_scores = {}
        self.arbitration_mode = "winner"  # or "softmax"
        self.k_vol = 0.1  # Volatility penalty coefficient
        self.beta = 2.0   # Softmax temperature
        
    def compute_net_alpha(self, alpha: np.ndarray, costs: np.ndarray) -> np.ndarray:
        """Compute net alpha after subtracting costs."""
        return alpha - costs
    
    def arbitrate_horizons_winner(self, net_alphas: Dict[str, np.ndarray], 
                                market_data: Dict) -> Tuple[np.ndarray, str]:
        """
        Winner-takes-most arbitration with adjacent blending.
        
        Args:
            net_alphas: Dict mapping horizon -> net alpha [N]
            market_data: Market data for volatility penalty
            
        Returns:
            (final_alpha, selected_horizon)
        """
        if not net_alphas:
            return np.array([]), ""
        
        horizons = list(net_alphas.keys())
        n_symbols = len(next(iter(net_alphas.values())))
        
        # Compute scores with volatility penalty
        scores = {}
        for horizon, alpha in net_alphas.items():
            horizon_minutes = self._horizon_to_minutes(horizon)
            vol_penalty = self.k_vol * market_data.get('vol_short', np.ones(n_symbols)) * np.sqrt(horizon_minutes / 5.0)
            scores[horizon] = alpha - vol_penalty
        
        # Find best horizon per symbol
        score_matrix = np.column_stack([scores[h] for h in horizons])
        best_horizons = np.argmax(score_matrix, axis=1)
        
        # Adjacent blending (70% primary, 30% adjacent)
        final_alpha = np.zeros(n_symbols)
        for i, best_horizon_idx in enumerate(best_horizons):
            primary_horizon = horizons[best_horizon_idx]
            primary_alpha = net_alphas[primary_horizon][i]
            
            # Find adjacent horizon
            adjacent_horizon = self._find_adjacent_horizon(primary_horizon, horizons)
            if adjacent_horizon and adjacent_horizon in net_alphas:
                adjacent_alpha = net_alphas[adjacent_horizon][i]
                final_alpha[i] = 0.7 * primary_alpha + 0.3 * adjacent_alpha
            else:
                final_alpha[i] = primary_alpha
        
        # Return most common horizon
        most_common_horizon = max(set(best_horizons), key=list(best_horizons).count)
        selected_horizon = horizons[most_common_horizon]
        
        return final_alpha, selected_horizon
    
    def arbitrate_horizons_softmax(self, net_alphas: Dict[str, np.ndarray], 
                                  market_data: Dict) -> Tuple[np.ndarray, str]:
        """
        Softmax arbitration over horizons.
        
        Args:
            net_alphas: Dict mapping horizon -> net alpha [N]
            market_data: Market data for scaling
            
        Returns:
            (final_alpha, selected_horizon)
        """
        if not net_alphas:
            return np.array([]), ""
        
        horizons = list(net_alphas.keys())
        n_symbols = len(next(iter(net_alphas.values())))
        
        if CPP_AVAILABLE and n_symbols >= 4:
            # Use C++ SIMD implementation for large vectors
            try:
                alpha_matrix = np.column_stack([net_alphas[h] for h in horizons])
                vol_scales = []
                for horizon in horizons:
                    horizon_minutes = self._horizon_to_minutes(horizon)
                    vol_scale = market_data.get('vol_short', np.ones(n_symbols)) * np.sqrt(horizon_minutes / 5.0)
                    vol_scales.append(vol_scale)
                vol_matrix = np.column_stack(vol_scales)
                
                final_alpha, selected_idx = cpp_engine.horizon_softmax(
                    alpha_matrix.astype(np.float64),
                    vol_matrix.astype(np.float64),
                    self.beta
                )
                selected_horizon = horizons[selected_idx]
                return final_alpha, selected_horizon
            except Exception as e:
                logger.warning(f"C++ softmax arbitration failed: {e}, falling back to Python")
        
        # Python fallback implementation
        # Compute softmax weights
        alpha_matrix = np.column_stack([net_alphas[h] for h in horizons])
        
        # Scale by horizon-specific volatility
        scaled_alphas = np.zeros_like(alpha_matrix)
        for i, horizon in enumerate(horizons):
            horizon_minutes = self._horizon_to_minutes(horizon)
            vol_scale = market_data.get('vol_short', np.ones(n_symbols)) * np.sqrt(horizon_minutes / 5.0)
            scaled_alphas[:, i] = alpha_matrix[:, i] / (vol_scale + 1e-8)
        
        # Softmax
        exp_alphas = np.exp(self.beta * scaled_alphas)
        weights = exp_alphas / np.sum(exp_alphas, axis=1, keepdims=True)
        
        # Weighted combination
        final_alpha = np.sum(weights * alpha_matrix, axis=1)
        
        # Return most weighted horizon
        avg_weights = np.mean(weights, axis=0)
        selected_horizon = horizons[np.argmax(avg_weights)]
        
        return final_alpha, selected_horizon
    
    def arbitrate_horizons(self, net_alphas: Dict[str, np.ndarray], 
                          market_data: Dict) -> Tuple[np.ndarray, str]:
        """Main arbitration method."""
        if self.arbitration_mode == "winner":
            return self.arbitrate_horizons_winner(net_alphas, market_data)
        else:
            return self.arbitrate_horizons_softmax(net_alphas, market_data)
    
    def _horizon_to_minutes(self, horizon: str) -> int:
        """Convert horizon string to minutes."""
        if horizon.endswith('m'):
            return int(horizon[:-1])
        elif horizon == '1d':
            return 1440  # 24 * 60
        elif horizon == '5d':
            return 7200  # 5 * 24 * 60
        elif horizon == '20d':
            return 28800  # 20 * 24 * 60
        else:
            return 5  # Default to 5 minutes
    
    def _find_adjacent_horizon(self, primary_horizon: str, all_horizons: List[str]) -> Optional[str]:
        """Find adjacent horizon for blending."""
        horizon_order = ['5m', '10m', '15m', '30m', '60m', '120m', '1d', '5d', '20d']
        
        try:
            primary_idx = horizon_order.index(primary_horizon)
            # Try next horizon first, then previous
            if primary_idx + 1 < len(horizon_order):
                next_horizon = horizon_order[primary_idx + 1]
                if next_horizon in all_horizons:
                    return next_horizon
            if primary_idx - 1 >= 0:
                prev_horizon = horizon_order[primary_idx - 1]
                if prev_horizon in all_horizons:
                    return prev_horizon
        except ValueError:
            pass
        
        return None

class AdvancedCostModel(CostModel):
    """Advanced cost model with session-aware and correlation-aware adjustments."""
    
    def __init__(self, k1: float = 0.5, k2: float = 0.3, k3: float = 0.2):
        super().__init__(k1, k2, k3)
        self.session_adjustments = {}
        self.correlation_penalties = {}
        
    def get_session_adjustment(self, current_time: datetime) -> float:
        """Get session-based cost adjustment."""
        hour = current_time.hour
        
        # Higher costs during market open/close
        if 9 <= hour <= 10 or 15 <= hour <= 16:
            return 1.5
        elif 10 <= hour <= 15:
            return 1.0
        else:
            return 0.8
    
    def get_correlation_penalty(self, symbol: str, other_symbols: List[str], 
                               correlation_matrix: np.ndarray) -> float:
        """Get correlation-based penalty for position sizing."""
        if symbol not in other_symbols:
            return 1.0
        
        symbol_idx = other_symbols.index(symbol)
        correlations = correlation_matrix[symbol_idx, :]
        
        # Penalty based on average correlation
        avg_correlation = np.mean(np.abs(correlations))
        penalty = 1.0 + 0.5 * avg_correlation
        
        return penalty
