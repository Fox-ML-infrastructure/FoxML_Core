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
Cost Arbitrator - Cost Model & Horizon Arbitration
================================================

Handle costs and horizon arbitration for multi-horizon trading.
"""


import numpy as np
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class CostArbitrator:
    """Handle costs and horizon arbitration."""
    
    def __init__(self, config):
        self.config = config
        self.cost_model = CostModel(config)
        self.arbitration_mode = config.get('arbitration_mode', 'winner')
        self.softmax_beta = config.get('softmax_beta', 2.0)
        self.timing_penalty_k = config.get('timing_penalty_k', 0.1)
        
        logger.info(f"CostArbitrator initialized: mode={self.arbitration_mode}")
    
    def arbitrate_horizons(self, alpha_by_horizon: Dict[str, Dict[str, float]],
                          market_data: Dict) -> Dict[str, float]:
        """
        Arbitrate between horizons.
        
        Args:
            alpha_by_horizon: Alpha by horizon and symbol
            market_data: Market data for cost calculation
        
        Returns:
            Final alpha after arbitration
        """
        try:
            # Apply costs
            alpha_net = self._apply_costs(alpha_by_horizon, market_data)
            
            if not alpha_net:
                logger.warning("No alpha after cost adjustment")
                return {}
            
            # Arbitrate horizons
            if self.arbitration_mode == 'winner':
                return self._winner_takes_most(alpha_net, market_data)
            elif self.arbitration_mode == 'softmax':
                return self._softmax_blend(alpha_net, market_data)
            else:
                logger.warning(f"Unknown arbitration mode: {self.arbitration_mode}")
                return self._winner_takes_most(alpha_net, market_data)
                
        except Exception as e:
            logger.error(f"Error in horizon arbitration: {e}")
            return {}
    
    def _apply_costs(self, alpha_by_horizon: Dict[str, Dict[str, float]], 
                    market_data: Dict) -> Dict[str, Dict[str, float]]:
        """Apply costs to alpha by horizon."""
        alpha_net = {}
        
        for horizon, alpha in alpha_by_horizon.items():
            try:
                costs = self.cost_model.estimate_costs(horizon, market_data)
                alpha_net[horizon] = {s: alpha[s] - costs.get(s, 0) 
                                    for s in alpha.keys()}
                
                logger.debug(f"Applied costs to horizon {horizon}: {len(alpha_net[horizon])} symbols")
                
            except Exception as e:
                logger.warning(f"Error applying costs to horizon {horizon}: {e}")
                alpha_net[horizon] = alpha
        
        return alpha_net
    
    def _winner_takes_most(self, alpha_net: Dict[str, Dict[str, float]], 
                          market_data: Dict) -> Dict[str, float]:
        """Winner-takes-most with adjacent blending."""
        try:
            # Calculate scores with timing penalty
            scores = {}
            for horizon, alpha in alpha_net.items():
                timing_penalty = self._calculate_timing_penalty(horizon, market_data)
                scores[horizon] = {s: alpha[s] - timing_penalty.get(s, 0) 
                                 for s in alpha.keys()}
            
            # Pick best horizon per symbol
            symbols = list(alpha_net[list(alpha_net.keys())[0]].keys())
            final_alpha = {}
            
            for symbol in symbols:
                # Find best horizon for this symbol
                best_horizon = max(scores.keys(), 
                                 key=lambda h: scores[h].get(symbol, -np.inf))
                
                # Adjacent blending (70% best, 30% adjacent)
                alpha_primary = alpha_net[best_horizon][symbol]
                alpha_adjacent = self._get_adjacent_alpha(alpha_net, best_horizon, symbol)
                
                final_alpha[symbol] = 0.7 * alpha_primary + 0.3 * alpha_adjacent
                
                logger.debug(f"Symbol {symbol}: best_horizon={best_horizon}, "
                           f"primary={alpha_primary:.4f}, adjacent={alpha_adjacent:.4f}")
            
            logger.info(f"Winner-takes-most arbitration: {len(final_alpha)} symbols")
            return final_alpha
            
        except Exception as e:
            logger.error(f"Error in winner-takes-most: {e}")
            return {}
    
    def _softmax_blend(self, alpha_net: Dict[str, Dict[str, float]], 
                      market_data: Dict) -> Dict[str, float]:
        """Softmax blending across horizons."""
        try:
            symbols = list(alpha_net[list(alpha_net.keys())[0]].keys())
            final_alpha = {}
            
            for symbol in symbols:
                # Get alphas for this symbol across all horizons
                alphas = [alpha_net[h][symbol] for h in alpha_net.keys()]
                horizons = list(alpha_net.keys())
                
                # Calculate softmax weights
                weights = self._softmax_weights(alphas, self.softmax_beta)
                
                # Blend
                final_alpha[symbol] = sum(w * a for w, a in zip(weights, alphas))
                
                logger.debug(f"Symbol {symbol}: softmax weights={weights}, "
                           f"alphas={alphas}, final={final_alpha[symbol]:.4f}")
            
            logger.info(f"Softmax arbitration: {len(final_alpha)} symbols")
            return final_alpha
            
        except Exception as e:
            logger.error(f"Error in softmax blend: {e}")
            return {}
    
    def _calculate_timing_penalty(self, horizon: str, market_data: Dict) -> Dict[str, float]:
        """Calculate timing penalty for a horizon."""
        try:
            # Convert horizon to minutes
            h_minutes = self._horizon_to_minutes(horizon)
            
            # Get volatility data
            vol_short = market_data.get('vol_short', {})
            
            # Calculate penalty
            penalty = {}
            for symbol, vol in vol_short.items():
                penalty[symbol] = self.timing_penalty_k * vol * np.sqrt(h_minutes / 5.0)
            
            return penalty
            
        except Exception as e:
            logger.warning(f"Error calculating timing penalty: {e}")
            return {}
    
    def _get_adjacent_alpha(self, alpha_net: Dict[str, Dict[str, float]], 
                           best_horizon: str, symbol: str) -> float:
        """Get alpha from adjacent horizon."""
        try:
            horizons = sorted(alpha_net.keys(), key=self._horizon_to_minutes)
            best_idx = horizons.index(best_horizon)
            
            # Get adjacent horizon
            if best_idx > 0:
                adjacent_horizon = horizons[best_idx - 1]
            elif best_idx < len(horizons) - 1:
                adjacent_horizon = horizons[best_idx + 1]
            else:
                # No adjacent horizon, use same horizon
                adjacent_horizon = best_horizon
            
            return alpha_net[adjacent_horizon].get(symbol, 0.0)
            
        except Exception as e:
            logger.warning(f"Error getting adjacent alpha: {e}")
            return 0.0
    
    def _softmax_weights(self, alphas: List[float], beta: float) -> List[float]:
        """Calculate softmax weights."""
        try:
            if not alphas:
                return []
            
            # Normalize alphas
            alphas = np.array(alphas)
            alphas_norm = alphas / (np.std(alphas) + 1e-8)
            
            # Calculate softmax
            exp_alphas = np.exp(beta * alphas_norm)
            weights = exp_alphas / np.sum(exp_alphas)
            
            return weights.tolist()
            
        except Exception as e:
            logger.warning(f"Error calculating softmax weights: {e}")
            # Return equal weights
            return [1.0 / len(alphas)] * len(alphas)
    
    def _horizon_to_minutes(self, horizon: str) -> float:
        """Convert horizon string to minutes."""
        try:
            if horizon.endswith('m'):
                return float(horizon[:-1])
            elif horizon.endswith('h'):
                return float(horizon[:-1]) * 60
            elif horizon.endswith('d'):
                return float(horizon[:-1]) * 1440
            else:
                # Default to minutes
                return float(horizon)
        except:
            return 60.0  # Default to 1 hour

class CostModel:
    """Estimate trading costs."""
    
    def __init__(self, config):
        self.k1 = config.get('cost_k1', 0.5)  # Spread cost
        self.k2 = config.get('cost_k2', 0.3)  # Vol cost
        self.k3 = config.get('cost_k3', 0.1)  # Participation cost
        self.use_adaptive_costs = config.get('use_adaptive_costs', False)
        
        logger.info(f"CostModel initialized: k1={self.k1}, k2={self.k2}, k3={self.k3}")
    
    def estimate_costs(self, horizon: str, market_data: Dict) -> Dict[str, float]:
        """
        Estimate costs in bps.
        
        Formula: c = k1 * spread_bps + k2 * vol * sqrt(h/5) + k3 * participation^0.6
        """
        try:
            costs = {}
            
            # Get market data
            spreads = market_data.get('spreads', {})
            vol_short = market_data.get('vol_short', {})
            participation = market_data.get('participation', {})
            
            # Convert horizon to minutes
            h_minutes = self._horizon_to_minutes(horizon)
            
            for symbol in market_data.get('symbols', []):
                # Get symbol-specific data
                spread_bps = spreads.get(symbol, 2.0)
                vol = vol_short.get(symbol, 0.15)
                part = participation.get(symbol, 0.01)
                
                # Calculate cost components
                spread_cost = self.k1 * spread_bps
                vol_cost = self.k2 * vol * np.sqrt(h_minutes / 5.0)
                participation_cost = self.k3 * (part ** 0.6)
                
                # Total cost
                total_cost = spread_cost + vol_cost + participation_cost
                
                # Apply adaptive adjustment if enabled
                if self.use_adaptive_costs:
                    total_cost = self._apply_adaptive_adjustment(total_cost, symbol, market_data)
                
                costs[symbol] = total_cost
                
                logger.debug(f"Symbol {symbol}: spread={spread_cost:.2f}, "
                           f"vol={vol_cost:.2f}, part={participation_cost:.2f}, "
                           f"total={total_cost:.2f}")
            
            logger.info(f"Estimated costs for {len(costs)} symbols, horizon {horizon}")
            return costs
            
        except Exception as e:
            logger.error(f"Error estimating costs: {e}")
            return {}
    
    def _apply_adaptive_adjustment(self, base_cost: float, symbol: str, 
                                  market_data: Dict) -> float:
        """Apply adaptive cost adjustment based on market conditions."""
        try:
            # Get recent volatility
            recent_vol = market_data.get('recent_vol', {}).get(symbol, 0.15)
            
            # Adjust cost based on volatility
            if recent_vol > 0.3:  # High volatility
                adjustment = 1.2
            elif recent_vol < 0.1:  # Low volatility
                adjustment = 0.8
            else:
                adjustment = 1.0
            
            return base_cost * adjustment
            
        except Exception as e:
            logger.warning(f"Error in adaptive adjustment: {e}")
            return base_cost
    
    def _horizon_to_minutes(self, horizon: str) -> float:
        """Convert horizon string to minutes."""
        try:
            if horizon.endswith('m'):
                return float(horizon[:-1])
            elif horizon.endswith('h'):
                return float(horizon[:-1]) * 60
            elif horizon.endswith('d'):
                return float(horizon[:-1]) * 1440
            else:
                return float(horizon)
        except:
            return 60.0

class AdvancedCostModel(CostModel):
    """Advanced cost model with additional features."""
    
    def __init__(self, config):
        super().__init__(config)
        self.use_market_impact = config.get('use_market_impact', False)
        self.use_liquidity_adjustment = config.get('use_liquidity_adjustment', False)
        self.market_impact_k = config.get('market_impact_k', 0.1)
    
    def estimate_costs_advanced(self, horizon: str, market_data: Dict) -> Dict[str, float]:
        """Estimate costs with advanced features."""
        try:
            # Get base costs
            base_costs = self.estimate_costs(horizon, market_data)
            
            # Apply market impact
            if self.use_market_impact:
                base_costs = self._apply_market_impact(base_costs, market_data)
            
            # Apply liquidity adjustment
            if self.use_liquidity_adjustment:
                base_costs = self._apply_liquidity_adjustment(base_costs, market_data)
            
            return base_costs
            
        except Exception as e:
            logger.error(f"Error in advanced cost estimation: {e}")
            return base_costs
    
    def _apply_market_impact(self, costs: Dict[str, float], 
                           market_data: Dict) -> Dict[str, float]:
        """Apply market impact adjustment."""
        adjusted_costs = {}
        
        for symbol, cost in costs.items():
            # Get market impact data
            impact = market_data.get('market_impact', {}).get(symbol, 0.0)
            
            # Adjust cost
            adjusted_costs[symbol] = cost + self.market_impact_k * impact
        
        return adjusted_costs
    
    def _apply_liquidity_adjustment(self, costs: Dict[str, float], 
                                  market_data: Dict) -> Dict[str, float]:
        """Apply liquidity adjustment."""
        adjusted_costs = {}
        
        for symbol, cost in costs.items():
            # Get liquidity data
            liquidity = market_data.get('liquidity', {}).get(symbol, 1.0)
            
            # Adjust cost (higher liquidity = lower cost)
            adjusted_costs[symbol] = cost / liquidity
        
        return adjusted_costs
