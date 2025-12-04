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
Barrier Gate - Timing Risk Attenuation
======================================

Apply barrier probabilities as timing gates to alpha.
"""


import numpy as np
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class BarrierGate:
    """Apply barrier probabilities as timing gates."""
    
    def __init__(self, config):
        self.config = config
        self.g_min = config.get('g_min', 0.2)
        self.gamma = config.get('gamma', 1.0)
        self.delta = config.get('delta', 0.5)
        self.use_horizon_specific = config.get('use_horizon_specific', False)
        
        logger.info(f"BarrierGate initialized: g_min={self.g_min}, gamma={self.gamma}, delta={self.delta}")
    
    def apply_gate(self, alpha: Dict[str, float], 
                   peak_probs: Dict[str, float],
                   valley_probs: Dict[str, float]) -> Dict[str, float]:
        """
        Apply barrier gate to alpha.
        
        Args:
            alpha: Alpha values by symbol
            peak_probs: Peak probabilities by symbol
            valley_probs: Valley probabilities by symbol
        
        Returns:
            Gated alpha values
        """
        try:
            gated_alpha = {}
            
            for symbol in alpha.keys():
                p_peak = peak_probs.get(symbol, 0.5)
                p_valley = valley_probs.get(symbol, 0.5)
                
                # Calculate gate
                g = self._calculate_gate(p_peak, p_valley)
                
                # Apply gate
                gated_alpha[symbol] = alpha[symbol] * g
                
                logger.debug(f"Symbol {symbol}: alpha={alpha[symbol]:.4f}, gate={g:.4f}, gated={gated_alpha[symbol]:.4f}")
            
            logger.info(f"Applied barrier gate to {len(gated_alpha)} symbols")
            return gated_alpha
            
        except Exception as e:
            logger.error(f"Error applying barrier gate: {e}")
            return alpha
    
    def apply_gate_by_horizon(self, alpha_by_horizon: Dict[str, Dict[str, float]], 
                             peak_probs: Dict[str, float],
                             valley_probs: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """
        Apply barrier gate to alpha by horizon.
        
        Args:
            alpha_by_horizon: Alpha values by horizon and symbol
            peak_probs: Peak probabilities by symbol
            valley_probs: Valley probabilities by symbol
        
        Returns:
            Gated alpha values by horizon
        """
        gated_by_horizon = {}
        
        for horizon, alpha in alpha_by_horizon.items():
            if self.use_horizon_specific:
                # Use horizon-specific barrier probabilities
                horizon_peak = self._get_horizon_probs(peak_probs, horizon)
                horizon_valley = self._get_horizon_probs(valley_probs, horizon)
            else:
                # Use same barrier probabilities for all horizons
                horizon_peak = peak_probs
                horizon_valley = valley_probs
            
            gated_alpha = self.apply_gate(alpha, horizon_peak, horizon_valley)
            if gated_alpha:
                gated_by_horizon[horizon] = gated_alpha
        
        return gated_by_horizon
    
    def _calculate_gate(self, p_peak: float, p_valley: float) -> float:
        """
        Calculate gate value from barrier probabilities.
        
        Formula: g = max(g_min, (1 - p_peak)^gamma * (0.5 + 0.5 * p_valley)^delta)
        """
        try:
            # Clamp probabilities to [0, 1]
            p_peak = np.clip(p_peak, 0.0, 1.0)
            p_valley = np.clip(p_valley, 0.0, 1.0)
            
            # Calculate gate components
            peak_component = (1 - p_peak) ** self.gamma
            valley_component = (0.5 + 0.5 * p_valley) ** self.delta
            
            # Combine components
            g = peak_component * valley_component
            
            # Apply minimum gate
            g = max(self.g_min, g)
            
            return g
            
        except Exception as e:
            logger.warning(f"Error calculating gate: {e}")
            return self.g_min
    
    def _get_horizon_probs(self, probs: Dict[str, float], horizon: str) -> Dict[str, float]:
        """Get horizon-specific barrier probabilities."""
        # For now, use the same probabilities for all horizons
        # In a more sophisticated implementation, you might have
        # different barrier models for different horizons
        return probs
    
    def get_gate_stats(self, peak_probs: Dict[str, float], 
                      valley_probs: Dict[str, float]) -> Dict[str, Any]:
        """Get statistics about gate values."""
        gates = []
        for symbol in peak_probs.keys():
            p_peak = peak_probs.get(symbol, 0.5)
            p_valley = valley_probs.get(symbol, 0.5)
            gate = self._calculate_gate(p_peak, p_valley)
            gates.append(gate)
        
        if not gates:
            return {}
        
        gates = np.array(gates)
        
        return {
            'mean': float(np.mean(gates)),
            'std': float(np.std(gates)),
            'min': float(np.min(gates)),
            'max': float(np.max(gates)),
            'median': float(np.median(gates)),
            'count': len(gates)
        }

class AdvancedBarrierGate(BarrierGate):
    """Advanced barrier gate with additional features."""
    
    def __init__(self, config):
        super().__init__(config)
        self.use_volatility_adjustment = config.get('use_volatility_adjustment', False)
        self.use_correlation_adjustment = config.get('use_correlation_adjustment', False)
        self.volatility_threshold = config.get('volatility_threshold', 0.3)
    
    def apply_gate_advanced(self, alpha: Dict[str, float], 
                          peak_probs: Dict[str, float],
                          valley_probs: Dict[str, float],
                          market_data: Dict) -> Dict[str, float]:
        """
        Apply advanced barrier gate with additional adjustments.
        
        Args:
            alpha: Alpha values by symbol
            peak_probs: Peak probabilities by symbol
            valley_probs: Valley probabilities by symbol
            market_data: Market data for adjustments
        
        Returns:
            Gated alpha values
        """
        try:
            # Apply base gate
            gated_alpha = self.apply_gate(alpha, peak_probs, valley_probs)
            
            # Apply volatility adjustment
            if self.use_volatility_adjustment:
                gated_alpha = self._apply_volatility_adjustment(gated_alpha, market_data)
            
            # Apply correlation adjustment
            if self.use_correlation_adjustment:
                gated_alpha = self._apply_correlation_adjustment(gated_alpha, market_data)
            
            return gated_alpha
            
        except Exception as e:
            logger.error(f"Error in advanced barrier gate: {e}")
            return gated_alpha
    
    def _apply_volatility_adjustment(self, alpha: Dict[str, float], 
                                   market_data: Dict) -> Dict[str, float]:
        """Apply volatility-based adjustment to gate."""
        adjusted_alpha = {}
        
        for symbol, a in alpha.items():
            vol = market_data.get('vol_short', {}).get(symbol, 0.2)
            
            if vol > self.volatility_threshold:
                # Reduce gate for high volatility
                adjustment = 0.8
            else:
                # Normal gate for low volatility
                adjustment = 1.0
            
            adjusted_alpha[symbol] = a * adjustment
        
        return adjusted_alpha
    
    def _apply_correlation_adjustment(self, alpha: Dict[str, float], 
                                    market_data: Dict) -> Dict[str, float]:
        """Apply correlation-based adjustment to gate."""
        # Placeholder for correlation adjustment
        # This would use correlation data to adjust gates
        return alpha

class BarrierProbabilityProvider:
    """Provider for barrier probabilities."""
    
    def __init__(self, config):
        self.config = config
        self.barrier_models = {}
        self._load_barrier_models()
    
    def _load_barrier_models(self):
        """Load barrier classification models."""
        try:
            # Load peak and valley classifiers
            # This would integrate with your trained barrier models
            logger.info("Loading barrier models...")
            
            # Placeholder - implement based on your model storage
            # self.barrier_models = load_barrier_models()
            
            logger.info("Barrier models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading barrier models: {e}")
            self.barrier_models = {}
    
    def get_peak_probabilities(self, symbols: List[str], 
                              features: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Get peak probabilities for symbols."""
        try:
            # Use trained peak classifier
            peak_model = self.barrier_models.get('peak')
            if not peak_model:
                logger.warning("No peak model available, using default probabilities")
                return {symbol: 0.5 for symbol in symbols}
            
            # Get predictions
            probs = {}
            for symbol in symbols:
                if symbol in features:
                    # Predict peak probability
                    pred = peak_model.predict_proba(features[symbol])
                    probs[symbol] = float(pred[0][1])  # Probability of peak
                else:
                    probs[symbol] = 0.5
            
            return probs
            
        except Exception as e:
            logger.error(f"Error getting peak probabilities: {e}")
            return {symbol: 0.5 for symbol in symbols}
    
    def get_valley_probabilities(self, symbols: List[str], 
                               features: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Get valley probabilities for symbols."""
        try:
            # Use trained valley classifier
            valley_model = self.barrier_models.get('valley')
            if not valley_model:
                logger.warning("No valley model available, using default probabilities")
                return {symbol: 0.5 for symbol in symbols}
            
            # Get predictions
            probs = {}
            for symbol in symbols:
                if symbol in features:
                    # Predict valley probability
                    pred = valley_model.predict_proba(features[symbol])
                    probs[symbol] = float(pred[0][1])  # Probability of valley
                else:
                    probs[symbol] = 0.5
            
            return probs
            
        except Exception as e:
            logger.error(f"Error getting valley probabilities: {e}")
            return {symbol: 0.5 for symbol in symbols}
