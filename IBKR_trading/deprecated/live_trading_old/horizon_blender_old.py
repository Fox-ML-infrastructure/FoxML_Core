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
Horizon Blender - Per-Horizon Model Blending
===========================================

Blend all models for a specific horizon using OOF-trained weights.
"""


import numpy as np
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class HorizonBlender:
    """Blend all models for a specific horizon."""
    
    def __init__(self, config):
        self.config = config
        self.blend_weights = self._load_blend_weights()
        
        logger.info(f"HorizonBlender initialized with {len(self.blend_weights)} horizons")
    
    def blend_horizon(self, horizon: str, predictions: Dict[str, Dict[str, float]]) -> Optional[Dict[str, float]]:
        """
        Blend all model predictions for a horizon.
        
        Args:
            horizon: Target horizon
            predictions: {model_name: {symbol: prediction}}
        
        Returns:
            Blended alpha for each symbol
        """
        try:
            # Get blend weights for this horizon
            w_h = self.blend_weights.get(horizon, {})
            
            if not w_h:
                logger.warning(f"No blend weights for horizon {horizon}")
                return None
            
            # Filter to available models
            available_models = [name for name in w_h.keys() if name in predictions]
            if not available_models:
                logger.warning(f"No available models for horizon {horizon}")
                return None
            
            logger.debug(f"Blending {len(available_models)} models for horizon {horizon}")
            
            # Renormalize weights
            W = np.array([w_h[name] for name in available_models])
            W = W / W.sum()
            
            # Get predictions matrix
            symbols = list(predictions[available_models[0]].keys())
            M = np.column_stack([predictions[name] for name in available_models])
            
            # Blend
            alpha_h = M @ W
            
            result = dict(zip(symbols, alpha_h))
            logger.info(f"Blended {len(result)} symbols for horizon {horizon}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error blending horizon {horizon}: {e}")
            return None
    
    def blend_all_horizons(self, predictions_by_horizon: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
        """
        Blend all horizons.
        
        Args:
            predictions_by_horizon: {horizon: {model_name: {symbol: prediction}}}
        
        Returns:
            {horizon: {symbol: blended_alpha}}
        """
        blended_results = {}
        
        for horizon, predictions in predictions_by_horizon.items():
            blended = self.blend_horizon(horizon, predictions)
            if blended:
                blended_results[horizon] = blended
        
        logger.info(f"Blended {len(blended_results)} horizons")
        return blended_results
    
    def _load_blend_weights(self) -> Dict[str, Dict[str, float]]:
        """Load OOF-trained blend weights."""
        try:
            # Load from config or file
            weights = self.config.get('blend_weights', {})
            
            # Default weights if none provided
            if not weights:
                weights = self._get_default_weights()
            
            logger.info(f"Loaded blend weights for {len(weights)} horizons")
            return weights
            
        except Exception as e:
            logger.error(f"Error loading blend weights: {e}")
            return self._get_default_weights()
    
    def _get_default_weights(self) -> Dict[str, Dict[str, float]]:
        """Get default blend weights."""
        return {
            '5m': {
                'LightGBM': 0.3, 'XGBoost': 0.25, 'CNN1D': 0.2, 
                'LSTM': 0.15, 'Transformer': 0.1
            },
            '15m': {
                'LightGBM': 0.35, 'XGBoost': 0.3, 'CNN1D': 0.2, 'LSTM': 0.15
            },
            '30m': {
                'LightGBM': 0.4, 'XGBoost': 0.3, 'CNN1D': 0.2, 'LSTM': 0.1
            },
            '60m': {
                'LightGBM': 0.45, 'XGBoost': 0.35, 'CNN1D': 0.2
            },
            '1d': {
                'LightGBM': 0.5, 'XGBoost': 0.3, 'MLP': 0.2
            }
        }
    
    def update_weights(self, horizon: str, weights: Dict[str, float]):
        """Update blend weights for a horizon."""
        self.blend_weights[horizon] = weights
        logger.info(f"Updated weights for horizon {horizon}")
    
    def get_weights(self, horizon: str) -> Dict[str, float]:
        """Get blend weights for a horizon."""
        return self.blend_weights.get(horizon, {})
    
    def list_horizons(self) -> List[str]:
        """List all available horizons."""
        return list(self.blend_weights.keys())

class AdvancedBlender:
    """Advanced blending with dynamic weights and model selection."""
    
    def __init__(self, config):
        self.config = config
        self.base_blender = HorizonBlender(config)
        self.use_dynamic_weights = config.get('use_dynamic_weights', False)
        self.model_selection_threshold = config.get('model_selection_threshold', 0.1)
    
    def blend_horizon_advanced(self, horizon: str, predictions: Dict[str, Dict[str, float]], 
                              market_data: Dict) -> Optional[Dict[str, float]]:
        """
        Advanced blending with dynamic weights and model selection.
        
        Args:
            horizon: Target horizon
            predictions: Model predictions
            market_data: Market data for dynamic weighting
        
        Returns:
            Blended alpha
        """
        try:
            if self.use_dynamic_weights:
                # Dynamic weighting based on recent performance
                weights = self._calculate_dynamic_weights(horizon, predictions, market_data)
            else:
                # Use static weights
                weights = self.base_blender.get_weights(horizon)
            
            # Model selection - drop low-performing models
            if self.model_selection_threshold > 0:
                predictions = self._select_models(predictions, weights)
                weights = {name: w for name, w in weights.items() if name in predictions}
            
            # Normalize weights
            if weights:
                total_weight = sum(weights.values())
                weights = {name: w / total_weight for name, w in weights.items()}
            
            # Blend
            return self._blend_with_weights(predictions, weights)
            
        except Exception as e:
            logger.error(f"Error in advanced blending: {e}")
            return self.base_blender.blend_horizon(horizon, predictions)
    
    def _calculate_dynamic_weights(self, horizon: str, predictions: Dict[str, Dict[str, float]], 
                                  market_data: Dict) -> Dict[str, float]:
        """Calculate dynamic weights based on recent performance."""
        # Placeholder for dynamic weighting logic
        # This would use recent performance metrics to adjust weights
        return self.base_blender.get_weights(horizon)
    
    def _select_models(self, predictions: Dict[str, Dict[str, float]], 
                      weights: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Select models based on performance threshold."""
        selected = {}
        for name, preds in predictions.items():
            if weights.get(name, 0) >= self.model_selection_threshold:
                selected[name] = preds
        return selected
    
    def _blend_with_weights(self, predictions: Dict[str, Dict[str, float]], 
                           weights: Dict[str, float]) -> Dict[str, float]:
        """Blend predictions with given weights."""
        if not predictions or not weights:
            return {}
        
        # Get symbols
        symbols = list(predictions[list(predictions.keys())[0]].keys())
        
        # Calculate weighted average
        blended = {}
        for symbol in symbols:
            weighted_sum = 0.0
            total_weight = 0.0
            
            for model_name, preds in predictions.items():
                if model_name in weights and symbol in preds:
                    weight = weights[model_name]
                    weighted_sum += weight * preds[symbol]
                    total_weight += weight
            
            if total_weight > 0:
                blended[symbol] = weighted_sum / total_weight
            else:
                blended[symbol] = 0.0
        
        return blended
