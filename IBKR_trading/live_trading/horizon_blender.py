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

Blends all regression models per horizon using OOF-trained ridge → simplex weights.
Uses C++ kernels for hot path operations.
"""


import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import sys
import os

# Add C++ engine to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'cpp_engine', 'python_bindings'))

try:
    import ibkr_trading_engine_py as cpp_engine
    CPP_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("C++ engine available for hot path operations")
except ImportError:
    CPP_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("C++ engine not available, falling back to Python implementations")

logger = logging.getLogger(__name__)

class HorizonBlender:
    """Per-horizon model blender using ridge regression + simplex projection."""
    
    def __init__(self, blender_dir: str = "TRAINING/blenders"):
        self.blender_dir = Path(blender_dir)
        self.blender_dir.mkdir(exist_ok=True)
        self.blenders = {}
        self.horizons = ["5m", "10m", "15m", "30m", "60m", "120m", "1d", "5d", "20d"]
        self._load_or_fit_blenders()
    
    def _load_or_fit_blenders(self):
        """Load existing blenders or fit new ones."""
        for horizon in self.horizons:
            blender_path = self.blender_dir / f"blender_{horizon}.pkl"
            if blender_path.exists():
                try:
                    blender_data = joblib.load(blender_path)
                    self.blenders[horizon] = blender_data
                    logger.info(f"Loaded blender for horizon {horizon}")
                except Exception as e:
                    logger.warning(f"Failed to load blender for {horizon}: {e}")
                    self.blenders[horizon] = None
            else:
                self.blenders[horizon] = None
    
    def fit_blender(self, horizon: str, oof_predictions: Dict[str, np.ndarray], 
                   targets: np.ndarray, ridge_alpha: float = 0.01) -> Dict[str, float]:
        """
        Fit a ridge blender for a specific horizon.
        
        Args:
            horizon: Target horizon
            oof_predictions: Dict mapping model_name -> OOF predictions [N]
            targets: True targets [N]
            ridge_alpha: Ridge regularization parameter
            
        Returns:
            Dict mapping model_name -> weight
        """
        if not oof_predictions:
            return {}
        
        # Prepare data
        model_names = list(oof_predictions.keys())
        X = np.column_stack([oof_predictions[name] for name in model_names])
        y = targets
        
        # Remove any rows with NaN
        valid_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) == 0:
            logger.warning(f"No valid data for horizon {horizon}")
            return {}
        
        # Fit ridge regression
        ridge = Ridge(alpha=ridge_alpha, fit_intercept=False)
        ridge.fit(X, y)
        
        # Get raw weights
        raw_weights = ridge.coef_
        
        # Project to simplex (non-negative, sum to 1)
        weights = self._project_to_simplex(raw_weights)
        
        # Create weight dict
        weight_dict = {name: weight for name, weight in zip(model_names, weights)}
        
        # Save blender
        blender_data = {
            'weights': weight_dict,
            'ridge_alpha': ridge_alpha,
            'model_names': model_names,
            'fitted_on': len(X)
        }
        
        blender_path = self.blender_dir / f"blender_{horizon}.pkl"
        joblib.dump(blender_data, blender_path)
        self.blenders[horizon] = blender_data
        
        logger.info(f"Fitted blender for horizon {horizon}: {weight_dict}")
        return weight_dict
    
    def _project_to_simplex(self, weights: np.ndarray) -> np.ndarray:
        """Project weights to simplex (non-negative, sum to 1)."""
        if CPP_AVAILABLE and len(weights) >= 4:
            # Use C++ SIMD implementation for large vectors
            try:
                return cpp_engine.project_simplex(weights.astype(np.float64))
            except Exception as e:
                logger.warning(f"C++ simplex projection failed: {e}, falling back to Python")
        
        # Python fallback implementation
        # Sort weights in descending order
        sorted_indices = np.argsort(weights)[::-1]
        sorted_weights = weights[sorted_indices]
        
        # Find the number of positive weights
        n = len(weights)
        for i in range(n):
            if sorted_weights[i] * (i + 1) <= np.sum(sorted_weights[:i+1]) - 1:
                break
        else:
            i = n - 1
        
        # Compute threshold
        threshold = (np.sum(sorted_weights[:i+1]) - 1) / (i + 1)
        
        # Project to simplex
        projected = np.maximum(0, weights - threshold)
        
        # Renormalize
        if np.sum(projected) > 0:
            projected = projected / np.sum(projected)
        else:
            projected = np.ones_like(weights) / len(weights)
        
        return projected
    
    def blend_horizon(self, horizon: str, predictions: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """
        Blend predictions for a specific horizon.
        
        Args:
            horizon: Target horizon
            predictions: Dict mapping model_name -> predictions [N]
            
        Returns:
            Blended predictions [N] or None if no blender available
        """
        if horizon not in self.blenders or self.blenders[horizon] is None:
            logger.warning(f"No blender available for horizon {horizon}")
            return None
        
        blender_data = self.blenders[horizon]
        weights = blender_data['weights']
        
        # Get available models
        available_models = [name for name in weights.keys() if name in predictions]
        if not available_models:
            logger.warning(f"No available models for horizon {horizon}")
            return None
        
        # Renormalize weights for available models
        available_weights = np.array([weights[name] for name in available_models])
        available_weights = available_weights / np.sum(available_weights)
        
        # Blend predictions
        blended = np.zeros_like(predictions[available_models[0]])
        for name, weight in zip(available_models, available_weights):
            pred = predictions[name]
            if np.isfinite(pred).all():
                blended += weight * pred
        
        return blended
    
    def blend_all_horizons(self, all_predictions: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """Blend predictions for all horizons."""
        blended_predictions = {}
        
        for horizon, predictions in all_predictions.items():
            blended = self.blend_horizon(horizon, predictions)
            if blended is not None:
                blended_predictions[horizon] = blended
                
        return blended_predictions

class AdvancedBlender(HorizonBlender):
    """Advanced blender with adaptive weights and ensemble methods."""
    
    def __init__(self, blender_dir: str = "TRAINING/blenders"):
        super().__init__(blender_dir)
        self.performance_history = {}
        self.adaptive_weights = {}
    
    def update_performance(self, horizon: str, model_name: str, 
                          actual_returns: np.ndarray, predictions: np.ndarray):
        """Update performance history for adaptive weighting."""
        if horizon not in self.performance_history:
            self.performance_history[horizon] = {}
        
        if model_name not in self.performance_history[horizon]:
            self.performance_history[horizon][model_name] = []
        
        # Calculate performance metric (e.g., correlation)
        correlation = np.corrcoef(actual_returns, predictions)[0, 1]
        if not np.isnan(correlation):
            self.performance_history[horizon][model_name].append(correlation)
            
            # Keep only recent history
            if len(self.performance_history[horizon][model_name]) > 100:
                self.performance_history[horizon][model_name] = \
                    self.performance_history[horizon][model_name][-100:]
    
    def get_adaptive_weights(self, horizon: str, model_names: List[str]) -> Dict[str, float]:
        """Get adaptive weights based on recent performance."""
        if horizon not in self.performance_history:
            return {name: 1.0 / len(model_names) for name in model_names}
        
        weights = {}
        for name in model_names:
            if name in self.performance_history[horizon]:
                recent_perf = self.performance_history[horizon][name][-10:]
                weights[name] = np.mean(recent_perf) if recent_perf else 0.0
            else:
                weights[name] = 0.0
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {name: weight / total_weight for name, weight in weights.items()}
        else:
            weights = {name: 1.0 / len(model_names) for name in model_names}
        
        return weights
