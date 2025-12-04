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
Barrier Gate - Timing and Risk Attenuator
=========================================

Uses barrier probabilities to scale final alpha multiplicatively.
Uses C++ kernels for hot path operations.
"""


import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import sys
import os

# Add C++ engine to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'cpp_engine', 'python_bindings'))

try:
    import ibkr_trading_engine_py as cpp_engine
    CPP_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("C++ engine available for barrier gate operations")
except ImportError:
    CPP_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("C++ engine not available, falling back to Python implementations")

logger = logging.getLogger(__name__)

class BarrierGate:
    """Barrier probability gating for timing and risk control."""
    
    def __init__(self, g_min: float = 0.2, gamma: float = 1.0, delta: float = 0.5):
        self.g_min = g_min
        self.gamma = gamma
        self.delta = delta
        self.calibration_params = {}
        
    def calibrate_barrier_probs(self, barrier_preds: Dict[str, np.ndarray], 
                              actual_peaks: np.ndarray, actual_valleys: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calibrate barrier probabilities using Platt scaling or isotonic regression.
        
        Args:
            barrier_preds: Dict mapping barrier_type -> raw probabilities [N]
            actual_peaks: Actual peak indicators [N]
            actual_valleys: Actual valley indicators [N]
            
        Returns:
            Dict mapping barrier_type -> calibrated probabilities [N]
        """
        calibrated = {}
        
        for barrier_type, probs in barrier_preds.items():
            if 'peak' in barrier_type.lower():
                actual = actual_peaks
            elif 'valley' in barrier_type.lower():
                actual = actual_valleys
            else:
                continue
            
            # Simple Platt scaling: P_calibrated = 1 / (1 + exp(-(a * P_raw + b)))
            # For now, use identity mapping (no calibration)
            # In production, fit a and b on validation data
            calibrated[barrier_type] = np.clip(probs, 0.0, 1.0)
            
        return calibrated
    
    def compute_gate(self, p_peak: np.ndarray, p_valley: np.ndarray) -> np.ndarray:
        """
        Compute multiplicative gate from barrier probabilities.
        
        Args:
            p_peak: Peak probabilities [N]
            p_valley: Valley probabilities [N]
            
        Returns:
            Gate values [N] in [g_min, 1.0]
        """
        if CPP_AVAILABLE and len(p_peak) >= 4:
            # Use C++ SIMD implementation for large vectors
            try:
                return cpp_engine.barrier_gate_batch(
                    p_peak.astype(np.float64), 
                    p_valley.astype(np.float64),
                    self.g_min, self.gamma, self.delta
                )
            except Exception as e:
                logger.warning(f"C++ barrier gate failed: {e}, falling back to Python")
        
        # Python fallback implementation
        # Gate formula: g = max(g_min, (1 - p_peak)^gamma * (0.5 + 0.5 * p_valley)^delta)
        peak_term = np.power(1.0 - np.clip(p_peak, 0.0, 1.0), self.gamma)
        valley_term = np.power(0.5 + 0.5 * np.clip(p_valley, 0.0, 1.0), self.delta)
        
        gate = peak_term * valley_term
        gate = np.clip(gate, self.g_min, 1.0)
        
        return gate
    
    def apply_gate(self, alpha: np.ndarray, barrier_preds: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Apply barrier gate to alpha values.
        
        Args:
            alpha: Alpha values [N]
            barrier_preds: Dict mapping barrier_type -> probabilities [N]
            
        Returns:
            Gated alpha values [N]
        """
        # Extract peak and valley probabilities
        p_peak = barrier_preds.get('will_peak_5m', np.zeros_like(alpha))
        p_valley = barrier_preds.get('will_valley_5m', np.zeros_like(alpha))
        
        # If specific barrier types not available, try alternatives
        if 'will_peak_5m' not in barrier_preds:
            for key in barrier_preds.keys():
                if 'peak' in key.lower():
                    p_peak = barrier_preds[key]
                    break
        
        if 'will_valley_5m' not in barrier_preds:
            for key in barrier_preds.keys():
                if 'valley' in key.lower():
                    p_valley = barrier_preds[key]
                    break
        
        # Compute and apply gate
        gate = self.compute_gate(p_peak, p_valley)
        gated_alpha = alpha * gate
        
        return gated_alpha

class AdvancedBarrierGate(BarrierGate):
    """Advanced barrier gate with horizon-specific gates and adaptive parameters."""
    
    def __init__(self, g_min: float = 0.2, gamma: float = 1.0, delta: float = 0.5):
        super().__init__(g_min, gamma, delta)
        self.horizon_gates = {}
        self.adaptive_params = {}
        
    def set_horizon_gate(self, horizon: str, g_min: float, gamma: float, delta: float):
        """Set gate parameters for a specific horizon."""
        self.horizon_gates[horizon] = {
            'g_min': g_min,
            'gamma': gamma,
            'delta': delta
        }
    
    def apply_horizon_gate(self, alpha: np.ndarray, barrier_preds: Dict[str, np.ndarray], 
                          horizon: str) -> np.ndarray:
        """Apply horizon-specific gate."""
        if horizon in self.horizon_gates:
            params = self.horizon_gates[horizon]
            # Temporarily override parameters
            old_params = (self.g_min, self.gamma, self.delta)
            self.g_min = params['g_min']
            self.gamma = params['gamma']
            self.delta = params['delta']
            
            gated_alpha = self.apply_gate(alpha, barrier_preds)
            
            # Restore parameters
            self.g_min, self.gamma, self.delta = old_params
            return gated_alpha
        else:
            return self.apply_gate(alpha, barrier_preds)
    
    def update_adaptive_params(self, performance_metrics: Dict[str, float]):
        """Update gate parameters based on performance metrics."""
        # Simple adaptive logic - in production, use more sophisticated methods
        if 'sharpe' in performance_metrics:
            sharpe = performance_metrics['sharpe']
            if sharpe < 0.5:
                # Increase gating when performance is poor
                self.gamma = min(2.0, self.gamma * 1.1)
                self.g_min = min(0.5, self.g_min * 1.1)
            elif sharpe > 2.0:
                # Decrease gating when performance is good
                self.gamma = max(0.5, self.gamma * 0.95)
                self.g_min = max(0.1, self.g_min * 0.95)

class BarrierProbabilityProvider:
    """Provider for barrier probabilities from trained models."""
    
    def __init__(self, model_predictor):
        self.model_predictor = model_predictor
        self.barrier_models = {}
        self._load_barrier_models()
    
    def _load_barrier_models(self):
        """Load models trained on barrier targets."""
        # Look for models trained on barrier targets
        for model_name, model_info in self.model_predictor.registry.models.items():
            if any(target in str(model_info['path']) for target in 
                   ['will_peak', 'will_valley', 'y_will_peak', 'y_will_valley']):
                self.barrier_models[model_name] = model_info
    
    def get_barrier_probabilities(self, features: pd.DataFrame, symbols: List[str]) -> Dict[str, np.ndarray]:
        """Get barrier probabilities from trained models."""
        barrier_probs = {}
        
        for model_name, model_info in self.barrier_models.items():
            try:
                # Extract target from model path or name
                target = None
                for t in ['will_peak', 'will_valley', 'y_will_peak', 'y_will_valley']:
                    if t in str(model_info['path']):
                        target = t
                        break
                
                if target:
                    pred = self.model_predictor._predict_single_model(
                        model_info, features, symbols, target
                    )
                    if pred is not None:
                        barrier_probs[target] = pred
                        
            except Exception as e:
                logger.warning(f"Failed to get barrier probability from {model_name}: {e}")
        
        return barrier_probs
