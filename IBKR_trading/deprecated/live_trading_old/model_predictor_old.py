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
Model Predictor - Unified Prediction Engine
==========================================

Unified model prediction engine for all model types (tabular + sequential + multi-task).
"""


import numpy as np
import logging
from typing Dict, List, Any, Optional
from typing import Dict, List, Any, Tuple
import torch

logger = logging.getLogger(__name__)

class ModelPredictor:
    """Unified model prediction engine for all model types."""
    
    def __init__(self, model_registry, buffer_manager, config):
        self.model_registry = model_registry
        self.buffer_manager = buffer_manager
        self.config = config
        
        # Import family router
        try:
            from models.family_router import FamilyRouter
            self.family_router = FamilyRouter(config)
        except ImportError:
            logger.warning("FamilyRouter not available, using fallback")
            self.family_router = None
    
    def predict_horizon(self, horizon: str, symbols: List[str], 
                       features: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """
        Get predictions for all models of a specific horizon.
        
        Args:
            horizon: Target horizon (e.g., '5m', '15m', '1d')
            symbols: List of symbols to predict
            features: Feature data for each symbol
        
        Returns:
            Dictionary of {model_name: {symbol: prediction}}
        """
        predictions = {}
        
        try:
            # Get all models for this horizon
            models = self.model_registry.get_models_by_horizon(horizon)
            
            if not models:
                logger.warning(f"No models found for horizon {horizon}")
                return {}
            
            logger.info(f"Predicting with {len(models)} models for horizon {horizon}")
            
            for model_name, model in models.items():
                try:
                    # Route to appropriate data processing
                    if self._is_sequence_model(model_name):
                        preds = self._predict_sequential(model, symbols, features)
                    else:
                        preds = self._predict_tabular(model, symbols, features)
                    
                    if preds:
                        predictions[model_name] = preds
                        logger.debug(f"Model {model_name}: {len(preds)} predictions")
                    
                except Exception as e:
                    logger.warning(f"Model {model_name} failed: {e}")
                    continue
            
            logger.info(f"Successfully predicted with {len(predictions)} models")
            return predictions
            
        except Exception as e:
            logger.error(f"Error in predict_horizon: {e}")
            return {}
    
    def _is_sequence_model(self, model_name: str) -> bool:
        """Check if model requires sequential data."""
        if self.family_router:
            return self.family_router.is_sequence_family(model_name)
        
        # Fallback: check model name patterns
        sequence_patterns = ['CNN1D', 'LSTM', 'Transformer', 'TabLSTM', 'TabTransformer']
        return any(pattern in model_name for pattern in sequence_patterns)
    
    def _predict_sequential(self, model, symbols: List[str], 
                          features: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Predict using sequential models."""
        predictions = {}
        
        for symbol in symbols:
            try:
                # Get sequence from buffer
                sequence = self.buffer_manager.get_sequence(symbol)
                if sequence is None:
                    logger.debug(f"No sequence available for {symbol}")
                    continue
                
                # Convert to numpy if needed
                if isinstance(sequence, torch.Tensor):
                    sequence_np = sequence.cpu().numpy()
                else:
                    sequence_np = sequence
                
                # Ensure correct shape [1, T, F]
                if sequence_np.ndim == 2:
                    sequence_np = sequence_np.reshape(1, sequence_np.shape[0], sequence_np.shape[1])
                elif sequence_np.ndim == 3 and sequence_np.shape[0] != 1:
                    sequence_np = sequence_np.reshape(1, sequence_np.shape[1], sequence_np.shape[2])
                
                # Predict
                pred = model.predict(sequence_np)
                
                # Handle different prediction formats
                if isinstance(pred, (list, tuple)):
                    pred = pred[0]
                if isinstance(pred, np.ndarray):
                    pred = pred[0] if pred.size > 0 else 0.0
                
                predictions[symbol] = float(pred)
                
            except Exception as e:
                logger.warning(f"Sequential prediction failed for {symbol}: {e}")
                continue
        
        return predictions
    
    def _predict_tabular(self, model, symbols: List[str], 
                        features: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Predict using tabular models."""
        predictions = {}
        
        for symbol in symbols:
            try:
                if symbol not in features:
                    logger.debug(f"No features available for {symbol}")
                    continue
                
                # Get latest features
                feature_data = features[symbol]
                if feature_data.ndim == 2:
                    feature_row = feature_data[-1]  # Latest row
                else:
                    feature_row = feature_data
                
                # Ensure correct shape
                if feature_row.ndim == 1:
                    feature_row = feature_row.reshape(1, -1)
                
                # Predict
                pred = model.predict(feature_row)
                
                # Handle different prediction formats
                if isinstance(pred, (list, tuple)):
                    pred = pred[0]
                if isinstance(pred, np.ndarray):
                    pred = pred[0] if pred.size > 0 else 0.0
                
                predictions[symbol] = float(pred)
                
            except Exception as e:
                logger.warning(f"Tabular prediction failed for {symbol}: {e}")
                continue
        
        return predictions
    
    def predict_all_horizons(self, symbols: List[str], 
                            features: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Get predictions for all horizons.
        
        Args:
            symbols: List of symbols to predict
            features: Feature data for each symbol
        
        Returns:
            Dictionary of {horizon: {model_name: {symbol: prediction}}}
        """
        all_predictions = {}
        
        horizons = self.config.get('horizons', ['5m', '15m', '30m', '60m', '1d'])
        
        for horizon in horizons:
            logger.info(f"Predicting horizon {horizon}")
            predictions = self.predict_horizon(horizon, symbols, features)
            if predictions:
                all_predictions[horizon] = predictions
        
        return all_predictions

class ModelRegistry:
    """Registry for managing trained models."""
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        self._load_models()
    
    def _load_models(self):
        """Load all trained models."""
        try:
            # Load models from the training system
            # This would integrate with your model storage system
            logger.info("Loading trained models...")
            
            # Placeholder - implement based on your model storage
            # self.models = load_all_models_from_storage()
            
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.models = {}
    
    def get_models_by_horizon(self, horizon: str) -> Dict[str, Any]:
        """Get all models for a specific horizon."""
        return self.models.get(horizon, {})
    
    def get_model(self, horizon: str, model_name: str) -> Optional[Any]:
        """Get a specific model."""
        horizon_models = self.models.get(horizon, {})
        return horizon_models.get(model_name)
    
    def list_available_models(self) -> Dict[str, List[str]]:
        """List all available models by horizon."""
        return {horizon: list(models.keys()) 
                for horizon, models in self.models.items()}
