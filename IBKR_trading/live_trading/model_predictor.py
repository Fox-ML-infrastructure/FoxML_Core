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
Model Predictor - Unified Model Prediction Engine
================================================

Handles prediction from all trained models (tabular + sequential + multi-task)
across all horizons and strategies.
"""


import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import joblib
import torch
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX Runtime not available - some models may not load")

class ModelRegistry:
    """Registry for all trained models across horizons and strategies."""
    
    def __init__(self, model_dir: str = "TRAINING/models"):
        self.model_dir = Path(model_dir)
        self.models = {}
        self.horizons = ["5m", "10m", "15m", "30m", "60m", "120m", "1d", "5d", "20d"]
        self.strategies = ["single_task", "multi_task", "cascade"]
        self.families = [
            "LightGBM", "XGBoost", "MLP", "CNN1D", "LSTM", "Transformer",
            "Ensemble", "TabCNN", "TabLSTM", "TabTransformer", "RewardBased",
            "QuantileLightGBM", "NGBoost", "GMMRegime", "ChangePoint",
            "FTRLProximal", "VAE", "GAN", "MetaLearning", "MultiTask"
        ]
        self._load_models()
    
    def _load_models(self):
        """Load all available models from disk."""
        for horizon in self.horizons:
            for strategy in self.strategies:
                for family in self.families:
                    model_path = self.model_dir / f"{family}_{strategy}_{horizon}.pkl"
                    if model_path.exists():
                        try:
                            model = joblib.load(model_path)
                            key = f"{family}_{strategy}_{horizon}"
                            self.models[key] = {
                                'model': model,
                                'family': family,
                                'strategy': strategy,
                                'horizon': horizon,
                                'path': model_path
                            }
                            logger.info(f"Loaded model: {key}")
                        except Exception as e:
                            logger.warning(f"Failed to load {model_path}: {e}")
    
    def get_models_for_horizon(self, horizon: str) -> Dict[str, Any]:
        """Get all models for a specific horizon."""
        return {k: v for k, v in self.models.items() if v['horizon'] == horizon}
    
    def get_models_for_family(self, family: str) -> Dict[str, Any]:
        """Get all models for a specific family."""
        return {k: v for k, v in self.models.items() if v['family'] == family}

class ModelPredictor:
    """Unified model prediction engine."""
    
    def __init__(self, model_registry: ModelRegistry, device: str = 'cpu'):
        self.registry = model_registry
        self.device = torch.device(device)
        self.prediction_cache = {}
        self.cache_ttl = {}
        
    def predict_horizon(self, horizon: str, features: pd.DataFrame, 
                       symbols: List[str], target_col: str) -> Dict[str, np.ndarray]:
        """
        Get predictions from all models for a specific horizon.
        
        Args:
            horizon: Target horizon (e.g., '5m', '1d')
            features: Feature matrix [N, F] for all symbols
            symbols: List of symbols
            target_col: Target column name
            
        Returns:
            Dict mapping model_name -> predictions [N]
        """
        models = self.registry.get_models_for_horizon(horizon)
        predictions = {}
        
        for model_name, model_info in models.items():
            try:
                # Check cache first
                cache_key = f"{model_name}_{hash(str(features.values.tobytes()))}"
                if cache_key in self.prediction_cache:
                    cache_time = self.cache_ttl.get(cache_key, datetime.min)
                    if datetime.now() - cache_time < timedelta(minutes=5):
                        predictions[model_name] = self.prediction_cache[cache_key]
                        continue
                
                # Get prediction
                pred = self._predict_single_model(model_info, features, symbols, target_col)
                if pred is not None and np.isfinite(pred).all():
                    predictions[model_name] = pred
                    # Cache the result
                    self.prediction_cache[cache_key] = pred
                    self.cache_ttl[cache_key] = datetime.now()
                    
            except Exception as e:
                logger.warning(f"Failed to predict with {model_name}: {e}")
                continue
        
        return predictions
    
    def _predict_single_model(self, model_info: Dict, features: pd.DataFrame, 
                            symbols: List[str], target_col: str) -> Optional[np.ndarray]:
        """Predict with a single model."""
        model = model_info['model']
        family = model_info['family']
        
        try:
            if family in ['LightGBM', 'XGBoost', 'MLP']:
                # Tabular models
                X = features.values.astype(np.float32)
                pred = model.predict(X)
                return pred
                
            elif family in ['CNN1D', 'LSTM', 'Transformer', 'TabCNN', 'TabLSTM', 'TabTransformer']:
                # Sequential models - need to reshape to [N, T, F]
                # For now, assume features are already in sequence format
                X = features.values.astype(np.float32)
                if X.ndim == 2:
                    # Add time dimension if needed
                    X = X[:, np.newaxis, :]  # [N, 1, F]
                
                if hasattr(model, 'predict'):
                    pred = model.predict(X)
                else:
                    # PyTorch model
                    X_tensor = torch.from_numpy(X).float().to(self.device)
                    with torch.no_grad():
                        pred = model(X_tensor).cpu().numpy().flatten()
                
                return pred
                
            else:
                # Other models
                X = features.values.astype(np.float32)
                if hasattr(model, 'predict'):
                    pred = model.predict(X)
                else:
                    # Try calling the model directly
                    pred = model(X)
                
                return pred
                
        except Exception as e:
            logger.error(f"Error predicting with {family}: {e}")
            return None
    
    def predict_all_horizons(self, features: pd.DataFrame, symbols: List[str], 
                           target_col: str) -> Dict[str, Dict[str, np.ndarray]]:
        """Get predictions for all horizons."""
        all_predictions = {}
        
        for horizon in self.registry.horizons:
            horizon_preds = self.predict_horizon(horizon, features, symbols, target_col)
            if horizon_preds:
                all_predictions[horizon] = horizon_preds
                
        return all_predictions
    
    def get_barrier_predictions(self, features: pd.DataFrame, symbols: List[str]) -> Dict[str, np.ndarray]:
        """Get barrier probability predictions."""
        barrier_targets = ['will_peak_5m', 'will_valley_5m', 'y_will_peak_5m', 'y_will_valley_5m']
        barrier_preds = {}
        
        for target in barrier_targets:
            try:
                # Look for models trained on this barrier target
                models = {k: v for k, v in self.registry.models.items() 
                         if target in str(v['path'])}
                
                if models:
                    # Use the first available model
                    model_name = list(models.keys())[0]
                    model_info = models[model_name]
                    pred = self._predict_single_model(model_info, features, symbols, target)
                    if pred is not None:
                        barrier_preds[target] = pred
                        
            except Exception as e:
                logger.warning(f"Failed to get barrier predictions for {target}: {e}")
                
        return barrier_preds
