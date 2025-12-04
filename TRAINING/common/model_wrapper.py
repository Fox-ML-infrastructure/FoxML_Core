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
Model wrapper classes for saving compatibility.
This ensures all models have the proper attributes for the saving system.
"""


import joblib
import numpy as np
from typing import Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

class ModelWrapper:
    """Base wrapper for all models to ensure saving compatibility."""
    
    def __init__(self, model: Any, scaler: Optional[Any] = None, imputer: Optional[Any] = None):
        self.model = model
        self.scaler = scaler
        self.imputer = imputer
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the wrapped model."""
        return self.model.predict(X)
    
    def save(self, path: str) -> None:
        """Save the model (default implementation)."""
        joblib.dump(self.model, path)

class LightGBMWrapper(ModelWrapper):
    """Wrapper for LightGBM models with save_model method."""
    
    def save_model(self, path: str) -> None:
        """Save LightGBM model using its native save_model method."""
        if hasattr(self.model, 'save_model'):
            self.model.save_model(path)
        else:
            # Fallback to joblib if save_model not available
            joblib.dump(self.model, path)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using LightGBM model."""
        return self.model.predict(X)

class TensorFlowWrapper(ModelWrapper):
    """Wrapper for TensorFlow/Keras models with save method."""
    
    def save(self, path: str) -> None:
        """Save TensorFlow model using its native save method."""
        if hasattr(self.model, 'save'):
            self.model.save(path)
        else:
            # Fallback to joblib if save not available
            joblib.dump(self.model, path)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using TensorFlow model."""
        return self.model.predict(X)

class XGBoostWrapper(ModelWrapper):
    """Wrapper for XGBoost models (uses joblib saving)."""
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using XGBoost model."""
        return self.model.predict(X)

def wrap_model_for_saving(model: Any, model_type: str, scaler: Optional[Any] = None, imputer: Optional[Any] = None) -> ModelWrapper:
    """
    Wrap a model for saving compatibility based on its type.
    
    Args:
        model: The trained model
        model_type: Type of model ('lightgbm', 'tensorflow', 'xgboost', 'sklearn')
        scaler: Optional scaler object
        imputer: Optional imputer object
    
    Returns:
        Wrapped model with proper saving methods
    """
    
    if model_type.lower() in ['lightgbm', 'lgb']:
        return LightGBMWrapper(model, scaler, imputer)
    elif model_type.lower() in ['tensorflow', 'keras', 'mlp', 'cnn1d', 'lstm', 'transformer', 'tabcnn', 'tablstm', 'tabtransformer', 'vae', 'gan', 'neural_network', 'meta_learning', 'multi_task', 'ensemble']:
        return TensorFlowWrapper(model, scaler, imputer)
    elif model_type.lower() in ['xgboost', 'xgb']:
        return XGBoostWrapper(model, scaler, imputer)
    else:
        # Default wrapper for sklearn and other models
        return ModelWrapper(model, scaler, imputer)

def get_model_saving_info(model: Any) -> dict:
    """
    Get information about how to save a model.
    
    Args:
        model: The model to analyze
    
    Returns:
        Dictionary with saving information
    """
    
    # Detect PyTorch models
    is_pytorch = 'torch' in str(type(model)).lower() or hasattr(model, 'state_dict')
    
    info = {
        'has_save_model': hasattr(model, 'save_model'),
        'has_save': hasattr(model, 'save'),
        'model_type': 'pytorch' if is_pytorch else type(model).__name__,
        'is_lightgbm': hasattr(model, 'save_model') and 'lightgbm' in str(type(model)).lower(),
        'is_tensorflow': hasattr(model, 'save') and ('tensorflow' in str(type(model)).lower() or 'keras' in str(type(model)).lower()),
        'is_xgboost': 'xgboost' in str(type(model)).lower() or 'xgb' in str(type(model)).lower(),
        'is_pytorch': is_pytorch
    }
    
    return info
