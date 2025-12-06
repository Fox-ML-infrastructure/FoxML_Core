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
Base 3D Model Trainer

Base class for models that work with 3D input data (samples, timesteps, features).
This includes sequential/time-series models like:
- CNN1D
- LSTM
- Transformer
- TabCNN
- TabLSTM
- TabTransformer

These models reshape 2D input (samples, features) to 3D (samples, timesteps, 1)
where timesteps = features.

TODO: Refactor existing 3D trainers to inherit from this base class.
This will centralize 3D-specific preprocessing, reshaping, and prediction logic.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import logging
from .base_trainer import BaseModelTrainer

logger = logging.getLogger(__name__)


class Base3DTrainer(BaseModelTrainer):
    """
    Base class for 3D model trainers (samples, timesteps, features).
    
    Handles 3D-specific preprocessing and prediction patterns.
    These models reshape 2D input (samples, features) to 3D (samples, timesteps, 1).
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize 3D trainer.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        # TODO: Add 3D-specific initialization if needed
    
    def preprocess_data(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Preprocess data for 3D models.
        
        This method handles:
        - Input validation (accepts 2D or 3D)
        - Standard preprocessing pipeline from BaseModelTrainer (requires 2D)
        - Returns 2D data (will be reshaped to 3D in train/predict)
        
        Args:
            X: Input features, shape (n_samples, n_features) or (n_samples, timesteps, 1)
            y: Optional target values, shape (n_samples,)
            
        Returns:
            Preprocessed (X, y) tuple, where X is 2D (n_samples, n_features)
        """
        # Handle 3D input by reshaping to 2D for preprocessing
        # This is needed when post_fit_sanity passes 3D data
        if len(X.shape) == 3:
            # Reshape from (samples, timesteps, 1) to (samples, timesteps)
            X = X.reshape(X.shape[0], X.shape[1])
        
        # Ensure input is 2D for preprocessing
        if len(X.shape) != 2:
            raise ValueError(f"Expected 2D or 3D input, got shape {X.shape}")
        
        # Use parent preprocessing (handles imputation, column masking, etc.)
        return super().preprocess_data(X, y)
    
    def _reshape_to_3d(self, X: np.ndarray) -> np.ndarray:
        """
        Reshape 2D input to 3D for model consumption.
        
        Args:
            X: Input features, shape (n_samples, n_features)
            
        Returns:
            Reshaped input, shape (n_samples, n_features, 1)
        """
        if len(X.shape) == 3:
            # Already 3D, return as-is
            return X
        if len(X.shape) != 2:
            raise ValueError(f"Expected 2D input, got shape {X.shape}")
        return X.reshape(X.shape[0], X.shape[1], 1)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict on 2D or 3D input data.
        
        This method handles:
        - Input validation (accepts 2D or 3D)
        - Preprocessing via preprocess_data() (returns 2D)
        - Reshaping to 3D for model
        - Model prediction
        - Output validation
        
        Args:
            X: Input features, shape (n_samples, n_features) or (n_samples, timesteps, 1)
            
        Returns:
            Predictions, shape (n_samples,)
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Handle 3D input by reshaping to 2D for preprocessing
        # This is needed when post_fit_sanity passes 3D data
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], X.shape[1])
        
        # Preprocess (returns 2D)
        Xp, _ = self.preprocess_data(X, None)
        
        # Reshape to 3D for model
        Xp = self._reshape_to_3d(Xp)
        
        # TODO: Implement model-specific prediction
        # This should be overridden by subclasses
        raise NotImplementedError("Subclasses must implement model-specific prediction logic")
    
    @abstractmethod
    def _build_model(self, input_dim: int) -> Any:
        """
        Build the 3D model architecture.
        
        Args:
            input_dim: Number of timesteps (features after preprocessing)
            
        Returns:
            Model instance
        """
        pass
    
    @abstractmethod
    def train(self, X_tr: np.ndarray, y_tr: np.ndarray, 
              X_va=None, y_va=None, feature_names: List[str] = None, **kwargs) -> Any:
        """
        Train the 3D model.
        
        Args:
            X_tr: Training features, shape (n_samples, n_features)
            y_tr: Training targets, shape (n_samples,)
            X_va: Optional validation features
            y_va: Optional validation targets
            feature_names: Optional feature names
            **kwargs: Additional training arguments
            
        Returns:
            Trained model
        """
        pass

