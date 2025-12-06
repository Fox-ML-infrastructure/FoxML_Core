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
Base 2D Model Trainer

Base class for models that work with 2D input data (samples, features).
This includes cross-sectional models like:
- MLP
- LightGBM
- XGBoost
- NGBoost
- Ensemble
- GMMRegime
- ChangePoint
- FTRLProximal
- RewardBased
- QuantileLightGBM
- VAE
- GAN
- MetaLearning
- MultiTask

TODO: Refactor existing 2D trainers to inherit from this base class.
This will centralize 2D-specific preprocessing and prediction logic.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import logging
from .base_trainer import BaseModelTrainer

logger = logging.getLogger(__name__)


class Base2DTrainer(BaseModelTrainer):
    """
    Base class for 2D model trainers (samples, features).
    
    Handles 2D-specific preprocessing and prediction patterns.
    All 2D models expect input shape (n_samples, n_features).
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize 2D trainer.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        # TODO: Add 2D-specific initialization if needed
    
    def preprocess_data(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Preprocess 2D data (samples, features).
        
        This method handles:
        - Input validation (must be 2D)
        - Standard preprocessing pipeline from BaseModelTrainer
        - 2D-specific transformations if needed
        
        Args:
            X: Input features, shape (n_samples, n_features)
            y: Optional target values, shape (n_samples,)
            
        Returns:
            Preprocessed (X, y) tuple
        """
        # TODO: Add 2D-specific validation
        # Ensure input is 2D
        if len(X.shape) != 2:
            raise ValueError(f"Expected 2D input (samples, features), got shape {X.shape}")
        
        # Use parent preprocessing (handles imputation, column masking, etc.)
        return super().preprocess_data(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict on 2D input data.
        
        This method handles:
        - Input validation (must be 2D)
        - Preprocessing via preprocess_data()
        - Model prediction
        - Output validation
        
        Args:
            X: Input features, shape (n_samples, n_features)
            
        Returns:
            Predictions, shape (n_samples,)
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # TODO: Add 2D-specific validation
        # Ensure input is 2D
        if len(X.shape) != 2:
            raise ValueError(f"Expected 2D input (samples, features), got shape {X.shape}")
        
        # Use parent preprocessing
        Xp, _ = self.preprocess_data(X, None)
        
        # TODO: Implement model-specific prediction
        # This should be overridden by subclasses
        raise NotImplementedError("Subclasses must implement model-specific prediction logic")
    
    @abstractmethod
    def _build_model(self, input_dim: int) -> Any:
        """
        Build the 2D model architecture.
        
        Args:
            input_dim: Number of input features
            
        Returns:
            Model instance
        """
        pass
    
    @abstractmethod
    def train(self, X_tr: np.ndarray, y_tr: np.ndarray, 
              X_va=None, y_va=None, feature_names: List[str] = None, **kwargs) -> Any:
        """
        Train the 2D model.
        
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

