# MIT License - see LICENSE file

"""
Neural Network Wrapper

Wrapper for neural network models with consistent interface.
"""


import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class NeuralNetworkWrapper:
    """Wrapper for neural network models"""
    
    def __init__(self, **kwargs):
        try:
            from sklearn.neural_network import MLPRegressor, MLPClassifier
            self.MLPRegressor = MLPRegressor
            self.MLPClassifier = MLPClassifier
        except ImportError:
            raise ImportError("Neural network models not available")
        
        self.model = None
        self.config = kwargs
    
    def fit(self, X, y):
        """Fit the model"""
        if self.model is None:
            raise ValueError("Model not initialized")
        
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities (for classification)"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # For regression models, return dummy probabilities
            pred = self.predict(X)
            return np.column_stack([1 - pred, pred])
    
    @property
    def feature_importances_(self):
        """Get feature importances"""
        if self.model is None:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        else:
            return None
