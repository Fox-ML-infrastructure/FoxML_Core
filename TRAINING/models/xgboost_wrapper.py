# MIT License - see LICENSE file

"""
XGBoost Wrapper

Wrapper for XGBoost models with consistent interface.
"""


import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class XGBoostWrapper:
    """Wrapper for XGBoost models"""
    
    def __init__(self, **kwargs):
        try:
            import xgboost as xgb
            self.xgb = xgb
        except ImportError:
            raise ImportError("XGBoost not available")
        
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
