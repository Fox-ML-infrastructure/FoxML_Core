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
LightGBM Wrapper

Wrapper for LightGBM models with consistent interface.
"""


import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class LightGBMWrapper:
    """Wrapper for LightGBM models"""
    
    def __init__(self, **kwargs):
        try:
            import lightgbm as lgb
            self.lgb = lgb
        except ImportError:
            raise ImportError("LightGBM not available")
        
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
