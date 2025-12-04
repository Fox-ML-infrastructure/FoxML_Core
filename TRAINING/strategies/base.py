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
Base Training Strategy

Abstract base class for all training strategies.
"""


from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class BaseTrainingStrategy(ABC):
    """Base class for all training strategies"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.metadata = {}
        
    @abstractmethod
    def train(self, X: np.ndarray, y_dict: Dict[str, np.ndarray], 
              feature_names: List[str], **kwargs) -> Dict[str, Any]:
        """Train models based on strategy"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """Make predictions using trained models"""
        pass
    
    @abstractmethod
    def get_target_types(self) -> Dict[str, str]:
        """Return target types (regression/classification) for each target"""
        pass
    
    def validate_data(self, X: np.ndarray, y_dict: Dict[str, np.ndarray]) -> bool:
        """Validate input data"""
        if X.shape[0] != len(list(y_dict.values())[0]):
            raise ValueError("X and y dimensions don't match")
        
        for target_name, y in y_dict.items():
            if len(y) != X.shape[0]:
                raise ValueError(f"Target {target_name} length doesn't match X")
                
        return True
    
    def save_models(self, filepath: str) -> None:
        """Save trained models to disk"""
        import joblib
        joblib.dump({
            'models': self.models,
            'metadata': self.metadata,
            'config': self.config
        }, filepath)
        logger.info(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str) -> None:
        """Load trained models from disk"""
        import joblib
        data = joblib.load(filepath)
        self.models = data['models']
        self.metadata = data['metadata']
        self.config = data['config']
        logger.info(f"Models loaded from {filepath}")
    
    def get_feature_importance(self, target_name: str = None) -> Optional[Dict[str, np.ndarray]]:
        """
        Get feature importance from trained models.
        
        Args:
            target_name: Specific target to get importance for. If None, returns all.
            
        Returns:
            Dictionary mapping target names to feature importance arrays,
            or None if no models support feature importance.
        """
        if target_name is not None:
            # Get importance for specific target
            if target_name not in self.models:
                logger.warning(f"Target {target_name} not found in trained models")
                return None
            
            model = self.models[target_name]
            importance = self._extract_feature_importance(model)
            return {target_name: importance} if importance is not None else None
        else:
            # Get importance for all targets
            importances = {}
            for name, model in self.models.items():
                importance = self._extract_feature_importance(model)
                if importance is not None:
                    importances[name] = importance
            return importances if importances else None
    
    def _extract_feature_importance(self, model) -> Optional[np.ndarray]:
        """Extract feature importance from a model"""
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For linear models, use absolute coefficient values
            coef = np.array(model.coef_)
            if len(coef.shape) > 1:
                # Multi-class, take mean across classes
                return np.abs(coef).mean(axis=0)
            return np.abs(coef)
        else:
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about trained models"""
        info = {
            'strategy': self.__class__.__name__,
            'n_models': len(self.models),
            'targets': list(self.models.keys()),
            'config': self.config
        }
        return info
