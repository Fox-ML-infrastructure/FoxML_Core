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
Model Factory

Creates appropriate models based on configuration and target type.
"""


from typing import Dict, Any, Optional, Union
import logging
from .registry import ModelRegistry

logger = logging.getLogger(__name__)

class ModelFactory:
    """Factory for creating models based on configuration - Singleton pattern"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelFactory, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.registry = ModelRegistry()
            self._initialized = True
    
    def create_model(self, model_type: str, target_type: str, 
                    config: Dict[str, Any] = None) -> Any:
        """Create a model instance"""
        
        if config is None:
            config = {}
        
        # Get model class from registry
        model_class = self.registry.get_model_class(model_type, target_type)
        
        if model_class is None:
            logger.warning(f"Unknown model type {model_type}, using default")
            model_class = self.registry.get_default_model_class(target_type)
        
        # Create model instance with configuration
        try:
            # Check if it's a trainer class (has train method)
            if hasattr(model_class, 'train'):
                # It's a trainer class, create instance
                model = model_class(config)
            else:
                # It's a scikit-learn style class, create with parameters
                model = model_class(**config)
            
            logger.info(f"Created {model_type} model for {target_type}")
            return model
        except Exception as e:
            logger.error(f"Error creating {model_type} model: {e}")
            # Fallback to default
            default_class = self.registry.get_default_model_class(target_type)
            if hasattr(default_class, 'train'):
                return default_class(config)
            else:
                return default_class(**config)
    
    def create_models_for_targets(self, targets: Dict[str, str], 
                                 model_configs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Create models for multiple targets"""
        
        models = {}
        
        for target_name, target_type in targets.items():
            # Get model type for this target
            model_type = model_configs.get(target_name, {}).get('type', 'auto')
            
            if model_type == 'auto':
                model_type = self._auto_select_model_type(target_type)
            
            # Get configuration for this target
            config = model_configs.get(target_name, {})
            config.pop('type', None)  # Remove type from config
            
            # Create model
            model = self.create_model(model_type, target_type, config)
            models[target_name] = model
        
        return models
    
    def _auto_select_model_type(self, target_type: str) -> str:
        """Auto-select model type based on target type"""
        
        if target_type == 'classification':
            return 'random_forest'  # Good default for classification
        else:
            return 'random_forest'  # Good default for regression
    
    def get_available_models(self, target_type: str) -> list:
        """Get list of available models for target type"""
        return self.registry.get_available_models(target_type)
    
    def validate_model_config(self, model_type: str, target_type: str, 
                             config: Dict[str, Any]) -> bool:
        """Validate model configuration"""
        try:
            model_class = self.registry.get_model_class(model_type, target_type)
            if model_class is None:
                return False
            
            # Try to create model with config to validate
            model_class(**config)
            return True
        except Exception:
            return False
