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
Single Task Strategy

Option A: Train separate models for each target.
Best for: Tree models, when targets have different scales, avoiding leakage.
"""


import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
from .base import BaseTrainingStrategy

logger = logging.getLogger(__name__)

class SingleTaskStrategy(BaseTrainingStrategy):
    """Train separate models for each target"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.target_types = {}
        
    def train(self, X: np.ndarray, y_dict: Dict[str, np.ndarray], 
              feature_names: List[str], **kwargs) -> Dict[str, Any]:
        """Train separate models for each target"""
        logger.info("🎯 Training single-task models (Option A)")
        
        self.validate_data(X, y_dict)
        
        results = {}
        
        for target_name, y in y_dict.items():
            logger.info(f"Training model for target: {target_name}")
            
            # Determine model type based on target
            target_type = self._determine_target_type(target_name, y)
            self.target_types[target_name] = target_type
            
            # Create appropriate model
            model = self._create_model(target_type, target_name)
            
            # Train model with early stopping (if supported)
            try:
                # Check if model supports early stopping
                if hasattr(model, 'fit') and ('eval_set' in model.fit.__code__.co_varnames or 
                                              'X_valid' in model.fit.__code__.co_varnames):
                    # Split for validation
                    from sklearn.model_selection import train_test_split
                    X_train, X_val, y_train, y_val = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
                    
                    # Fit with early stopping
                    if 'eval_set' in model.fit.__code__.co_varnames:
                        # LightGBM/XGBoost style
                        import lightgbm as lgb
                        early_stopping_rounds = self.config.get('early_stopping_rounds', 50)
                        model.fit(
                            X_train, y_train,
                            eval_set=[(X_val, y_val)],
                            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
                        )
                        logger.info(f"✅ {target_name}: Early stopping at iteration {getattr(model, 'best_iteration_', 'N/A')}")
                    else:
                        # Fallback
                        model.fit(X, y)
                else:
                    # Standard fit for models that don't support early stopping
                    model.fit(X, y)
            except Exception as e:
                logger.warning(f"Early stopping failed for {target_name}, using standard fit: {e}")
                model.fit(X, y)
            
            # Store model and metadata
            self.models[target_name] = model
            results[target_name] = {
                'model': model,
                'target_type': target_type,
                'feature_names': feature_names,
                'n_features': X.shape[1],
                'n_samples': len(y),
                'feature_importance': self._get_feature_importance(model)
            }
            
            logger.info(f"✅ {target_name} ({target_type}) trained successfully")
            
        return results
    
    def predict(self, X: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """Make predictions using trained models"""
        predictions = {}
        
        for target_name, model in self.models.items():
            try:
                if self.target_types[target_name] == 'classification':
                    # For classification, return probabilities
                    if hasattr(model, 'predict_proba'):
                        pred = model.predict_proba(X)[:, 1]  # Probability of positive class
                    else:
                        pred = model.predict(X)
                else:
                    # For regression, return continuous values
                    pred = model.predict(X)
                
                predictions[target_name] = pred
                
            except Exception as e:
                logger.error(f"Error predicting {target_name}: {e}")
                predictions[target_name] = np.zeros(len(X))
        
        return predictions
    
    def get_target_types(self) -> Dict[str, str]:
        """Return target types for each target"""
        return self.target_types.copy()
    
    def _determine_target_type(self, target_name: str, y: np.ndarray) -> str:
        """Determine if target is regression or classification"""
        
        # Check target name patterns
        if target_name.startswith('fwd_ret_'):
            return 'regression'
        elif any(target_name.startswith(prefix) for prefix in 
                ['will_peak', 'will_valley', 'mdd', 'mfe', 'y_will_']):
            return 'classification'
        
        # Check data characteristics
        unique_values = np.unique(y[~np.isnan(y)])
        
        if len(unique_values) <= 2 and all(v in [0, 1] for v in unique_values):
            return 'classification'
        else:
            return 'regression'
    
    def _create_model(self, target_type: str, target_name: str):
        """Create appropriate model for target type"""
        
        # Get model configuration
        model_config = self.config.get('models', {})
        model_type = model_config.get('type', 'auto')
        
        if model_type == 'auto':
            # Auto-select based on target type
            if target_type == 'classification':
                return self._create_classification_model(target_name)
            else:
                return self._create_regression_model(target_name)
        else:
            # Use specified model type
            return self._create_model_by_type(model_type, target_type, target_name)
    
    def _create_model_with_factory(self, model_type: str, target_type: str, target_name: str):
        """Create model using the factory system"""
        try:
            from models.factory import ModelFactory
            
            # Use global singleton factory (no re-initialization)
            factory = ModelFactory()
            
            model_config = self.config.get('models', {}).get(model_type, {})
            
            # Remove type from config
            model_config = {k: v for k, v in model_config.items() if k != 'type'}
            
            return factory.create_model(model_type, target_type, model_config)
            
        except Exception as e:
            logger.warning(f"Could not create {model_type} with factory: {e}")
            # Fallback to direct creation
            return self._create_model_by_type(model_type, target_type, target_name)
    
    def _create_regression_model(self, target_name: str):
        """Create regression model"""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import Ridge
        
        model_config = self.config.get('models', {}).get('regression', {})
        model_type = model_config.get('type', 'random_forest')
        
        if model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=model_config.get('n_estimators', 100),
                max_depth=model_config.get('max_depth', None),
                random_state=42
            )
        elif model_type == 'ridge':
            return Ridge(alpha=model_config.get('alpha', 1.0))
        else:
            return RandomForestRegressor(n_estimators=100, random_state=42)
    
    def _create_classification_model(self, target_name: str):
        """Create classification model"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        
        model_config = self.config.get('models', {}).get('classification', {})
        model_type = model_config.get('type', 'random_forest')
        
        if model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=model_config.get('n_estimators', 100),
                max_depth=model_config.get('max_depth', None),
                random_state=42
            )
        elif model_type == 'logistic':
            return LogisticRegression(
                C=model_config.get('C', 1.0),
                random_state=42
            )
        else:
            return RandomForestClassifier(n_estimators=100, random_state=42)
    
    def _create_model_by_type(self, model_type: str, target_type: str, target_name: str):
        """Create model by specific type"""
        # Try factory first for MTF model families
        if model_type in ['LightGBM', 'XGBoost', 'MLP', 'CNN1D', 'LSTM', 'Transformer', 'Ensemble']:
            return self._create_model_with_factory(model_type, target_type, target_name)
        
        # Fallback to direct creation for other types
        if model_type == 'lightgbm':
            return self._create_lightgbm_model(target_type)
        elif model_type == 'xgboost':
            return self._create_xgboost_model(target_type)
        elif model_type == 'neural_network':
            return self._create_neural_network_model(target_type)
        else:
            # Fallback to default
            if target_type == 'classification':
                return self._create_classification_model(target_name)
            else:
                return self._create_regression_model(target_name)
    
    def _create_lightgbm_model(self, target_type: str):
        """Create LightGBM model with Spec 2 high regularization"""
        try:
            import lightgbm as lgb
            
            # Get model config or use Spec 2 defaults
            model_config = self.config.get('models', {}).get('lightgbm', {})
            
            # Spec 2: High Regularization defaults
            params = {
                'num_leaves': model_config.get('num_leaves', 96),  # 64-128 range
                'max_depth': model_config.get('max_depth', 8),  # 7-9 range
                'min_child_weight': model_config.get('min_child_weight', 0.5),
                'learning_rate': model_config.get('learning_rate', 0.03),
                'n_estimators': model_config.get('n_estimators', 1000),
                'subsample': model_config.get('bagging_fraction', 0.75),  # 0.7-0.8 range
                'colsample_bytree': model_config.get('feature_fraction', 0.75),
                'subsample_freq': 1,
                'reg_alpha': model_config.get('lambda_l1', 0.1),
                'reg_lambda': model_config.get('lambda_l2', 0.1),
                'random_state': 42,
                'verbose': -1,
            }
            
            if target_type == 'classification':
                return lgb.LGBMClassifier(**params)
            else:
                return lgb.LGBMRegressor(**params)
                
        except ImportError:
            logger.warning("LightGBM not available, falling back to RandomForest")
            if target_type == 'classification':
                return self._create_classification_model('fallback')
            else:
                return self._create_regression_model('fallback')
    
    def _create_xgboost_model(self, target_type: str):
        """Create XGBoost model with Spec 2 high regularization"""
        try:
            import xgboost as xgb
            
            # Get model config or use Spec 2 defaults
            model_config = self.config.get('models', {}).get('xgboost', {})
            
            # Spec 2: High Regularization defaults
            params = {
                'max_depth': model_config.get('max_depth', 7),  # 5-8 range
                'min_child_weight': model_config.get('min_child_weight', 0.5),
                'gamma': model_config.get('gamma', 0.3),  # min_split_gain
                'learning_rate': model_config.get('eta', 0.03),
                'n_estimators': model_config.get('n_estimators', 1000),
                'subsample': model_config.get('subsample', 0.75),  # 0.7-0.8 range
                'colsample_bytree': model_config.get('colsample_bytree', 0.75),
                'reg_alpha': model_config.get('reg_alpha', 0.1),
                'reg_lambda': model_config.get('reg_lambda', 0.1),
                'random_state': 42,
                'verbosity': 0,
            }
            
            if target_type == 'classification':
                return xgb.XGBClassifier(**params)
            else:
                return xgb.XGBRegressor(**params)
                
        except ImportError:
            logger.warning("XGBoost not available, falling back to RandomForest")
            if target_type == 'classification':
                return self._create_classification_model('fallback')
            else:
                return self._create_regression_model('fallback')
    
    def _create_neural_network_model(self, target_type: str):
        """Create neural network model"""
        try:
            from sklearn.neural_network import MLPRegressor, MLPClassifier
            
            if target_type == 'classification':
                return MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    random_state=42,
                    max_iter=500
                )
            else:
                return MLPRegressor(
                    hidden_layer_sizes=(100, 50),
                    random_state=42,
                    max_iter=500
                )
        except ImportError:
            logger.warning("Neural network not available, falling back to RandomForest")
            if target_type == 'classification':
                return self._create_classification_model('fallback')
            else:
                return self._create_regression_model('fallback')
    
    def _get_feature_importance(self, model) -> Optional[np.ndarray]:
        """Get feature importance from model"""
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        elif hasattr(model, 'coef_'):
            return np.abs(model.coef_)
        else:
            return None
