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
Model Registry

Registry for different model types and their configurations.
"""


from typing import Dict, Any, Optional, Type, List
import logging

logger = logging.getLogger(__name__)

class ModelRegistry:
    """Registry for model types and their classes - Singleton pattern"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._models = {
                'classification': {},
                'regression': {}
            }
            self._default_models = {
                'classification': 'random_forest',
                'regression': 'random_forest'
            }
            self._register_default_models()
            self._initialized = True
    
    def register_model(self, model_type: str, target_type: str, model_class: Type, 
                      is_default: bool = False):
        """Register a model class"""
        if target_type not in self._models:
            self._models[target_type] = {}
        
        self._models[target_type][model_type] = model_class
        
        if is_default:
            self._default_models[target_type] = model_type
        
        logger.info(f"Registered {model_type} for {target_type}")
    
    def get_model_class(self, model_type: str, target_type: str) -> Optional[Type]:
        """Get model class for type and target type"""
        return self._models.get(target_type, {}).get(model_type)
    
    def get_default_model_class(self, target_type: str) -> Optional[Type]:
        """Get default model class for target type"""
        default_type = self._default_models.get(target_type)
        if default_type:
            return self.get_model_class(default_type, target_type)
        return None
    
    def get_available_models(self, target_type: str) -> List[str]:
        """Get list of available models for target type"""
        return list(self._models.get(target_type, {}).keys())
    
    def _register_default_models(self):
        """Register default models"""
        # Register scikit-learn models
        self._register_sklearn_models()
        
        # Register LightGBM if available
        self._register_lightgbm_models()
        
        # Register XGBoost if available
        self._register_xgboost_models()
        
        # Register neural network models
        self._register_neural_network_models()
        
        # Register all model families from MTF training
        self._register_mtf_model_families()
    
    def _register_sklearn_models(self):
        """Register scikit-learn models"""
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.linear_model import LogisticRegression, Ridge
            from sklearn.neural_network import MLPClassifier, MLPRegressor
            
            # Random Forest
            self.register_model('random_forest', 'classification', RandomForestClassifier, is_default=True)
            self.register_model('random_forest', 'regression', RandomForestRegressor, is_default=True)
            
            # Linear models
            self.register_model('logistic_regression', 'classification', LogisticRegression)
            self.register_model('ridge', 'regression', Ridge)
            
            # Neural networks
            self.register_model('mlp_classifier', 'classification', MLPClassifier)
            self.register_model('mlp_regressor', 'regression', MLPRegressor)
            
        except ImportError as e:
            logger.warning(f"Could not register scikit-learn models: {e}")
    
    def _register_lightgbm_models(self):
        """Register LightGBM models if available"""
        try:
            import lightgbm as lgb
            
            self.register_model('lightgbm', 'classification', lgb.LGBMClassifier)
            self.register_model('lightgbm', 'regression', lgb.LGBMRegressor)
            
        except ImportError:
            logger.debug("LightGBM not available")
    
    def _register_xgboost_models(self):
        """Register XGBoost models if available"""
        try:
            import xgboost as xgb
            
            self.register_model('xgboost', 'classification', xgb.XGBClassifier)
            self.register_model('xgboost', 'regression', xgb.XGBRegressor)
            
        except ImportError:
            logger.debug("XGBoost not available")
    
    def _register_neural_network_models(self):
        """Register neural network models"""
        try:
            from sklearn.neural_network import MLPClassifier, MLPRegressor
            
            self.register_model('neural_network', 'classification', MLPClassifier)
            self.register_model('neural_network', 'regression', MLPRegressor)
            
        except ImportError:
            logger.debug("Neural network models not available")
    
    def _register_mtf_model_families(self):
        """Register all model families from MTF training"""
        try:
            from model_fun import (
                LightGBMTrainer, XGBoostTrainer, MLPTrainer, CNN1DTrainer,
                LSTMTrainer, TransformerTrainer, TabCNNTrainer, TabLSTMTrainer,
                TabTransformerTrainer, RewardBasedTrainer, QuantileLightGBMTrainer,
                NGBoostTrainer, GMMRegimeTrainer, ChangePointTrainer, FTRLProximalTrainer,
                VAETrainer, GANTrainer, EnsembleTrainer, MetaLearningTrainer,
                MultiTaskTrainer, NeuralNetworkTrainer
            )
            
            # Register all model families
            model_families = [
                ('LightGBM', LightGBMTrainer),
                ('XGBoost', XGBoostTrainer),
                ('MLP', MLPTrainer),
                ('CNN1D', CNN1DTrainer),
                ('LSTM', LSTMTrainer),
                ('Transformer', TransformerTrainer),
                ('TabCNN', TabCNNTrainer),
                ('TabLSTM', TabLSTMTrainer),
                ('TabTransformer', TabTransformerTrainer),
                ('RewardBased', RewardBasedTrainer),
                ('QuantileLightGBM', QuantileLightGBMTrainer),
                ('NGBoost', NGBoostTrainer),
                ('GMMRegime', GMMRegimeTrainer),
                ('ChangePoint', ChangePointTrainer),
                ('FTRLProximal', FTRLProximalTrainer),
                ('VAE', VAETrainer),
                ('GAN', GANTrainer),
                ('Ensemble', EnsembleTrainer),
                ('MetaLearning', MetaLearningTrainer),
                ('MultiTask', MultiTaskTrainer),
                ('NeuralNetwork', NeuralNetworkTrainer)
            ]
            
            for family_name, trainer_class in model_families:
                # Register for both classification and regression
                self.register_model(family_name, 'classification', trainer_class)
                self.register_model(family_name, 'regression', trainer_class)
            
            logger.info(f"Registered {len(model_families)} MTF model families")
            
        except ImportError as e:
            logger.warning(f"Could not register MTF model families: {e}")
    
    def get_mtf_model_families(self) -> List[str]:
        """Get list of available MTF model families"""
        return [
            'LightGBM', 'XGBoost', 'MLP', 'CNN1D', 'LSTM', 'Transformer',
            'TabCNN', 'TabLSTM', 'TabTransformer', 'RewardBased',
            'QuantileLightGBM', 'NGBoost', 'GMMRegime', 'ChangePoint',
            'FTRLProximal', 'VAE', 'GAN', 'Ensemble', 'MetaLearning', 'MultiTask'
        ]
    
    def get_family_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """Get capabilities for each model family"""
        return {
            "LightGBM": {"nan_ok": True, "needs_tf": False, "experimental": False},
            "XGBoost": {"nan_ok": True, "needs_tf": False, "experimental": False},
            "MLP": {"nan_ok": False, "needs_tf": False, "experimental": False},
            "CNN1D": {"nan_ok": False, "needs_tf": True, "experimental": False},
            "LSTM": {"nan_ok": False, "needs_tf": True, "experimental": False},
            "Transformer": {"nan_ok": False, "needs_tf": True, "experimental": False},
            "TabCNN": {"nan_ok": False, "needs_tf": True, "experimental": False},
            "TabLSTM": {"nan_ok": False, "needs_tf": True, "experimental": False},
            "TabTransformer": {"nan_ok": False, "needs_tf": True, "experimental": False},
            "RewardBased": {"nan_ok": False, "needs_tf": False, "experimental": False},
            "QuantileLightGBM": {"nan_ok": True, "needs_tf": False, "experimental": False},
            "NGBoost": {"nan_ok": False, "needs_tf": False, "experimental": True},
            "GMMRegime": {"nan_ok": False, "needs_tf": False, "experimental": True},
            "ChangePoint": {"nan_ok": False, "needs_tf": False, "experimental": True},
            "FTRLProximal": {"nan_ok": False, "needs_tf": False, "experimental": False},
            "VAE": {"nan_ok": False, "needs_tf": True, "experimental": True},
            "GAN": {"nan_ok": False, "needs_tf": True, "experimental": True},
            "Ensemble": {"nan_ok": False, "needs_tf": False, "experimental": False},
            "MetaLearning": {"nan_ok": False, "needs_tf": True, "experimental": True},
            "MultiTask": {"nan_ok": False, "needs_tf": True, "experimental": True}
        }
    
    def get_model_info(self, model_type: str, target_type: str) -> Dict[str, Any]:
        """Get information about a model type"""
        model_class = self.get_model_class(model_type, target_type)
        
        if model_class is None:
            return {'available': False}
        
        info = {
            'available': True,
            'class': model_class,
            'module': model_class.__module__,
            'name': model_class.__name__
        }
        
        # Add default parameters if available
        try:
            import inspect
            sig = inspect.signature(model_class.__init__)
            params = {}
            for param_name, param in sig.parameters.items():
                if param_name != 'self':
                    params[param_name] = {
                        'default': param.default if param.default != inspect.Parameter.empty else None,
                        'annotation': param.annotation if param.annotation != inspect.Parameter.empty else None
                    }
            info['parameters'] = params
        except Exception:
            info['parameters'] = {}
        
        return info

# Trainer Registry for the new modular system
TRAINER_REGISTRY = {
    "LightGBM": ("model_fun.lightgbm_trainer", "LightGBMTrainer"),
    "QuantileLightGBM": ("model_fun.quantile_lightgbm_trainer", "QuantileLightGBMTrainer"),
    "XGBoost": ("model_fun.xgboost_trainer", "XGBoostTrainer"),
    "RewardBased": ("model_fun.reward_based_trainer", "RewardBasedTrainer"),
    "NGBoost": ("model_fun.ngboost_trainer", "NGBoostTrainer"),
    "GMMRegime": ("model_fun.gmm_regime_trainer", "GMMRegimeTrainer"),
    "ChangePoint": ("model_fun.change_point_trainer", "ChangePointTrainer"),
    "FTRLProximal": ("model_fun.ftrl_proximal_trainer", "FTRLProximalTrainer"),
    "Ensemble": ("model_fun.ensemble_trainer", "EnsembleTrainer"),
    "MLP": ("model_fun.mlp_trainer", "MLPTrainer"),
    "CNN1D": ("model_fun.cnn1d_trainer", "CNN1DTrainer"),
    "LSTM": ("model_fun.lstm_trainer", "LSTMTrainer"),
    "Transformer": ("model_fun.transformer_trainer", "TransformerTrainer"),
    "TabCNN": ("model_fun.tabcnn_trainer", "TabCNNTrainer"),
    "TabLSTM": ("model_fun.tablstm_trainer", "TabLSTMTrainer"),
    "TabTransformer": ("model_fun.tabtransformer_trainer", "TabTransformerTrainer"),
    "VAE": ("model_fun.vae_trainer", "VAETrainer"),
    "GAN": ("model_fun.gan_trainer", "GANTrainer"),
    "MetaLearning": ("model_fun.meta_learning_trainer", "MetaLearningTrainer"),
    "MultiTask": ("model_fun.multi_task_trainer", "MultiTaskTrainer"),
}

# Family capabilities mapping for the new system
FAMILY_CAPABILITIES = {
    "LightGBM": {"needs_tf": False, "needs_isolation": "maybe_omp", "experimental": False},
    "QuantileLightGBM": {"needs_tf": False, "needs_isolation": "omp", "experimental": False},
    "XGBoost": {"needs_tf": False, "needs_isolation": False, "experimental": False},
    "RewardBased": {"needs_tf": False, "needs_isolation": False, "experimental": False},
    "NGBoost": {"needs_tf": False, "needs_isolation": False, "experimental": True},
    "GMMRegime": {"needs_tf": False, "needs_isolation": "mkl", "experimental": True},
    "ChangePoint": {"needs_tf": False, "needs_isolation": "mkl", "experimental": True},
    "FTRLProximal": {"needs_tf": False, "needs_isolation": False, "experimental": False},
    "Ensemble": {"needs_tf": False, "needs_isolation": False, "experimental": False},
    "MLP": {"needs_tf": True, "needs_isolation": False, "experimental": False},
    "CNN1D": {"needs_tf": True, "needs_isolation": False, "experimental": False},
    "LSTM": {"needs_tf": True, "needs_isolation": False, "experimental": False},
    "Transformer": {"needs_tf": True, "needs_isolation": False, "experimental": False},
    "TabCNN": {"needs_tf": True, "needs_isolation": False, "experimental": False},
    "TabLSTM": {"needs_tf": True, "needs_isolation": False, "experimental": False},
    "TabTransformer": {"needs_tf": True, "needs_isolation": False, "experimental": False},
    "VAE": {"needs_tf": True, "needs_isolation": False, "experimental": True},
    "GAN": {"needs_tf": True, "needs_isolation": False, "experimental": True},
    "MetaLearning": {"needs_tf": True, "needs_isolation": False, "experimental": True},
    "MultiTask": {"needs_tf": True, "needs_isolation": False, "experimental": True},
}

def get_trainer_info(family_name: str):
    """Get trainer module and class info for a family"""
    if family_name not in TRAINER_REGISTRY:
        raise ValueError(f"Unknown family: {family_name}")
    return TRAINER_REGISTRY[family_name]

def get_family_capabilities(family_name: str):
    """Get capabilities for a family"""
    if family_name not in FAMILY_CAPABILITIES:
        raise ValueError(f"Unknown family: {family_name}")
    return FAMILY_CAPABILITIES[family_name]

def get_all_families():
    """Get list of all available families"""
    return list(TRAINER_REGISTRY.keys())

def get_tf_families():
    """Get list of TensorFlow families"""
    return [f for f, caps in FAMILY_CAPABILITIES.items() if caps["needs_tf"]]

def get_isolation_families():
    """Get list of families that need isolation"""
    return [f for f, caps in FAMILY_CAPABILITIES.items() if caps["needs_isolation"]]

def get_experimental_families():
    """Get list of experimental families"""
    return [f for f, caps in FAMILY_CAPABILITIES.items() if caps["experimental"]]
