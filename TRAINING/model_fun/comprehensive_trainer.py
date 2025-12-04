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
Comprehensive Model Trainer

Implements all training functions from the original train_mtf_cross_sectional_gpu.py
with the same functionality but in a modular way.
"""


import numpy as np
from TRAINING.common.determinism import get_deterministic_params, seed_for
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from .base_trainer import BaseModelTrainer

logger = logging.getLogger(__name__)

class ComprehensiveTrainer(BaseModelTrainer):
    """Comprehensive trainer that implements all original training functions"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.model_type = None
        self.scaler = None
        self.tf_device = None
        
    def train(self, X_tr: np.ndarray, y_tr: np.ndarray, 
              feature_names: List[str] = None, **kwargs) -> Dict[str, Any]:
        """Train model using comprehensive training approach"""
        
        # Get model family from config
        family = self.config.get('family', 'LightGBM')
        
        # Determine model type
        unique_values = np.unique(y_tr[~np.isnan(y_tr)])
        if len(unique_values) <= 2 and all(v in [0, 1] for v in unique_values):
            self.model_type = 'classification'
        else:
            self.model_type = 'regression'
        
        # Store feature names
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X_tr.shape[1])]
        self.target_name = kwargs.get('target_name', 'target')
        
        # Preprocess data
        X_clean, y_clean = self.preprocess_data(X_tr, y_tr)
        
        # Train based on family
        if family == 'LightGBM':
            return self._train_lightgbm(X_clean, y_clean)
        elif family == 'XGBoost':
            return self._train_xgboost(X_clean, y_clean)
        elif family == 'MLP':
            return self._train_mlp(X_clean, y_clean)
        elif family == 'CNN1D':
            return self._train_cnn1d(X_clean, y_clean)
        elif family == 'LSTM':
            return self._train_lstm(X_clean, y_clean)
        elif family == 'Transformer':
            return self._train_transformer(X_clean, y_clean)
        elif family == 'Ensemble':
            return self._train_ensemble(X_clean, y_clean)
        else:
            logger.warning(f"Family {family} not implemented yet")
            return {'model': None, 'model_type': self.model_type, 'train_score': 0.0}
    
    def _train_lightgbm(self, X_tr: np.ndarray, y_tr: np.ndarray) -> Dict[str, Any]:
        """Train LightGBM model"""
        try:

            import lightgbm as lgb
        except ImportError:
            raise ImportError("LightGBM not available")
        
        # Create model
        if self.model_type == 'classification':
            model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            )
        else:
            model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            )
        
        # Train model
        model.fit(X_tr, y_tr)
        self.model = model
        
        # Get training score
        train_pred = model.predict(X_tr)
        if self.model_type == 'classification':
            train_score = model.score(X_tr, y_tr)
        else:
            from sklearn.metrics import r2_score
            train_score = r2_score(y_tr, train_pred)
        
        return {
            'model': model,
            'model_type': self.model_type,
            'train_score': train_score,
            'n_features': X_tr.shape[1],
            'n_samples': len(X_tr),
            'feature_names': self.feature_names
        }
    
    def _train_xgboost(self, X_tr: np.ndarray, y_tr: np.ndarray) -> Dict[str, Any]:
        """Train XGBoost model"""
        try:

            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost not available")
        
        # Create model
        if self.model_type == 'classification':
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbosity=0
            )
        else:
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbosity=0
            )
        
        # Train model
        model.fit(X_tr, y_tr)
        self.model = model
        
        # Get training score
        train_pred = model.predict(X_tr)
        if self.model_type == 'classification':
            train_score = model.score(X_tr, y_tr)
        else:
            from sklearn.metrics import r2_score
            train_score = r2_score(y_tr, train_pred)
        
        return {
            'model': model,
            'model_type': self.model_type,
            'train_score': train_score,
            'n_features': X_tr.shape[1],
            'n_samples': len(X_tr),
            'feature_names': self.feature_names
        }
    
    def _train_mlp(self, X_tr: np.ndarray, y_tr: np.ndarray) -> Dict[str, Any]:
        """Train MLP model with GPU acceleration"""
        try:

            import tensorflow as tf
            from tensorflow.keras import layers, Model
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            raise ImportError("TensorFlow not available")
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_tr)
        
        # Get device
        self.tf_device = self._get_tf_device()
        
        logger.info(f"🧠 MLP training on {self.tf_device}")
        
        # Create MLP with GPU acceleration
        with tf.device(self.tf_device):
            inputs = layers.Input(shape=(X_scaled.shape[1],))
            x = layers.Dense(512, activation='relu')(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
            x = layers.Dense(256, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
            x = layers.Dense(128, activation='relu')(x)
            x = layers.Dropout(0.2)(x)
            x = layers.Dense(64, activation='relu')(x)
            x = layers.Dropout(0.1)(x)
            outputs = layers.Dense(1, dtype="float32")(x)
            
            model = Model(inputs, outputs)
            model.compile(

                optimizer=Adam(learning_rate=0.001),
                loss='mse' if self.model_type == 'regression' else 'binary_crossentropy',
                metrics=['mae'] if self.model_type == 'regression' else ['accuracy']
            )
        
        # Clear GPU memory
        tf.keras.backend.clear_session()
        import gc
        gc.collect()
        
        # Training callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        # Train model
        history = model.fit(
            X_scaled, y_tr,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0
        )
        
        self.model = model
        
        # Get training score
        train_pred = model.predict(X_scaled, verbose=0)
        if self.model_type == 'classification':
            train_pred = (train_pred > 0.5).astype(int).flatten()
            train_score = np.mean(train_pred == y_tr)
        else:
            from sklearn.metrics import r2_score
            train_score = r2_score(y_tr, train_pred.flatten())
        
        return {
            'model': model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'train_score': train_score,
            'n_features': X_tr.shape[1],
            'n_samples': len(X_tr),
            'feature_names': self.feature_names,
            'history': history.history
        }
    
    def _train_cnn1d(self, X_tr: np.ndarray, y_tr: np.ndarray) -> Dict[str, Any]:
        """Train CNN1D model"""
        try:

            import tensorflow as tf
            from tensorflow.keras import layers, Model
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            raise ImportError("TensorFlow not available")
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_tr)
        
        # Reshape for CNN (add sequence dimension)
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        
        # Get device
        self.tf_device = self._get_tf_device()
        
        logger.info(f"🧠 CNN1D training on {self.tf_device}")
        
        # Create CNN model
        with tf.device(self.tf_device):
            inputs = layers.Input(shape=(X_reshaped.shape[1], 1))
            x = layers.Conv1D(64, 3, activation='relu')(inputs)
            x = layers.MaxPooling1D(2)(x)
            x = layers.Conv1D(32, 3, activation='relu')(x)
            x = layers.GlobalMaxPooling1D()(x)
            x = layers.Dense(50, activation='relu')(x)
            x = layers.Dropout(0.2)(x)
            outputs = layers.Dense(1, dtype="float32")(x)
            
            model = Model(inputs, outputs)
            model.compile(

                optimizer=Adam(learning_rate=0.001),
                loss='mse' if self.model_type == 'regression' else 'binary_crossentropy',
                metrics=['mae'] if self.model_type == 'regression' else ['accuracy']
            )
        
        # Clear GPU memory
        tf.keras.backend.clear_session()
        import gc
        gc.collect()
        
        # Training callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        # Train model
        history = model.fit(
            X_reshaped, y_tr,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0
        )
        
        self.model = model
        
        # Get training score
        train_pred = model.predict(X_reshaped, verbose=0)
        if self.model_type == 'classification':
            train_pred = (train_pred > 0.5).astype(int).flatten()
            train_score = np.mean(train_pred == y_tr)
        else:
            from sklearn.metrics import r2_score
            train_score = r2_score(y_tr, train_pred.flatten())
        
        return {
            'model': model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'train_score': train_score,
            'n_features': X_tr.shape[1],
            'n_samples': len(X_tr),
            'feature_names': self.feature_names,
            'history': history.history
        }
    
    def _train_lstm(self, X_tr: np.ndarray, y_tr: np.ndarray) -> Dict[str, Any]:
        """Train LSTM model"""
        # Placeholder implementation
        logger.warning("LSTM trainer not fully implemented yet")
        return {'model': None, 'model_type': self.model_type, 'train_score': 0.0}
    
    def _train_transformer(self, X_tr: np.ndarray, y_tr: np.ndarray) -> Dict[str, Any]:
        """Train Transformer model"""
        # Placeholder implementation
        logger.warning("Transformer trainer not fully implemented yet")
        return {'model': None, 'model_type': self.model_type, 'train_score': 0.0}
    
    def _train_ensemble(self, X_tr: np.ndarray, y_tr: np.ndarray) -> Dict[str, Any]:
        """Train Ensemble model"""
        # Placeholder implementation
        logger.warning("Ensemble trainer not fully implemented yet")
        return {'model': None, 'model_type': self.model_type, 'train_score': 0.0}
    
    def _get_tf_device(self) -> str:
        """Get TensorFlow device"""
        if os.getenv("CPU_ONLY", "0") == "1":
            return "/CPU:0"
        try:

            import tensorflow as tf
            gpus = tf.config.list_logical_devices("GPU")
            return "/GPU:0" if gpus else "/CPU:0"
        except Exception:
            return "/CPU:0"
    
    def predict(self, X_tr: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        if hasattr(self.model, 'predict'):
            if hasattr(self.model, 'predict_proba') and self.model_type == 'classification':
                # For classification, return probabilities
                return self.model.predict_proba(X_tr)[:, 1]
            else:
                # For regression or models without predict_proba
                return self.model.predict(X_tr)
        else:
            # For TensorFlow models
            if self.scaler is not None:
                X_scaled = self.scaler.transform(X_tr)
            else:
                X_scaled = X_tr
            
            # Handle different model types
            if hasattr(self.model, 'predict'):
                pred = self.model.predict(X_scaled, verbose=0)
                if self.model_type == 'classification':
                    return (pred > 0.5).astype(int).flatten()
                else:
                    return pred.flatten()
            else:
                return np.zeros(len(X_tr))
    
    def predict_proba(self, X_tr: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        if self.model_type == 'classification':
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(X_tr)
            else:
                # For TensorFlow models
                if self.scaler is not None:
                    X_scaled = self.scaler.transform(X_tr)
                else:
                    X_scaled = X_tr
                
                pred = self.model.predict(X_scaled, verbose=0)
                return np.column_stack([1 - pred.flatten(), pred.flatten()])
        else:
            # For regression, return dummy probabilities
            pred = self.predict(X_tr)
            return np.column_stack([1 - pred, pred])
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance"""
        if not self.is_trained:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_)
        else:
            return None
