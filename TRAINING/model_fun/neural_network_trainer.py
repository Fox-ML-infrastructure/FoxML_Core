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

"""Simple working stub for Neuralnetwork trainer."""

import numpy as np
from TRAINING.common.determinism import get_deterministic_params, seed_for
import logging
from typing import Dict, Any, Optional
from .base_trainer import BaseModelTrainer

logger = logging.getLogger(__name__)

class NeuralNetworkTrainer(BaseModelTrainer):
    """Simple working stub for Neuralnetwork trainer."""


    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

    def fit(self, X_tr: np.ndarray, y_tr: np.ndarray, seed: Optional[int] = None, **kwargs) -> Any:
        """Fit method for compatibility with sklearn-style interface."""
        return self.train(X_tr, y_tr, seed, **kwargs)
    
    def train(self, X_tr: np.ndarray, y_tr: np.ndarray, X_va=None, y_va=None, cpu_only=False, num_threads=12, feat_cols=None, seed: Optional[int] = None, **kwargs) -> Any:
        """Train Neural Network model using self-contained logic."""
        return self._train_neural_network(X_tr, y_tr, X_va, y_va, cpu_only, num_threads, feat_cols, seed, **kwargs)
    
    def _train_neural_network(self, X_tr, y_tr, X_va=None, y_va=None, cpu_only=False, num_threads=12, feat_cols=None, seed=None, epochs=50, **kwargs):
        """Train Neural Network model."""
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, Model
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import StandardScaler
            import numpy as np
            
            # Set device
            TF_DEVICE = '/GPU:0' if not cpu_only else '/CPU:0'
            
            # Preprocess data
            imputer = SimpleImputer(strategy='median')
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(imputer.fit_transform(X_tr))
            
            if X_va is not None:
                X_va_scaled = scaler.transform(imputer.transform(X_va))
            else:
                X_va_scaled = None
            
            n_features = X_scaled.shape[1]
            
            logger.info(f"🧠 Neural Network training on {TF_DEVICE}")
            
            with tf.device(TF_DEVICE):
                inputs = layers.Input(shape=(n_features,))
                x = layers.Dense(256, activation='relu')(inputs)
                x = layers.BatchNormalization()(x)
                x = layers.Dropout(0.3)(x)
                x = layers.Dense(128, activation='relu')(x)
                x = layers.BatchNormalization()(x)
                x = layers.Dropout(0.3)(x)
                x = layers.Dense(64, activation='relu')(x)
                x = layers.BatchNormalization()(x)
                x = layers.Dropout(0.2)(x)
                outputs = layers.Dense(1, dtype="float32")(x)
                
                model = Model(inputs, outputs)
                model.compile(
                    optimizer=Adam(learning_rate=0.001),
                    loss='mse',
                    metrics=['mae']
                )
            
            # Callbacks
            callbacks = [
                EarlyStopping(patience=20, restore_best_weights=True),
                ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-6)
            ]
            
            # Train model
            if X_va_scaled is not None and y_va is not None:
                model.fit(
                    X_scaled, y_tr,
                    validation_data=(X_va_scaled, y_va),
                    epochs=epochs,
                    batch_size=256,
                    callbacks=callbacks,
                    verbose=0
                )
            else:
                model.fit(
                    X_scaled, y_tr,
                    epochs=epochs,
                    batch_size=256,
                    callbacks=callbacks,
                    verbose=0
                )
            
            # Store preprocessors for inference
            self.scaler = scaler
            self.imputer = imputer
            self.model = model
            self.is_trained = True
            return model
            
        except ImportError:
            logger.error("TensorFlow not available for Neural Network")
            return None
        except Exception as e:
            logger.error(f"Neural Network training failed: {e}")
            return None
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained neural network model."""
        if not hasattr(self, 'model') or not hasattr(self, 'is_trained') or not self.is_trained:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        try:
            # Clean input data using stored preprocessors
            X_float = X.astype(np.float64)
            X_clean = self.imputer.transform(X_float)
            X_scaled = self.scaler.transform(X_clean)
            
            # Make predictions
            predictions = self.model.predict(X_scaled, verbose=0)
            
            # Flatten if needed
            if predictions.ndim > 1:
                predictions = predictions.flatten()
            
            # Ensure predictions are finite
            predictions = np.nan_to_num(predictions, nan=0.0)
            
            return predictions.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Neural Network prediction failed: {e}")
            # Return zeros as fallback
            return np.zeros(X.shape[0], dtype=np.float32)