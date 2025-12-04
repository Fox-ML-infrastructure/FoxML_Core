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

import numpy as np, logging, sys
from typing import Any, Dict, List, Optional
from pathlib import Path
from sklearn.model_selection import train_test_split
import keras
from keras import layers, ops, Model, optimizers, callbacks
from keras.saving import register_keras_serializable
from .base_trainer import BaseModelTrainer
from TRAINING.common.safety import configure_tf
logger = logging.getLogger(__name__)

# Add CONFIG directory to path for centralized config loading
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "CONFIG"
if str(_CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(_CONFIG_DIR))

# Try to import config loader
_USE_CENTRALIZED_CONFIG = False
try:
    from config_loader import load_model_config
    _USE_CENTRALIZED_CONFIG = True
except ImportError:
    logger.debug("config_loader not available; using hardcoded defaults")

@register_keras_serializable(package="trader")
class Sampling(layers.Layer):
    """Keras 3 compatible sampling layer for VAE
    z = z_mean + exp(0.5 * z_log_var) * eps
    
    Registered with Keras for proper serialization across process boundaries.
    """
    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = keras.random.normal(shape=ops.shape(z_mean))
        return z_mean + ops.exp(0.5 * z_log_var) * eps
    
    def get_config(self):
        """Required for Keras serialization."""
        return super().get_config()

@register_keras_serializable(package="trader")
class KLLossLayer(layers.Layer):
    """
    Keras 3 compatible KL divergence loss layer.
    
    Functional models in Keras 3 don't support model.add_loss() directly,
    so we use a layer that calls self.add_loss() instead.
    
    Registered with Keras for proper serialization across process boundaries.
    """
    def __init__(self, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
    
    def call(self, inputs):
        z_mean, z_log_var = inputs
        # KL divergence: -0.5 * sum(1 + log(var) - mean^2 - var)
        kl = -0.5 * ops.sum(1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var), axis=1)
        kl_loss = ops.mean(ops.cast(kl, "float32")) * self.beta
        # Register loss on the model
        self.add_loss(kl_loss)
        # Pass through z_mean, z_log_var for downstream use
        return inputs
    
    def get_config(self):
        """Required for Keras serialization."""
        config = super().get_config()
        config.update({"beta": self.beta})
        return config

class VAETrainer(BaseModelTrainer):
    def __init__(self, config: Dict[str, Any] = None):
        # Load centralized config if available and no config provided
        if config is None and _USE_CENTRALIZED_CONFIG:
            try:
                config = load_model_config("vae")
                logger.info("✅ [VAE] Loaded centralized config from CONFIG/model_config/vae.yaml")
            except Exception as e:
                logger.warning(f"Failed to load centralized config: {e}. Using hardcoded defaults.")
                config = {}
        
        super().__init__(config or {})
        
        # DEPRECATED: Hardcoded defaults kept for backward compatibility
        # To change these, edit CONFIG/model_config/vae.yaml
        self.config.setdefault("epochs", 50)
        self.config.setdefault("batch_size", 512)
        self.config.setdefault("latent_dim", self.config.get("z_dim", 16))  # Support old "z_dim" key
        self.config.setdefault("hidden_dim", 128)
        self.config.setdefault("dropout", 0.2)
        self.config.setdefault("learning_rate", 1e-3)
        self.config.setdefault("beta", 1.0)  # KL weight
        self.config.setdefault("patience", 10)

    def train(self, X_tr: np.ndarray, y_tr: np.ndarray, 
              X_va=None, y_va=None, feature_names: List[str] = None, **kwargs) -> Any:
        from common.threads import ensure_gpu_visible
        
        # Ensure GPU is visible (restore if hidden by prior CPU-only family)
        gpu_available = ensure_gpu_visible("VAE")
        
        # 1) Preprocess data
        X_tr, y_tr = self.preprocess_data(X_tr, y_tr)
        self.feature_names = feature_names or [f"f{i}" for i in range(X_tr.shape[1])]
        
        # 2) Configure TensorFlow
        configure_tf(cpu_only=kwargs.get("cpu_only", False) or not gpu_available)
        
        # 3) Split only if no external validation provided
        if X_va is None or y_va is None:
            X_tr, X_va, y_tr, y_va = train_test_split(
                X_tr, y_tr, test_size=0.2, random_state=42
            )
        
        # 4) Build model with safe defaults
        model = self._build_model(X_tr.shape[1])
        
        # 5) Train with callbacks
        cbs = [
            callbacks.EarlyStopping(patience=self.config["patience"], restore_best_weights=True),
            callbacks.ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6)
        ]
        
        model.fit(
            X_tr, [X_tr, y_tr],
            validation_data=(X_va, [X_va, y_va]),
            epochs=self.config["epochs"],
            batch_size=self.config["batch_size"],
            callbacks=cbs,
            verbose=0
        )
        
        # 6) Store state and sanity check
        self.model = model
        self.is_trained = True
        self.post_fit_sanity(X_tr, "VAE")
        return self.model

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        Xp, _ = self.preprocess_data(X, None)
        _, yhat = self.model.predict(Xp, verbose=0)
        return np.nan_to_num(yhat.ravel(), nan=0.0).astype(np.float32)

    def _build_model(self, input_dim: int) -> Model:
        """Build VAE model with Keras 3 compatible ops"""
        inputs = layers.Input(shape=(input_dim,), dtype="float32", name="x")
        
        # Encoder
        h = layers.Dense(self.config["hidden_dim"], activation="relu")(inputs)
        h = layers.BatchNormalization()(h)
        h = layers.Dropout(self.config["dropout"])(h)
        
        # Latent space
        latent_dim = self.config.get("latent_dim", self.config.get("z_dim", 16))
        z_mean = layers.Dense(latent_dim, name="z_mean")(h)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(h)
        
        # Sampling layer (Keras 3 compatible)
        z = Sampling(name="z")([z_mean, z_log_var])
        
        # Decoder
        h2 = layers.Dense(self.config["hidden_dim"], activation="relu")(z)
        h2 = layers.BatchNormalization()(h2)
        h2 = layers.Dropout(self.config["dropout"])(h2)
        
        # Reconstruction and prediction
        recon = layers.Dense(input_dim, name="recon")(h2)
        yhat = layers.Dense(1, name="yhat")(z)
        
        # Add KL loss via layer (Keras 3 Functional API requirement)
        _ = KLLossLayer(beta=self.config["beta"], name="kl_loss")([z_mean, z_log_var])
        
        model = Model(inputs, [recon, yhat], name="vae")
        
        # Compile with gradient clipping
        opt = optimizers.Adam(
            learning_rate=self.config["learning_rate"],
            clipnorm=1.0
        )
        
        # First loss is reconstruction (MSE), second is prediction
        model.compile(
            optimizer=opt,
            loss=["mse", "mse"],
            loss_weights=[1.0, 1.0]
        )
        
        return model