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

import numpy as np, logging, tensorflow as tf, sys
from typing import Any, Dict, List, Optional
from pathlib import Path
from sklearn.model_selection import train_test_split
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

class TransformerTrainer(BaseModelTrainer):
    def __init__(self, config: Dict[str, Any] = None):
        # Load centralized config if available and no config provided
        if config is None and _USE_CENTRALIZED_CONFIG:
            try:
                config = load_model_config("transformer")
                logger.info("✅ [Transformer] Loaded centralized config from CONFIG/model_config/transformer.yaml")
            except Exception as e:
                logger.warning(f"Failed to load centralized config: {e}. Using hardcoded defaults.")
                config = {}
        
        super().__init__(config or {})
        
        # DEPRECATED: Hardcoded defaults kept for backward compatibility
        # To change these, edit CONFIG/model_config/transformer.yaml
        self.config.setdefault("epochs", 50)
        self.config.setdefault("batch_size", 512)
        self.config.setdefault("d_model", 128)
        self.config.setdefault("heads", 8)
        self.config.setdefault("ff_dim", 256)
        self.config.setdefault("dropout", 0.1)
        self.config.setdefault("learning_rate", 1e-3)
        self.config.setdefault("patience", 10)

    def train(self, X_tr: np.ndarray, y_tr: np.ndarray, 
              X_va=None, y_va=None, feature_names: List[str] = None, **kwargs) -> Any:
        from common.threads import ensure_gpu_visible
        
        # Ensure GPU is visible (restore if hidden by prior CPU-only family)
        gpu_available = ensure_gpu_visible("Transformer")
        
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
        
        # 4) Reshape for Transformer
        X_tr = X_tr.reshape(X_tr.shape[0], X_tr.shape[1], 1)
        X_va = X_va.reshape(X_va.shape[0], X_va.shape[1], 1)
        
        # 5) Build model with safe defaults
        model = self._build_model(X_tr.shape[1])
        
        # 6) Train with callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=self.config["patience"], restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6)
        ]
        
        model.fit(
            X_tr, y_tr,
            validation_data=(X_va, y_va),
            epochs=self.config["epochs"],
            batch_size=self.config["batch_size"],
            callbacks=callbacks,
            verbose=0
        )
        
        # 7) Store state and sanity check
        self.model = model
        self.is_trained = True
        self.post_fit_sanity(X_tr, "Transformer")
        return self.model

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        Xp, _ = self.preprocess_data(X, None)
        Xp = Xp.reshape(Xp.shape[0], Xp.shape[1], 1)
        preds = self.model.predict(Xp, verbose=0).ravel()
        return np.nan_to_num(preds, nan=0.0).astype(np.float32)

    def _build_model(self, input_dim: int) -> tf.keras.Model:
        """Build Transformer model with safe defaults"""
        inputs = tf.keras.Input(shape=(input_dim, 1), name="x")
        x = inputs
        
        # Project to d_model
        x = tf.keras.layers.Conv1D(self.config["d_model"], 1, activation="linear")(x)
        
        # Multi-head attention
        attn = tf.keras.layers.MultiHeadAttention(
            num_heads=self.config["heads"], 
            key_dim=self.config["d_model"]
        )(x, x)
        
        # Add & Norm
        x = tf.keras.layers.Add()([x, attn])
        x = tf.keras.layers.LayerNormalization()(x)
        
        # Feed forward
        ff = tf.keras.layers.Dense(self.config["ff_dim"], activation="relu")(x)
        ff = tf.keras.layers.Dropout(self.config["dropout"])(ff)
        ff = tf.keras.layers.Dense(self.config["d_model"])(ff)
        
        # Add & Norm
        x = tf.keras.layers.Add()([x, ff])
        x = tf.keras.layers.LayerNormalization()(x)
        
        # Global pooling and output
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(self.config["dropout"])(x)
        outputs = tf.keras.layers.Dense(1, name="y")(x)
        
        model = tf.keras.Model(inputs, outputs)
        
        # Compile with gradient clipping
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config["learning_rate"],
            clipnorm=1.0
        )
        
        model.compile(
            optimizer=optimizer,
            loss="mse",
            metrics=["mae"]
        )
        
        return model