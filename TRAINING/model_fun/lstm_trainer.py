# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

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

class LSTMTrainer(BaseModelTrainer):
    def __init__(self, config: Dict[str, Any] = None):
        # Load centralized config if available and no config provided
        if config is None and _USE_CENTRALIZED_CONFIG:
            try:
                config = load_model_config("lstm")
                logger.info("✅ [LSTM] Loaded centralized config from CONFIG/model_config/lstm.yaml")
            except Exception as e:
                logger.warning(f"Failed to load centralized config: {e}. Using hardcoded defaults.")
                config = {}
        
        super().__init__(config or {})
        
        # DEPRECATED: Hardcoded defaults kept for backward compatibility
        # To change these, edit CONFIG/model_config/lstm.yaml
        self.config.setdefault("epochs", 30)  # Reduced from 50 to prevent timeouts
        self.config.setdefault("batch_size", 256)  # Reduced from 512 to speed up training
        self.config.setdefault("lstm_units", self.config.get("units", 128))  # Support old "units" key
        self.config.setdefault("dropout", 0.2)
        self.config.setdefault("recurrent_dropout", 0.1)  # Reduced from 0.2 to speed up training
        self.config.setdefault("learning_rate", 1e-3)
        self.config.setdefault("patience", 5)  # Reduced from 10 for faster early stopping

    def train(self, X_tr: np.ndarray, y_tr: np.ndarray, 
              X_va=None, y_va=None, feature_names: List[str] = None, **kwargs) -> Any:
        from common.threads import ensure_gpu_visible
        
        # Ensure GPU is visible (restore if hidden by prior CPU-only family)
        gpu_available = ensure_gpu_visible("LSTM")
        
        # 1) Preprocess data
        X_tr, y_tr = self.preprocess_data(X_tr, y_tr)
        self.feature_names = feature_names or [f"f{i}" for i in range(X_tr.shape[1])]
        
        # 2) Configure TensorFlow
        configure_tf(cpu_only=kwargs.get("cpu_only", False) or not gpu_available)
        
        # Enable mixed precision for Ampere GPUs (compute capability 8.6+)
        if not kwargs.get("cpu_only", False):
            try:
                from tensorflow.keras import mixed_precision
                mixed_precision.set_global_policy("mixed_float16")
                logger.info("🚀 [LSTM] Enabled mixed precision for faster training")
            except Exception as e:
                logger.debug(f"Mixed precision not available: {e}")
        
        # 3) Split only if no external validation provided
        if X_va is None or y_va is None:
            # Load test split params from config
            test_size, seed = self._get_test_split_params()
            X_tr, X_va, y_tr, y_va = train_test_split(
                X_tr, y_tr, test_size=test_size, random_state=seed
            )
        
        # 4) Reshape for LSTM
        X_tr = X_tr.reshape(X_tr.shape[0], X_tr.shape[1], 1)
        X_va = X_va.reshape(X_va.shape[0], X_va.shape[1], 1)
        
        # 4.5) Adjust batch size and epochs dynamically based on sequence length to prevent timeouts
        seq_len = X_tr.shape[1]
        base_batch_size = self.config["batch_size"]
        base_epochs = self.config["epochs"]
        max_seq_for_full_batch = self.config.get("max_sequence_length_for_full_batch", 200)
        
        # For sequences > 200, reduce batch size to speed up training
        if seq_len > max_seq_for_full_batch:
            # Scale batch size inversely with sequence length
            adjusted_batch_size = max(32, int(base_batch_size * (max_seq_for_full_batch / seq_len)))
            if adjusted_batch_size < base_batch_size:
                logger.warning(f"⚠️ [LSTM] Reducing batch size from {base_batch_size} to {adjusted_batch_size} "
                             f"to speed up training (sequence length: {seq_len})")
                self.config["batch_size"] = adjusted_batch_size
            
            # For very long sequences (>300), also reduce epochs to prevent timeouts
            if seq_len > 300:
                adjusted_epochs = max(15, int(base_epochs * (300 / seq_len)))
                if adjusted_epochs < base_epochs:
                    logger.warning(f"⚠️ [LSTM] Reducing epochs from {base_epochs} to {adjusted_epochs} "
                                 f"to prevent timeout (sequence length: {seq_len})")
                    self.config["epochs"] = adjusted_epochs
        
        # 5) Build model with safe defaults
        model = self._build_model(X_tr.shape[1])
        
        # 6) Train with callbacks (load from config if available)
        callbacks = self.get_callbacks("LSTM")
        
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
        self.post_fit_sanity(X_tr, "LSTM")
        return self.model

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        # Handle 3D input (from post_fit_sanity) by reshaping to 2D for preprocessing
        # LSTM uses shape (samples, timesteps, 1), so reshape to (samples, timesteps)
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], X.shape[1])
        Xp, _ = self.preprocess_data(X, None)
        Xp = Xp.reshape(Xp.shape[0], Xp.shape[1], 1)
        preds = self.model.predict(Xp, verbose=0).ravel()
        return np.nan_to_num(preds, nan=0.0).astype(np.float32)

    def _build_model(self, input_dim: int) -> tf.keras.Model:
        """Build LSTM model with safe defaults"""
        inputs = tf.keras.Input(shape=(input_dim, 1), name="x")
        x = inputs
        
        # LSTM layers with dropout
        lstm_units = self.config.get("lstm_units", self.config.get("units", 128))
        x = tf.keras.layers.LSTM(
            lstm_units,
            return_sequences=True,
            dropout=self.config["dropout"],
            recurrent_dropout=self.config["recurrent_dropout"]
        )(x)
        x = tf.keras.layers.LSTM(
            lstm_units // 2,
            dropout=self.config["dropout"],
            recurrent_dropout=self.config["recurrent_dropout"]
        )(x)
        
        # Dense layers
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(self.config["dropout"])(x)
        outputs = tf.keras.layers.Dense(1, name="y")(x)
        
        model = tf.keras.Model(inputs, outputs)
        
        # Compile with gradient clipping (load from config if available)
        clipnorm = self._get_clipnorm()
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config["learning_rate"],
            clipnorm=clipnorm
        )
        
        model.compile(
            optimizer=optimizer,
            loss="mse",
            metrics=["mae"]
        )
        
        return model