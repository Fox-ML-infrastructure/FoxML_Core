# MIT License - see LICENSE file

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

class CNN1DTrainer(BaseModelTrainer):
    def __init__(self, config: Dict[str, Any] = None):
        # Load centralized config if available and no config provided
        if config is None and _USE_CENTRALIZED_CONFIG:
            try:
                config = load_model_config("cnn1d")
                logger.info("✅ [CNN1D] Loaded centralized config from CONFIG/model_config/cnn1d.yaml")
            except Exception as e:
                logger.warning(f"Failed to load centralized config: {e}. Using hardcoded defaults.")
                config = {}
        
        super().__init__(config or {})
        
        # DEPRECATED: Hardcoded defaults kept for backward compatibility
        # To change these, edit CONFIG/model_config/cnn1d.yaml
        self.config.setdefault("epochs", 30)  # Reduced from 50 to prevent timeouts
        self.config.setdefault("batch_size", 256)  # Reduced from 512 to prevent OOM
        self.config.setdefault("filters", [64, 64])
        self.config.setdefault("dropout", 0.2)
        self.config.setdefault("learning_rate", 1e-3)
        self.config.setdefault("patience", 10)

    def train(self, X_tr: np.ndarray, y_tr: np.ndarray, 
              X_va=None, y_va=None, feature_names: List[str] = None, **kwargs) -> Any:
        from common.threads import ensure_gpu_visible
        
        # Ensure GPU is visible (restore if hidden by prior CPU-only family)
        gpu_available = ensure_gpu_visible("CNN1D")
        
        # 1) Preprocess data
        X_tr, y_tr = self.preprocess_data(X_tr, y_tr)
        self.feature_names = feature_names or [f"f{i}" for i in range(X_tr.shape[1])]
        
        # 2) Configure TensorFlow
        configure_tf(cpu_only=kwargs.get("cpu_only", False) or not gpu_available)
        
        # 3) Split only if no external validation provided
        if X_va is None or y_va is None:
            # Load test split params from config
            test_size, random_state = self._get_test_split_params()
            X_tr, X_va, y_tr, y_va = train_test_split(
                X_tr, y_tr, test_size=test_size, random_state=random_state
            )
        
        # 4) Reshape for CNN1D
        X_tr = X_tr.reshape(X_tr.shape[0], X_tr.shape[1], 1)
        X_va = X_va.reshape(X_va.shape[0], X_va.shape[1], 1)
        
        # 5) Build model with safe defaults
        model = self._build_model(X_tr.shape[1])
        
        # 6) Train with callbacks
        callbacks = [
            # Use callbacks from config if available
            *self.get_callbacks("CNN1D")
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
        self.post_fit_sanity(X_tr, "CNN1D")
        return self.model

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        # Handle 3D input (from post_fit_sanity) by reshaping to 2D for preprocessing
        # CNN1D uses shape (samples, timesteps, 1), so reshape to (samples, timesteps)
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], X.shape[1])
        Xp, _ = self.preprocess_data(X, None)
        Xp = Xp.reshape(Xp.shape[0], Xp.shape[1], 1)
        preds = self.model.predict(Xp, verbose=0).ravel()
        return np.nan_to_num(preds, nan=0.0).astype(np.float32)

    def _build_model(self, input_dim: int) -> tf.keras.Model:
        """Build CNN1D model with safe defaults"""
        inputs = tf.keras.Input(shape=(input_dim, 1), name="x")
        x = inputs
        
        for filters in self.config["filters"]:
            x = tf.keras.layers.Conv1D(filters, 3, padding="same", activation="relu")(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(self.config["dropout"])(x)
            x = tf.keras.layers.MaxPool1D(2)(x)
        
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
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