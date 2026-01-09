# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

import numpy as np, logging, sys
from typing import Any, Dict, List, Optional
from pathlib import Path
from sklearn.model_selection import train_test_split
import keras
from keras import layers, ops, optimizers, callbacks
from keras import Model
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

class MetaLearningTrainer(BaseModelTrainer):
    def __init__(self, config: Dict[str, Any] = None):
        # Load centralized config if available and no config provided
        if config is None and _USE_CENTRALIZED_CONFIG:
            try:
                config = load_model_config("meta_learning")
                logger.info("✅ [MetaLearning] Loaded centralized config from CONFIG/model_config/meta_learning.yaml")
            except Exception as e:
                logger.warning(f"Failed to load centralized config: {e}. Using hardcoded defaults.")
                config = {}
        
        super().__init__(config or {})
        
        # DEPRECATED: Hardcoded defaults kept for backward compatibility
        # To change these, edit CONFIG/model_config/meta_learning.yaml
        self.config.setdefault("epochs", 50)
        self.config.setdefault("batch_size", 512)
        self.config.setdefault("hidden_dim", 128)
        self.config.setdefault("dropout", 0.2)
        self.config.setdefault("learning_rate", 1e-3)
        self.config.setdefault("patience", 10)

    def train(self, X_tr: np.ndarray, y_tr: np.ndarray, 
              X_va=None, y_va=None, feature_names: List[str] = None, **kwargs) -> Any:
        from common.threads import ensure_gpu_visible
        
        # Ensure GPU is visible (restore if hidden by prior CPU-only family)
        gpu_available = ensure_gpu_visible("MetaLearning")
        
        # 1) Preprocess data
        X_tr, y_tr = self.preprocess_data(X_tr, y_tr)
        self.feature_names = feature_names or [f"f{i}" for i in range(X_tr.shape[1])]
        
        # 2) Configure TensorFlow
        configure_tf(cpu_only=kwargs.get("cpu_only", False) or not gpu_available)
        
        # TensorFlow is already initialized by _bootstrap_family_runtime in isolation_runner
        # Just import it here - threading and GPU config already done
        import tensorflow as tf
        
        # Check if we have GPUs (already configured by bootstrap)
        gpus = tf.config.list_physical_devices("GPU")
        logger.info("[MetaLearning] Starting training with %d GPUs available", len(gpus))
        
        if not gpus and gpu_available:
            logger.warning("[MetaLearning] GPU was visible but TensorFlow cannot access it - check CUDA installation")
        
        # Enable mixed precision for Ampere GPUs (compute capability 8.6+)
        if gpus:
            try:
                tf.keras.mixed_precision.set_global_policy("mixed_float16")
                logger.info("🚀 [MetaLearning] Enabled mixed precision (float16) for faster training")
            except Exception as e:
                logger.debug("Mixed precision not available: %s", e)
        
        # 3) Split only if no external validation provided
        if X_va is None or y_va is None:
            # Load test split params from config
            test_size, seed = self._get_test_split_params()
            X_tr, X_va, y_tr, y_va = train_test_split(
                X_tr, y_tr, test_size=test_size, random_state=seed
            )
        
        # 4) Build model with safe defaults
        model = self._build_model(X_tr.shape[1])
        
        # 5) Train with callbacks
        cbs = [
            callbacks.EarlyStopping(patience=self.config.get("patience", 10), restore_best_weights=True),
            callbacks.ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6)
        ]
        
        model.fit(
            X_tr, y_tr,
            validation_data=(X_va, y_va),
            epochs=self.config.get("epochs", 50),
            batch_size=self.config.get("batch_size", 512),
            callbacks=cbs,
            verbose=0
        )
        
        # 6) Store state and sanity check
        self.model = model
        self.is_trained = True
        self.post_fit_sanity(X_tr, "MetaLearning")
        return self.model

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        Xp, _ = self.preprocess_data(X, None)
        preds = self.model.predict(Xp)
        return np.nan_to_num(preds, nan=0.0).astype(np.float32)

    def _build_model(self, input_dim: int) -> Model:
        """Build MetaLearning model with TensorFlow/Keras - GPU accelerated"""
        inputs = layers.Input(shape=(input_dim,), dtype="float32", name="x")
        
        # Base learner 1: Linear path
        linear = layers.Dense(self.config["hidden_dim"], activation="relu", name="linear_base")(inputs)
        linear = layers.BatchNormalization()(linear)
        linear = layers.Dropout(self.config["dropout"])(linear)
        linear_out = layers.Dense(1, name="linear_pred")(linear)
        
        # Base learner 2: Non-linear path (deeper network)
        nonlinear = layers.Dense(self.config["hidden_dim"] * 2, activation="relu", name="nonlinear_base1")(inputs)
        nonlinear = layers.BatchNormalization()(nonlinear)
        nonlinear = layers.Dropout(self.config["dropout"])(nonlinear)
        nonlinear = layers.Dense(self.config["hidden_dim"], activation="relu", name="nonlinear_base2")(nonlinear)
        nonlinear = layers.BatchNormalization()(nonlinear)
        nonlinear = layers.Dropout(self.config["dropout"])(nonlinear)
        nonlinear_out = layers.Dense(1, name="nonlinear_pred")(nonlinear)
        
        # Meta-learner: Combine base learners
        combined = layers.Concatenate(name="meta_combine")([linear_out, nonlinear_out])
        meta = layers.Dense(self.config["hidden_dim"], activation="relu", name="meta_layer")(combined)
        meta = layers.BatchNormalization()(meta)
        meta = layers.Dropout(self.config["dropout"])(meta)
        outputs = layers.Dense(1, name="meta_pred")(meta)
        
        model = Model(inputs, outputs, name="meta_learning")
        
        # Compile with gradient clipping
        opt = optimizers.Adam(
            learning_rate=self.config["learning_rate"],
            clipnorm=self._get_clipnorm()
        )
        
        model.compile(optimizer=opt, loss="mse", metrics=["mae"])
        
        return model