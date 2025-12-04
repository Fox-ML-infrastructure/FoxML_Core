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
from sklearn.model_selection import train_test_split
from pathlib import Path
from .base_trainer import BaseModelTrainer
from TRAINING.common.safety import configure_tf
logger = logging.getLogger(__name__)

# Add CONFIG to path for centralized configuration loading
_CONFIG_DIR = Path(__file__).resolve().parents[2] / "CONFIG"
if str(_CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(_CONFIG_DIR))

try:
    from config_loader import load_model_config
    _USE_CENTRALIZED_CONFIG = True
except ImportError:
    logger.warning("Could not import config_loader, falling back to hardcoded defaults")
    _USE_CENTRALIZED_CONFIG = False

class MultiTaskTrainer(BaseModelTrainer):
    """
    Multi-task trainer with support for multiple output heads (Spec 1: MTL).
    Supports both single-target (backward compatible) and multi-target training.
    For correlated targets (TTH, MDD, MFE), use multi-target mode with loss weights.
    """
    def __init__(self, config: Dict[str, Any] = None):
        # Load from centralized CONFIG if not provided
        if config is None and _USE_CENTRALIZED_CONFIG:
            try:
                config = load_model_config("multi_task")
                logger.info("✅ Loaded MultiTask config from CONFIG/model_config/multi_task.yaml")
            except Exception as e:
                logger.warning(f"Failed to load centralized config: {e}, using hardcoded defaults")
                config = {}
        
        super().__init__(config or {})
        
        # DEPRECATED: Hardcoded defaults (kept for backward compatibility)
        # These values are now defined in CONFIG/model_config/multi_task.yaml
        # Spec 1: Multitask Learning defaults
        self.config.setdefault("epochs", 50)
        self.config.setdefault("batch_size", 512)
        self.config.setdefault("hidden_dim", 256)  # Shared hidden layer size
        self.config.setdefault("dropout", 0.2)
        self.config.setdefault("learning_rate", 3e-4)  # 1e-4 to 5e-4 range, using middle
        self.config.setdefault("patience", 10)
        # Multi-task specific
        self.config.setdefault("use_multi_head", None)  # Auto-detect from y shape
        self.config.setdefault("loss_weights", None)  # Dict mapping target names to weights
        self.config.setdefault("target_names", None)  # List of target names for multi-head

    def train(self, X_tr: np.ndarray, y_tr: np.ndarray, 
              X_va=None, y_va=None, feature_names: List[str] = None, **kwargs) -> Any:
        from common.threads import ensure_gpu_visible
        
        # Ensure GPU is visible (restore if hidden by prior CPU-only family)
        gpu_available = ensure_gpu_visible("MultiTask")
        
        # 1) Preprocess data
        X_tr, y_tr = self.preprocess_data(X_tr, y_tr)
        self.feature_names = feature_names or [f"f{i}" for i in range(X_tr.shape[1])]
        
        # 2) Detect multi-target mode
        # Check if y is 2D with multiple columns (multi-target)
        is_multi_target = len(y_tr.shape) > 1 and y_tr.shape[1] > 1
        use_multi_head = self.config.get("use_multi_head")
        if use_multi_head is None:
            use_multi_head = is_multi_target
        
        # 3) Get target names
        if use_multi_head:
            target_names = self.config.get("target_names")
            if target_names is None:
                n_targets = y_tr.shape[1] if is_multi_target else 1
                target_names = [f"task_{i+1}" for i in range(n_targets)]
            self.target_names = target_names
            logger.info(f"MultiTask: Using multi-head mode with {len(target_names)} targets: {target_names}")
        else:
            self.target_names = ["y"]
            logger.info("MultiTask: Using single-head mode (backward compatible)")
        
        # 4) Configure TensorFlow
        configure_tf(cpu_only=kwargs.get("cpu_only", False) or not gpu_available)
        
        # 5) Split only if no external validation provided
        if X_va is None or y_va is None:
            X_tr, X_va, y_tr, y_va = train_test_split(
                X_tr, y_tr, test_size=0.2, random_state=42
            )
        
        # 6) Prepare targets for multi-head mode
        if use_multi_head and is_multi_target:
            # Convert 2D y to dict format for multi-head training
            y_tr_dict = {name: y_tr[:, i] for i, name in enumerate(self.target_names)}
            y_va_dict = {name: y_va[:, i] for i, name in enumerate(self.target_names)}
        else:
            # Single target mode
            y_tr_dict = y_tr
            y_va_dict = y_va
        
        # 7) Build model with safe defaults
        model = self._build_model(X_tr.shape[1], use_multi_head=use_multi_head)
        
        # 8) Prepare loss and loss_weights
        if use_multi_head:
            loss_dict = {name: "mse" for name in self.target_names}
            loss_weights = self.config.get("loss_weights")
            if loss_weights is None:
                # Default: equal weights for all targets
                loss_weights = {name: 1.0 for name in self.target_names}
            # Ensure all targets have weights
            for name in self.target_names:
                if name not in loss_weights:
                    loss_weights[name] = 1.0
            logger.info(f"MultiTask loss weights: {loss_weights}")
        else:
            loss_dict = "mse"
            loss_weights = None
        
        # 9) Train with callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss" if not use_multi_head else "val_loss",
                patience=self.config["patience"],
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6)
        ]
        
        model.fit(
            X_tr, y_tr_dict if use_multi_head else y_tr,
            validation_data=(X_va, y_va_dict if use_multi_head else y_va),
            epochs=self.config["epochs"],
            batch_size=self.config["batch_size"],
            callbacks=callbacks,
            verbose=0
        )
        
        # 10) Store state and sanity check
        self.model = model
        self.is_trained = True
        self.use_multi_head = use_multi_head
        self.post_fit_sanity(X_tr, "MultiTask")
        return self.model

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        Xp, _ = self.preprocess_data(X, None)
        preds = self.model.predict(Xp, verbose=0)
        
        # Handle multi-head predictions
        if self.use_multi_head:
            # If multi-head, preds is a list of arrays (one per head)
            # Stack them into a 2D array
            if isinstance(preds, (list, tuple)):
                preds = np.column_stack([p.ravel() for p in preds])
            else:
                # Already stacked or single output
                preds = preds.ravel() if len(preds.shape) > 1 and preds.shape[1] == 1 else preds
        else:
            preds = preds.ravel()
        
        return np.nan_to_num(preds, nan=0.0).astype(np.float32)

    def _build_model(self, input_dim: int, use_multi_head: bool = False) -> tf.keras.Model:
        """
        Build MultiTask model with safe defaults (Spec 1: MTL).
        
        For multi-head mode:
        - Shared hidden layers: Dense(256, ReLU), BN, Dropout(0.2), Dense(128, ReLU), BN, Dropout(0.2)
        - Separate output heads: one Dense(1, linear) per target
        - Loss: dict mapping each output name to 'mse'
        - Loss weights: configurable per target (default: all 1.0)
        """
        inputs = tf.keras.Input(shape=(input_dim,), name="x")
        x = inputs
        
        # Shared layers (Spec 1 architecture)
        x = tf.keras.layers.Dense(self.config["hidden_dim"], activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(self.config["dropout"])(x)
        
        x = tf.keras.layers.Dense(self.config["hidden_dim"] // 2, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(self.config["dropout"])(x)
        
        # Output layer(s)
        if use_multi_head:
            # Multiple output heads (one per target)
            outputs = []
            for target_name in self.target_names:
                output = tf.keras.layers.Dense(1, activation="linear", name=target_name)(x)
                outputs.append(output)
            # Model expects list of outputs for multi-head
            model = tf.keras.Model(inputs, outputs)
        else:
            # Single output head (backward compatible)
            outputs = tf.keras.layers.Dense(1, activation="linear", name="y")(x)
            model = tf.keras.Model(inputs, outputs)
        
        # Compile with gradient clipping
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config["learning_rate"],
            clipnorm=1.0
        )
        
        # Prepare loss and loss_weights
        if use_multi_head:
            loss_dict = {name: "mse" for name in self.target_names}
            loss_weights = self.config.get("loss_weights")
            if loss_weights is None:
                loss_weights = {name: 1.0 for name in self.target_names}
            # Ensure all targets have weights
            for name in self.target_names:
                if name not in loss_weights:
                    loss_weights[name] = 1.0
        else:
            loss_dict = "mse"
            loss_weights = None
        
        model.compile(
            optimizer=optimizer,
            loss=loss_dict,
            loss_weights=loss_weights,
            metrics=["mae"] if not use_multi_head else {name: "mae" for name in self.target_names}
        )
        
        return model