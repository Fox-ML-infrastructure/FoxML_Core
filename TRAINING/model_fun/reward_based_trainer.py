# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

import numpy as np, logging, sys
from typing import Any, Dict, List, Optional
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from .base_trainer import BaseModelTrainer
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

class RewardBasedTrainer(BaseModelTrainer):
    def __init__(self, config: Dict[str, Any] = None):
        # Load centralized config if available and no config provided
        if config is None and _USE_CENTRALIZED_CONFIG:
            try:
                config = load_model_config("reward_based")
                logger.info("✅ [RewardBased] Loaded centralized config from CONFIG/model_config/reward_based.yaml")
            except Exception as e:
                logger.warning(f"Failed to load centralized config: {e}. Using hardcoded defaults.")
                config = {}
        
        super().__init__(config or {})
        
        # DEPRECATED: Hardcoded defaults kept for backward compatibility
        # To change these, edit CONFIG/model_config/reward_based.yaml
        self.config.setdefault("alpha", 1.0)
        self.config.setdefault("reward_power", 1.0)  # weights ~ |y|^p
        self.config.setdefault("max_iter", 1000)
        self.config.setdefault("tol", 1e-4)

    def train(self, X_tr: np.ndarray, y_tr: np.ndarray, 
              X_va=None, y_va=None, feature_names: List[str] = None, **kwargs) -> Any:
        # 1) Preprocess data
        X_tr, y_tr = self.preprocess_data(X_tr, y_tr)
        self.feature_names = feature_names or [f"f{i}" for i in range(X_tr.shape[1])]
        
        # 2) Split only if no external validation provided
        if X_va is None or y_va is None:
            test_size, seed = self._get_test_split_params()
            X_tr, X_va, y_tr, y_va = train_test_split(
                X_tr, y_tr, test_size=test_size, random_state=seed
            )
        
        # 3) Build model with safe defaults
        model = self._build_model()
        
        # 4) Calculate sample weights based on reward power
        w = np.power(np.abs(y_tr) + 1e-6, float(self.config["reward_power"]))
        
        # 5) Train with sample weights
        self.fit_with_threads(model, X_tr, y_tr, sample_weight=w)
        
        # 6) Store state and sanity check
        self.model = model
        self.is_trained = True
        self.post_fit_sanity(X_tr, "RewardBased")
        return self.model

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        Xp, _ = self.preprocess_data(X, None)
        preds = self.model.predict(Xp)
        return np.nan_to_num(preds, nan=0.0).astype(np.float32)

    def _build_model(self):
        """Build RewardBased model with safe defaults"""
        model = Ridge(
            alpha=self.config["alpha"],
            max_iter=self.config["max_iter"],
            tol=self.config["tol"],
            seed=self._get_seed(),
            **self.config.get("ridge_params", {})
        )
        return model