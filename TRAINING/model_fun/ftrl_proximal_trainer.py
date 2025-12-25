# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

import numpy as np, logging, sys
from typing import Any, Dict, List, Optional
from pathlib import Path
from sklearn.linear_model import SGDRegressor
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

class FTRLProximalTrainer(BaseModelTrainer):
    def __init__(self, config: Dict[str, Any] = None):
        # Load centralized config if available and no config provided
        if config is None and _USE_CENTRALIZED_CONFIG:
            try:
                config = load_model_config("ftrl_proximal")
                logger.info("✅ [FTRLProximal] Loaded centralized config from CONFIG/model_config/ftrl_proximal.yaml")
            except Exception as e:
                logger.warning(f"Failed to load centralized config: {e}. Using hardcoded defaults.")
                config = {}
        
        super().__init__(config or {})
        
        # DEPRECATED: Hardcoded defaults kept for backward compatibility
        # To change these, edit CONFIG/model_config/ftrl_proximal.yaml
        self.config.setdefault("learning_rate", self.config.get("alpha", 0.0001))  # Support old "alpha" key
        self.config.setdefault("l1_regularization_strength", self.config.get("l1_ratio", 0.15))  # Support old key
        self.config.setdefault("max_iter", 2000)
        self.config.setdefault("tol", 1e-4)
        self.config.setdefault("learning_rate_schedule", "optimal")

    def train(self, X_tr: np.ndarray, y_tr: np.ndarray, 
              X_va=None, y_va=None, feature_names: List[str] = None, **kwargs) -> Any:
        # 1) Preprocess data
        X_tr, y_tr = self.preprocess_data(X_tr, y_tr)
        self.feature_names = feature_names or [f"f{i}" for i in range(X_tr.shape[1])]
        
        # 2) Split only if no external validation provided
        if X_va is None or y_va is None:
            test_size, random_state = self._get_test_split_params()
            X_tr, X_va, y_tr, y_va = train_test_split(
                X_tr, y_tr, test_size=test_size, random_state=random_state
            )
        
        # 3) Build model with safe defaults
        model = self._build_model()
        
        # 4) Train
        self.fit_with_threads(model, X_tr, y_tr)
        
        # 5) Store state and sanity check
        self.model = model
        self.is_trained = True
        self.post_fit_sanity(X_tr, "FTRLProximal")
        return self.model

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        Xp, _ = self.preprocess_data(X, None)
        preds = self.model.predict(Xp)
        return np.nan_to_num(preds, nan=0.0).astype(np.float32)

    def _build_model(self):
        """Build FTRLProximal model with safe defaults"""
        alpha = self.config.get("learning_rate", self.config.get("alpha", 0.0001))
        l1_ratio = self.config.get("l1_regularization_strength", self.config.get("l1_ratio", 0.15))
        learning_rate_schedule = self.config.get("learning_rate_schedule", self.config.get("learning_rate", "optimal"))
        
        model = SGDRegressor(
            penalty="elasticnet",
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=self.config["max_iter"],
            tol=self.config["tol"],
            learning_rate=learning_rate_schedule,
            random_state=self._get_random_state(),
            **self.config.get("sgd_params", {})
        )
        return model