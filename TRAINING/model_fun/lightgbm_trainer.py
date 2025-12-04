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

import numpy as np, logging, lightgbm as lgb, os, sys
from typing import Any, Dict, List, Optional
from sklearn.model_selection import train_test_split
from pathlib import Path
from .base_trainer import BaseModelTrainer
from TRAINING.common.threads import thread_guard
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

class LightGBMTrainer(BaseModelTrainer):
    def __init__(self, config: Dict[str, Any] = None):
        # Load from centralized CONFIG if not provided
        if config is None and _USE_CENTRALIZED_CONFIG:
            try:
                config = load_model_config("lightgbm")
                logger.info("✅ Loaded LightGBM config from CONFIG/model_config/lightgbm.yaml")
            except Exception as e:
                logger.warning(f"Failed to load centralized config: {e}, using hardcoded defaults")
                config = {}
        
        super().__init__(config or {})
        
        # DEPRECATED: Hardcoded defaults (kept for backward compatibility)
        # These values are now defined in CONFIG/model_config/lightgbm.yaml
        # Spec 2: High Regularization defaults for LightGBM (fixes overfitting)
        # Recommended: max_depth 7-9, num_leaves 64-128, learning_rate 0.01-0.05
        self.config.setdefault("num_leaves", 96)  # 64-128 range, using middle
        self.config.setdefault("max_depth", 8)  # 7-9 range, using 8
        self.config.setdefault("min_data_in_leaf", 200)  # Keep existing
        self.config.setdefault("min_child_weight", 0.5)  # New: 0.1-1.0 range
        self.config.setdefault("feature_fraction", 0.75)  # 0.7-0.8 range (colsample_bytree)
        self.config.setdefault("bagging_fraction", 0.75)  # 0.7-0.8 range (subsample)
        self.config.setdefault("bagging_freq", 1)  # Enable bagging every iteration
        self.config.setdefault("lambda_l1", 0.1)  # Reduced from 1.0 (0.1-1.0 range)
        self.config.setdefault("lambda_l2", 0.1)  # Reduced from 2.0 (0.1-1.0 range)
        self.config.setdefault("learning_rate", 0.03)  # 0.01-0.05 range, using middle
        self.config.setdefault("n_estimators", 1000)  # Use early stopping instead
        self.config.setdefault("early_stopping_rounds", 50)  # Reduced from 200
        # Get optimal thread count from config or environment
        import os
        self.num_threads = int(self.config.get("num_threads", os.getenv("OMP_NUM_THREADS", "4")))
        self.config.setdefault("threads", self.num_threads)

    def train(self, X_tr: np.ndarray, y_tr: np.ndarray, 
              X_va=None, y_va=None, feature_names: List[str] = None, **kwargs) -> Any:
        # 1) Preprocess data
        X_tr, y_tr = self.preprocess_data(X_tr, y_tr)
        self.feature_names = feature_names or [f"f{i}" for i in range(X_tr.shape[1])]
        
        # 2) Split only if no external validation provided
        if X_va is None or y_va is None:
            X_tr, X_va, y_tr, y_va = train_test_split(
                X_tr, y_tr, test_size=0.2, random_state=42
            )
        
        # 3) Build model with safe defaults
        model = self._build_model(kwargs.get("cpu_only", False))
        
        # 4) Train with early stopping
        logger.info(f"LightGBM num_threads={self.num_threads} | OMP={os.getenv('OMP_NUM_THREADS')}")
        
        # Diagnostic: check threadpool state before training
        try:
            from threadpoolctl import threadpool_info
            for tp in threadpool_info():
                logger.info(f"TP: lib={tp.get('internal_api')}, api={tp.get('user_api')}, num_threads={tp.get('num_threads')}")
        except Exception:
            pass
        
        # Train the model (thread_guard is already applied by isolation runner)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(self.config["early_stopping_rounds"], verbose=False)]
        )
        
        # 5) Store state and sanity check
        self.model = model
        self.is_trained = True
        self.post_fit_sanity(X_tr, "LightGBM")
        return self.model

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        Xp, _ = self.preprocess_data(X, None)
        preds = self.model.predict(Xp)
        return np.nan_to_num(preds, nan=0.0).astype(np.float32)

    def _build_model(self, cpu_only: bool = False):
        """Build LightGBM model with safe defaults and explicit threading"""
        # Get thread count from unified system
        from common.threads import plan_for_family
        plan = plan_for_family("LightGBM", self.num_threads)
        threads = plan["OMP"]
        
        model = lgb.LGBMRegressor(
            objective="regression",
            num_leaves=self.config["num_leaves"],
            max_depth=self.config["max_depth"],
            min_data_in_leaf=self.config["min_data_in_leaf"],
            min_child_weight=self.config.get("min_child_weight", 0.5),
            feature_fraction=self.config["feature_fraction"],
            bagging_fraction=self.config["bagging_fraction"],
            bagging_freq=self.config.get("bagging_freq", 1),
            lambda_l1=self.config["lambda_l1"],
            lambda_l2=self.config["lambda_l2"],
            learning_rate=self.config["learning_rate"],
            n_estimators=self.config["n_estimators"],
            n_jobs=threads,          # sklearn alias
            num_threads=threads,     # LightGBM native (belt and suspenders)
            random_state=42,
            verbose=-1,
            # Speed optimizations (don't change model quality)
            feature_pre_filter=True,
            bin_construct_sample_cnt=200000,  # limits binning cost
            two_round=True,  # speed up IO
            **self.config.get("lgbm_params", {})
        )
        logger.info(f"LightGBM model configured with {threads} threads (n_jobs={threads}, num_threads={threads})")
        return model