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

# TRAINING/model_fun/quantile_lightgbm_trainer.py
"""
QuantileLightGBM trainer with robust defaults and Huber fallback.
Quantile loss is slower and more sensitive than L2, so we use:
- Safe params for big data (smaller max_bin, moderate num_leaves, subsampling)
- Early stopping with validation split
- Time budget to prevent hanging
- Huber fallback if quantile fails
"""
import os, time, logging, sys
import numpy as np
import lightgbm as lgb
from typing import Any, Dict
from pathlib import Path
from sklearn.model_selection import train_test_split

from .base_trainer import BaseModelTrainer
from TRAINING.common.threads import thread_guard, log_thread_state, plan_for_family

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

def _time_budget_cb(budget_sec: int):
    """Callback to enforce time budget during training."""
    start = time.time()
    def _cb(env):
        if time.time() - start > budget_sec:
            raise RuntimeError(f"TIME_BUDGET_EXCEEDED({budget_sec}s)")
    _cb.order = 0
    return _cb

class QuantileLightGBMTrainer(BaseModelTrainer):
    def __init__(self, config: Dict[str, Any] = None):
        # Load centralized config if available and no config provided
        if config is None and _USE_CENTRALIZED_CONFIG:
            try:
                config = load_model_config("quantile_lightgbm")
                logger.info("✅ [QuantileLightGBM] Loaded centralized config from CONFIG/model_config/quantile_lightgbm.yaml")
            except Exception as e:
                logger.warning(f"Failed to load centralized config: {e}. Using hardcoded defaults.")
                config = {}
        
        super().__init__(config or {})
        
        # Initialize num_threads from config or environment
        self.num_threads = int(self.config.get("num_threads", os.getenv("OMP_NUM_THREADS", "4")))
        self.config.setdefault("threads", self.num_threads)
        
        # DEPRECATED: Hardcoded defaults kept for backward compatibility
        # To change these, edit CONFIG/model_config/quantile_lightgbm.yaml
        self.config.setdefault("alpha", 0.5)
        self.config.setdefault("learning_rate", 0.05)
        self.config.setdefault("n_estimators", 2000)
        self.config.setdefault("num_leaves", 64)
        self.config.setdefault("max_depth", -1)
        self.config.setdefault("min_data_in_leaf", 2048)  # Stabilizes quantile
        self.config.setdefault("feature_fraction", 0.7)
        self.config.setdefault("bagging_fraction", 0.7)
        self.config.setdefault("lambda_l1", 0.0)
        self.config.setdefault("lambda_l2", 2.0)
        self.config.setdefault("early_stopping_rounds", 100)
        self.config.setdefault("max_bin", 63)  # Faster histogram build
        self.config.setdefault("time_budget_sec", 1800)  # 30 min default
        
    def _threads(self) -> int:
        """Get thread count, clamped for quantile (4-8 is sweet spot)."""
        t = self.num_threads
        return max(1, min(8, int(t)))
    
    def _safe_params(self, num_threads: int, alpha: float):
        """Big-data safe defaults for quantile."""
        return {
            "objective": "quantile",
            "alpha": float(alpha),
            "metric": "quantile",
            "num_threads": num_threads,
            "learning_rate": self.config["learning_rate"],
            "num_leaves": self.config["num_leaves"],
            "min_data_in_leaf": self.config["min_data_in_leaf"],
            "min_sum_hessian_in_leaf": 1.0,
            "feature_fraction": self.config["feature_fraction"],
            "bagging_fraction": self.config["bagging_fraction"],
            "bagging_freq": 1,
            "lambda_l1": self.config["lambda_l1"],
            "lambda_l2": self.config["lambda_l2"],
            "max_depth": self.config["max_depth"],
            "max_bin": self.config["max_bin"],
            "bin_construct_sample_cnt": 200000,
            "verbosity": -1,
            "force_col_wise": True,  # Faster on wide tables
            "feature_pre_filter": True,
            "two_round": True,
        }
    
    def _fallback_params(self, num_threads: int):
        """Robust-but-fast alternative if quantile fails."""
        return {
            "objective": "huber",
            "alpha": 0.9,  # Huber quantile-ish
            "n_estimators": self.config["n_estimators"],
            "learning_rate": self.config["learning_rate"],
            "num_leaves": self.config["num_leaves"],
            "min_data_in_leaf": self.config["min_data_in_leaf"],
            "feature_fraction": self.config["feature_fraction"],
            "bagging_fraction": self.config["bagging_fraction"],
            "bagging_freq": 1,
            "lambda_l2": self.config["lambda_l2"],
            "max_bin": self.config["max_bin"],
            "n_jobs": num_threads,
            "num_threads": num_threads,
            "verbose": -1,
        }
    
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Any:
        """Train QuantileLightGBM with robust defaults and fallback."""
        # 1) Preprocess and guard features/targets
        X, y = self.preprocess_data(X, y)
        
        # 2) Split for validation (crucial for quantile + early stopping)
        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y, test_size=0.1, random_state=42, shuffle=False  # Chronological for time series
        )
        
        # 3) Get thread count
        omp = self._threads()
        
        logger.info(f"QuantileLightGBM num_threads={omp} | OMP={os.getenv('OMP_NUM_THREADS')}")
        
        # Diagnostic: check threadpool state before training
        try:
            from threadpoolctl import threadpool_info
            for tp in threadpool_info():
                logger.info(f"TP: lib={tp.get('internal_api')}, api={tp.get('user_api')}, num_threads={tp.get('num_threads')}")
        except Exception:
            pass
        
        # 4) Build LightGBM datasets
        lgb_tr = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
        lgb_va = lgb.Dataset(X_va, label=y_va, reference=lgb_tr, free_raw_data=False)
        
        # 5) Setup params and callbacks
        params = self._safe_params(num_threads=omp, alpha=self.config["alpha"])
        rounds = self.config["n_estimators"]
        esr = self.config["early_stopping_rounds"]
        budget = self.config["time_budget_sec"]
        
        callbacks = [
            lgb.early_stopping(stopping_rounds=esr, verbose=False),
            lgb.log_evaluation(period=200),
            _time_budget_cb(budget)
        ]
        
        logger.info(f"[QuantileLGBM] Training with alpha={self.config['alpha']}, rounds={rounds}, ESR={esr}, budget={budget}s")
        
        try:
            # 6) Train with time budget and early stopping
            booster = lgb.train(
                params=params,
                train_set=lgb_tr,
                num_boost_round=rounds,
                valid_sets=[lgb_va],
                callbacks=callbacks
            )
            best_iter = booster.best_iteration or rounds
            logger.info(f"✅ QuantileLGBM trained | best_iter={best_iter} | best_score={booster.best_score}")
            
            # Store state
            self.model = booster
            self.is_trained = True
            self.post_fit_sanity(X_tr, "QuantileLightGBM")
            
            return self.model
            
        except Exception as e:
            logger.warning(f"❌ QuantileLGBM failed ({e}) → falling back to Huber LGBM")
            
            # Fallback: Huber objective (robust regression, much faster)
            from lightgbm import LGBMRegressor
            fb_params = self._fallback_params(num_threads=omp)
            fb = LGBMRegressor(**fb_params)
            fb.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[lgb.early_stopping(stopping_rounds=esr, verbose=False)]
            )
            logger.info(f"✅ Huber fallback trained | best_iter={fb.best_iteration_}")
            
            # Store state
            self.model = fb
            self.is_trained = True
            self.post_fit_sanity(X_tr, "QuantileLightGBM(Huber)")
            
            return self.model
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using trained model."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        Xp, _ = self.preprocess_data(X, None)
        preds = self.model.predict(Xp)
        return np.nan_to_num(preds, nan=0.0).astype(np.float32)