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

import numpy as np, logging, xgboost as xgb, os, sys
from typing import Any, Dict, List, Optional
from sklearn.model_selection import train_test_split
from pathlib import Path
from .base_trainer import BaseModelTrainer
from TRAINING.common.threads import thread_guard
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

class XGBoostTrainer(BaseModelTrainer):
    def __init__(self, config: Dict[str, Any] = None):
        # Load from centralized CONFIG if not provided
        if config is None and _USE_CENTRALIZED_CONFIG:
            try:
                config = load_model_config("xgboost")
                logger.info("✅ Loaded XGBoost config from CONFIG/model_config/xgboost.yaml")
            except Exception as e:
                logger.warning(f"Failed to load centralized config: {e}, using hardcoded defaults")
                config = {}
        
        super().__init__(config or {})
        
        # DEPRECATED: Hardcoded defaults (kept for backward compatibility)
        # These values are now defined in CONFIG/model_config/xgboost.yaml
        # Spec 2: High Regularization defaults for XGBoost (fixes overfitting)
        # Recommended: max_depth 5-8, learning_rate 0.01-0.05, subsample/colsample 0.7-0.8
        self.config.setdefault("max_depth", 7)  # 5-8 range, using 7
        self.config.setdefault("min_child_weight", 0.5)  # Reduced from 10 (0.1-1.0 range)
        self.config.setdefault("subsample", 0.75)  # 0.7-0.8 range
        self.config.setdefault("colsample_bytree", 0.75)  # 0.7-0.8 range
        self.config.setdefault("gamma", 0.3)  # min_split_gain: 0.1-0.5 range
        self.config.setdefault("reg_alpha", 0.1)  # L1 reg: reduced from 1.0 (0.1-1.0 range)
        self.config.setdefault("reg_lambda", 0.1)  # L2 reg: reduced from 2.0 (0.1-1.0 range)
        self.config.setdefault("eta", 0.03)  # learning_rate: 0.01-0.05 range, using middle
        self.config.setdefault("n_estimators", 1000)  # Use early stopping instead
        self.config.setdefault("early_stopping_rounds", 50)  # Reduced from 200
        # Get optimal thread count from config or environment
        import os
        self.num_threads = int(self.config.get("num_threads", os.getenv("OMP_NUM_THREADS", "4")))
        self.config.setdefault("threads", self.num_threads)

    def train(self, X_tr: np.ndarray, y_tr: np.ndarray, 
              X_va=None, y_va=None, feature_names: List[str] = None, **kwargs) -> Any:
        import gc
        
        # 1) Preprocess data
        X_tr, y_tr = self.preprocess_data(X_tr, y_tr)
        self.feature_names = feature_names or [f"f{i}" for i in range(X_tr.shape[1])]
        
        # 2) Split only if no external validation provided
        if X_va is None or y_va is None:
            X_tr, X_va, y_tr, y_va = train_test_split(
                X_tr, y_tr, test_size=0.2, random_state=42
            )
        
        # 3) Determine if GPU is available
        cpu_only = kwargs.get("cpu_only", False)
        use_gpu = not cpu_only and self._check_gpu_available()
        
        # 4) Build model with memory-efficient settings
        model = self._build_model(cpu_only=not use_gpu)
        
        # 5) Train with early stopping (thread_guard already applied by in-process runner)
        logger.info(f"[XGBoost] num_threads={self.num_threads} | OMP={os.getenv('OMP_NUM_THREADS')} | GPU={use_gpu}")
        
        try:
            if use_gpu:
                # Try DeviceQuantileDMatrix (XGBoost >= 1.7), fallback to DMatrix
                has_device_quantile = hasattr(xgb, 'DeviceQuantileDMatrix')
                
                if has_device_quantile:
                    logger.info("[XGBoost] Using DeviceQuantileDMatrix for GPU training")
                    dtrain = xgb.DeviceQuantileDMatrix(X_tr, label=y_tr, max_bin=256)
                    dvalid = xgb.DeviceQuantileDMatrix(X_va, label=y_va, max_bin=256)
                else:
                    logger.info("[XGBoost] Using DMatrix for GPU training (DeviceQuantileDMatrix not available)")
                    dtrain = xgb.DMatrix(X_tr, label=y_tr)
                    dvalid = xgb.DMatrix(X_va, label=y_va)
                
                # Get booster directly for more control
                params = {
                    'tree_method': 'gpu_hist',
                    'predictor': 'gpu_predictor',
                    'max_depth': self.config["max_depth"],
                    'min_child_weight': self.config["min_child_weight"],
                    'subsample': self.config["subsample"],
                    'colsample_bytree': self.config["colsample_bytree"],
                    'gamma': self.config.get("gamma", 0.3),
                    'reg_alpha': self.config["reg_alpha"],
                    'reg_lambda': self.config["reg_lambda"],
                    'eta': self.config["eta"],
                    'objective': 'reg:squarederror',
                    'max_bin': 128,  # Reduce VRAM usage
                    'single_precision_histogram': True,  # Half precision histograms
                }
                
                evals = [(dtrain, 'train'), (dvalid, 'valid')]
                
                try:
                    # Try GPU training
                    booster = xgb.train(
                        params,
                        dtrain,
                        num_boost_round=self.config["n_estimators"],
                        evals=evals,
                        early_stopping_rounds=self.config["early_stopping_rounds"],
                        verbose_eval=False
                    )
                    
                    # Wrap booster in XGBRegressor for consistent interface
                    model.set_params(n_estimators=booster.best_iteration + 1)
                    model._Booster = booster
                    
                    # CRITICAL: Explicitly free GPU memory
                    del dtrain, dvalid, booster, evals
                    gc.collect()
                    logger.info("[XGBoost] GPU memory cleanup completed")
                    
                except Exception as gpu_error:
                    # Check if it's a CUDA OOM error
                    error_msg = str(gpu_error).lower()
                    if "cudaerror" in error_msg or "out of memory" in error_msg or "memory allocation" in error_msg:
                        logger.warning(f"[XGBoost] GPU OOM detected, falling back to CPU")
                        
                        # Free GPU objects
                        try:
                            del dtrain, dvalid, evals
                        except:
                            pass
                        
                        # Hide GPU to prevent CUDA from being touched again
                        os.environ["CUDA_VISIBLE_DEVICES"] = ""
                        logger.info("[XGBoost] Hidden GPU (CUDA_VISIBLE_DEVICES='') for CPU fallback")
                        
                        gc.collect()
                        gc.collect()
                        
                        # Retry on CPU - MUST update model to CPU mode and rebuild DMatrix
                        logger.info("[XGBoost] Retrying with CPU (hist method)")
                        
                        # Set CPU parameters (critical - model still has gpu_hist!)
                        model.set_params(
                            tree_method='hist',
                            predictor='cpu_predictor',
                            device='cpu'
                        )
                        
                        # Use standard fit() which will create CPU DMatrix internally
                        model.fit(
                            X_tr, y_tr,
                            eval_set=[(X_va, y_va)],
                            verbose=False
                        )
                        logger.info("[XGBoost] CPU training completed successfully")
                    else:
                        # Not an OOM error, re-raise
                        raise
                
            else:
                # CPU training - use standard fit
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    verbose=False
                )
        
        except Exception as e:
            # Cleanup on error
            logger.error(f"[XGBoost] Training failed: {e}")
            gc.collect()
            raise
        
        # 6) Store state and sanity check (before cleanup)
        self.model = model
        self.is_trained = True
        
        # Run sanity check on small sample to avoid holding full data
        sample_size = min(1000, X_tr.shape[0])
        sample_idx = np.random.choice(X_tr.shape[0], size=sample_size, replace=False)
        X_sample = X_tr[sample_idx]
        self.post_fit_sanity(X_sample, "XGBoost")
        
        # 7) Final cleanup of training data
        del X_tr, X_va, y_tr, y_va, X_sample
        gc.collect()
        
        return self.model

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        Xp, _ = self.preprocess_data(X, None)
        preds = self.model.predict(Xp)
        return np.nan_to_num(preds, nan=0.0).astype(np.float32)

    def _check_gpu_available(self) -> bool:
        """Check if GPU is available for XGBoost"""
        import os
        cvd = os.getenv("CUDA_VISIBLE_DEVICES", "unset")
        logger.info(f"[XGBoost] Checking GPU availability (CVD={cvd})")
        
        try:
            import xgboost as xgb
            # Log XGBoost build info
            build_info = xgb.build_info()
            logger.info(f"[XGBoost] Build info: USE_CUDA={build_info.get('USE_CUDA', 'unknown')}")
            
            # Quick test with tiny data
            test_data = xgb.DMatrix([[1, 2, 3]], label=[1])
            xgb.train({"tree_method": "gpu_hist", "max_depth": 1}, test_data, num_boost_round=1)
            logger.info("[XGBoost] ✅ GPU available and working!")
            return True
        except Exception as e:
            logger.warning(f"[XGBoost] ❌ GPU not available: {e}")
            logger.info(f"[XGBoost] Falling back to CPU (tree_method='hist')")
            return False
    
    def _build_model(self, cpu_only: bool = False):
        """Build XGBoost model with safe defaults"""
        tree_method = "hist" if cpu_only else "gpu_hist"
        
        # Memory-efficient settings for GPU
        extra_params = {}
        if not cpu_only:
            extra_params.update({
                'max_bin': 256,  # Reduce memory usage
                'single_precision_histogram': True,  # Half precision for histograms
            })
        
        model = xgb.XGBRegressor(
            tree_method=tree_method,
            max_depth=self.config["max_depth"],
            min_child_weight=self.config["min_child_weight"],
            subsample=self.config["subsample"],
            colsample_bytree=self.config["colsample_bytree"],
            gamma=self.config.get("gamma", 0.3),
            reg_alpha=self.config["reg_alpha"],
            reg_lambda=self.config["reg_lambda"],
            eta=self.config["eta"],
            n_estimators=self.config["n_estimators"],
            n_jobs=self.num_threads,
            random_state=42,
            **extra_params,
            **self.config.get("xgb_params", {})
        )
        return model