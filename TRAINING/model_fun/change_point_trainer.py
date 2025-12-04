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

import numpy as np, logging, sys
from typing import Any, Dict, List
from pathlib import Path
from sklearn.cluster import KMeans
from .base_trainer import BaseModelTrainer, safe_ridge_fit
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

class ChangePointTrainer(BaseModelTrainer):
    def __init__(self, config: Dict[str, Any] = None):
        # Load centralized config if available and no config provided
        if config is None and _USE_CENTRALIZED_CONFIG:
            try:
                config = load_model_config("change_point")
                logger.info("✅ [ChangePoint] Loaded centralized config from CONFIG/model_config/change_point.yaml")
            except Exception as e:
                logger.warning(f"Failed to load centralized config: {e}. Using hardcoded defaults.")
                config = {}
        
        super().__init__(config or {})
        
        # DEPRECATED: Hardcoded defaults kept for backward compatibility
        # To change these, edit CONFIG/model_config/change_point.yaml
        self.config.setdefault("n_regimes", 2)

    def train(self, X_tr, y_tr, feature_names: List[str] = None, **kw) -> Any:
        from common.threads import blas_threads, compute_blas_threads_for_family
        
        self.validate_data(X_tr, y_tr)
        X, y = self.preprocess_data(X_tr, y_tr)
        self.feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        
        # Use BLAS threading for KMeans and Ridge fits
        n_threads = self._threads()
        blas_n = max(12, n_threads - 2)  # Use most cores for BLAS-heavy operations
        
        # VERIFY what's actually loaded
        try:
            from threadpoolctl import threadpool_info, threadpool_limits
            libs_before = [f"{i['internal_api']}:{i['num_threads']}" for i in threadpool_info()]
            logger.info(f"[ChangePoint] BLAS/OpenMP BEFORE: {libs_before}")
        except ImportError:
            from common.threads import blas_threads
            threadpool_limits = blas_threads
            logger.warning("[ChangePoint] threadpoolctl not available, using fallback")
        
        # FORCE threads for BLAS-heavy operations (applies to all: BLAS + OpenMP)
        with threadpool_limits(limits=blas_n):
            # Verify threads were set
            try:
                libs_during = [f"{i['internal_api']}:{i['num_threads']}" for i in threadpool_info()]
                logger.info(f"[ChangePoint] BLAS/OpenMP DURING (limits={blas_n}): {libs_during}")
            except:
                pass
            
            logger.info(f"[ChangePoint] Training with {blas_n} BLAS threads")
            z = KMeans(n_clusters=self.config["n_regimes"], n_init=10, random_state=42).fit_predict(np.c_[X.mean(1), y])
            self.models = []
            for k in range(self.config["n_regimes"]):
                # Use safe_ridge_fit to avoid scipy.linalg.solve segfaults
                m = safe_ridge_fit(X[z==k], y[z==k], alpha=1.0)
                self.models.append(m)
        
        self.model = (self.models, )
        self.is_trained = True
        return self.model

    def predict(self, X):
        if not self.is_trained: raise ValueError("Not trained")
        Xp, _ = self.preprocess_data(X, None)
        # simple averaging of regime models for inference
        preds = np.stack([m.predict(Xp) for m in self.models], axis=1).mean(1)
        return np.nan_to_num(preds, nan=0.0).astype(np.float32)