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
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from .base_trainer import BaseModelTrainer
from TRAINING.common.threads import thread_guard
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

class _Clipper(BaseEstimator, TransformerMixin):
    """Clips features to [-clip, clip] after scaling."""
    def __init__(self, clip=10.0):
        self.clip = float(clip)
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(X, -self.clip, self.clip, out=X)
        return X

class NGBoostTrainer(BaseModelTrainer):
    def __init__(self, config: Dict[str, Any] = None):
        # Load centralized config if available and no config provided
        if config is None and _USE_CENTRALIZED_CONFIG:
            try:
                config = load_model_config("ngboost")
                logger.info("✅ [NGBoost] Loaded centralized config from CONFIG/model_config/ngboost.yaml")
            except Exception as e:
                logger.warning(f"Failed to load centralized config: {e}. Using hardcoded defaults.")
                config = {}
        
        super().__init__(config or {})
        
        # DEPRECATED: Hardcoded defaults kept for backward compatibility
        # To change these, edit CONFIG/model_config/ngboost.yaml
        self.config.setdefault("n_estimators", 700)
        self.config.setdefault("learning_rate", 0.03)
        self.config.setdefault("clip", 10.0)
        self.config.setdefault("early_stopping_rounds", 50)

    def train(self, X_tr, y_tr, feature_names: List[str] = None, **kw) -> Any:
        from ngboost import NGBRegressor
        from ngboost.distns import Normal
        from ngboost.scores import MLE
        
        self.validate_data(X_tr, y_tr)
        X, y = self.preprocess_data(X_tr, y_tr)
        self.feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        
        # Split for early stopping (prevents blowups)
        s = int(self.config.get("seed", 42))
        X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.15, random_state=s)

        # Use histogram single-tree base learner (much faster)
        from sklearn.ensemble import HistGradientBoostingRegressor
        base_learner = HistGradientBoostingRegressor(
            max_iter=1,  # Single tree per boosting round
            max_depth=6,
            learning_rate=0.1,
            random_state=s
        )
        
        ngb = NGBRegressor(
            Dist=Normal,             # Normal with log-link for scale
            Score=MLE,               # Use class, not instance
            Base=base_learner,        # Use histogram single-tree base
            n_estimators=self.config["n_estimators"],
            learning_rate=self.config["learning_rate"],
            natural_gradient=True,
            col_sample=self.config.get("col_sample", 0.6),
            minibatch_frac=self.config.get("minibatch_frac", 0.2),
            random_state=s,
            verbose=False,
            tol=1e-4
        )

        pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler(quantile_range=(5, 95))),
            ("clip", _Clipper(clip=self.config["clip"])),
            ("ngb", ngb),
        ])

        logger.info("Fitting NGBoost pipeline with early stopping…")
        # NGBoost doesn't benefit from high BLAS threads (sequential algorithm)
        # Use 1 thread - if you need speed, run multiple NGBoost jobs in parallel
        from common.threads import blas_threads
        with blas_threads(1):
            logger.info("[NGBoost] Using 1 BLAS thread (sequential boosting - parallelize externally for speed)")
            pipe.fit(
                X_tr, y_tr,
                ngb__X_val=X_va, ngb__Y_val=y_va,
                ngb__early_stopping_rounds=self.config["early_stopping_rounds"]
            )

        # Quick sanity check on finite preds
        preds = pipe.predict(X_va[:1000])
        if not np.isfinite(preds).all():
            raise RuntimeError("NGBoost produced non-finite predictions; try smaller learning_rate / larger clip.")

        self.model = pipe
        self.is_trained = True
        self.model.imputer = self.imputer
        self.model.colmask = self.colmask
        
        # Post-fit sanity check
        self.post_fit_sanity(X_tr, "NGBoost")
        return self.model

    def predict(self, X):
        if not self.is_trained: raise ValueError("Not trained")
        Xp, _ = self.preprocess_data(X, None)
        return self.model.predict(Xp).astype(np.float32)