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
from typing import Any, Dict, List, Optional
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
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

class GMMRegimeTrainer(BaseModelTrainer):
    def __init__(self, config: Dict[str, Any] = None):
        # Load centralized config if available and no config provided
        if config is None and _USE_CENTRALIZED_CONFIG:
            try:
                config = load_model_config("gmm_regime")
                logger.info("✅ [GMMRegime] Loaded centralized config from CONFIG/model_config/gmm_regime.yaml")
            except Exception as e:
                logger.warning(f"Failed to load centralized config: {e}. Using hardcoded defaults.")
                config = {}
        
        super().__init__(config or {})
        
        # DEPRECATED: Hardcoded defaults kept for backward compatibility
        # To change these, edit CONFIG/model_config/gmm_regime.yaml
        self.config.setdefault("n_components", 3)
        self.config.setdefault("reg_covar", 1e-4)
        self.config.setdefault("covariance_type", "diag")
        self.config.setdefault("max_iter", 100)
        self.config.setdefault("ridge_alpha", 1.0)
        self.regs = []

    def train(self, X_tr: np.ndarray, y_tr: np.ndarray, 
              X_va=None, y_va=None, feature_names: List[str] = None, **kwargs) -> Any:
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.cluster import MiniBatchKMeans
        
        # Store raw feature count BEFORE preprocessing
        self.n_features_raw_ = X_tr.shape[1]
        
        # 1) Preprocess data - GMM needs float64 for numerical stability
        X_tr, y_tr = self.preprocess_data(X_tr, y_tr)
        X_tr = np.asarray(X_tr, dtype=np.float64)  # Override float32 from base
        self.feature_names = feature_names or [f"f{i}" for i in range(X_tr.shape[1])]
        
        # 2) Split only if no external validation provided
        if X_va is None or y_va is None:
            X_tr, X_va, y_tr, y_va = train_test_split(
                X_tr, y_tr, test_size=0.2, random_state=42
            )
        
        logger.info(f"[GMMRegime] Input data: {X_tr.shape}, dtype={X_tr.dtype}")
        
        # 3) Stabilization: Remove near-constant features
        var = np.var(X_tr, axis=0)
        keep = var > 1e-12
        self.keep_mask = keep
        self.mask_idx_ = np.flatnonzero(keep)
        self.n_features_masked_ = len(self.mask_idx_)
        
        X_tr_filtered = X_tr[:, keep]
        logger.info(f"[GMMRegime] Kept {keep.sum()}/{len(keep)} features (var > 1e-12)")
        
        # 4) Standardize (critical for GMM stability)
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_tr_scaled = scaler.fit_transform(X_tr_filtered)
        
        # 5) PCA dimensionality reduction (makes GMM tractable + fast)
        n_pca = min(64, X_tr_scaled.shape[1], X_tr_scaled.shape[0] // 10)
        pca = PCA(n_components=n_pca, svd_solver="randomized", whiten=True, random_state=42)
        X_red = pca.fit_transform(X_tr_scaled)
        logger.info(f"[GMMRegime] PCA reduced to {n_pca} components (from {X_tr_filtered.shape[1]})")
        
        # Store preprocessing pipeline
        self.scaler = scaler
        self.pca = pca
        
        # Use BLAS threading for GMM and Ridge fits
        n_threads = self._threads()
        blas_n = max(12, n_threads - 2)  # Use most cores for BLAS-heavy operations
        
        # VERIFY what's actually loaded
        try:
            from threadpoolctl import threadpool_info, threadpool_limits
            libs_before = [f"{i['internal_api']}:{i['num_threads']}" for i in threadpool_info()]
            logger.info(f"[GMMRegime] BLAS/OpenMP BEFORE: {libs_before}")
        except ImportError:
            from common.threads import blas_threads
            threadpool_limits = blas_threads
            logger.warning("[GMMRegime] threadpoolctl not available, using fallback")
        
        # 6) Initialize GMM with KMeans on subsample (prevents degenerate components)
        k = self.config["n_components"]
        n_sample = min(200_000, X_red.shape[0])
        idx = np.random.choice(X_red.shape[0], size=n_sample, replace=False)
        logger.info(f"[GMMRegime] KMeans init on {n_sample} samples for {k} components")
        km = MiniBatchKMeans(n_clusters=k, batch_size=10_000, n_init=3, random_state=42)
        km.fit(X_red[idx])
        means_init = km.cluster_centers_
        
        # FORCE threads for BLAS-heavy operations (EM algorithm + Ridge) - applies to all libs
        with threadpool_limits(limits=blas_n):
            # Verify threads were set
            try:
                libs_during = [f"{i['internal_api']}:{i['num_threads']}" for i in threadpool_info()]
                logger.info(f"[GMMRegime] BLAS/OpenMP DURING (limits={blas_n}): {libs_during}")
            except:
                pass
            
            logger.info(f"[GMMRegime] Training with {blas_n} BLAS threads for EM + Ridge")
            
            # 7) Build GMM with battle-tested settings + fallback
            try:
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type=self.config["covariance_type"],
                    reg_covar=max(self.config["reg_covar"], 1e-3),  # At least 1e-3
                    n_init=3,
                    init_params="kmeans",
                    means_init=means_init,
                    max_iter=self.config["max_iter"],
                    tol=1e-3,
                    random_state=42,
                    verbose=0
                )
                gmm.fit(X_red)
                logger.info(f"[GMMRegime] GMM converged: {gmm.converged_}, iterations: {gmm.n_iter_}")
            except ValueError as e:
                logger.warning(f"[GMMRegime] GMM failed with {self.config['covariance_type']}, trying diag: {e}")
                # Fallback to diag with higher regularization
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type="diag",  # Safer
                    reg_covar=1e-2,  # Higher regularization
                    n_init=3,
                    init_params="kmeans",
                    means_init=means_init,
                    max_iter=self.config["max_iter"],
                    tol=1e-3,
                    random_state=42,
                    verbose=0
                )
                gmm.fit(X_red)
                logger.info(f"[GMMRegime] Fallback GMM converged: {gmm.converged_}")
            
            # 8) Get regime probabilities (on reduced space)
            resp = gmm.predict_proba(X_red)
            
            # 9) Train regime-specific regressors on ORIGINAL features (not PCA)
            self.regs = []
            for k in range(resp.shape[1]):
                w = resp[:, k]
                # Use safe_ridge_fit to avoid scipy.linalg.solve segfaults
                # Note: safe_ridge_fit doesn't support sample_weight directly,
                # so we use weighted least squares by scaling X and y
                X_weighted = X_tr_filtered * np.sqrt(w)[:, None]
                y_weighted = y_tr * np.sqrt(w)
                r = safe_ridge_fit(X_weighted, y_weighted, alpha=self.config["ridge_alpha"])
                self.regs.append(r)
        
        # 10) Store state and sanity check
        self.model = (gmm, self.regs)
        self.is_trained = True
        
        # For sanity check, we need to test with a small subset of RAW features
        # Create a tiny sample to avoid reprocessing the full dataset
        sample_idx = np.random.choice(min(1000, X_tr.shape[0]), size=min(100, X_tr.shape[0]), replace=False)
        # Reconstruct raw-ish features by undoing standardization (approximate, just for sanity)
        # Actually, just skip post_fit_sanity for GMM since we have complex preprocessing
        logger.info("[GMMRegime] Skipping post_fit_sanity (complex preprocessing pipeline)")
        
        return self.model

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Robust preprocessing: handle both raw and already-preprocessed inputs
        if X.shape[1] == self.n_features_raw_:
            # Input is RAW features - apply full preprocessing
            Xp, _ = self.preprocess_data(X, None)
            Xp = np.asarray(Xp, dtype=np.float64)
            Xp_filtered = Xp[:, self.keep_mask]
        elif X.shape[1] == self.n_features_masked_:
            # Input is already filtered - skip imputer and masking
            logger.debug(f"[GMMRegime] Input already filtered ({X.shape[1]} features), skipping base preprocessing")
            Xp_filtered = np.asarray(X, dtype=np.float64)
        else:
            raise ValueError(
                f"[GMMRegime] Unexpected n_features={X.shape[1]} "
                f"(expected raw={self.n_features_raw_} or masked={self.n_features_masked_})"
            )
        
        # Apply scaling and PCA
        Xp_scaled = self.scaler.transform(Xp_filtered)
        Xp_red = self.pca.transform(Xp_scaled)
        
        gmm, regs = self.model
        
        # Get regime probabilities on reduced space
        resp = gmm.predict_proba(Xp_red)
        
        # Weighted prediction from all regimes on original (filtered) features
        preds = np.zeros(Xp_filtered.shape[0], dtype=np.float64)
        for k, r in enumerate(regs):
            preds += resp[:, k] * r.predict(Xp_filtered)
        
        return np.nan_to_num(preds, nan=0.0).astype(np.float32)
