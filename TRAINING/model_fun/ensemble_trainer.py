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

"""
Optimized Ensemble trainer with lightweight blending (NO VotingRegressor refit).
- HistGradientBoosting: OMP=8-12 (OpenMP parallelism)
- RandomForest: process-parallel (n_jobs) with OMP=1
- Ridge: cheap baseline
- Blend: weighted average learned from validation set (instant, no refit)

Environment variables:
- RF_PARALLEL: "openmp" (default) or "trees" to control RF parallelization mode
"""

import logging
import numpy as np
import time
import os
from typing import Dict, Any, Optional, List
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, LinearRegression

from .base_trainer import BaseModelTrainer
from TRAINING.common.threads import thread_guard, predict_guard, default_threads, cpu_affinity_guard, reset_affinity

logger = logging.getLogger(__name__)

# Diagnostic helpers
try:
    from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
except Exception:
    def _openmp_effective_n_threads():
        return os.getenv("OMP_NUM_THREADS", "n/a")

def _cpu_affinity():
    """Get CPU affinity (how many CPUs this process can use)."""
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return None

def _log_threadpools(where: str):
    """Log current threadpool state for diagnostics."""
    try:
        from threadpoolctl import threadpool_info
        pools = "; ".join(f"{p.get('user_api')}={p.get('num_threads')}" for p in threadpool_info())
    except Exception:
        pools = "n/a"
    
    logger.info(
        "%s | OMP=%s MKL=%s | pools=[%s] | openmp_effective=%s | affinity=%s",
        where,
        os.getenv("OMP_NUM_THREADS"),
        os.getenv("MKL_NUM_THREADS"),
        pools or "n/a",
        _openmp_effective_n_threads(),
        _cpu_affinity()
    )

class EnsembleTrainer(BaseModelTrainer):
    """
    Ensemble trainer with optimized per-model threading.
    Combines HGB (OpenMP), RF (joblib), and Ridge for robust predictions.
    """

    def __init__(self, config: Dict[str, Any] = None):
        # Load from centralized CONFIG if not provided
        if config is None:
            try:
                import sys
                from pathlib import Path
                _CONFIG_DIR = Path(__file__).resolve().parents[2] / "CONFIG"
                if str(_CONFIG_DIR) not in sys.path:
                    sys.path.insert(0, str(_CONFIG_DIR))
                from config_loader import load_model_config
                config = load_model_config("ensemble")
                logger.info("✅ Loaded Ensemble config from CONFIG/model_config/ensemble.yaml")
            except Exception as e:
                logger.warning(f"Failed to load centralized config: {e}, using hardcoded defaults")
                config = {}
        
        super().__init__(config or {})
        
        # Initialize num_threads from config or environment
        import os
        self.num_threads = int(self.config.get("num_threads", os.getenv("OMP_NUM_THREADS", "12")))
        self.config.setdefault("threads", self.num_threads)
        
        # DEPRECATED: Hardcoded defaults (kept for backward compatibility)
        # These values are now defined in CONFIG/model_config/ensemble.yaml
        # HGB config (OpenMP-heavy)
        self.config.setdefault("hgb_max_iter", 300)
        self.config.setdefault("hgb_max_depth", 8)
        self.config.setdefault("hgb_learning_rate", 0.05)
        self.config.setdefault("hgb_max_bins", 255)
        self.config.setdefault("hgb_l2", 1e-4)
        self.config.setdefault("hgb_early_stop", True)
        
        # RF config (joblib-heavy) - Spec 3: Stacking Regressor
        self.config.setdefault("rf_n_estimators", 300)
        self.config.setdefault("rf_max_depth", 15)  # Spec 3: 15 (was 18)
        self.config.setdefault("rf_max_samples", 0.7)
        self.config.setdefault("rf_max_features", "sqrt")  # Spec 3: sqrt
        
        # Ridge config (cheap) - Spec 3: Final estimator
        self.config.setdefault("ridge_alpha", 1.0)  # Spec 3: 1.0-10.0, tune with CV
        
        # Stacking config - Spec 3: Use StackingRegressor with CV
        self.config.setdefault("use_stacking", True)  # Use StackingRegressor (Spec 3) vs weighted blend
        self.config.setdefault("stacking_cv", 5)  # K=5 or K=10 for cross-validation
        self.config.setdefault("final_estimator_alpha", 1.0)  # Ridge alpha for final estimator
        
        # Thread allocation - use full budget for OMP-heavy models
        T = default_threads()
        self.hgb_omp = T                             # Use all threads for HGB (OpenMP parallelism)
        self.rf_jobs = T                             # Use all threads for RF (joblib workers)
        self.rf_omp = 1                              # No OpenMP in RF (avoid oversubscription)

    def fit(self, X_tr: np.ndarray, y_tr: np.ndarray, seed: Optional[int] = None, **kwargs) -> Any:
        """Fit method for sklearn-style compatibility."""
        return self.train(X_tr, y_tr, seed=seed, **kwargs)

    def train(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_va=None,
        y_va=None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """
        Train Ensemble model with optimized per-model threading.
        
        Threading strategy:
        - HGB: OMP=T (all threads), MKL=1 (OpenMP parallelism for histogram building)
        - RF: n_jobs=T (all threads), OMP=1 (joblib threads, avoid OpenMP fight)
        - Ridge: default (very cheap, single-threaded)
        """
        t_start = time.perf_counter()
        
        # 0) System diagnostics
        logger.info("🔍 [Ensemble] System info: CPU count=%s | affinity cpus=%s", 
                   os.cpu_count(), _cpu_affinity())
        logger.info("🔍 [Ensemble] Initial env: OMP=%s MKL=%s", 
                   os.getenv("OMP_NUM_THREADS"), os.getenv("MKL_NUM_THREADS"))
        try:
            logger.info("🔍 [Ensemble] CPU affinity at start: %s", os.sched_getaffinity(0))
        except AttributeError:
            logger.info("🔍 [Ensemble] CPU affinity: not available on this platform")
        _log_threadpools("[Ensemble] Initial state")
        
        # 1) Preprocess
        self.validate_data(X_tr, y_tr)
        X, y = self.preprocess_data(X_tr, y_tr)
        n, d = X.shape
        logger.info(f"🚀 [Ensemble] Starting training on {n} samples, {d} features")
        
        # Get RF parallelization mode from environment
        rf_mode = os.getenv("RF_PARALLEL", "openmp").lower()
        logger.info(f"🔧 Thread plan: HGB OMP={self.hgb_omp}, RF mode={rf_mode}")
        
        # 2) Train/val split (only if no external val provided)
        # CRITICAL: Add purge gap to prevent temporal leakage
        if X_va is None or y_va is None:
            s = int(seed) if seed is not None else 42
            
            # Calculate split point with purge gap
            # Use conservative default: 60m target = 17 bars (60/5 + 5 buffer)
            # If target info available in kwargs, could extract horizon
            purge_overlap = 17  # Default: 60m target with 5m bars
            
            # Split chronologically (no shuffle) with purge gap
            split_idx = int(len(X) * 0.8)  # 80% train, 20% test
            purge_end = split_idx - purge_overlap  # Remove purge_overlap from train end
            
            if purge_end > 0:
                X_train, y_train = X[:purge_end], y[:purge_end]
                X_val, y_val = X[split_idx:], y[split_idx:]
            else:
                # If purge would make train set too small, use smaller purge
                purge_end = max(1, split_idx - 5)  # Minimum 5 bar purge
                X_train, y_train = X[:purge_end], y[:purge_end]
                X_val, y_val = X[split_idx:], y[split_idx:]
                logger.warning(f"⚠️  Purge gap reduced due to small dataset (purge={split_idx - purge_end} bars)")
            
            logger.info(f"📊 Split: train={len(X_train)}, val={len(X_val)}, purge_gap={split_idx - purge_end} bars")
        else:
            X_train, y_train = X, y
            X_val, y_val = X_va, y_va
        
        s = int(seed) if seed is not None else 42
        
        # 3) Train HGB with OpenMP threads (memory-bandwidth bound)
        logger.info(f"🔧 [HGB] Training with OMP={self.hgb_omp}...")
        t0 = time.perf_counter()
        _log_threadpools("[HGB] before guard")
        with cpu_affinity_guard():  # Ensure full core access
            with thread_guard(omp=self.hgb_omp, mkl=1):
                _log_threadpools(f"[HGB] guarded (omp={self.hgb_omp})")
                hgb = HistGradientBoostingRegressor(
                    max_iter=self.config["hgb_max_iter"],
                    max_depth=self.config["hgb_max_depth"],
                    learning_rate=self.config["hgb_learning_rate"],
                    max_bins=self.config["hgb_max_bins"],
                    l2_regularization=self.config["hgb_l2"],
                    early_stopping=self.config["hgb_early_stop"],
                    validation_fraction=0.1 if self.config["hgb_early_stop"] else None,
                    random_state=s,
                    verbose=0,
                )
                hgb.fit(X_train, y_train)
        logger.info(f"✅ [HGB] Trained in {time.perf_counter()-t0:.2f}s (iterations={hgb.n_iter_})")
        
        # 4) Train RF with switchable parallelization mode
        t0 = time.perf_counter()
        rf = RandomForestRegressor(
            n_estimators=self.config["rf_n_estimators"],
            max_depth=self.config["rf_max_depth"],
            max_features=self.config["rf_max_features"],
            max_samples=self.config["rf_max_samples"],
            bootstrap=True,
            random_state=s,
            verbose=0,
        )
        
        # CRITICAL: Use max_samples to dramatically speed up RF on large datasets
        # This caps per-tree cost without hurting quality
        max_samples_per_tree = min(200_000, len(X_train))
        rf.set_params(max_samples=max_samples_per_tree)
        
        if rf_mode == "openmp":
            # Parallelize WITHIN each tree via OpenMP (often better on big data)
            rf_omp = min(max(4, self.num_threads), 24)  # 4-24 threads for OpenMP
            rf.set_params(n_jobs=1)  # Single tree at a time, but parallelized internally
            logger.info(f"🔧 [RF] OpenMP mode: n_jobs=1, OMP={rf_omp}, max_samples={max_samples_per_tree}")
            _log_threadpools("[RF/openmp] before guard")
            with cpu_affinity_guard():  # Ensure full core access
                with thread_guard(omp=rf_omp, mkl=1):
                    _log_threadpools(f"[RF/openmp] guarded (omp={rf_omp})")
                    rf.fit(X_train, y_train)
        else:
            # Parallelize ACROSS trees via joblib (FAST mode with process parallelism)
            rf_jobs = min(max(4, self.num_threads), os.cpu_count() or 16)
            rf.set_params(n_jobs=rf_jobs)  # Multiple trees in parallel
            logger.info(f"🔧 [RF] Trees mode: n_jobs={rf_jobs}, OMP=1, max_samples={max_samples_per_tree}")
            _log_threadpools("[RF/trees] before guard")
            
            # Use process parallelism with inner OMP=1 for maximum speed
            from sklearn.utils import parallel_backend
            with cpu_affinity_guard():  # Ensure full core access
                with thread_guard(omp=1, mkl=1):  # OMP=1 to avoid oversubscription
                    with parallel_backend("loky", inner_max_num_threads=1):
                        _log_threadpools(f"[RF/trees] guarded (jobs={rf_jobs}, omp=1)")
                        rf.fit(X_train, y_train)
        
        logger.info(f"✅ [RF] Trained in {time.perf_counter()-t0:.2f}s")
        
        # 5) Train Ridge (use BLAS threads for speedup)
        logger.info(f"🔧 [Ridge] Training...")
        t0 = time.perf_counter()
        from common.threads import blas_threads
        ridge = Ridge(alpha=self.config["ridge_alpha"], random_state=s)
        with blas_threads(min(8, self.num_threads)):
            ridge.fit(X_train, y_train)
        logger.info(f"✅ [Ridge] Trained in {time.perf_counter()-t0:.2f}s")
        
        # 6) Build ensemble using StackingRegressor (Spec 3) or weighted blend
        use_stacking = self.config.get("use_stacking", True)
        
        if use_stacking:
            # Spec 3: Use StackingRegressor with CV
            logger.info(f"🔧 [Ensemble] Using StackingRegressor with CV={self.config['stacking_cv']} (Spec 3)")
            t0 = time.perf_counter()
            
            # Base estimators: diverse models (HGB, RF, LinearRegression)
            # Note: We can't reuse already-fitted models, so we'll create new ones
            # But we can use the same hyperparameters
            base_estimators = [
                ("hgb", HistGradientBoostingRegressor(
                    max_iter=self.config["hgb_max_iter"],
                    max_depth=self.config["hgb_max_depth"],
                    learning_rate=self.config["hgb_learning_rate"],
                    max_bins=self.config["hgb_max_bins"],
                    l2_regularization=self.config["hgb_l2"],
                    early_stopping=self.config["hgb_early_stop"],
                    validation_fraction=0.1 if self.config["hgb_early_stop"] else None,
                    random_state=s,
                    verbose=0,
                )),
                ("rf", RandomForestRegressor(
                    n_estimators=self.config["rf_n_estimators"],
                    max_depth=self.config["rf_max_depth"],
                    max_features=self.config["rf_max_features"],
                    max_samples=self.config["rf_max_samples"],
                    bootstrap=True,
                    random_state=s,
                    verbose=0,
                )),
                ("lr", LinearRegression()),  # Spec 3: LinearRegression as base
            ]
            
            # Final estimator: Ridge with L2 regularization (Spec 3)
            final_estimator = Ridge(
                alpha=self.config.get("final_estimator_alpha", self.config["ridge_alpha"]),
                random_state=s
            )
            
            # Create StackingRegressor with CV
            stacking_cv = self.config.get("stacking_cv", 5)
            stacker = StackingRegressor(
                estimators=base_estimators,
                final_estimator=final_estimator,
                cv=stacking_cv,  # K-fold CV to prevent data leakage
                n_jobs=1,  # StackingRegressor handles parallelism internally
                verbose=0
            )
            
            # Train StackingRegressor (it will fit base models and final estimator with CV)
            logger.info(f"🔧 [Stacking] Training with {stacking_cv}-fold CV...")
            with cpu_affinity_guard():
                with thread_guard(omp=min(self.num_threads, 8), mkl=1):
                    stacker.fit(X_train, y_train)
            
            logger.info(f"✅ [Stacking] Trained in {time.perf_counter()-t0:.2f}s")
            
            # Store the stacker
            self.model = stacker
            self._members = None  # Not used in stacking mode
            self._weights = None
            self._member_names = ["StackingRegressor"]
            
            # Validation score
            try:
                val_preds = stacker.predict(X_val)
                from sklearn.metrics import mean_squared_error, r2_score
                mse = mean_squared_error(y_val, val_preds)
                r2 = r2_score(y_val, val_preds)
                logger.info(f"📊 [Stacking] Validation: MSE={mse:.6f}, R²={r2:.4f}")
            except Exception as e:
                logger.warning(f"Stacking validation score failed: {e}")
        
        else:
            # Legacy: Lightweight blending (weighted average)
            logger.info(f"🤝 [Ensemble] Using weighted blend (legacy mode)...")
            t0 = time.perf_counter()
            
            # Get validation predictions (fast, multi-threaded)
            with predict_guard(omp=1, mkl=1):
                rf.set_params(n_jobs=self.rf_jobs)  # parallel across trees
                y_rf_val = rf.predict(X_val)
            
            with predict_guard(omp=self.hgb_omp, mkl=1):
                y_hgb_val = hgb.predict(X_val)
            
            y_rg_val = ridge.predict(X_val)  # cheap, no guard needed
            
            # Stack predictions and learn optimal weights via Ridge
            P = np.column_stack([y_hgb_val, y_rf_val, y_rg_val])
            meta = Ridge(alpha=1.0, fit_intercept=False, positive=True)
            with blas_threads(1):  # Meta-learning is tiny, 1 thread is fine
                meta.fit(P, y_val)
            weights = meta.coef_
            weights = weights / (weights.sum() + 1e-12)  # normalize
            
            logger.info(f"🤝 Blend weights: HGB={weights[0]:.3f} | RF={weights[1]:.3f} | Ridge={weights[2]:.3f} (learned in {time.perf_counter()-t0:.2f}s)")
            
            # Store members and weights
            self._members = [hgb, rf, ridge]
            self._weights = weights
            self._member_names = ["HGB", "RF", "Ridge"]
            self.model = self  # for compatibility
        
        self.is_trained = True
        
        logger.info(f"✅ [Ensemble] Total training time: {time.perf_counter()-t_start:.2f}s")
        if use_stacking:
            logger.info(f"📋 [Stacking] Base models: HGB, RF, LinearRegression | Final: Ridge(alpha={self.config.get('final_estimator_alpha', self.config['ridge_alpha'])})")
        else:
            logger.info(f"📋 Base models: HGB (iter={hgb.n_iter_}), RF (n={self.config['rf_n_estimators']}), Ridge")
        
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using StackingRegressor (Spec 3) or weighted blend (legacy).
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        # Use base class preprocessing for inference
        Xc, _ = self.preprocess_data(X, None)
        
        use_stacking = self.config.get("use_stacking", True)
        
        if use_stacking and hasattr(self, 'model') and self.model is not None and not isinstance(self.model, type(self)):
            # StackingRegressor mode
            preds = self.model.predict(Xc)
        else:
            # Legacy weighted blend mode
            if self._members is None or self._weights is None:
                raise ValueError("Ensemble not properly trained in blend mode")
            
            hgb, rf, ridge = self._members
            
            # RF: process-parallel across trees (n_jobs), OMP=1
            with predict_guard(omp=1, mkl=1):
                rf.set_params(n_jobs=self.rf_jobs)
                y_rf = rf.predict(Xc)
            
            # HGB: OpenMP-parallel (histogram evaluation)
            with predict_guard(omp=self.hgb_omp, mkl=1):
                y_hgb = hgb.predict(Xc)
            
            # Ridge: cheap, no guard needed
            y_rg = ridge.predict(Xc)
            
            # Weighted blend (instant)
            w = self._weights
            preds = w[0]*y_hgb + w[1]*y_rf + w[2]*y_rg
        
        return np.nan_to_num(preds, nan=0.0).astype(np.float32)
