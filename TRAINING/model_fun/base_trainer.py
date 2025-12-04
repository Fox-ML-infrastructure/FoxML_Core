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
Base Model Trainer

Abstract base class for all model trainers with improved preprocessing and imputer handling.
"""


from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import joblib
import logging
import os
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from TRAINING.common.safety import guard_features, guard_targets, finite_preds_or_raise
from TRAINING.common.threads import thread_guard, set_estimator_threads, default_threads, guard_for_estimator

logger = logging.getLogger(__name__)

def safe_ridge_fit(X, y, alpha=1.0):
    """
    Safely fit a Ridge model with fallback to lsqr solver.
    
    This avoids scipy.linalg.solve segfaults in MKL/OpenMP conflict scenarios.
    The lsqr solver bypasses the Cholesky/direct solve path that causes crashes.
    
    Args:
        X: Feature matrix
        y: Target vector
        alpha: Regularization strength
    
    Returns:
        Fitted Ridge model
    """
    solver_pref = os.getenv("SKLEARN_RIDGE_SOLVER", "auto")
    try:
        model = Ridge(alpha=alpha, solver=solver_pref, random_state=42)
        return model.fit(X, y)
    except Exception as e:
        # Fall back to lsqr solver (bypasses Cholesky/MKL path)
        logger.warning("⚠️  Ridge(solver='%s') failed: %s. Falling back to 'lsqr' solver.", solver_pref, e)
        model = Ridge(alpha=alpha, solver="lsqr", random_state=42)
        return model.fit(X, y)

class BaseModelTrainer(ABC):
    """Base class for all model trainers with robust preprocessing"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.model = None
        self.is_trained = False
        self.feature_names: List[str] = []
        self.target_name: str = ""
        self.imputer: Optional[SimpleImputer] = None
        self.colmask: Optional[np.ndarray] = None
        self.family_name: str = getattr(self, '__class__', type(self)).__name__.replace("Trainer", "")
        
    def _threads(self) -> int:
        """Get number of threads from config or default"""
        return int(self.config.get("num_threads", default_threads()))

    def fit_with_threads(self, estimator, X, y, sample_weight=None, *, phase: str = "fit", blas_threads_override: int = None):
        """
        Universal fit method with smart threading.
        Automatically detects estimator type and applies correct OMP/MKL settings.
        
        Args:
            estimator: The model to fit (RF, HGB, Ridge, etc.)
            X, y: Training data
            sample_weight: Optional sample weights
            phase: "fit", "meta", "linear_solve" (hints for BLAS-heavy phases)
            blas_threads_override: Override BLAS thread count (None = use default)
        
        Returns:
            Fitted estimator
        """
        from common.threads import blas_threads, compute_blas_threads_for_family
        
        n = self._threads()
        
        # Compute BLAS threads if not overridden
        if blas_threads_override is None:
            blas_threads_override = compute_blas_threads_for_family(self.family_name, n)
        
        # Log for verification
        logger.info(f"[{self.family_name}] fit_with_threads: using {blas_threads_override} BLAS threads (total cores: {n})")
        
        # Use BLAS threading context for BLAS-heavy operations
        with guard_for_estimator(estimator, family=self.family_name, threads=n, phase=phase):
            with blas_threads(blas_threads_override):
                if sample_weight is not None:
                    return estimator.fit(X, y, sample_weight=sample_weight)
                return estimator.fit(X, y)
    
    def predict_with_threads(self, estimator, X, *, phase: str = "predict"):
        """
        Universal predict method with smart threading.
        Ensures predictions use all cores (no single-core bottleneck).
        
        Args:
            estimator: The fitted model
            X: Data to predict on
            phase: "predict" or custom phase name
        
        Returns:
            Predictions
        """
        n = self._threads()
        with guard_for_estimator(estimator, family=self.family_name, threads=n, phase=phase):
            return estimator.predict(X)
        
    @abstractmethod
    def train(self, X_tr: np.ndarray, y_tr: np.ndarray, feature_names: List[str] = None, **kwargs) -> Dict[str, Any]:
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X_tr: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass
    
    def predict_proba(self, X_tr: np.ndarray) -> np.ndarray:
        """Predict probabilities (for classification models)"""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_tr)
        else:
            raise NotImplementedError("predict_proba not supported for this trainer")
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance"""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.atleast_1d(self.model.coef_).ravel()
        else:
            return None
    
    def save_model(self, filepath: str) -> None:
        """Save trained model with preprocessors"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        model_data = {
            'model': self.model,
            'config': self.config,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'imputer': self.imputer,
            'colmask': self.colmask,
            'trainer_class': self.__class__.__name__
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load trained model with preprocessors"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.config = model_data['config']
        self.feature_names = model_data.get('feature_names', [])
        self.target_name = model_data.get('target_name', '')
        self.imputer = model_data.get('imputer', None)
        self.colmask = model_data.get('colmask', None)
        self.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'trainer_class': self.__class__.__name__,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'config': self.config
        }
    
    def validate_data(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Validate input data"""
        if X is None or y is None:
            raise ValueError("X/y missing")
        if X.shape[0] != len(y):
            raise ValueError("X and y length mismatch")
        if X.shape[0] == 0:
            raise ValueError("Empty dataset")
        return True
    
    def preprocess_data(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Preprocess data with robust imputer and column handling"""
        # Cast to float32 for speed (2x memory reduction + faster BLAS)
        # Use C-contiguous for optimal cache performance
        X = np.ascontiguousarray(X, dtype=np.float32)
        
        if y is not None:
            # Training mode: fit imputer and colmask
            y = np.asarray(y, dtype=np.float64).ravel()
            mask = np.isfinite(y)
            if not mask.any():
                raise ValueError("No finite targets after filtering")
            X, y = X[mask], y[mask]
            
            # Drop all-NaN columns on TRAIN only
            self.colmask = np.isfinite(X).any(axis=0)
            if not self.colmask.any():
                raise ValueError("All columns are NaN")
            X = X[:, self.colmask]
            
            # Fit imputer on training data
            self.imputer = SimpleImputer(strategy="median")
            X = self.imputer.fit_transform(X)
            
            # Apply global safety guards
            X = guard_features(X)
            y = guard_targets(y)
            
            logger.info("Preprocessed train: %d rows, %d cols", X.shape[0], X.shape[1])
            return X, y
        
        # Inference mode: reuse colmask + imputer
        if self.colmask is not None:
            if X.shape[1] >= self.colmask.size:
                X = X[:, self.colmask]
            else:
                logger.warning("Incoming feature count < trained colmask; skipping column mask")
        if self.imputer is not None:
            X = self.imputer.transform(X)
        
        # Apply safety guards for inference too
        X = guard_features(X)
        return X, None
    
    def post_fit_sanity(self, X: np.ndarray, name: str):
        """Post-fit sanity check for finite predictions"""
        try:
            preds = self.predict(X[:min(1024, len(X))])
            finite_preds_or_raise(name, preds)
        except Exception as e:
            raise RuntimeError(f"{name} post-fit sanity failed: {e}")