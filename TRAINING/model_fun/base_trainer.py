# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

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
import sys
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from TRAINING.common.safety import guard_features, guard_targets, finite_preds_or_raise
from TRAINING.common.threads import thread_guard, set_estimator_threads, default_threads, guard_for_estimator

logger = logging.getLogger(__name__)

# Add CONFIG directory to path for centralized config loading
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIR = _REPO_ROOT / "CONFIG"
if str(_CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(_CONFIG_DIR))

# Try to import config loader
_CONFIG_AVAILABLE = False
try:
    from config_loader import get_callbacks_config, get_cfg
    _CONFIG_AVAILABLE = True
except ImportError:
    logger.debug("Config loader not available; using hardcoded callback defaults")

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
    # Get deterministic seed from determinism system
    try:
        from TRAINING.common.determinism import BASE_SEED
        ridge_seed = BASE_SEED if BASE_SEED is not None else 42  # FALLBACK_DEFAULT_OK
    except:
        ridge_seed = 42  # FALLBACK_DEFAULT_OK
    
    solver_pref = os.getenv("SKLEARN_RIDGE_SOLVER", "auto")
    try:
        model = Ridge(alpha=alpha, solver=solver_pref)
        return model.fit(X, y)
    except Exception as e:
        # Fall back to lsqr solver (bypasses Cholesky/MKL path)
        logger.warning("⚠️  Ridge(solver='%s') failed: %s. Falling back to 'lsqr' solver.", solver_pref, e)
        model = Ridge(alpha=alpha, solver="lsqr")
        return model.fit(X, y)

class BaseModelTrainer(ABC):
    """Base class for all model trainers with robust preprocessing"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.model = None
        self.is_trained = False
        self.feature_names: List[str] = []
        self.target: str = ""
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
    
    def _get_clipnorm(self) -> float:
        """Get gradient clipping norm from config, with fallback to default."""
        if _CONFIG_AVAILABLE:
            try:
                clipnorm = get_cfg("safety.gradient_clipping.clipnorm", default=1.0, config_name="safety_config")
                return float(clipnorm)
            except Exception as e:
                logger.debug(f"Failed to load clipnorm from config: {e}")
        return 1.0  # Default fallback
    
    def _get_test_split_params(self) -> tuple[float, int]:
        """Get test_size and seed from config, with fallback to defaults.
        
        Uses determinism system to ensure consistent seeds across the pipeline.
        """
        # Load test_size from config
        test_size = 0.2  # FALLBACK_DEFAULT_OK
        if _CONFIG_AVAILABLE:
            try:
                test_size = float(get_cfg("preprocessing.validation.test_size", default=0.2, config_name="preprocessing_config"))
            except Exception as e:
                logger.debug(f"Failed to load test_size from config: {e}")
        
        # Load seed from determinism system (preferred) or config
        seed = None
        try:
            from TRAINING.common.determinism import BASE_SEED
            if BASE_SEED is not None:
                seed = BASE_SEED
            elif _CONFIG_AVAILABLE:
                try:
                    seed = int(get_cfg("preprocessing.validation.seed", default=42, config_name="preprocessing_config"))
                except Exception:
                    seed = 42  # FALLBACK_DEFAULT_OK
        except Exception as e:
            logger.debug(f"Failed to load seed from determinism system: {e}")
            seed = 42  # FALLBACK_DEFAULT_OK
        
        # Final fallback if still None
        if seed is None:
            seed = 42  # FALLBACK_DEFAULT_OK
        
        return float(test_size), int(seed)
    
    def _get_seed(self) -> int:
        """Get seed from determinism system, with fallback to config or default."""
        try:
            from TRAINING.common.determinism import BASE_SEED
            if BASE_SEED is not None:
                return BASE_SEED
        except Exception:
            pass
        
        # Fallback to config
        if _CONFIG_AVAILABLE:
            try:
                return int(get_cfg("preprocessing.validation.seed", default=42, config_name="preprocessing_config"))
            except Exception:
                pass
        
        return 42  # Final fallback
    
    def _get_learning_rate(self, default: float = 0.001) -> float:
        """Get learning_rate from config, with fallback to default."""
        if _CONFIG_AVAILABLE:
            try:
                # Try to get from model-specific config first
                family_lr = get_cfg(f"models.{self.family_name.lower()}.learning_rate", default=None, config_name="model_config")
                if family_lr is not None:
                    return float(family_lr)
                # Fallback to general optimizer config
                return float(get_cfg("optimizer.learning_rate", default=default, config_name="optimizer_config"))
            except Exception as e:
                logger.debug(f"Failed to load learning_rate from config: {e}")
        return default  # Final fallback
    
    def _get_imputation_strategy(self) -> str:
        """Get imputation strategy from config, with fallback to default."""
        if _CONFIG_AVAILABLE:
            try:
                strategy = get_cfg("preprocessing.imputation.strategy", default="median", config_name="preprocessing_config")
                return str(strategy)
            except Exception as e:
                logger.debug(f"Failed to load imputation strategy from config: {e}")
        return "median"  # Default fallback
    
    def get_callbacks(self, family_name: str = None) -> List[Any]:
        """
        Get training callbacks from centralized config.
        
        Args:
            family_name: Model family name (defaults to self.family_name)
            
        Returns:
            List of Keras callbacks (EarlyStopping, ReduceLROnPlateau, etc.)
        """
        if family_name is None:
            family_name = self.family_name
        
        # Try to load from config
        if _CONFIG_AVAILABLE:
            try:
                callbacks_cfg = get_callbacks_config()
                early_stop_cfg = callbacks_cfg.get('callbacks', {}).get('early_stopping', {})
                lr_reduction_cfg = callbacks_cfg.get('callbacks', {}).get('lr_reduction', {})
                
                # Get family-specific patience or use default
                model_patience = early_stop_cfg.get('model_patience', {})
                patience = model_patience.get(family_name, model_patience.get('default', 10))
                
                # Build callbacks
                callbacks = []
                
                # Early stopping
                if early_stop_cfg.get('enabled', True):
                    try:
                        import tensorflow as tf
                        # Ensure patience is int
                        patience = int(patience)
                        callbacks.append(tf.keras.callbacks.EarlyStopping(
                            monitor=early_stop_cfg.get('monitor', 'val_loss'),
                            patience=patience,
                            restore_best_weights=early_stop_cfg.get('restore_best_weights', True),
                            min_delta=float(early_stop_cfg.get('min_delta', 0.0)),
                            mode=early_stop_cfg.get('mode', 'min')
                        ))
                    except ImportError:
                        logger.warning("TensorFlow not available for callbacks")
                
                # Learning rate reduction
                if lr_reduction_cfg.get('enabled', True):
                    try:
                        import tensorflow as tf
                        # Convert numeric values to proper types (YAML may load 1e-6 as string)
                        min_lr_val = lr_reduction_cfg.get('min_lr', 1e-6)
                        if isinstance(min_lr_val, str):
                            min_lr_val = float(min_lr_val)
                        else:
                            min_lr_val = float(min_lr_val)
                        
                        factor_val = lr_reduction_cfg.get('factor', 0.5)
                        if isinstance(factor_val, str):
                            factor_val = float(factor_val)
                        else:
                            factor_val = float(factor_val)
                        
                        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
                            monitor=lr_reduction_cfg.get('monitor', 'val_loss'),
                            patience=int(lr_reduction_cfg.get('patience', 5)),
                            factor=factor_val,
                            min_lr=min_lr_val,
                            mode=lr_reduction_cfg.get('mode', 'min'),
                            cooldown=int(lr_reduction_cfg.get('cooldown', 0)),
                            min_delta=float(lr_reduction_cfg.get('min_delta', 0.0))
                        ))
                    except ImportError:
                        pass
                
                if callbacks:
                    return callbacks
            except Exception as e:
                logger.debug(f"Failed to load callbacks from config: {e}, using defaults")
        
        # Fallback to hardcoded defaults
        try:
            import tensorflow as tf
            patience = self.config.get('patience', 10)
            return [
                tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6)
            ]
        except ImportError:
            return []
    
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
            'target': self.target,
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
        self.target = model_data.get('target', '')
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
            'target': self.target,
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