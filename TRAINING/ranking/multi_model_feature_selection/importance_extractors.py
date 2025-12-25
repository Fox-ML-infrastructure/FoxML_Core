# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

"""
Importance Extraction Functions

Functions for extracting feature importance from trained models using various methods.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple
from sklearn.inspection import permutation_importance

logger = logging.getLogger(__name__)


def safe_load_dataframe(file_path: Path) -> pd.DataFrame:
    """Safely load DataFrame with error handling."""
    try:
        return pd.read_parquet(file_path)
    except Exception:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            raise


def extract_native_importance(model, feature_names: List[str]) -> pd.Series:
    """
    Extract native feature importance from tree-based models.
    
    Supports:
    - LightGBM: feature_importances_ (gain or split)
    - XGBoost: feature_importances_ (gain, weight, or cover)
    - Random Forest: feature_importances_ (gini impurity)
    - CatBoost: feature_importances_ (PredictionValuesChange)
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError(f"Model {type(model)} does not have feature_importances_ attribute")
    
    importance = model.feature_importances_
    
    # Ensure correct length
    if len(importance) != len(feature_names):
        raise ValueError(
            f"Importance length {len(importance)} != feature_names length {len(feature_names)}"
        )
    
    return pd.Series(importance, index=feature_names)


def extract_shap_importance(
    model, X: np.ndarray, feature_names: List[str],
    n_samples: Optional[int] = 1000
) -> pd.Series:
    """
    Extract SHAP importance from model.
    
    Uses TreeExplainer for tree models, LinearExplainer for linear models,
    or KernelExplainer as fallback.
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP is required for SHAP importance extraction. Install with: pip install shap")
    
    # Limit samples for performance
    if n_samples and X.shape[0] > n_samples:
        sample_idx = np.random.choice(X.shape[0], n_samples, replace=False)
        X_sample = X[sample_idx]
    else:
        X_sample = X
    
    # Try TreeExplainer first (fastest for tree models)
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # Handle multi-class output (use mean across classes)
        if isinstance(shap_values, list):
            shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        else:
            shap_values = np.abs(shap_values)
        
        # Average across samples
        importance = np.mean(shap_values, axis=0)
    except Exception:
        # Fallback to LinearExplainer for linear models
        try:
            explainer = shap.LinearExplainer(model, X_sample)
            shap_values = explainer.shap_values(X_sample)
            importance = np.mean(np.abs(shap_values), axis=0)
        except Exception:
            # Final fallback: KernelExplainer (slow but universal)
            explainer = shap.KernelExplainer(model.predict, X_sample[:100])  # Limit to 100 for speed
            shap_values = explainer.shap_values(X_sample[:50])  # Limit to 50 for speed
            importance = np.mean(np.abs(shap_values), axis=0)
    
    if len(importance) != len(feature_names):
        raise ValueError(
            f"SHAP importance length {len(importance)} != feature_names length {len(feature_names)}"
        )
    
    return pd.Series(importance, index=feature_names)


def extract_permutation_importance(
    model, X: np.ndarray, y: np.ndarray,
    feature_names: List[str],
    scoring: str = 'r2',
    n_repeats: int = 5,
    random_state: Optional[int] = None
) -> pd.Series:
    """
    Extract permutation importance from model.
    
    This is the most reliable method but also the slowest, as it requires
    retraining the model multiple times with permuted features.
    """
    # Limit samples for performance (permutation importance is expensive)
    max_samples = 1000
    if X.shape[0] > max_samples:
        sample_idx = np.random.RandomState(random_state).choice(
            X.shape[0], max_samples, replace=False
        )
        X_sample = X[sample_idx]
        y_sample = y[sample_idx]
    else:
        X_sample = X
        y_sample = y
    
    # Compute permutation importance
    perm_importance = permutation_importance(
        model, X_sample, y_sample,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1  # Avoid nested parallelism
    )
    
    # Use mean importance across repeats
    importance = perm_importance.importances_mean
    
    if len(importance) != len(feature_names):
        raise ValueError(
            f"Permutation importance length {len(importance)} != feature_names length {len(feature_names)}"
        )
    
    return pd.Series(importance, index=feature_names)

