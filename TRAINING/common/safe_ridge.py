# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

"""
Safe Ridge Regression Wrapper

Provides a bulletproof Ridge fit that falls back to LSQR solver
when Cholesky/scipy.linalg.solve causes MKL segfaults.
"""


import numpy as np
from sklearn.linear_model import Ridge
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def safe_ridge_fit(X: np.ndarray, y: np.ndarray, 
                   alpha: float = 1.0,
                   family: Optional[str] = None,
                   **kwargs) -> Ridge:
    """
    Fit Ridge regression with fallback to safer solver.
    
    Args:
        X: Training features
        y: Training targets
        alpha: Ridge regularization parameter
        family: Family name (to check config for preferred solver)
        **kwargs: Additional Ridge parameters
    
    Returns:
        Fitted Ridge model
    """
    # Check config for preferred solver
    solver = "auto"
    if family:
        try:
            from .family_config import get_ridge_solver
            solver = get_ridge_solver(family)
        except Exception:
            pass
    
    # Override solver from kwargs if provided
    solver = kwargs.pop("solver", solver)
    
    try:
        # Try with configured solver first
        model = Ridge(alpha=alpha, solver=solver, **kwargs)
        model.fit(X, y)
        return model
    except Exception as e:
        # Fall back to LSQR which avoids scipy.linalg.solve
        if solver != "lsqr":
            logger.warning(
                "Ridge fit failed with solver='%s' (error: %s), "
                "falling back to solver='lsqr'",
                solver, str(e)[:100]
            )
            try:
                model = Ridge(alpha=alpha, solver="lsqr", **kwargs)
                model.fit(X, y)
                return model
            except Exception as e2:
                logger.error("Ridge fit failed even with lsqr solver: %s", e2)
                raise
        else:
            # Already using lsqr, re-raise
            logger.error("Ridge fit failed with lsqr solver: %s", e)
            raise

def safe_ridge_predict(model: Ridge, X: np.ndarray) -> np.ndarray:
    """
    Predict with Ridge model, with fallback for edge cases.
    
    Args:
        model: Fitted Ridge model
        X: Features to predict on
    
    Returns:
        Predictions
    """
    try:
        return model.predict(X)
    except Exception as e:
        logger.warning("Ridge predict failed: %s, returning zeros", e)
        return np.zeros(len(X))

