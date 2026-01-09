# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Sklearn-Safe Data Conversion

Converts tabular data (pandas, polars, numpy) into dense float32 numpy arrays
suitable for sklearn models that don't handle NaNs or non-float dtypes.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from typing import Tuple, List, Union, Optional, Any

# Shared imputer instance (fitted per-call, but reuse the object)
_SKLEARN_IMPUTER = SimpleImputer(strategy="median")


def make_sklearn_dense_X(
    X: Union[pd.DataFrame, np.ndarray, Any],  # Any to support polars
    feature_names: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Convert tabular object into a dense float32 numpy array with NaNs imputed.
    
    Safe for sklearn models that don't handle NaNs: Lasso, mutual information,
    univariate selection, Boruta, etc.
    
    Args:
        X: Input data (pandas DataFrame, polars DataFrame, or numpy array)
        feature_names: Optional list of feature names (if X is numpy array)
    
    Returns:
        Tuple of (dense_array, feature_names)
        - dense_array: float32 numpy array with NaNs imputed to median
        - feature_names: List of feature names (preserved from input)
    
    Examples:
        >>> X_df = pd.DataFrame({'a': [1, 2, np.nan], 'b': [4, 5, 6]})
        >>> X_dense, names = make_sklearn_dense_X(X_df)
        >>> X_dense.shape
        (3, 2)
        >>> np.isnan(X_dense).any()
        False
    """
    # Normalize to pandas DataFrame
    if hasattr(X, "to_pandas"):  # polars DataFrame
        X = X.to_pandas()
    elif isinstance(X, np.ndarray):
        # Convert numpy array to DataFrame
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        X = pd.DataFrame(X, columns=feature_names)
    elif not isinstance(X, pd.DataFrame):
        # Try to convert to DataFrame
        X = pd.DataFrame(X)
    
    # Preserve feature names
    if feature_names is None:
        feature_names = list(X.columns)
    
    # Replace infs with NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Force numeric dtypes; non-numeric -> NaN
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    
    # Convert to float32 dense and impute
    arr = X.to_numpy(dtype=np.float32)
    
    # Fit and transform with imputer
    arr_imputed = _SKLEARN_IMPUTER.fit_transform(arr)
    
    return arr_imputed, feature_names

