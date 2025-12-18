"""
Copyright (c) 2025-2026 Fox ML Infrastructure LLC

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
Shared utilities for target type detection and model building.

Used across ranking and feature selection to ensure consistent behavior.
"""

import numpy as np
import pandas as pd
from typing import Union


def is_classification_target(y: Union[np.ndarray, list, 'pd.Series'], max_classes: int = 20) -> bool:
    """
    Detect if target is classification (discrete) vs regression (continuous).
    
    Args:
        y: Target array (numpy, pandas, or list)
        max_classes: Maximum number of unique values to consider as classification
    
    Returns:
        True if classification, False if regression
    
    Examples:
        >>> is_classification_target([0, 1, 0, 1])
        True
        >>> is_classification_target([0.5, 1.2, 0.8, 1.5])
        False
        >>> is_classification_target([0, 1, 2, 3, 4, 5])  # 6 classes
        True
        >>> is_classification_target([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])  # 22 classes
        False
    """
    arr = np.asarray(y)
    
    # Filter out NaNs
    if arr.dtype.kind == "f":  # float array
        valid_arr = arr[~np.isnan(arr)]
    else:
        valid_arr = arr
    
    if len(valid_arr) == 0:
        return False  # Can't determine from empty array
    
    # Get unique classes
    classes = np.unique(valid_arr)
    
    # Classification if:
    # 1. Integer/boolean/unsigned integer dtype
    # 2. AND number of unique values <= max_classes
    is_discrete = classes.dtype.kind in ("b", "i", "u")
    is_few_classes = len(classes) <= max_classes
    
    return is_discrete and is_few_classes


def is_binary_classification_target(y: Union[np.ndarray, list, 'pd.Series']) -> bool:
    """
    Detect if target is binary classification (exactly 2 classes, typically 0/1).
    
    Args:
        y: Target array
    
    Returns:
        True if binary classification, False otherwise
    """
    arr = np.asarray(y)
    
    # Filter out NaNs
    if arr.dtype.kind == "f":
        valid_arr = arr[~np.isnan(arr)]
    else:
        valid_arr = arr
    
    if len(valid_arr) == 0:
        return False
    
    unique_vals = np.unique(valid_arr)
    
    # Binary if exactly 2 unique values and they're in {0, 1, 0.0, 1.0}
    if len(unique_vals) == 2:
        val_set = set(unique_vals)
        return val_set.issubset({0, 1, 0.0, 1.0})
    
    return False


def is_multiclass_target(y: Union[np.ndarray, list, 'pd.Series'], max_classes: int = 10) -> bool:
    """
    Detect if target is multiclass classification (3+ classes, but not too many).
    
    Args:
        y: Target array
        max_classes: Maximum number of classes to consider as multiclass
    
    Returns:
        True if multiclass classification, False otherwise
    """
    if is_binary_classification_target(y):
        return False
    
    arr = np.asarray(y)
    
    # Filter out NaNs
    if arr.dtype.kind == "f":
        valid_arr = arr[~np.isnan(arr)]
    else:
        valid_arr = arr
    
    if len(valid_arr) == 0:
        return False
    
    unique_vals = np.unique(valid_arr)
    
    # Multiclass if:
    # 1. More than 2 classes
    # 2. All values are integers (or integer-like floats)
    # 3. Number of classes <= max_classes
    if len(unique_vals) > 2 and len(unique_vals) <= max_classes:
        # Check if all values are integer-like
        all_integer_like = all(
            isinstance(v, (int, np.integer)) or (isinstance(v, float) and v.is_integer())
            for v in unique_vals
        )
        return all_integer_like
    
    return False

