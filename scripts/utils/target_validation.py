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
Shared utilities for target validation and degenerate target detection
Used across all ranking scripts for consistent behavior
"""


import numpy as np
import warnings
from typing import Tuple, Optional
from scripts.utils.task_types import TaskType


def validate_target(
    y: np.ndarray, 
    task_type: Optional[TaskType] = None,
    min_samples: int = 10, 
    min_class_samples: int = 2
) -> Tuple[bool, Optional[str]]:
    """
    Validate that a target is suitable for model training
    
    Args:
        y: Target array
        task_type: Optional TaskType (if None, will be inferred)
        min_samples: Minimum total samples required
        min_class_samples: Minimum samples per class (for classification)
    
    Returns:
        (is_valid, error_message)
        is_valid: True if target is valid, False otherwise
        error_message: None if valid, error description if invalid
    """
    # Remove NaN
    y_clean = y[~np.isnan(y)]
    
    if len(y_clean) < min_samples:
        return False, f"Too few samples: {len(y_clean)} < {min_samples}"
    
    # Infer task type if not provided
    if task_type is None:
        unique_vals = np.unique(y_clean)
        n_unique = len(unique_vals)
        
        if n_unique == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
            task_type = TaskType.BINARY_CLASSIFICATION
        elif n_unique <= 10:
            # Check if integer-like (classification) vs continuous (regression)
            if all(isinstance(v, (int, np.integer)) or 
                   (isinstance(v, float) and float(v).is_integer()) 
                   for v in unique_vals):
                task_type = TaskType.MULTICLASS_CLASSIFICATION
            else:
                task_type = TaskType.REGRESSION
        else:
            task_type = TaskType.REGRESSION
    
    # Check unique values
    unique_vals = np.unique(y_clean)
    n_unique = len(unique_vals)
    
    if n_unique < 2:
        return False, f"Only {n_unique} unique value(s) (degenerate target)"
    
    # Task-specific validation
    if task_type == TaskType.BINARY_CLASSIFICATION:
        if n_unique != 2:
            return False, f"Binary classification requires exactly 2 classes, found {n_unique}"
        
        class_counts = np.bincount(y_clean.astype(int))
        min_class_count = class_counts[class_counts > 0].min()
        
        if min_class_count < min_class_samples:
            return False, f"Smallest class has only {min_class_count} sample(s) (too few for CV)"
    
    elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
        if n_unique < 2:
            return False, f"Multiclass requires at least 2 classes, found {n_unique}"
        
        try:
            class_counts = np.bincount(y_clean.astype(int))
            min_class_count = class_counts[class_counts > 0].min()
            
            if min_class_count < min_class_samples:
                return False, f"Smallest class has only {min_class_count} sample(s) (too few for CV)"
        except (ValueError, OverflowError):
            return False, "Invalid class labels for multiclass classification"
    
    elif task_type == TaskType.REGRESSION:
        # For regression, check variance
        std = y_clean.std()
        if std < 1e-6:
            return False, f"Zero or near-zero variance (std={std:.2e})"
    
    return True, None


def check_cv_compatibility(
    y: np.ndarray, 
    task_type: Optional[TaskType] = None,
    cv_folds: int = 3
) -> Tuple[bool, Optional[str]]:
    """
    Check if target is compatible with cross-validation
    
    For classification targets, ensures each class has enough samples
    to be split across CV folds.
    
    Args:
        y: Target array
        task_type: Optional TaskType (if None, will be inferred)
        cv_folds: Number of CV folds
    
    Returns:
        (is_compatible, error_message)
    """
    y_clean = y[~np.isnan(y)]
    unique_vals = np.unique(y_clean)
    n_unique = len(unique_vals)
    
    # Infer task type if not provided
    if task_type is None:
        if n_unique == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
            task_type = TaskType.BINARY_CLASSIFICATION
        elif n_unique <= 10:
            if all(isinstance(v, (int, np.integer)) or 
                   (isinstance(v, float) and float(v).is_integer()) 
                   for v in unique_vals):
                task_type = TaskType.MULTICLASS_CLASSIFICATION
            else:
                task_type = TaskType.REGRESSION
        else:
            task_type = TaskType.REGRESSION
    
    # For classification targets, check class balance
    if task_type in {TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION}:
        try:
            if all(isinstance(v, (int, np.integer)) or (isinstance(v, float) and float(v).is_integer())
                   for v in unique_vals):
                class_counts = np.bincount(y_clean.astype(int))
                min_class_count = class_counts[class_counts > 0].min()
                
                # Need at least 1 sample per class per fold (less conservative)
                # Changed from cv_folds * 2 to cv_folds * 1 to avoid rejecting valid targets unnecessarily
                min_required = cv_folds * 1
                if min_class_count < min_required:
                    return False, f"Smallest class has {min_class_count} samples, need {min_required} for {cv_folds}-fold CV"
        except (ValueError, OverflowError):
            return False, "Invalid class labels for classification"
    
    return True, None


def safe_cross_val_score(model, X, y, cv=3, scoring='r2', **kwargs):
    """
    Wrapper around cross_val_score with better error handling
    
    Returns:
        scores array (may contain NaN for failed folds)
    """
    from sklearn.model_selection import cross_val_score
    
    # Always use error_score=np.nan to handle fold failures gracefully
    kwargs['error_score'] = np.nan
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, **kwargs)
            return scores
        except ValueError as e:
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ['least populated class', 'too few', 'invalid classes', 'expected']):
                # Degenerate target in CV - return all NaN
                return np.array([np.nan] * cv)
            raise

