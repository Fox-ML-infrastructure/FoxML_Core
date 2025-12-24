# MIT License - see LICENSE file

"""
Feature Analysis for Leakage Detection

Functions for analyzing features to detect potential data leakage.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Optional

from TRAINING.common.utils.task_types import TaskType

logger = logging.getLogger(__name__)


def find_near_copy_features(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: TaskType,
    tol: float = 1e-4,
    min_match: Optional[float] = None,
    min_corr: Optional[float] = None
) -> List[str]:
    """
    Find features that are basically copies of y (or 1 - y) for binary targets,
    or highly correlated for regression targets.
    
    This is a pre-training leak scan that catches obvious leaks before models are trained.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        task_type: TaskType enum (BINARY_CLASSIFICATION, REGRESSION, etc.)
        tol: Tolerance for numerical comparison
        min_match: For binary classification, minimum match ratio (default: 99.9%)
        min_corr: For regression, minimum correlation (default: 0.99)
    
    Returns:
        List of feature names that are near-copies of the target
    """
    leaking_features = []
    
    if task_type == TaskType.BINARY_CLASSIFICATION:
        # For binary classification, check for features that match y or 1-y
        if min_match is None:
            min_match = 0.999  # Default: 99.9% match
        
        y_values = y.values
        y_complement = 1 - y_values
        
        for col in X.columns:
            feature_values = X[col].values
            
            # Check if feature matches y
            match_y = np.abs(feature_values - y_values) < tol
            match_ratio_y = match_y.sum() / len(y_values) if len(y_values) > 0 else 0
            
            # Check if feature matches 1-y
            match_complement = np.abs(feature_values - y_complement) < tol
            match_ratio_complement = match_complement.sum() / len(y_values) if len(y_values) > 0 else 0
            
            if match_ratio_y >= min_match or match_ratio_complement >= min_match:
                leaking_features.append(col)
                logger.warning(
                    f"  🚨 LEAK DETECTED: {col} is a near-copy of target "
                    f"(match_y={match_ratio_y:.1%}, match_1-y={match_ratio_complement:.1%})"
                )
    
    elif task_type == TaskType.REGRESSION:
        # For regression, check for high correlation
        if min_corr is None:
            min_corr = 0.99  # Default: 99% correlation
        
        for col in X.columns:
            try:
                corr = np.abs(np.corrcoef(X[col].values, y.values)[0, 1])
                if not np.isnan(corr) and corr >= min_corr:
                    leaking_features.append(col)
                    logger.warning(
                        f"  🚨 LEAK DETECTED: {col} is highly correlated with target "
                        f"(corr={corr:.3f})"
                    )
            except Exception as e:
                logger.debug(f"Could not compute correlation for {col}: {e}")
    
    return leaking_features


def is_calendar_feature(feature_name: str) -> bool:
    """
    Check if a feature is a calendar/time feature that should be excluded from leak detection.
    
    Calendar features (hour, day_of_week, etc.) are not considered leaks even if they
    correlate with time-based targets.
    """
    calendar_patterns = [
        'hour', 'day_of_week', 'day_of_month', 'month', 'year',
        'is_weekend', 'is_holiday', 'quarter', 'week_of_year'
    ]
    
    feature_lower = feature_name.lower()
    return any(pattern in feature_lower for pattern in calendar_patterns)


def detect_leaking_features(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: List[str],
    task_type: TaskType,
    target_column: str,
    tol: float = 1e-4
) -> List[str]:
    """
    Detect features that leak information about the target.
    
    This is a comprehensive leak scan that checks for:
    1. Near-copy features (exact or near-exact matches to target)
    2. Calendar features (excluded from leak detection)
    3. High correlation features (for regression)
    
    Args:
        X: Feature DataFrame
        y: Target Series
        feature_names: List of feature names to check
        task_type: TaskType enum
        target_column: Target column name (for logging)
        tol: Tolerance for numerical comparison
    
    Returns:
        List of leaking feature names
    """
    leaking_features = []
    
    # Step 1: Find near-copy features
    near_copies = find_near_copy_features(X, y, task_type, tol=tol)
    leaking_features.extend(near_copies)
    
    # Step 2: Filter out calendar features (they're not leaks)
    leaking_features = [
        f for f in leaking_features
        if not is_calendar_feature(f)
    ]
    
    if leaking_features:
        logger.warning(
            f"  🚨 Found {len(leaking_features)} leaking features for {target_column}: "
            f"{', '.join(leaking_features[:5])}"
            + (f" ... and {len(leaking_features) - 5} more" if len(leaking_features) > 5 else "")
        )
    
    return leaking_features

