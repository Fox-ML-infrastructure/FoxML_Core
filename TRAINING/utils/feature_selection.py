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
Feature Selection Utilities

Helpers for selecting the most important features to reduce dimensionality
and improve model performance.
"""


import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def select_top_features(
    X: np.ndarray,
    feature_names: List[str],
    feature_importances: np.ndarray,
    n_features: int = 50,
    min_importance: float = 0.0
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Select top N features based on importance scores.
    
    Args:
        X: Feature matrix (N, F)
        feature_names: List of feature names
        feature_importances: Feature importance scores (F,)
        n_features: Number of top features to select
        min_importance: Minimum importance threshold
        
    Returns:
        X_selected: Selected features (N, n_features)
        selected_names: Names of selected features
        selected_importances: Importance scores of selected features
    """
    # Validate inputs
    if len(feature_names) != X.shape[1]:
        raise ValueError(f"Feature names length ({len(feature_names)}) doesn't match X columns ({X.shape[1]})")
    
    if len(feature_importances) != X.shape[1]:
        raise ValueError(f"Importance length ({len(feature_importances)}) doesn't match X columns ({X.shape[1]})")
    
    # Sort features by importance
    sorted_indices = np.argsort(feature_importances)[::-1]
    
    # Apply minimum importance threshold
    valid_indices = sorted_indices[feature_importances[sorted_indices] >= min_importance]
    
    # Take top N
    selected_indices = valid_indices[:n_features]
    
    # Extract selected data
    X_selected = X[:, selected_indices]
    selected_names = [feature_names[i] for i in selected_indices]
    selected_importances = feature_importances[selected_indices]
    
    logger.info(f"Selected {len(selected_names)} features out of {len(feature_names)}")
    logger.info(f"Importance range: {selected_importances.min():.6f} to {selected_importances.max():.6f}")
    
    return X_selected, selected_names, selected_importances


def get_feature_importance_from_strategy(
    strategy,
    target_name: str,
    feature_names: List[str],
    aggregate: str = 'mean'
) -> Tuple[List[str], np.ndarray]:
    """
    Get feature importance from a trained strategy and return sorted features.
    
    Args:
        strategy: Trained strategy (SingleTaskStrategy, etc.)
        target_name: Target to get importance for
        feature_names: List of feature names
        aggregate: How to aggregate multiple targets ('mean', 'max', 'median')
        
    Returns:
        sorted_features: Feature names sorted by importance (descending)
        sorted_importances: Importance scores (descending)
    """
    importances_dict = strategy.get_feature_importance(target_name)
    
    if importances_dict is None:
        logger.warning(f"No feature importance available for {target_name}")
        return [], np.array([])
    
    # Get importance array
    if target_name in importances_dict:
        importances = importances_dict[target_name]
    else:
        logger.warning(f"Target {target_name} not found in importance dict")
        return [], np.array([])
    
    # Sort by importance
    sorted_indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_importances = importances[sorted_indices]
    
    return sorted_features, sorted_importances


def create_feature_report(
    feature_names: List[str],
    feature_importances: np.ndarray,
    top_n: int = 50,
    output_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Create a feature importance report.
    
    Args:
        feature_names: List of feature names
        feature_importances: Feature importance scores
        top_n: Number of top features to include in report
        output_file: If provided, save report to this CSV file
        
    Returns:
        DataFrame with feature importance report
    """
    # Create DataFrame
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    })
    
    # Sort by importance
    df = df.sort_values('importance', ascending=False).reset_index(drop=True)
    
    # Add rank
    df['rank'] = range(1, len(df) + 1)
    
    # Calculate cumulative importance
    df['cumulative_importance'] = df['importance'].cumsum() / df['importance'].sum()
    
    # Take top N
    df_top = df.head(top_n)
    
    # Log summary
    logger.info(f"Top {top_n} features account for {df_top['cumulative_importance'].iloc[-1]:.2%} of total importance")
    logger.info(f"Top 10 features: {', '.join(df_top['feature'].head(10).tolist())}")
    
    # Save if requested
    if output_file:
        df_top.to_csv(output_file, index=False)
        logger.info(f"Feature report saved to {output_file}")
    
    return df_top


def select_features_by_correlation(
    X: np.ndarray,
    feature_names: List[str],
    target: np.ndarray,
    n_features: int = 50,
    correlation_threshold: float = 0.95
) -> Tuple[np.ndarray, List[str]]:
    """
    Select features by removing highly correlated features.
    
    Args:
        X: Feature matrix (N, F)
        feature_names: List of feature names
        target: Target variable (N,)
        n_features: Maximum number of features to select
        correlation_threshold: Threshold for removing correlated features
        
    Returns:
        X_selected: Selected features
        selected_names: Names of selected features
    """
    # Calculate correlation with target
    correlations = np.array([np.corrcoef(X[:, i], target)[0, 1] for i in range(X.shape[1])])
    correlations = np.abs(correlations)  # Use absolute correlation
    
    # Sort by correlation with target
    sorted_indices = np.argsort(correlations)[::-1]
    
    # Remove highly correlated features
    selected_indices = []
    for idx in sorted_indices:
        if len(selected_indices) >= n_features:
            break
        
        # Check correlation with already selected features
        is_redundant = False
        for selected_idx in selected_indices:
            corr = np.corrcoef(X[:, idx], X[:, selected_idx])[0, 1]
            if abs(corr) > correlation_threshold:
                is_redundant = True
                break
        
        if not is_redundant:
            selected_indices.append(idx)
    
    # Extract selected data
    X_selected = X[:, selected_indices]
    selected_names = [feature_names[i] for i in selected_indices]
    
    logger.info(f"Selected {len(selected_names)} features (removed {X.shape[1] - len(selected_names)} correlated features)")
    
    return X_selected, selected_names


def auto_select_features(
    strategy,
    X: np.ndarray,
    y_dict: Dict[str, np.ndarray],
    feature_names: List[str],
    target_name: str,
    n_features: int = 50,
    method: str = 'importance'
) -> Tuple[np.ndarray, List[str]]:
    """
    Automatically select features using a trained strategy.
    
    Args:
        strategy: Trained strategy (must have been trained first)
        X: Full feature matrix
        y_dict: Target dictionary
        feature_names: List of feature names
        target_name: Target to use for feature selection
        n_features: Number of features to select
        method: Selection method ('importance' or 'correlation')
        
    Returns:
        X_selected: Selected feature matrix
        selected_names: Names of selected features
    """
    if method == 'importance':
        # Get feature importance from strategy
        sorted_features, sorted_importances = get_feature_importance_from_strategy(
            strategy, target_name, feature_names
        )
        
        if len(sorted_features) == 0:
            logger.error("No feature importance available. Make sure model is trained.")
            return X, feature_names
        
        # Take top N features
        top_features = sorted_features[:n_features]
        top_indices = [feature_names.index(f) for f in top_features]
        
        X_selected = X[:, top_indices]
        selected_names = top_features
        
    elif method == 'correlation':
        # Use correlation-based selection
        X_selected, selected_names = select_features_by_correlation(
            X, feature_names, y_dict[target_name], n_features
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return X_selected, selected_names

