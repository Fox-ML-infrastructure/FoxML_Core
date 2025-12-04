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
Feature Pruning Utilities

Pre-processes feature sets to remove low-importance features before expensive model training.

The "Curse of Dimensionality" Problem:
- You have ~280 features, but ~100 are statistically irrelevant (bottom 1% importance)
- Passing all 280 features to models dilutes split candidates
- Garbage features increase noise floor and cause overfitting

Solution: Quick importance-based pruning using a fast model (LightGBM) to identify
features with < 0.01% cumulative importance, then drop them before the heavy training loops.
"""


import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def quick_importance_prune(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    cumulative_threshold: float = 0.0001,  # 0.01% cumulative importance
    min_features: int = 50,  # Always keep at least this many
    task_type: str = 'regression',
    n_estimators: int = 50,  # Fast model for quick pruning
    random_state: int = 42
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """
    Prune features with very low cumulative importance using a fast LightGBM model.
    
    This is a PRE-PROCESSING step to reduce dimensionality before expensive
    multi-model training. Uses a lightweight model to quickly identify garbage features.
    
    Args:
        X: Feature matrix (N, F)
        y: Target array (N,)
        feature_names: List of feature names (F,)
        cumulative_threshold: Drop features below this cumulative importance (default: 0.01%)
        min_features: Always keep at least this many features (default: 50)
        task_type: 'regression' or 'classification'
        n_estimators: Number of trees for quick importance (default: 50, fast)
        random_state: Random seed
    
    Returns:
        X_pruned: Pruned feature matrix (N, F_pruned)
        pruned_names: Names of kept features
        pruning_stats: Dict with statistics about pruning
    """
    if len(feature_names) != X.shape[1]:
        raise ValueError(f"Feature names length ({len(feature_names)}) doesn't match X columns ({X.shape[1]})")
    
    original_count = len(feature_names)
    
    # Skip pruning if we already have few features
    if original_count <= min_features:
        logger.info(f"  Skipping pruning: only {original_count} features (min={min_features})")
        return X, feature_names, {
            'original_count': original_count,
            'pruned_count': original_count,
            'dropped_count': 0,
            'dropped_features': []
        }
    
    try:
        import lightgbm as lgb
    except ImportError:
        logger.warning("LightGBM not available for feature pruning, skipping")
        return X, feature_names, {
            'original_count': original_count,
            'pruned_count': original_count,
            'dropped_count': 0,
            'dropped_features': []
        }
    
    logger.info(f"  Quick importance pruning: {original_count} features → target: {min_features}+")
    
    # Train a fast LightGBM model to get importance
    if task_type == 'regression':
        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=5,  # Shallow for speed
            learning_rate=0.1,
            verbosity=-1,
            random_state=random_state,
            n_jobs=1  # Single thread for quick pruning
        )
    else:
        # Classification
        unique_vals = np.unique(y[~np.isnan(y)])
        if len(unique_vals) == 2:
            objective = 'binary'
        else:
            objective = 'multiclass'
        
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=5,
            learning_rate=0.1,
            objective=objective,
            verbosity=-1,
            random_state=random_state,
            n_jobs=1
        )
    
    try:
        # Quick training
        model.fit(X, y)
        
        # Get feature importance (normalized to sum to 1)
        importances = model.feature_importances_
        total_importance = importances.sum()
        if total_importance > 0:
            normalized_importance = importances / total_importance
        else:
            logger.warning("  All feature importances are zero, skipping pruning")
            return X, feature_names, {
                'original_count': original_count,
                'pruned_count': original_count,
                'dropped_count': 0,
                'dropped_features': []
            }
        
        # Sort by importance (descending)
        sorted_indices = np.argsort(normalized_importance)[::-1]
        sorted_importance = normalized_importance[sorted_indices]
        
        # Calculate cumulative importance
        cumulative_importance = np.cumsum(sorted_importance)
        
        # Find features to keep:
        # 1. Keep features above cumulative threshold
        # 2. Always keep at least min_features
        keep_mask = cumulative_importance <= (1.0 - cumulative_threshold)
        keep_mask[:min_features] = True  # Always keep top N
        
        # Get indices of features to keep (in original order)
        keep_indices = sorted_indices[keep_mask]
        keep_indices = np.sort(keep_indices)  # Restore original order
        
        # Extract pruned data
        X_pruned = X[:, keep_indices]
        pruned_names = [feature_names[i] for i in keep_indices]
        
        dropped_count = original_count - len(pruned_names)
        dropped_features = [feature_names[i] for i in range(len(feature_names)) if i not in keep_indices]
        
        # Log statistics
        top_10_importance = sorted_importance[:10]
        logger.info(f"  Pruned: {original_count} → {len(pruned_names)} features (dropped {dropped_count})")
        logger.info(f"  Top 10 importance range: {top_10_importance[-1]:.4%} to {top_10_importance[0]:.4%}")
        if dropped_count > 0:
            logger.info(f"  Dropped features (sample): {dropped_features[:10]}")
            if len(dropped_features) > 10:
                logger.info(f"    ... and {len(dropped_features) - 10} more")
        
        return X_pruned, pruned_names, {
            'original_count': original_count,
            'pruned_count': len(pruned_names),
            'dropped_count': dropped_count,
            'dropped_features': dropped_features,
            'top_10_features': [feature_names[sorted_indices[i]] for i in range(min(10, len(sorted_indices)))],
            'top_10_importance': top_10_importance.tolist()
        }
        
    except Exception as e:
        logger.warning(f"  Feature pruning failed: {e}, using all features")
        return X, feature_names, {
            'original_count': original_count,
            'pruned_count': original_count,
            'dropped_count': 0,
            'dropped_features': [],
            'error': str(e)
        }


def prune_features_by_importance_csv(
    importance_csv_path: str,
    feature_names: List[str],
    cumulative_threshold: float = 0.0001,
    min_features: int = 50
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Prune features based on pre-computed importance from CSV file.
    
    Useful when you have already computed feature importance and want to reuse it.
    
    Args:
        importance_csv_path: Path to CSV with columns: feature_name, importance
        feature_names: List of all feature names to filter
        cumulative_threshold: Drop features below this cumulative importance
        min_features: Always keep at least this many
    
    Returns:
        pruned_names: List of kept feature names
        pruning_stats: Dict with statistics
    """
    try:
        df_importance = pd.read_csv(importance_csv_path)
        
        # Ensure we have required columns
        if 'feature_name' not in df_importance.columns or 'importance' not in df_importance.columns:
            raise ValueError(f"CSV must have 'feature_name' and 'importance' columns")
        
        # Normalize importance
        total = df_importance['importance'].sum()
        if total > 0:
            df_importance['normalized_importance'] = df_importance['importance'] / total
        else:
            logger.warning("All importances are zero in CSV")
            return feature_names, {'error': 'zero_importance'}
        
        # Sort by importance
        df_importance = df_importance.sort_values('normalized_importance', ascending=False)
        
        # Calculate cumulative
        df_importance['cumulative'] = df_importance['normalized_importance'].cumsum()
        
        # Find features to keep
        keep_mask = df_importance['cumulative'] <= (1.0 - cumulative_threshold)
        keep_mask.iloc[:min_features] = True  # Always keep top N
        
        kept_features = df_importance[keep_mask]['feature_name'].tolist()
        
        # Filter to only features that exist in our feature_names
        pruned_names = [f for f in feature_names if f in kept_features]
        
        dropped_count = len(feature_names) - len(pruned_names)
        
        return pruned_names, {
            'original_count': len(feature_names),
            'pruned_count': len(pruned_names),
            'dropped_count': dropped_count
        }
        
    except Exception as e:
        logger.warning(f"Failed to prune from CSV: {e}")
        return feature_names, {'error': str(e)}

