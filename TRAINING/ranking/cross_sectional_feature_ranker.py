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
Cross-sectional feature ranking module.

This module computes feature importance using a true cross-sectional (panel) model,
where rows = (symbol, timestamp) and features are ranked based on their ability
to predict the target across the entire universe simultaneously.

This provides a complementary view to per-symbol feature selection:
- Per-symbol: "Does this feature work on AAPL? On MSFT?"
- Cross-sectional: "Does this feature work across the universe?"

Features can then be tagged as:
- CORE: Strong in both per-symbol AND cross-sectional
- SYMBOL-SPECIFIC: Strong per-symbol, weak cross-sectional
- WEAK: Weak in both
"""


import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

logger = logging.getLogger(__name__)


def normalize_cross_sectional_per_date(
    X: np.ndarray,
    time_vals: np.ndarray,
    method: str = 'zscore'
) -> np.ndarray:
    """
    Normalize features per timestamp (cross-sectional normalization).
    
    This makes features comparable across symbols at each point in time,
    which is useful for cross-sectional ranking where we care about
    relative position within the universe.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        time_vals: Timestamp array (n_samples,)
        method: Normalization method ('zscore' or 'rank')
    
    Returns:
        Normalized feature matrix
    """
    if method not in ['zscore', 'rank']:
        raise ValueError(f"Unknown normalization method: {method}")
    
    X_norm = X.copy()
    time_df = pd.DataFrame({'time': time_vals})
    
    # Group by timestamp and normalize within each group
    for t in time_df['time'].unique():
        mask = time_df['time'] == t
        if mask.sum() < 2:
            continue  # Need at least 2 samples for normalization
        
        if method == 'zscore':
            # Z-score: (x - mean) / std
            X_norm[mask] = (X[mask] - X[mask].mean(axis=0)) / (X[mask].std(axis=0) + 1e-9)
        elif method == 'rank':
            # Rank transform: rank / n_samples (0 to 1)
            from scipy.stats import rankdata
            for feat_idx in range(X.shape[1]):
                ranks = rankdata(X[mask, feat_idx], method='average')
                X_norm[mask, feat_idx] = ranks / len(ranks)
    
    return X_norm


def train_panel_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    model_family: str = 'lightgbm',
    model_config: Optional[Dict] = None,
    target_column: Optional[str] = None  # For deterministic seed generation
) -> Tuple[Any, pd.Series]:
    """
    Train a single panel model and extract feature importance.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        feature_names: List of feature names
        model_family: Model family ('lightgbm', 'xgboost', etc.)
        model_config: Optional model hyperparameters
    
    Returns:
        Tuple of (trained_model, importance_series)
    """
    if model_config is None:
        model_config = {}
    
    # Generate deterministic seed for cross-sectional panel model
    from TRAINING.common.determinism import stable_seed_from
    seed_parts = ['cross_sectional', model_family]
    if target_column:
        seed_parts.append(target_column)
    cs_seed = stable_seed_from(seed_parts)
    
    # Load model configs from YAML files (single source of truth)
    default_configs = {}
    try:
        from CONFIG.config_loader import load_model_config
        
        # Load LightGBM config
        try:
            lgb_config = load_model_config('lightgbm')
            default_configs['lightgbm'] = {
                'n_estimators': lgb_config.get('n_estimators', 100),
                'max_depth': lgb_config.get('max_depth', 6),
                'learning_rate': lgb_config.get('learning_rate', 0.05),
                'random_state': cs_seed,
                'verbosity': -1,
                'n_jobs': 1
            }
        except Exception:
            default_configs['lightgbm'] = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.05,
                'random_state': cs_seed,
                'verbosity': -1,
                'n_jobs': 1
            }
        
        # Load XGBoost config
        try:
            xgb_config = load_model_config('xgboost')
            default_configs['xgboost'] = {
                'n_estimators': xgb_config.get('n_estimators', 100),
                'max_depth': xgb_config.get('max_depth', 6),
                'learning_rate': xgb_config.get('learning_rate', 0.05),
                'random_state': cs_seed,
                'n_jobs': 1
            }
        except Exception:
            default_configs['xgboost'] = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.05,
                'random_state': cs_seed,
                'n_jobs': 1
            }
    except Exception:
        # Fallback to hardcoded defaults
        default_configs = {
            'lightgbm': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.05,
                'random_state': cs_seed,
                'verbosity': -1,
                'n_jobs': 1
            },
            'xgboost': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.05,
                'random_state': cs_seed,
                'n_jobs': 1
            }
        }
    
    # Merge with defaults
    config = {**default_configs.get(model_family, {}), **model_config}
    
    # Determine task type
    unique_vals = np.unique(y[~np.isnan(y)])
    is_binary = len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0})
    is_multiclass = len(unique_vals) <= 10 and all(
        isinstance(v, (int, np.integer)) or (isinstance(v, float) and v.is_integer())
        for v in unique_vals
    )
    
    # Train model
    if model_family == 'lightgbm':
        try:
            from lightgbm import LGBMRegressor, LGBMClassifier
            
            if is_binary or is_multiclass:
                model = LGBMClassifier(**config)
            else:
                model = LGBMRegressor(**config)
            
            model.fit(X, y)
            
            # Get feature importance (gain-based)
            if hasattr(model, 'feature_importances_'):
                importance = pd.Series(model.feature_importances_, index=feature_names)
            else:
                importance = pd.Series(0.0, index=feature_names)
            
            return model, importance
            
        except Exception as e:
            logger.warning(f"LightGBM panel model failed: {e}")
            return None, pd.Series(0.0, index=feature_names)
    
    elif model_family == 'xgboost':
        try:
            import xgboost as xgb
            
            if is_binary or is_multiclass:
                model = xgb.XGBClassifier(**config)
            else:
                model = xgb.XGBRegressor(**config)
            
            model.fit(X, y)
            
            # Get feature importance (gain-based)
            if hasattr(model, 'feature_importances_'):
                importance = pd.Series(model.feature_importances_, index=feature_names)
            else:
                importance = pd.Series(0.0, index=feature_names)
            
            return model, importance
            
        except Exception as e:
            logger.warning(f"XGBoost panel model failed: {e}")
            return None, pd.Series(0.0, index=feature_names)
    
    else:
        logger.warning(f"Unknown model family: {model_family}")
        return None, pd.Series(0.0, index=feature_names)


def compute_cross_sectional_importance(
    candidate_features: List[str],
    target_column: str,
    symbols: List[str],
    data_dir: Path,
    # Load defaults from config
    try:
        from CONFIG.config_loader import get_cfg
        default_min_cs = int(get_cfg("pipeline.data_limits.min_cross_sectional_samples", default=10, config_name="pipeline_config"))
        default_max_cs_samples = int(get_cfg("pipeline.data_limits.max_cs_samples", default=1000, config_name="pipeline_config"))
    except Exception:
        default_min_cs = 10
        default_max_cs_samples = 1000
    
    model_families: List[str] = ['lightgbm'],
    min_cs: int = default_min_cs,
    max_cs_samples: int = default_max_cs_samples,
    normalization: Optional[str] = None,
    model_configs: Optional[Dict[str, Dict]] = None
) -> pd.Series:
    """
    Compute cross-sectional feature importance using panel model.
    
    This trains a single model across all symbols simultaneously (panel data)
    and ranks features by their importance in predicting the target across
    the universe.
    
    Args:
        candidate_features: List of feature names to evaluate (top_k from per-symbol selection)
        target_column: Target column name
        symbols: List of symbols to include
        data_dir: Directory containing symbol data
        model_families: List of model families to use (e.g., ['lightgbm', 'xgboost'])
        min_cs: Minimum cross-sectional size per timestamp
        max_cs_samples: Maximum samples per timestamp
        normalization: Optional normalization method ('zscore' or 'rank')
        model_configs: Optional dict of model_family -> config overrides
    
    Returns:
        Series with feature -> CS importance score (aggregated across model families)
    """
    logger.info(f"🔍 Computing cross-sectional importance for {len(candidate_features)} candidate features")
    logger.info(f"   Symbols: {len(symbols)}, Model families: {model_families}")
    
    # Load panel data (reuse existing utility)
    from TRAINING.utils.cross_sectional_data import (
        load_mtf_data_for_ranking,
        prepare_cross_sectional_data_for_ranking
    )
    
    # Load data
    mtf_data = load_mtf_data_for_ranking(data_dir, symbols)
    if not mtf_data:
        logger.warning("No data loaded, returning zero importance")
        return pd.Series(0.0, index=candidate_features)
    
    # Build panel with candidate features only
    X, y, feature_names, symbols_array, time_vals = prepare_cross_sectional_data_for_ranking(
        mtf_data, target_column,
        min_cs=min_cs,
        max_cs_samples=max_cs_samples,
        feature_names=candidate_features  # Only candidate features
    )
    
    if X is None or y is None:
        logger.warning("Failed to prepare cross-sectional data, returning zero importance")
        return pd.Series(0.0, index=candidate_features)
    
    logger.info(f"   Panel data: {len(X)} samples, {X.shape[1]} features")
    
    # Optional normalization (per-date z-score or rank)
    if normalization:
        logger.info(f"   Applying {normalization} normalization per timestamp")
        X = normalize_cross_sectional_per_date(X, time_vals, method=normalization)
    
    # Train panel model(s) and get importance
    importances = {}
    for model_family in model_families:
        logger.debug(f"   Training {model_family} panel model...")
        model_config = (model_configs or {}).get(model_family, {})
        model, importance = train_panel_model(
            X, y, feature_names, model_family, model_config,
            target_column=target_column  # Pass target for deterministic seed
        )
        
        if model is not None:
            importances[model_family] = importance
            logger.debug(f"   {model_family}: top feature = {importance.idxmax()} ({importance.max():.4f})")
        else:
            logger.warning(f"   {model_family} failed, skipping")
    
    if not importances:
        logger.warning("All panel models failed, returning zero importance")
        return pd.Series(0.0, index=candidate_features)
    
    # Aggregate across model families (mean)
    cs_importance = pd.Series(0.0, index=feature_names)
    for imp in importances.values():
        # Align indices (handle missing features)
        aligned = imp.reindex(feature_names, fill_value=0.0)
        cs_importance += aligned
    cs_importance /= len(importances)
    
    # Normalize to 0-1 range for easier comparison with per-symbol scores
    if cs_importance.max() > 0:
        cs_importance = cs_importance / cs_importance.max()
    
    logger.info(f"   ✅ Cross-sectional importance computed: top feature = {cs_importance.idxmax()} ({cs_importance.max():.4f})")
    
    return cs_importance


def tag_features_by_importance(
    symbol_importance: pd.Series,
    cs_importance: pd.Series,
    symbol_threshold: float = 0.1,
    cs_threshold: float = 0.1
) -> pd.Series:
    """
    Tag features based on per-symbol vs cross-sectional importance.
    
    Categories:
    - CORE: Strong in both (>= threshold in both)
    - SYMBOL_SPECIFIC: Strong per-symbol, weak cross-sectional
    - CS_SPECIFIC: Strong cross-sectional, weak per-symbol
    - WEAK: Weak in both
    
    Args:
        symbol_importance: Per-symbol importance scores (from aggregation)
        cs_importance: Cross-sectional importance scores
        symbol_threshold: Threshold for "strong" per-symbol importance (relative, 0-1)
        cs_threshold: Threshold for "strong" CS importance (relative, 0-1)
    
    Returns:
        Series with feature -> category string
    """
    # Normalize both to 0-1 range if needed
    if symbol_importance.max() > 1.0:
        symbol_importance = symbol_importance / symbol_importance.max()
    if cs_importance.max() > 1.0:
        cs_importance = cs_importance / cs_importance.max()
    
    # Align indices
    all_features = symbol_importance.index.union(cs_importance.index)
    symbol_aligned = symbol_importance.reindex(all_features, fill_value=0.0)
    cs_aligned = cs_importance.reindex(all_features, fill_value=0.0)
    
    # Tag features
    categories = pd.Series('UNKNOWN', index=all_features)
    
    strong_symbol = symbol_aligned >= symbol_threshold
    strong_cs = cs_aligned >= cs_threshold
    
    categories[strong_symbol & strong_cs] = 'CORE'
    categories[strong_symbol & ~strong_cs] = 'SYMBOL_SPECIFIC'
    categories[~strong_symbol & strong_cs] = 'CS_SPECIFIC'
    categories[~strong_symbol & ~strong_cs] = 'WEAK'
    
    return categories

