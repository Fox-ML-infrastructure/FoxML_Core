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
Multi-Model Feature Selection Pipeline

Combines feature importance from multiple model families to find robust features
that have predictive power across diverse architectures.

Strategy:
1. Train multiple model families (tree-based, neural, specialized)
2. Extract importance using best method per family:
   - Tree models: Native feature_importances_ (gain/split)
   - Neural networks: SHAP TreeExplainer approximation or permutation
   - Linear models: Absolute coefficients
3. Aggregate across models AND symbols
4. Rank by consensus: features important across multiple model types

This avoids model-specific biases and finds truly predictive features.
"""


import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
import lightgbm as lgb
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import json
from collections import defaultdict
from dataclasses import dataclass, asdict
import warnings

# Add project root FIRST (before any scripts.* imports)
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from CONFIG.config_loader import load_model_config
import yaml

# Import checkpoint utility (after path is set)
from scripts.utils.checkpoint import CheckpointManager

# Setup logging with journald support (after path is set)
from scripts.utils.logging_setup import setup_logging
logger = setup_logging(
    script_name="multi_model_feature_selection",
    level=logging.INFO,
    use_journald=True
)

# Suppress warnings from SHAP/sklearn
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='X does not have valid feature names')


@dataclass
class ModelFamilyConfig:
    """Configuration for a model family"""
    name: str
    importance_method: str  # 'native', 'shap', 'permutation'
    enabled: bool
    config: Dict[str, Any]
    weight: float = 1.0  # Weight in final aggregation


@dataclass
class ImportanceResult:
    """Result from a single model's feature importance calculation"""
    model_family: str
    symbol: str
    importance_scores: pd.Series
    method: str
    train_score: float


def load_multi_model_config(config_path: Path = None) -> Dict[str, Any]:
    """Load multi-model feature selection configuration"""
    if config_path is None:
        config_path = _REPO_ROOT / "CONFIG" / "multi_model_feature_selection.yaml"
    
    if not config_path.exists():
        logger.warning(f"Config not found: {config_path}, using defaults")
        return get_default_config()
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded multi-model config from {config_path}")
    return config


def get_default_config() -> Dict[str, Any]:
    """Default configuration if file doesn't exist"""
    return {
        'model_families': {
            'lightgbm': {
                'enabled': True,
                'importance_method': 'native',
                'weight': 1.0,
                'config': {
                    'objective': 'regression_l1',
                    'metric': 'mae',
                    'n_estimators': 300,
                    'learning_rate': 0.05,
                    'num_leaves': 31,
                    'verbose': -1
                }
            },
            'xgboost': {
                'enabled': True,
                'importance_method': 'native',
                'weight': 1.0,
                'config': {
                    'objective': 'reg:squarederror',
                    'n_estimators': 300,
                    'learning_rate': 0.05,
                    'max_depth': 6,
                    'verbosity': 0
                }
            },
            'random_forest': {
                'enabled': True,
                'importance_method': 'native',
                'weight': 0.8,
                'config': {
                    'n_estimators': 200,
                    'max_depth': 15,
                    'max_features': 'sqrt',
                    'n_jobs': 4
                }
            },
            'neural_network': {
                'enabled': True,
                'importance_method': 'permutation',
                'weight': 1.2,
                'config': {
                    'hidden_layer_sizes': (128, 64),
                    'max_iter': 300,
                    'early_stopping': True,
                    'validation_fraction': 0.1
                }
            }
        },
        'aggregation': {
            'per_symbol_method': 'mean',
            'cross_model_method': 'weighted_mean',
            'require_min_models': 2,
            'consensus_threshold': 0.5
        },
        'sampling': {
            'max_samples_per_symbol': 50000,
            'validation_split': 0.2
        }
    }


def safe_load_dataframe(file_path: Path) -> pd.DataFrame:
    """Safely load a parquet file"""
    try:
        return pd.read_parquet(file_path)
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        raise


def extract_native_importance(model, feature_names: List[str]) -> pd.Series:
    """Extract native feature importance from various model types"""
    if hasattr(model, 'feature_importance'):
        # LightGBM
        importance = model.feature_importance(importance_type='gain')
    elif hasattr(model, 'get_feature_importance'):
        # CatBoost
        importance = model.get_feature_importance()
    elif hasattr(model, 'feature_importances_'):
        # sklearn models (RF, XGBoost sklearn API, HistGradientBoosting, etc.)
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Linear models (Lasso, Ridge, ElasticNet)
        importance = np.abs(model.coef_)
    elif hasattr(model, 'get_score'):
        # XGBoost native API
        score_dict = model.get_score(importance_type='gain')
        importance = np.array([score_dict.get(f, 0.0) for f in feature_names])
    elif hasattr(model, 'importance'):
        # Dummy model for mutual information
        importance = model.importance
    else:
        raise ValueError(f"Model does not have native feature importance. Available attributes: {dir(model)}")
    
    # Ensure importance matches feature_names length
    if len(importance) != len(feature_names):
        logger.warning(f"Importance length ({len(importance)}) doesn't match features ({len(feature_names)})")
        # Pad or truncate if needed
        if len(importance) < len(feature_names):
            importance = np.pad(importance, (0, len(feature_names) - len(importance)), 'constant')
        else:
            importance = importance[:len(feature_names)]
    
    return pd.Series(importance, index=feature_names)


def extract_shap_importance(model, X: np.ndarray, feature_names: List[str],
                           max_samples: int = 1000) -> pd.Series:
    """Extract SHAP-based feature importance"""
    try:
        import shap
    except ImportError:
        logger.warning("SHAP not available, falling back to permutation importance")
        return extract_permutation_importance(model, X, None, feature_names)
    
    # Sample for computational efficiency
    if len(X) > max_samples:
        indices = np.random.choice(len(X), max_samples, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X
    
    try:
        # TreeExplainer for tree models
        if hasattr(model, 'tree_'):
            explainer = shap.TreeExplainer(model)
        else:
            # KernelExplainer for other models (slower)
            explainer = shap.KernelExplainer(model.predict, X_sample[:100])
        
        shap_values = explainer.shap_values(X_sample)
        
        # Handle multi-output or single output
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # Mean absolute SHAP value per feature
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
        return pd.Series(mean_abs_shap, index=feature_names)
    
    except Exception as e:
        logger.warning(f"SHAP extraction failed: {e}, falling back to permutation")
        return extract_permutation_importance(model, X, None, feature_names)


def extract_permutation_importance(model, X: np.ndarray, y: np.ndarray,
                                   feature_names: List[str],
                                   n_repeats: int = 5) -> pd.Series:
    """Extract permutation importance"""
    try:
        from sklearn.inspection import permutation_importance
        
        # Need y for permutation importance
        if y is None:
            logger.warning("No y provided for permutation importance, returning zeros")
            return pd.Series(0.0, index=feature_names)
        
        result = permutation_importance(
            model, X, y,
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=1
        )
        
        return pd.Series(result.importances_mean, index=feature_names)
    
    except Exception as e:
        logger.error(f"Permutation importance failed: {e}")
        return pd.Series(0.0, index=feature_names)


def train_model_and_get_importance(
    model_family: str,
    family_config: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    data_interval_minutes: int = 5  # Data bar interval (default: 5 minutes)
) -> Tuple[Any, pd.Series, str]:
    """Train a single model family and extract importance"""
    
    # Validate target before training
    sys.path.insert(0, str(_REPO_ROOT / "scripts" / "utils"))
    try:
        from target_validation import validate_target
        is_valid, error_msg = validate_target(y, min_samples=10, min_class_samples=2)
        if not is_valid:
            logger.debug(f"    {model_family}: {error_msg}")
            return None, pd.Series(0.0, index=feature_names), family_config['importance_method'], 0.0
    except ImportError:
        # Fallback
        unique_vals = np.unique(y[~np.isnan(y)])
        if len(unique_vals) < 2:
            return None, pd.Series(0.0, index=feature_names), family_config['importance_method'], 0.0
    
    importance_method = family_config['importance_method']
    model_config = family_config['config']
    
    # Train model based on family
    if model_family == 'lightgbm':
        lgb_data = lgb.Dataset(X, label=y, feature_name=feature_names)
        model = lgb.train(
            params=model_config,
            train_set=lgb_data,
            num_boost_round=model_config.get('n_estimators', 300),
            callbacks=[lgb.log_evaluation(period=0)]
        )
        train_score = model.best_score.get('training', {}).get(model_config.get('metric', 'l1'), 0.0)
    
    elif model_family == 'xgboost':
        try:
            import xgboost as xgb
            model = xgb.XGBRegressor(**model_config)
            try:
                model.fit(X, y)
                train_score = model.score(X, y)
            except (ValueError, TypeError) as e:
                error_str = str(e).lower()
                if any(kw in error_str for kw in ['invalid classes', 'expected', 'too few']):
                    logger.debug(f"    XGBoost: Target degenerate")
                    return None, pd.Series(0.0, index=feature_names), importance_method, 0.0
                raise
        except ImportError:
            logger.error("XGBoost not available")
            return None, pd.Series(0.0, index=feature_names), importance_method, 0.0
    
    elif model_family == 'random_forest':
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(**model_config, random_state=42)
        model.fit(X, y)
        train_score = model.score(X, y)
    
    elif model_family == 'neural_network':
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        
        # Handle NaN values (neural networks can't handle them)
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        
        # Scale for neural networks
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        
        model = MLPRegressor(**model_config, random_state=42)
        try:
            model.fit(X_scaled, y)
            train_score = model.score(X_scaled, y)
        except (ValueError, TypeError) as e:
            error_str = str(e).lower()
            if any(kw in error_str for kw in ['least populated class', 'too few', 'invalid classes']):
                logger.debug(f"    Neural Network: Target too imbalanced")
                return None, pd.Series(0.0, index=feature_names), importance_method, 0.0
            raise
        
        # Use scaled data for importance
        X = X_scaled
    
    elif model_family == 'catboost':
        try:
            import catboost as cb
            # Determine task type
            unique_vals = np.unique(y[~np.isnan(y)])
            is_binary = len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0})
            is_multiclass = len(unique_vals) <= 10 and all(
                isinstance(v, (int, np.integer)) or (isinstance(v, float) and v.is_integer())
                for v in unique_vals
            )
            
            if is_binary:
                model = cb.CatBoostClassifier(**model_config, verbose=False)
            elif is_multiclass:
                model = cb.CatBoostClassifier(**model_config, verbose=False)
            else:
                model = cb.CatBoostRegressor(**model_config, verbose=False)
            
            try:
                model.fit(X, y)
                train_score = model.score(X, y)
            except (ValueError, TypeError) as e:
                error_str = str(e).lower()
                if any(kw in error_str for kw in ['invalid classes', 'expected', 'too few']):
                    logger.debug(f"    CatBoost: Target degenerate")
                    return None, pd.Series(0.0, index=feature_names), importance_method, 0.0
                raise
        except ImportError:
            logger.error("CatBoost not available (pip install catboost)")
            return None, pd.Series(0.0, index=feature_names), importance_method, 0.0
    
    elif model_family == 'lasso':
        from sklearn.linear_model import Lasso
        model = Lasso(**model_config, random_state=42)
        model.fit(X, y)
        train_score = model.score(X, y)
    
    elif model_family == 'mutual_information':
        # Mutual information doesn't train a model, just calculates information
        from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
        # Determine task type
        unique_vals = np.unique(y[~np.isnan(y)])
        is_binary = len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0})
        is_multiclass = len(unique_vals) <= 10 and all(
            isinstance(v, (int, np.integer)) or (isinstance(v, float) and v.is_integer())
            for v in unique_vals
        )
        
        if is_binary or is_multiclass:
            importance_values = mutual_info_classif(X, y, random_state=42, discrete_features='auto')
        else:
            importance_values = mutual_info_regression(X, y, random_state=42, discrete_features='auto')
        
        # Create a dummy model for compatibility (mutual info doesn't need a model)
        class DummyModel:
            def __init__(self, importance):
                self.importance = importance
            def get_feature_importance(self):
                return self.importance
        
        model = DummyModel(importance_values)
        train_score = 0.0  # No model to score
    
    elif model_family == 'univariate_selection':
        # Univariate Feature Selection (f_regression/f_classif)
        from sklearn.feature_selection import f_regression, f_classif, chi2
        
        # Determine task type
        unique_vals = np.unique(y[~np.isnan(y)])
        is_binary = len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0})
        is_multiclass = len(unique_vals) <= 10 and all(
            isinstance(v, (int, np.integer)) or (isinstance(v, float) and v.is_integer())
            for v in unique_vals
        )
        
        if is_binary or is_multiclass:
            scores, pvalues = f_classif(X, y)
        else:
            scores, pvalues = f_regression(X, y)
        
        # Normalize scores (F-statistics can be very large)
        if np.max(scores) > 0:
            importance_values = scores / np.max(scores)
        else:
            importance_values = scores
        
        class DummyModel:
            def __init__(self, importance):
                self.importance = importance
        
        model = DummyModel(importance_values)
        train_score = 0.0  # No model to score
    
    elif model_family == 'rfe':
        # Recursive Feature Elimination
        from sklearn.feature_selection import RFE
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        
        # Determine task type
        unique_vals = np.unique(y[~np.isnan(y)])
        is_binary = len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0})
        is_multiclass = len(unique_vals) <= 10 and all(
            isinstance(v, (int, np.integer)) or (isinstance(v, float) and v.is_integer())
            for v in unique_vals
        )
        
        n_features_to_select = min(model_config.get('n_features_to_select', 50), len(feature_names))
        step = model_config.get('step', 5)
        
        if is_binary or is_multiclass:
            estimator = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=1)
        else:
            estimator = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=1)
        
        selector = RFE(estimator, n_features_to_select=n_features_to_select, step=step)
        selector.fit(X, y)
        
        # Convert ranking to importance (lower rank = more important)
        # RFE ranking: 1 = selected, higher = eliminated
        # Convert to importance: 1/rank (higher importance for lower rank)
        ranking = selector.ranking_
        importance_values = 1.0 / (ranking + 1e-6)  # Avoid division by zero
        
        class DummyModel:
            def __init__(self, importance):
                self.importance = importance
        
        model = DummyModel(importance_values)
        train_score = selector.estimator_.score(X, y) if hasattr(selector, 'estimator_') else 0.0
    
    elif model_family == 'boruta':
        # Boruta - All-relevant feature selection
        try:
            from boruta import BorutaPy
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            
            # Determine task type
            unique_vals = np.unique(y[~np.isnan(y)])
            is_binary = len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0})
            is_multiclass = len(unique_vals) <= 10 and all(
                isinstance(v, (int, np.integer)) or (isinstance(v, float) and v.is_integer())
                for v in unique_vals
            )
            
            if is_binary or is_multiclass:
                rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=1)
            else:
                rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=1)
            
            boruta = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=42, max_iter=model_config.get('max_iter', 100))
            boruta.fit(X, y)
            
            # Convert to importance: selected features get high importance, rejected get low
            ranking = boruta.ranking_
            selected = boruta.support_
            
            # Importance: selected=1.0, tentative=0.5, rejected=0.1
            importance_values = np.where(selected, 1.0, np.where(ranking == 2, 0.5, 0.1))
            
            class DummyModel:
                def __init__(self, importance):
                    self.importance = importance
            
            model = DummyModel(importance_values)
            train_score = rf.score(X, y) if hasattr(rf, 'score') else 0.0
        except ImportError:
            logger.error("Boruta not available (pip install Boruta)")
            return None, pd.Series(0.0, index=feature_names), importance_method, 0.0
    
    elif model_family == 'stability_selection':
        # Stability Selection - Bootstrap-based feature selection
        from sklearn.linear_model import LassoCV
        from sklearn.linear_model import LogisticRegressionCV
        
        # Determine task type
        unique_vals = np.unique(y[~np.isnan(y)])
        is_binary = len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0})
        is_multiclass = len(unique_vals) <= 10 and all(
            isinstance(v, (int, np.integer)) or (isinstance(v, float) and v.is_integer())
            for v in unique_vals
        )
        
        # CRITICAL: Use PurgedTimeSeriesSplit to prevent temporal leakage
        from scripts.utils.purged_time_series_split import PurgedTimeSeriesSplit
        from scripts.utils.leakage_filtering import _extract_horizon, _load_leakage_config
        
        # Calculate purge_overlap from target horizon
        # CRITICAL: Use the data_interval_minutes parameter (detected in calling function)
        # Using wrong interval (e.g., assuming 5m when data is 1m) causes severe leakage
        purge_buffer_bars = 5  # Safety buffer
        
        leakage_config = _load_leakage_config()
        target_horizon_minutes = _extract_horizon(target_column, leakage_config) if target_column else None
        
        if target_horizon_minutes is not None:
            target_horizon_bars = target_horizon_minutes // data_interval_minutes
            purge_overlap = target_horizon_bars + purge_buffer_bars
        else:
            # Fallback: conservative default (60m = 12 bars + 5 buffer)
            purge_overlap = 17
        
        # Create purged CV splitter
        # NOTE: Using row-count based purging (legacy). For better accuracy, use time-based purging:
        # purged_cv = PurgedTimeSeriesSplit(n_splits=3, purge_overlap_time=pd.Timedelta(minutes=target_horizon_minutes), time_column_values=timestamps)
        purged_cv = PurgedTimeSeriesSplit(n_splits=3, purge_overlap=purge_overlap)
        
        n_bootstrap = model_config.get('n_bootstrap', 50)  # Reduced for speed
        stability_scores = np.zeros(X.shape[1])
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_boot, y_boot = X[indices], y[indices]
            
            try:
                if is_binary or is_multiclass:
                    model = LogisticRegressionCV(Cs=10, cv=purged_cv, random_state=42, max_iter=1000, n_jobs=1)
                else:
                    model = LassoCV(cv=purged_cv, random_state=42, max_iter=1000, n_jobs=1)
                
                model.fit(X_boot, y_boot)
                stability_scores += (np.abs(model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_) > 1e-6).astype(int)
            except:
                continue  # Skip failed bootstrap iterations
        
        # Normalize to 0-1 (fraction of times selected)
        importance_values = stability_scores / n_bootstrap
        
        class DummyModel:
            def __init__(self, importance):
                self.importance = importance
        
        model = DummyModel(importance_values)
        train_score = 0.0  # No single model to score
    
    else:
        logger.error(f"Unknown model family: {model_family}")
        return None, pd.Series(0.0, index=feature_names), importance_method, 0.0
    
    # Extract importance based on method
    if importance_method == 'native':
        importance = extract_native_importance(model, feature_names)
    elif importance_method == 'shap':
        importance = extract_shap_importance(model, X, feature_names)
    elif importance_method == 'permutation':
        importance = extract_permutation_importance(model, X, y, feature_names)
    else:
        logger.error(f"Unknown importance method: {importance_method}")
        importance = pd.Series(0.0, index=feature_names)
    
    return model, importance, importance_method, train_score


def process_single_symbol(
    symbol: str,
    data_path: Path,
    target_column: str,
    model_families_config: Dict[str, Dict[str, Any]],
    max_samples: int = 50000
) -> List[ImportanceResult]:
    """Process a single symbol with multiple model families"""
    
    results = []
    
    try:
        # Load data
        df = safe_load_dataframe(data_path)
        
        # Validate target
        if target_column not in df.columns:
            logger.warning(f"Skipping {symbol}: Target '{target_column}' not found")
            return results
        
        # Drop NaN in target
        df = df.dropna(subset=[target_column])
        if df.empty:
            logger.warning(f"Skipping {symbol}: No valid data after dropping NaN")
            return results
        
        # Sample if too large
        if len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42)
        
        # LEAKAGE PREVENTION: Filter out leaking features
        sys.path.insert(0, str(_REPO_ROOT / "scripts"))
        from filter_leaking_features import filter_features
        
        all_columns = df.columns.tolist()
        safe_columns = filter_features(all_columns, verbose=False)
        
        # Keep only safe features + target
        safe_columns_with_target = [c for c in safe_columns if c != target_column] + [target_column]
        df = df[safe_columns_with_target]
        
        # Prepare features (target already in safe list, so exclude it explicitly)
        X = df.drop(columns=[target_column], errors='ignore')
        
        # Drop object dtypes
        object_cols = X.select_dtypes(include=['object']).columns.tolist()
        if object_cols:
            X = X.drop(columns=object_cols)
        
        y = df[target_column]
        feature_names = X.columns.tolist()
        
        if not feature_names:
            logger.warning(f"Skipping {symbol}: No features after filtering")
            return results
        
        # Convert to numpy
        X_arr = X.to_numpy()
        y_arr = y.to_numpy()
        
        # CRITICAL: Auto-detect data interval to prevent leakage in PurgedTimeSeriesSplit
        from scripts.utils.data_interval import detect_interval_from_dataframe
        detected_interval = detect_interval_from_dataframe(df, timestamp_column='ts', default=5)
        if detected_interval != 5:
            logger.info(f"  Detected data interval: {detected_interval}m (was assuming 5m)")
        
        # Train each enabled model family
        for family_name, family_config in model_families_config.items():
            if not family_config.get('enabled', False):
                continue
            
            try:
                logger.info(f"  {symbol}: Training {family_name}...")
                model, importance, method, train_score = train_model_and_get_importance(
                    family_name, family_config, X_arr, y_arr, feature_names,
                    data_interval_minutes=detected_interval
                )
                
                if importance is not None and importance.sum() > 0:
                    result = ImportanceResult(
                        model_family=family_name,
                        symbol=symbol,
                        importance_scores=importance,
                        method=method,
                        train_score=train_score
                    )
                    results.append(result)
                    logger.info(f"    {family_name}: score={train_score:.4f}, "
                              f"top feature={importance.idxmax()} ({importance.max():.2f})")
                
            except Exception as e:
                logger.error(f"  {symbol}: {family_name} failed: {e}")
                continue
        
        logger.info(f"✅ {symbol}: Completed {len(results)}/{len(model_families_config)} models")
        
    except Exception as e:
        logger.error(f"❌ {symbol}: Processing failed: {e}", exc_info=True)
    
    return results


def aggregate_multi_model_importance(
    all_results: List[ImportanceResult],
    model_families_config: Dict[str, Dict[str, Any]],
    aggregation_config: Dict[str, Any],
    top_n: Optional[int] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Aggregate feature importance across models AND symbols
    
    Strategy:
    1. Group by model family
    2. Aggregate within each family across symbols
    3. Weight by family weight
    4. Combine across families
    5. Rank by consensus
    """
    
    if not all_results:
        return pd.DataFrame(), []
    
    # Group results by model family
    family_results = defaultdict(list)
    for result in all_results:
        family_results[result.model_family].append(result)
    
    # Aggregate within each family
    family_scores = {}
    for family_name, results in family_results.items():
        # Combine importances across symbols for this family
        importances_df = pd.concat(
            [r.importance_scores for r in results],
            axis=1,
            sort=False
        ).fillna(0)
        
        # Aggregate across symbols (mean by default)
        method = aggregation_config.get('per_symbol_method', 'mean')
        if method == 'mean':
            family_score = importances_df.mean(axis=1)
        elif method == 'median':
            family_score = importances_df.median(axis=1)
        else:
            family_score = importances_df.mean(axis=1)
        
        # Apply family weight
        weight = model_families_config[family_name].get('weight', 1.0)
        family_scores[family_name] = family_score * weight
        
        logger.info(f"📊 {family_name}: Aggregated {len(results)} symbols, "
                   f"weight={weight}, top={family_score.idxmax()}")
    
    # Combine across families
    combined_df = pd.DataFrame(family_scores)
    
    # Calculate consensus score
    cross_model_method = aggregation_config.get('cross_model_method', 'weighted_mean')
    if cross_model_method == 'weighted_mean':
        consensus_score = combined_df.mean(axis=1)
    elif cross_model_method == 'median':
        consensus_score = combined_df.median(axis=1)
    elif cross_model_method == 'geometric_mean':
        # Geometric mean (good for multiplicative effects)
        consensus_score = np.exp(np.log(combined_df + 1e-10).mean(axis=1))
    else:
        consensus_score = combined_df.mean(axis=1)
    
    # Calculate consensus metrics
    n_models = combined_df.shape[1]
    frequency = (combined_df > 0).sum(axis=1)
    frequency_pct = (frequency / n_models) * 100
    
    # Standard deviation across models (lower = more consensus)
    consensus_std = combined_df.std(axis=1)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame({
        'feature': consensus_score.index,
        'consensus_score': consensus_score.values,
        'n_models_agree': frequency,
        'consensus_pct': frequency_pct,
        'std_across_models': consensus_std,
    })
    
    # Add per-family scores
    for family_name in family_scores.keys():
        summary_df[f'{family_name}_score'] = combined_df[family_name].values
    
    # Sort by consensus score
    summary_df = summary_df.sort_values('consensus_score', ascending=False).reset_index(drop=True)
    
    # Filter by minimum consensus if specified
    min_models = aggregation_config.get('require_min_models', 1)
    summary_df = summary_df[summary_df['n_models_agree'] >= min_models]
    
    # Select top N
    if top_n:
        summary_df = summary_df.head(top_n)
    
    selected_features = summary_df['feature'].tolist()
    
    return summary_df, selected_features


def save_multi_model_results(
    summary_df: pd.DataFrame,
    selected_features: List[str],
    all_results: List[ImportanceResult],
    output_dir: Path,
    metadata: Dict[str, Any]
):
    """Save multi-model feature selection results"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Selected features list
    with open(output_dir / "selected_features.txt", "w") as f:
        for feature in selected_features:
            f.write(f"{feature}\n")
    logger.info(f"✅ Saved {len(selected_features)} features to selected_features.txt")
    
    # 2. Detailed summary CSV
    summary_df.to_csv(output_dir / "feature_importance_multi_model.csv", index=False)
    logger.info(f"✅ Saved detailed multi-model summary to feature_importance_multi_model.csv")
    
    # 3. Per-model-family breakdowns
    for family_name in summary_df.columns:
        if family_name.endswith('_score') and family_name not in ['consensus_score']:
            family_df = summary_df[['feature', family_name]].copy()
            family_df = family_df.sort_values(family_name, ascending=False)
            family_csv = output_dir / f"importance_{family_name.replace('_score', '')}.csv"
            family_df.to_csv(family_csv, index=False)
    
    # 4. Model agreement matrix
    model_families = list(set(r.model_family for r in all_results))
    agreement_matrix = pd.DataFrame(
        index=selected_features[:20],  # Top 20 for readability
        columns=model_families
    )
    
    for result in all_results:
        for feature in selected_features[:20]:
            if feature in result.importance_scores.index:
                current = agreement_matrix.loc[feature, result.model_family]
                score = result.importance_scores[feature]
                if pd.isna(current):
                    agreement_matrix.loc[feature, result.model_family] = score
                else:
                    agreement_matrix.loc[feature, result.model_family] = max(current, score)
    
    agreement_matrix.to_csv(output_dir / "model_agreement_matrix.csv")
    logger.info(f"✅ Saved model agreement matrix")
    
    # 5. Metadata JSON
    metadata['n_selected_features'] = len(selected_features)
    metadata['n_total_results'] = len(all_results)
    metadata['model_families_used'] = list(set(r.model_family for r in all_results))
    
    with open(output_dir / "multi_model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✅ Saved metadata")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Model Feature Selection: Find robust features across model families"
    )
    parser.add_argument("--symbols", type=str, 
                       help="Comma-separated symbols (default: all in data_dir)")
    parser.add_argument("--data-dir", type=Path,
                       default=_REPO_ROOT / "data/data_labeled/interval=5m",
                       help="Directory with labeled data")
    parser.add_argument("--output-dir", type=Path,
                       default=_REPO_ROOT / "DATA_PROCESSING/data/features/multi_model",
                       help="Output directory")
    parser.add_argument("--target-column", type=str,
                       default="y_will_peak_60m_0.8",
                       help="Target column for training")
    parser.add_argument("--top-n", type=int, default=60,
                       help="Number of features to select")
    parser.add_argument("--config", type=Path,
                       help="Path to multi-model config YAML")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Parallel workers (sequential per symbol)")
    parser.add_argument("--enable-families", type=str,
                       help="Comma-separated families to enable (e.g., lightgbm,xgboost,neural_network)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from checkpoint if available")
    parser.add_argument("--clear-checkpoint", action="store_true",
                       help="Clear existing checkpoint and start fresh")
    
    args = parser.parse_args()
    
    # Load config
    config = load_multi_model_config(args.config)
    
    # Override enabled families if specified
    if args.enable_families:
        enabled = [f.strip() for f in args.enable_families.split(',')]
        for family in config['model_families']:
            config['model_families'][family]['enabled'] = family in enabled
    
    # Count enabled families
    enabled_families = [f for f, cfg in config['model_families'].items() if cfg.get('enabled')]
    
    logger.info("="*80)
    logger.info("🚀 Multi-Model Feature Selection Pipeline")
    logger.info("="*80)
    logger.info(f"Target: {args.target_column}")
    logger.info(f"Top N: {args.top_n}")
    logger.info(f"Enabled model families ({len(enabled_families)}): {', '.join(enabled_families)}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("-"*80)
    
    # Find symbols
    if not args.data_dir.exists():
        logger.error(f"❌ Data directory not found: {args.data_dir}")
        return 1
    
    symbol_dirs = [d for d in args.data_dir.glob("symbol=*") if d.is_dir()]
    labeled_files = []
    for symbol_dir in symbol_dirs:
        symbol_name = symbol_dir.name.replace("symbol=", "")
        parquet_file = symbol_dir / f"{symbol_name}.parquet"
        if parquet_file.exists():
            labeled_files.append((symbol_name, parquet_file))
    
    if args.symbols:
        requested = [s.upper().strip() for s in args.symbols.split(',')]
        labeled_files = [(sym, path) for sym, path in labeled_files if sym.upper() in requested]
    
    if not labeled_files:
        logger.error("❌ No labeled files found")
        return 1
    
    logger.info(f"📊 Processing {len(labeled_files)} symbols")
    
    # Initialize checkpoint manager
    checkpoint_file = args.output_dir / "checkpoint.json"
    checkpoint = CheckpointManager(
        checkpoint_file=checkpoint_file,
        item_key_fn=lambda item: item if isinstance(item, str) else item[0]  # symbol name
    )
    
    # Clear checkpoint if requested
    if args.clear_checkpoint:
        checkpoint.clear()
        logger.info("Cleared checkpoint - starting fresh")
    
    # Load completed symbols
    completed = checkpoint.load_completed()
    logger.info(f"Found {len(completed)} completed symbols in checkpoint")
    
    # Process symbols (sequential to avoid GPU/memory conflicts)
    all_results = []
    for i, (symbol, path) in enumerate(labeled_files, 1):
        # Check if already completed
        if symbol in completed:
            if args.resume:
                logger.info(f"\n[{i}/{len(labeled_files)}] Skipping {symbol} (already completed)")
                symbol_results = completed[symbol]
                if isinstance(symbol_results, list):
                    # Reconstruct ImportanceResult objects from dicts
                    for r_dict in symbol_results:
                        # Convert importance_scores dict back to pd.Series
                        if isinstance(r_dict.get('importance_scores'), dict):
                            r_dict['importance_scores'] = pd.Series(r_dict['importance_scores'])
                        all_results.append(ImportanceResult(**r_dict))
                continue
            elif not args.resume:
                continue
        
        logger.info(f"\n[{i}/{len(labeled_files)}] Processing {symbol}...")
        try:
            results = process_single_symbol(
                symbol, path, args.target_column,
                config['model_families'],
                config['sampling']['max_samples_per_symbol']
            )
            all_results.extend(results)
            
            # Save checkpoint after each symbol
            # Convert results to dict for serialization (handle pd.Series)
            results_dict = []
            for r in results:
                r_dict = asdict(r)
                # Convert pd.Series to dict
                if isinstance(r_dict.get('importance_scores'), pd.Series):
                    r_dict['importance_scores'] = r_dict['importance_scores'].to_dict()
                results_dict.append(r_dict)
            checkpoint.save_item(symbol, results_dict)
        except Exception as e:
            logger.error(f"  Failed to process {symbol}: {e}")
            checkpoint.mark_failed(symbol, str(e))
            continue
    
    if not all_results:
        logger.error("❌ No results collected")
        return 1
    
    logger.info(f"\n{'='*80}")
    logger.info(f"📈 Aggregating {len(all_results)} model results...")
    logger.info(f"{'='*80}")
    
    # Aggregate across models and symbols
    summary_df, selected_features = aggregate_multi_model_importance(
        all_results,
        config['model_families'],
        config['aggregation'],
        args.top_n
    )
    
    if summary_df.empty:
        logger.error("❌ No features selected")
        return 1
    
    # Save results
    metadata = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'target_column': args.target_column,
        'top_n': args.top_n,
        'n_symbols': len(labeled_files),
        'enabled_families': enabled_families,
        'config': config
    }
    
    save_multi_model_results(
        summary_df, selected_features, all_results,
        args.output_dir, metadata
    )
    
    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info("✅ Multi-Model Feature Selection Complete!")
    logger.info(f"{'='*80}")
    logger.info(f"\n📊 Top 10 Features by Consensus:")
    for i, row in summary_df.head(10).iterrows():
        logger.info(f"  {i+1:2d}. {row['feature']:30s} | "
                   f"score={row['consensus_score']:8.2f} | "
                   f"agree={row['n_models_agree']}/{len(enabled_families)} | "
                   f"std={row['std_across_models']:6.2f}")
    
    logger.info(f"\n📁 Output files:")
    logger.info(f"  • {args.output_dir}/selected_features.txt")
    logger.info(f"  • {args.output_dir}/feature_importance_multi_model.csv")
    logger.info(f"  • {args.output_dir}/model_agreement_matrix.csv")
    logger.info(f"  • {args.output_dir}/importance_<family>.csv (per-family)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

