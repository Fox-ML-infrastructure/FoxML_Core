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
import inspect
import logging
import math
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
import pandas as pd
import numpy as np
import lightgbm as lgb
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import json
from collections import defaultdict
from dataclasses import dataclass, asdict
import warnings

# Add project root FIRST (before any TRAINING.* imports)
# TRAINING/ranking/multi_model_feature_selection.py -> parents[2] = repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# CRITICAL: Set up determinism BEFORE importing any ML libraries
# This ensures reproducible results across runs
try:
    from CONFIG.config_loader import get_cfg
    base_seed = get_cfg("pipeline.determinism.base_seed", default=42)
except ImportError:
    base_seed = 42  # FALLBACK_DEFAULT_OK

# Import determinism system FIRST (before any ML libraries)
from TRAINING.common.determinism import set_global_determinism, seed_for, stable_seed_from

# Set global determinism immediately
BASE_SEED = set_global_determinism(
    base_seed=base_seed,
    threads=None,  # Auto-detect optimal thread count
    deterministic_algorithms=False,  # Allow parallel algorithms for performance
    prefer_cpu_tree_train=False,  # Use GPU when available
    tf_on=False,  # TensorFlow not needed for feature selection
    strict_mode=False  # Allow optimizations
)

from CONFIG.config_loader import load_model_config
import yaml

# Import checkpoint utility (after path is set)
from TRAINING.utils.checkpoint import CheckpointManager
# Setup logging with journald support (after path is set)
from TRAINING.utils.logging_setup import setup_logging
logger = setup_logging(
    script_name="multi_model_feature_selection",
    level=logging.INFO,
    use_journald=True
)

# Suppress warnings from SHAP/sklearn
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='X does not have valid feature names')


# Import shared config cleaner utility
from TRAINING.utils.config_cleaner import clean_config_for_estimator as _clean_config_for_estimator


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


def normalize_importance(
    raw_importance: Optional[Union[np.ndarray, pd.Series]],
    n_features: int,
    family: str,
    feature_names: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, Optional[str]]:
    """
    Normalize and sanitize importance vector with fallback handling for no-signal cases.
    
    This function ensures importance vectors are always valid (non-None, correct shape, no NaN/inf)
    and provides a uniform fallback when there's truly no signal, preventing InvalidImportance errors.
    
    Args:
        raw_importance: Raw importance values (can be None, array, or Series)
        n_features: Expected number of features
        family: Model family name (for logging)
        feature_names: Optional feature names (for Series conversion)
        config: Optional config dict with fallback settings (from aggregation.fallback)
    
    Returns:
        Tuple of (normalized_importance, fallback_reason)
        - normalized_importance: np.ndarray of shape (n_features,) with non-zero sum
        - fallback_reason: None if no fallback used, or string reason if fallback applied
    """
    # Load fallback config from SST
    try:
        from CONFIG.config_loader import get_cfg
        fallback_cfg = get_cfg("preprocessing.multi_model_feature_selection.aggregation.fallback", default={}, config_name="preprocessing_config")
        uniform_importance = fallback_cfg.get('uniform_importance', 1e-6)
        normalize_after_fallback = fallback_cfg.get('normalize_after_fallback', True)
    except Exception as e:
        # Fallback defaults if config unavailable
        logger.debug(f"Failed to load fallback config: {e}, using defaults")
        uniform_importance = config.get('uniform_importance', 1e-6) if config else 1e-6
        normalize_after_fallback = config.get('normalize_after_fallback', True) if config else True
    
    # Handle None / empty
    if raw_importance is None:
        importance = np.zeros(n_features, dtype=float)
    else:
        # Convert to numpy array
        if isinstance(raw_importance, pd.Series):
            importance = raw_importance.values
        else:
            importance = np.asarray(raw_importance, dtype=float)
        
        # Flatten if needed
        importance = importance.flatten()
    
    # Clean NaN / inf
    importance = np.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Pad / truncate to match feature count
    if importance.size < n_features:
        importance = np.pad(importance, (0, n_features - importance.size), mode='constant', constant_values=0.0)
    elif importance.size > n_features:
        importance = importance[:n_features]
    
    # Fallback if truly no signal (all zeros)
    if not np.any(importance > 0):
        # No signal: treat as "no strong preference" instead of failure
        importance = np.full(n_features, uniform_importance, dtype=float)
        fallback_reason = f"{family}:fallback_uniform_no_signal"
        
        # Optional: normalize to sum to 1.0
        if normalize_after_fallback:
            importance = importance / importance.sum()
    else:
        fallback_reason = None
    
    # Final normalization: ensure sum > 0 (should always be true after fallback, but defensive)
    s = float(importance.sum())
    if s > 0 and normalize_after_fallback and fallback_reason is None:
        # Normalize existing signal (but not if we just applied uniform fallback)
        importance = importance / s
    
    # Guarantee: sum must be > 0
    assert importance.sum() > 0, f"Importance sum should be positive after normalization (family={family})"
    
    return importance, fallback_reason


def boruta_to_importance(
    support: np.ndarray,
    support_weak: Optional[np.ndarray] = None,
    ranking: Optional[np.ndarray] = None,
    n_features: Optional[int] = None,
) -> Optional[np.ndarray]:
    """
    Build a robust importance vector from Boruta outputs.
    
    This function ensures that when Boruta identifies confirmed or tentative features,
    the resulting importance array will have non-zero entries and a positive sum,
    preventing false "InvalidImportance" errors.
    
    Returns values compatible with gatekeeper logic:
    - confirmed: 1.0
    - tentative: 0.3
    - rejected: 0.0 (not -1.0, to ensure positive sum)
    
    Args:
        support: Boolean mask of confirmed features (True=confirmed, False=rejected/tentative)
        support_weak: Optional boolean mask of tentative features
        ranking: Optional integer ranking array (1=confirmed, 2=tentative, >2=rejected)
        n_features: Total number of features (inferred from support if not provided)
    
    Returns:
        importance: np.ndarray of shape (n_features,) with non-zero entries
                    for confirmed/tentative features, or None if truly no signal.
                    Values: confirmed=1.0, tentative=0.3, rejected=0.0
    
    Notes:
        - Only returns None when Boruta truly selects nothing (no confirmed, no tentative)
        - Guarantees sum(importance) > 0 when any features are confirmed/tentative
        - Uses gatekeeper-compatible scoring (1.0/0.3/0.0) instead of normalized values
        - Rejected features get 0.0 (not -1.0) to ensure positive sum for validation
    """
    support = np.asarray(support, dtype=bool)
    if n_features is None:
        n_features = support.shape[0]

    if support_weak is None:
        support_weak = np.zeros_like(support, dtype=bool)
    else:
        support_weak = np.asarray(support_weak, dtype=bool)

    # Defensive: ensure correct length
    if support.shape[0] != n_features:
        raise ValueError(f"support length {support.shape[0]} != n_features {n_features}")
    if support_weak.shape[0] != n_features:
        raise ValueError(f"support_weak length {support_weak.shape[0]} != n_features {n_features}")

    has_confirmed = support.any()
    has_tentative = support_weak.any()

    # Case 1: nothing selected at all → let caller handle as "no signal"
    if not has_confirmed and not has_tentative:
        return None

    # Initialize with zeros (rejected features)
    importance = np.zeros(n_features, dtype=float)
    
    # Assign importance: confirmed=1.0, tentative=0.3
    # Note: We use 0.0 for rejected (not -1.0) to ensure positive sum for validation
    # The gatekeeper logic will apply penalties separately in aggregation
    importance[support] = 1.0  # Confirmed features
    importance[support_weak] = 0.3  # Tentative features
    
    # If both confirmed and tentative exist, confirmed takes precedence
    # (support_weak should not overlap with support, but be defensive)
    overlap = support & support_weak
    if overlap.any():
        # If there's overlap, confirmed takes precedence (1.0 > 0.3)
        importance[overlap] = 1.0

    # Final safety check: ensure we have positive values
    if importance.sum() <= 0:
        # Something is very off; treat as "no signal"
        return None

    # Guarantee: if we have confirmed/tentative, sum must be > 0
    assert importance.sum() > 0, "Importance sum should be positive when features are selected"
    
    return importance


def load_multi_model_config(config_path: Path = None) -> Dict[str, Any]:
    """Load multi-model feature selection configuration
    
    Checks new location first (CONFIG/feature_selection/multi_model.yaml),
    then falls back to legacy location (CONFIG/multi_model_feature_selection.yaml).
    """
    if config_path is None:
        # Try new location first
        new_path = _REPO_ROOT / "CONFIG" / "feature_selection" / "multi_model.yaml"
        legacy_path = _REPO_ROOT / "CONFIG" / "multi_model_feature_selection.yaml"
        
        if new_path.exists():
            config_path = new_path
            logger.debug(f"Using new config location: {config_path}")
        elif legacy_path.exists():
            config_path = legacy_path
            logger.warning(
                f"⚠️  DEPRECATED: Using legacy config location: {legacy_path}\n"
                f"   Please migrate to: CONFIG/feature_selection/multi_model.yaml"
            )
        else:
            logger.warning(f"Config not found in new ({new_path}) or legacy ({legacy_path}) locations, using defaults")
            return get_default_config()
    
    if not config_path.exists():
        logger.warning(f"Config not found: {config_path}, using defaults")
        return get_default_config()
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Inject global defaults from defaults.yaml (SST)
    # This centralizes common settings like random_state, n_jobs, etc.
    try:
        from CONFIG.config_loader import inject_defaults
        
        # Inject defaults into each model family config
        if 'model_families' in config and config['model_families']:
            for family_name, family_config in config['model_families'].items():
                if 'config' in family_config and family_config['config'] is not None:
                    family_config['config'] = inject_defaults(
                        family_config['config'], 
                        model_family=family_name
                    )
                elif 'config' not in family_config or family_config.get('config') is None:
                    # Initialize empty config if missing/None
                    family_config['config'] = inject_defaults({}, model_family=family_name)
        
        # Inject defaults into top-level sections
        if 'sampling' in config and config.get('sampling') is not None:
            config['sampling'] = inject_defaults(config['sampling'])
        if 'permutation' in config and config.get('permutation') is not None:
            config['permutation'] = inject_defaults(config['permutation'])
            
    except Exception as e:
        logger.warning(f"Failed to inject defaults: {e}, continuing without defaults")
    
    logger.info(f"Loaded multi-model config from {config_path}")
    return config


def get_default_config() -> Dict[str, Any]:
    """Default configuration if file doesn't exist"""
    # Load default max_samples from config
    try:
        from CONFIG.config_loader import get_cfg, load_model_config
        default_max_samples = int(get_cfg("pipeline.data_limits.default_max_samples_feature_selection", default=50000, config_name="pipeline_config"))
        validation_split = float(get_cfg("preprocessing.validation.test_size", default=0.2, config_name="preprocessing_config"))
        
        # Load aggregation settings from config
        agg_cfg = get_cfg("preprocessing.multi_model_feature_selection.aggregation", default={}, config_name="preprocessing_config")
        model_weights_cfg = get_cfg("preprocessing.multi_model_feature_selection.model_weights", default={}, config_name="preprocessing_config")
        rf_cfg = get_cfg("preprocessing.multi_model_feature_selection.random_forest", default={}, config_name="preprocessing_config")
        nn_cfg = get_cfg("preprocessing.multi_model_feature_selection.neural_network", default={}, config_name="preprocessing_config")
        
        # Load model configs (load_model_config returns hyperparameters directly, like Phase 3)
        try:
            lgb_hyperparams = load_model_config('lightgbm')
        except Exception as e:
            logger.debug(f"Failed to load LightGBM config: {e}, using empty config")
            lgb_hyperparams = {}

        try:
            xgb_hyperparams = load_model_config('xgboost')
        except Exception as e:
            logger.debug(f"Failed to load XGBoost config: {e}, using empty config")
            xgb_hyperparams = {}

        try:
            mlp_hyperparams = load_model_config('mlp')
        except Exception as e:
            logger.debug(f"Failed to load MLP config: {e}, using empty config")
            mlp_hyperparams = {}
    except Exception as e:
        logger.warning(f"Failed to load default config values: {e}, using hardcoded defaults")
        default_max_samples = 50000
        validation_split = 0.2
        agg_cfg = {}
        model_weights_cfg = {}
        rf_cfg = {}
        nn_cfg = {}
        lgb_hyperparams = {}
        xgb_hyperparams = {}
        mlp_hyperparams = {}
    
    # Build model families config with defaults and config overrides
    return {
        'model_families': {
            'lightgbm': {
                'enabled': True,
                'importance_method': 'native',
                'weight': model_weights_cfg.get('lightgbm', 1.0),
                'config': {
                    'objective': 'regression_l1',
                    'metric': 'mae',
                    'n_estimators': lgb_hyperparams.get('n_estimators', 1000),  # Match Phase 3 default
                    'learning_rate': lgb_hyperparams.get('learning_rate', 0.03),  # Match Phase 3 default
                    'num_leaves': lgb_hyperparams.get('num_leaves', 96),  # Match Phase 3 default
                    'max_depth': lgb_hyperparams.get('max_depth', 8),  # Match Phase 3 default
                    'verbose': -1
                }
            },
            'xgboost': {
                'enabled': True,
                'importance_method': 'native',
                'weight': model_weights_cfg.get('xgboost', 1.0),
                'config': {
                    'objective': 'reg:squarederror',
                    'n_estimators': xgb_hyperparams.get('n_estimators', 1000),  # Match Phase 3 default
                    'learning_rate': xgb_hyperparams.get('eta', xgb_hyperparams.get('learning_rate', 0.03)),  # Match Phase 3 default (eta is XGBoost's learning_rate)
                    'max_depth': xgb_hyperparams.get('max_depth', 7),  # Match Phase 3 default
                    'verbosity': 0
                }
            },
            'random_forest': {
                'enabled': True,
                'importance_method': 'native',
                'weight': model_weights_cfg.get('random_forest', 0.8),
                'config': {
                    # Load from preprocessing config (no model_config file yet)
                    'n_estimators': rf_cfg.get('n_estimators', 200),
                    'max_depth': rf_cfg.get('max_depth', 15),
                    'max_features': rf_cfg.get('max_features', 'sqrt'),
                    'n_jobs': rf_cfg.get('n_jobs', 4)
                }
            },
            'neural_network': {
                'enabled': True,
                'importance_method': 'permutation',
                'weight': model_weights_cfg.get('neural_network', 1.2),
                'config': {
                    'hidden_layer_sizes': tuple(mlp_hyperparams.get('hidden_layers', [256, 128, 64])),  # Match Phase 3 default
                    'max_iter': mlp_hyperparams.get('epochs', mlp_hyperparams.get('max_iter', 50)),  # Match Phase 3 default
                    'early_stopping': True,
                    'validation_fraction': nn_cfg.get('validation_fraction', 0.1)  # Load from config
                }
            }
        },
        'aggregation': {
            'per_symbol_method': agg_cfg.get('per_symbol_method', 'mean'),
            'cross_model_method': agg_cfg.get('cross_model_method', 'weighted_mean'),
            'require_min_models': agg_cfg.get('require_min_models', 2),
            'consensus_threshold': agg_cfg.get('consensus_threshold', 0.5),
            'boruta_confirm_bonus': agg_cfg.get('boruta_confirm_bonus', 0.2),
            'boruta_reject_penalty': agg_cfg.get('boruta_reject_penalty', -0.3),
            'boruta_confirmed_threshold': agg_cfg.get('boruta_confirmed_threshold', 0.9),
            'boruta_tentative_threshold': agg_cfg.get('boruta_tentative_threshold', 0.0),
            'boruta_magnitude_warning_threshold': agg_cfg.get('boruta_magnitude_warning_threshold', 0.5)
        },
        'sampling': {
            'max_samples_per_symbol': default_max_samples,
            'validation_split': validation_split
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
                           max_samples: int = None,
                           model_family: Optional[str] = None,
                           target_column: Optional[str] = None,
                           symbol: Optional[str] = None) -> pd.Series:
    """Extract SHAP-based feature importance"""
    # Load default max_samples for SHAP from config if not provided
    if max_samples is None:
        try:
            from CONFIG.config_loader import get_cfg
            max_samples = int(get_cfg("pipeline.data_limits.max_cs_samples", default=1000, config_name="pipeline_config"))
        except Exception as e:
            logger.debug(f"Failed to load max_cs_samples from config: {e}, using default=1000")
            max_samples = 1000
    
    try:
        import shap
    except ImportError:
        logger.warning("SHAP not available, falling back to permutation importance")
        return extract_permutation_importance(model, X, None, feature_names,
                                             model_family=model_family,
                                             target_column=target_column,
                                             symbol=symbol)
    
    # Sample for computational efficiency - use deterministic sampling
    if len(X) > max_samples:
        # Generate deterministic seed for SHAP sampling
        shap_sample_seed = stable_seed_from(['shap', 'sampling'])
        np.random.seed(shap_sample_seed)
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
            # Load sample size from config
            try:
                from CONFIG.config_loader import get_cfg
                kernel_sample_size = int(get_cfg("preprocessing.multi_model_feature_selection.shap.kernel_explainer_sample_size", default=100, config_name="preprocessing_config"))
            except Exception as e:
                logger.debug(f"Failed to load kernel_explainer_sample_size from config: {e}, using default=100")
                kernel_sample_size = 100
            explainer = shap.KernelExplainer(model.predict, X_sample[:kernel_sample_size])
        
        shap_values = explainer.shap_values(X_sample)
        
        # Handle multi-output or single output
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # Mean absolute SHAP value per feature
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
        return pd.Series(mean_abs_shap, index=feature_names)
    
    except Exception as e:
        logger.warning(f"SHAP extraction failed: {e}, falling back to permutation")
        return extract_permutation_importance(model, X, None, feature_names, 
                                               model_family=model_family if 'model_family' in locals() else None,
                                               target_column=target_column if 'target_column' in locals() else None,
                                               symbol=symbol if 'symbol' in locals() else None)


def extract_permutation_importance(model, X: np.ndarray, y: np.ndarray,
                                   feature_names: List[str],
                                   n_repeats: int = 5,
                                   model_family: Optional[str] = None,
                                   target_column: Optional[str] = None,
                                   symbol: Optional[str] = None) -> pd.Series:
    """Extract permutation importance"""
    try:
        from sklearn.inspection import permutation_importance
        
        # Need y for permutation importance
        if y is None:
            logger.warning("No y provided for permutation importance, returning zeros")
            return pd.Series(0.0, index=feature_names)
        
        # Generate deterministic seed for permutation importance
        seed_parts = ['perm']
        if model_family:
            seed_parts.append(model_family)
        if symbol:
            seed_parts.append(symbol)
        if target_column:
            seed_parts.append(target_column)
        perm_seed = stable_seed_from(seed_parts)
        
        result = permutation_importance(
            model, X, y,
            n_repeats=n_repeats,
            random_state=perm_seed,
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
    data_interval_minutes: int = 5,  # Data bar interval (default: 5 minutes)
    target_column: Optional[str] = None,  # Target column name for horizon extraction
    symbol: Optional[str] = None  # Symbol name for deterministic seed generation
) -> Tuple[Any, pd.Series, str]:
    """Train a single model family and extract importance"""
    
    # Generate deterministic seed for this model/symbol/target combination
    seed_parts = [model_family]
    if symbol:
        seed_parts.append(symbol)
    if target_column:
        seed_parts.append(target_column)
    model_seed = stable_seed_from(seed_parts)
    
    # Validate target before training
    try:
        from TRAINING.utils.target_validation import validate_target
        is_valid, error_msg = validate_target(y, min_samples=10, min_class_samples=2)
        if not is_valid:
            logger.debug(f"    {model_family}: {error_msg}")
            return None, pd.Series(0.0, index=feature_names), family_config['importance_method'], 0.0
    except ImportError:
        # Fallback: validate_target not available, use simple check
        logger.debug(f"    {model_family}: validate_target not available, using simple validation")
        unique_vals = np.unique(y[~np.isnan(y)])
        if len(unique_vals) < 2:
            logger.debug(f"    {model_family}: Target has only {len(unique_vals)} unique value(s)")
            return None, pd.Series(0.0, index=feature_names), family_config['importance_method'], 0.0
    
    importance_method = family_config['importance_method']
    model_config = family_config['config']
    
    # Train model based on family
    if model_family == 'lightgbm':
        # Use sklearn wrapper for consistent scoring (same as xgboost)
        try:
            from lightgbm import LGBMRegressor, LGBMClassifier
            # Determine task type
            unique_vals = np.unique(y[~np.isnan(y)])
            is_binary = len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0})
            is_multiclass = len(unique_vals) <= 10 and all(
                isinstance(v, (int, np.integer)) or (isinstance(v, float) and v.is_integer())
                for v in unique_vals
            )
            
            # Clean config: remove params that don't apply to sklearn wrapper
            # Remove early stopping (requires eval_set) and other semantic params
            lgb_config = model_config.copy()
            lgb_config.pop('boosting_type', None)
            lgb_config.pop('device', None)
            lgb_config.pop('early_stopping_rounds', None)
            lgb_config.pop('early_stopping_round', None)
            lgb_config.pop('callbacks', None)
            lgb_config.pop('threads', None)
            lgb_config.pop('min_samples_split', None)
            
            # Determine estimator class
            est_cls = LGBMClassifier if (is_binary or is_multiclass) else LGBMRegressor
            
            # Clean config using systematic helper (removes duplicates and unknown params)
            extra = {"random_seed": model_seed}
            lgb_config = _clean_config_for_estimator(est_cls, lgb_config, extra, "lightgbm")
            
            # Instantiate with cleaned config + explicit params
            model = est_cls(**lgb_config, **extra)
            
            model.fit(X, y)
            train_score = model.score(X, y)  # R² for regression, accuracy for classification
            
            # Log metric for debugging
            metric_name = 'R²' if not (is_binary or is_multiclass) else 'accuracy'
            logger.debug(f"    lightgbm: metric={metric_name}, score={train_score:.4f}")
            
        except Exception as e:
            logger.error(f"LightGBM failed: {e}")
            return None, pd.Series(0.0, index=feature_names), importance_method, 0.0
    
    elif model_family == 'xgboost':
        try:
            import xgboost as xgb
            # Determine task type for proper model selection
            unique_vals = np.unique(y[~np.isnan(y)])
            is_binary = len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0})
            is_multiclass = len(unique_vals) <= 10 and all(
                isinstance(v, (int, np.integer)) or (isinstance(v, float) and v.is_integer())
                for v in unique_vals
            )
            
            # Remove early stopping params (requires eval_set) - feature selection doesn't need it
            # XGBoost 2.x requires eval_set if early_stopping_rounds is set, so we must remove it
            xgb_config = model_config.copy()
            xgb_config.pop('early_stopping_rounds', None)
            xgb_config.pop('early_stopping_round', None)  # Alternative name
            xgb_config.pop('callbacks', None)
            xgb_config.pop('eval_set', None)  # Remove if present
            xgb_config.pop('eval_metric', None)  # Often paired with early stopping
            
            # Determine estimator class
            est_cls = xgb.XGBClassifier if (is_binary or is_multiclass) else xgb.XGBRegressor
            
            # Clean config using systematic helper (removes duplicates and unknown params)
            extra = {"random_state": model_seed}
            xgb_config = _clean_config_for_estimator(est_cls, xgb_config, extra, "xgboost")
            
            # Instantiate with cleaned config + explicit params
            model = est_cls(**xgb_config, **extra)
            
            try:
                # No eval_set needed - early stopping params already removed from config
                model.fit(X, y)
                train_score = model.score(X, y)  # R² for regression, accuracy for classification
                
                # Log metric for debugging
                metric_name = 'R²' if not (is_binary or is_multiclass) else 'accuracy'
                logger.debug(f"    xgboost: metric={metric_name}, score={train_score:.4f}")
                
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
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        
        # Determine task type
        unique_vals = np.unique(y[~np.isnan(y)])
        is_binary = len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0})
        is_multiclass = len(unique_vals) <= 10 and all(
            isinstance(v, (int, np.integer)) or (isinstance(v, float) and v.is_integer())
            for v in unique_vals
        )
        
        # Determine estimator class
        est_cls = RandomForestClassifier if (is_binary or is_multiclass) else RandomForestRegressor
        
        # Clean config using systematic helper (removes duplicates and unknown params)
        # Note: RandomForest uses n_jobs, not num_threads/threads
        extra = {"random_state": model_seed}
        rf_config = _clean_config_for_estimator(est_cls, model_config, extra, "random_forest")
        
        # Instantiate with cleaned config + explicit params
        model = est_cls(**rf_config, **extra)
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
        
        # Clean config using systematic helper (removes duplicates and unknown params)
        # MLPRegressor doesn't accept n_jobs, num_threads, or threads
        extra = {"random_state": model_seed}
        nn_config = _clean_config_for_estimator(MLPRegressor, model_config, extra, "neural_network")
        
        # Instantiate with cleaned config + explicit params
        model = MLPRegressor(**nn_config, **extra)
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
            
            # Remove task-specific params (CatBoost uses thread_count, not n_jobs)
            cb_config = model_config.copy()
            cb_config.pop('verbose', None)
            cb_config.pop('loss_function', None)
            cb_config.pop('n_jobs', None)
            
            # Determine estimator class and loss function
            if is_binary:
                est_cls = cb.CatBoostClassifier
                loss_fn = 'Logloss'
            elif is_multiclass:
                est_cls = cb.CatBoostClassifier
                loss_fn = 'MultiClass'
            else:
                est_cls = cb.CatBoostRegressor
                loss_fn = 'RMSE'
            
            # Clean config using systematic helper (removes duplicates and unknown params)
            extra = {
                "random_seed": model_seed,
                "verbose": False,
                "loss_function": loss_fn
            }
            cb_config = _clean_config_for_estimator(est_cls, cb_config, extra, "catboost")
            
            # Instantiate with cleaned config + explicit params
            model = est_cls(**cb_config, **extra)
            
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
        from TRAINING.utils.sklearn_safe import make_sklearn_dense_X
        
        # Lasso doesn't handle NaNs - use sklearn-safe conversion
        X_dense, feature_names_dense = make_sklearn_dense_X(X, feature_names)
        
        # Clean config using systematic helper (removes duplicates and unknown params)
        # Note: Lasso's random_state only applies to 'saga' solver, but sklearn ignores it for others
        # We set it explicitly for determinism consistency (matches RandomForest/MLP pattern)
        extra = {"random_state": model_seed}
        lasso_config = _clean_config_for_estimator(Lasso, model_config, extra, "lasso")
        
        # Instantiate with cleaned config + explicit params
        model = Lasso(**lasso_config, **extra)
        model.fit(X_dense, y)
        train_score = model.score(X_dense, y)
        
        # Update feature_names to match dense array
        feature_names = feature_names_dense
    
    elif model_family == 'mutual_information':
        # Mutual information doesn't train a model, just calculates information
        from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
        from TRAINING.utils.sklearn_safe import make_sklearn_dense_X
        
        # Mutual information doesn't support NaN - use sklearn-safe conversion
        X_dense, feature_names_dense = make_sklearn_dense_X(X, feature_names)
        
        # Determine task type
        unique_vals = np.unique(y[~np.isnan(y)])
        is_binary = len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0})
        is_multiclass = len(unique_vals) <= 10 and all(
            isinstance(v, (int, np.integer)) or (isinstance(v, float) and v.is_integer())
            for v in unique_vals
        )
        
        if is_binary or is_multiclass:
            importance_values = mutual_info_classif(X_dense, y, random_state=model_seed, discrete_features='auto')
        else:
            importance_values = mutual_info_regression(X_dense, y, random_state=model_seed, discrete_features='auto')
        
        # Update feature_names to match dense array
        feature_names = feature_names_dense
        
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
        from TRAINING.utils.sklearn_safe import make_sklearn_dense_X
        
        # Univariate selection doesn't support NaN - use sklearn-safe conversion
        X_dense, feature_names_dense = make_sklearn_dense_X(X, feature_names)
        
        # Determine task type
        unique_vals = np.unique(y[~np.isnan(y)])
        is_binary = len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0})
        is_multiclass = len(unique_vals) <= 10 and all(
            isinstance(v, (int, np.integer)) or (isinstance(v, float) and v.is_integer())
            for v in unique_vals
        )
        
        # Suppress division by zero warnings (expected for zero-variance features)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if is_binary or is_multiclass:
                scores, pvalues = f_classif(X_dense, y)
            else:
                scores, pvalues = f_regression(X_dense, y)
        
        # Update feature_names to match dense array
        feature_names = feature_names_dense
        
        # Handle NaN/inf in scores (from zero-variance features)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize scores (F-statistics can be very large)
        max_score = np.max(scores)
        if max_score > 0:
            importance_values = scores / max_score
        else:
            # All scores are zero - assign uniform small importance to avoid all zeros
            importance_values = np.ones(len(scores)) * 1e-6
        
        # Ensure importance_values is a numpy array
        importance_values = np.asarray(importance_values)
        
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
        
        # Use config for RFE's internal estimator (load from preprocessing config if not in model_config)
        try:
            from CONFIG.config_loader import get_cfg
            rfe_cfg = get_cfg("preprocessing.multi_model_feature_selection.rfe", default={}, config_name="preprocessing_config")
            estimator_n_estimators = model_config.get('estimator_n_estimators', rfe_cfg.get('estimator_n_estimators', 100))
            estimator_max_depth = model_config.get('estimator_max_depth', rfe_cfg.get('estimator_max_depth', 10))
            estimator_n_jobs = model_config.get('estimator_n_jobs', rfe_cfg.get('estimator_n_jobs', 1))
            estimator_random_state = model_config.get('random_state', 42)  # Use model_config random_state if available
        except Exception as e:
            logger.debug(f"Failed to load RFE config: {e}, using model_config defaults")
            estimator_n_estimators = model_config.get('estimator_n_estimators', 100)
            estimator_max_depth = model_config.get('estimator_max_depth', 10)
            estimator_n_jobs = model_config.get('estimator_n_jobs', 1)
            estimator_random_state = model_config.get('random_state', 42)
        
        if is_binary or is_multiclass:
            estimator = RandomForestClassifier(
                n_estimators=estimator_n_estimators,
                max_depth=estimator_max_depth,
                random_state=estimator_random_state,
                n_jobs=estimator_n_jobs
            )
        else:
            estimator = RandomForestRegressor(
                n_estimators=estimator_n_estimators,
                max_depth=estimator_max_depth,
                random_state=estimator_random_state,
                n_jobs=estimator_n_jobs
            )
        
        selector = RFE(estimator, n_features_to_select=n_features_to_select, step=step)
        selector.fit(X, y)
        
        # Convert ranking to importance (lower rank = more important)
        # RFE ranking: 1 = selected, higher = eliminated
        # Convert to importance: 1/rank (higher importance for lower rank)
        ranking = selector.ranking_
        
        # Ensure ranking is valid
        if ranking is None or len(ranking) == 0:
            # Fallback: assign uniform importance if ranking is invalid
            importance_values = np.ones(len(feature_names)) * 1e-6
        else:
            # Convert ranking to importance, ensuring no division by zero
            importance_values = 1.0 / (ranking + 1e-6)
            # Normalize to ensure we have meaningful values
            max_importance = np.max(importance_values)
            if max_importance > 0:
                importance_values = importance_values / max_importance
            else:
                # All rankings are the same or invalid - assign uniform small importance
                importance_values = np.ones(len(feature_names)) * 1e-6
        
        # Ensure importance_values is a numpy array and matches feature count
        importance_values = np.asarray(importance_values)
        if len(importance_values) != len(feature_names):
            logger.warning(f"RFE importance length ({len(importance_values)}) doesn't match features ({len(feature_names)})")
            if len(importance_values) < len(feature_names):
                importance_values = np.pad(importance_values, (0, len(feature_names) - len(importance_values)), 'constant', constant_values=1e-6)
            else:
                importance_values = importance_values[:len(feature_names)]
        
        class DummyModel:
            def __init__(self, importance):
                self.importance = importance
        
        model = DummyModel(importance_values)
        # RFE's estimator was trained on transformed (reduced) features, so score with transformed X
        try:
            X_transformed = selector.transform(X)
            train_score = selector.estimator_.score(X_transformed, y) if hasattr(selector, 'estimator_') else 0.0
        except Exception as e:
            logger.debug(f"    rfe: Failed to compute train score: {e}")
            train_score = 0.0
    
    elif model_family == 'boruta':
        # Boruta - All-relevant feature selection (statistical gate, not just another importance scorer)
        # Uses ExtraTrees with more trees/shallower depth for stable importance testing
        try:
            from boruta import BorutaPy
            from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
            from TRAINING.utils.sklearn_safe import make_sklearn_dense_X
            
            # Boruta doesn't support NaN - use sklearn-safe conversion
            X_dense, feature_names_dense = make_sklearn_dense_X(X, feature_names)
            
            # Determine task type
            unique_vals = np.unique(y[~np.isnan(y)])
            is_binary = len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0})
            is_multiclass = len(unique_vals) <= 10 and all(
                isinstance(v, (int, np.integer)) or (isinstance(v, float) and v.is_integer())
                for v in unique_vals
            )
            
            # Use ExtraTrees (more random, better for stability testing) with Boruta-optimized hyperparams
            # More trees + shallower depth = stable importance signals, not best predictive performance
            # Load from preprocessing config if not in model_config
            try:
                from CONFIG.config_loader import get_cfg
                boruta_cfg = get_cfg("preprocessing.multi_model_feature_selection.boruta", default={}, config_name="preprocessing_config")
                boruta_n_estimators = model_config.get('n_estimators', boruta_cfg.get('n_estimators', 500))
                boruta_max_depth = model_config.get('max_depth', boruta_cfg.get('max_depth', 6))
                boruta_random_state = model_config.get('random_state', 42)
                boruta_max_iter = model_config.get('max_iter', boruta_cfg.get('max_iter', 100))
                boruta_n_jobs = model_config.get('n_jobs', boruta_cfg.get('n_jobs', 1))
                boruta_verbose = model_config.get('verbose', boruta_cfg.get('verbose', 0))
            except Exception as e:
                logger.debug(f"Failed to load Boruta config: {e}, using model_config defaults")
                boruta_n_estimators = model_config.get('n_estimators', 500)
                boruta_max_depth = model_config.get('max_depth', 6)
                boruta_random_state = model_config.get('random_state', 42)
                boruta_max_iter = model_config.get('max_iter', 100)
                boruta_n_jobs = model_config.get('n_jobs', 1)
                boruta_verbose = model_config.get('verbose', 0)
            boruta_class_weight = model_config.get('class_weight', 'auto')
            
            # Handle class_weight config
            if is_binary or is_multiclass:
                if boruta_class_weight == 'auto':
                    # Auto: balanced_subsample for binary, balanced for multiclass
                    class_weight_value = 'balanced_subsample' if is_binary else 'balanced'
                elif boruta_class_weight == 'none' or boruta_class_weight is None:
                    class_weight_value = None
                else:
                    # Use config value directly (could be dict, 'balanced', etc.)
                    class_weight_value = boruta_class_weight
                
                base_estimator = ExtraTreesClassifier(
                    n_estimators=boruta_n_estimators,
                    max_depth=boruta_max_depth,
                    random_state=boruta_random_state,
                    n_jobs=boruta_n_jobs,
                    class_weight=class_weight_value
                )
            else:
                base_estimator = ExtraTreesRegressor(
                    n_estimators=boruta_n_estimators,
                    max_depth=boruta_max_depth,
                    random_state=boruta_random_state,
                    n_jobs=boruta_n_jobs
                )
            
            boruta = BorutaPy(
                base_estimator,
                n_estimators='auto',
                verbose=boruta_verbose,
                random_state=boruta_random_state,
                max_iter=boruta_max_iter,
                perc=model_config.get('perc', 95)  # Higher threshold = more conservative (needs to beat shadow more decisively)
            )
            # Note: make_sklearn_dense_X imputes NaNs but doesn't filter rows, so y matches X_dense length
            boruta.fit(X_dense, y)
            
            # Update feature_names to match dense array
            feature_names = feature_names_dense
            
            # Boruta as statistical gate: confirmed/rejected/tentative labels
            ranking = boruta.ranking_  # 1=confirmed, 2=tentative, >2=rejected
            selected = boruta.support_  # True=confirmed, False=rejected or tentative
            
            # Get tentative mask (ranking == 2)
            tentative_mask = (ranking == 2)
            
            # Log Boruta gate results per symbol
            n_confirmed = selected.sum()
            n_tentative = tentative_mask.sum()
            n_rejected = (ranking > 2).sum()
            logger.info(f"    boruta: {n_confirmed} confirmed, {n_rejected} rejected, {n_tentative} tentative")
            
            # Use robust helper to convert Boruta outputs to importance vector
            # This ensures we never get all-zeros or negative-sum when there are confirmed/tentative features
            # Returns: confirmed=1.0, tentative=0.3, rejected=0.0 (positive sum for validation)
            importance_values = boruta_to_importance(
                support=selected,
                support_weak=tentative_mask,
                ranking=ranking,
                n_features=X_dense.shape[1]
            )
            
            # If Boruta truly found no signal, use fallback instead of failing
            fallback_reason = None
            if importance_values is None:
                logger.debug(f"    boruta: No stable features identified (all rejected), using uniform fallback")
                # Use normalize_importance fallback to avoid InvalidImportance error
                importance_values, fallback_reason = normalize_importance(
                    raw_importance=None,
                    n_features=X_dense.shape[1],
                    family="boruta",
                    feature_names=feature_names
                )
                # Note: selected and ranking are already defined above, so they're valid here
            
            # Convert to pandas Series for consistency with other model families
            importance_values = pd.Series(importance_values, index=feature_names)
            
            class DummyModel:
                def __init__(self, importance, fallback_reason=None):
                    self.importance = importance
                    self._fallback_reason = fallback_reason
                # Store Boruta metadata for aggregation
                def get_boruta_labels(self):
                    return {
                        'confirmed': selected,
                        'tentative': ranking == 2,
                        'rejected': ranking > 2,
                        'ranking': ranking
                    }
            
            model = DummyModel(importance_values, fallback_reason=fallback_reason)
            # CRITICAL: Do NOT call base_estimator.score(X_dense, y) for Boruta.
            # Boruta's internal ExtraTreesClassifier is trained on a transformed subset of features
            # (confirmed/rejected/tentative selection), not the full X_dense.
            # Attempting to score on full X_dense causes: "X has N features, but ExtraTreesClassifier is expecting M features"
            # Boruta's purpose is feature selection (gatekeeper), not prediction scoring.
            # Use NaN to indicate "not applicable" (not a predictive model, so no score exists)
            train_score = math.nan
        except ImportError:
            logger.error("Boruta not available (pip install Boruta)")
            return None, pd.Series(0.0, index=feature_names), importance_method, 0.0
        except Exception as e:
            logger.error(f"Boruta failed: {e}", exc_info=True)
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
        from TRAINING.utils.purged_time_series_split import PurgedTimeSeriesSplit
        from TRAINING.utils.leakage_filtering import _extract_horizon, _load_leakage_config
        
        # Calculate purge_overlap from target horizon
        # CRITICAL: Use the data_interval_minutes parameter (detected in calling function)
        # Using wrong interval (e.g., assuming 5m when data is 1m) causes severe leakage
        purge_buffer_bars = model_config.get('purge_buffer_bars', 5)  # Safety buffer (configurable)
        
        leakage_config = _load_leakage_config()
        target_horizon_minutes = _extract_horizon(target_column, leakage_config) if target_column else None
        
        if target_horizon_minutes is not None:
            target_horizon_bars = target_horizon_minutes // data_interval_minutes
            purge_overlap = target_horizon_bars + purge_buffer_bars
        else:
            # Fallback: conservative default (60m = 12 bars + buffer)
            purge_overlap = 17
        
        # Create purged CV splitter
        # NOTE: Using row-count based purging (legacy). For better accuracy, use time-based purging:
        # purged_cv = PurgedTimeSeriesSplit(n_splits=3, purge_overlap_time=pd.Timedelta(minutes=target_horizon_minutes), time_column_values=timestamps)
        n_splits = model_config.get('n_splits', 3)  # Number of CV splits (configurable)
        purged_cv = PurgedTimeSeriesSplit(n_splits=n_splits, purge_overlap=purge_overlap)
        
        n_bootstrap = model_config.get('n_bootstrap', 50)  # Reduced for speed
        stability_random_state = model_config.get('random_state', 42)
        stability_cs = model_config.get('Cs', 10)  # Number of C values for LogisticRegressionCV
        stability_max_iter = model_config.get('max_iter', 1000)  # Max iterations for LassoCV/LogisticRegressionCV
        stability_n_jobs = model_config.get('n_jobs', 1)  # Parallel jobs
        
        stability_scores = np.zeros(X.shape[1])
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_boot, y_boot = X[indices], y[indices]
            
            try:
                if is_binary or is_multiclass:
                    model = LogisticRegressionCV(
                        Cs=stability_cs,
                        cv=purged_cv,
                        random_state=stability_random_state,
                        max_iter=stability_max_iter,
                        n_jobs=stability_n_jobs
                    )
                else:
                    model = LassoCV(
                        cv=purged_cv,
                        random_state=stability_random_state,
                        max_iter=stability_max_iter,
                        n_jobs=stability_n_jobs
                    )
                
                model.fit(X_boot, y_boot)
                stability_scores += (np.abs(model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_) > 1e-6).astype(int)
            except Exception as e:
                # Skip failed bootstrap iterations (expected for some degenerate cases)
                logger.debug(f"    stability_selection: Bootstrap iteration failed: {e}")
                continue
        
        # Normalize to 0-1 (fraction of times selected)
        raw_importance = stability_scores / n_bootstrap
        
        # Use normalize_importance to handle edge cases (all zeros, NaN, etc.)
        importance_values, fallback_reason = normalize_importance(
            raw_importance=raw_importance,
            n_features=X.shape[1],
            family="stability_selection",
            feature_names=feature_names
        )
        
        if fallback_reason:
            logger.debug(f"    stability_selection: {fallback_reason}")
        
        class DummyModel:
            def __init__(self, importance, fallback_reason=None):
                self.importance = importance
                self._fallback_reason = fallback_reason
        
        model = DummyModel(importance_values, fallback_reason=fallback_reason)
        train_score = 0.0  # No single model to score
    
    else:
        logger.error(f"Unknown model family: {model_family}")
        return None, pd.Series(0.0, index=feature_names), importance_method, 0.0
    
    # Extract importance based on method
    try:
        if importance_method == 'native':
            importance = extract_native_importance(model, feature_names)
        elif importance_method == 'shap':
            importance = extract_shap_importance(model, X, feature_names,
                                                model_family=model_family,
                                                target_column=target_column,
                                                symbol=symbol)
        elif importance_method == 'permutation':
            importance = extract_permutation_importance(model, X, y, feature_names,
                                                        model_family=model_family,
                                                        target_column=target_column,
                                                        symbol=symbol)
        else:
            logger.error(f"Unknown importance method: {importance_method}")
            importance = pd.Series(0.0, index=feature_names)
        
        # Validate importance was extracted successfully
        if importance is None:
            logger.warning(f"    {model_family}: Importance extraction returned None, using zeros")
            importance = pd.Series(0.0, index=feature_names)
        elif not isinstance(importance, pd.Series):
            logger.warning(f"    {model_family}: Importance extraction returned {type(importance)}, converting to Series")
            importance = pd.Series(importance, index=feature_names) if hasattr(importance, '__len__') else pd.Series(0.0, index=feature_names)
        elif len(importance) != len(feature_names):
            logger.warning(f"    {model_family}: Importance length ({len(importance)}) != features ({len(feature_names)}), padding/truncating")
            if len(importance) < len(feature_names):
                importance = pd.concat([importance, pd.Series(0.0, index=feature_names[len(importance):])])
            else:
                importance = importance.iloc[:len(feature_names)]
        
    except Exception as e:
        logger.error(f"    {model_family}: Importance extraction failed: {e}")
        importance = pd.Series(0.0, index=feature_names)
    
    return model, importance, importance_method, train_score


def process_single_symbol(
    symbol: str,
    data_path: Path,
    target_column: str,
    model_families_config: Dict[str, Dict[str, Any]],
    max_samples: int = None,
    explicit_interval: Optional[Union[int, str]] = None,  # Optional explicit interval from config
    experiment_config: Optional[Any] = None,  # Optional ExperimentConfig (for data.bar_interval)
    output_dir: Optional[Path] = None  # Optional output directory for stability snapshots
) -> Tuple[List[ImportanceResult], List[Dict[str, Any]]]:
    """Process a single symbol with multiple model families"""
    
    # Load default max_samples from config if not provided
    if max_samples is None:
        try:
            from CONFIG.config_loader import get_cfg
            _CONFIG_AVAILABLE = True
        except ImportError:
            _CONFIG_AVAILABLE = False
        
        if _CONFIG_AVAILABLE:
            try:
                from CONFIG.config_loader import get_cfg
                max_samples = int(get_cfg("pipeline.data_limits.default_max_samples_feature_selection", default=50000, config_name="pipeline_config"))
            except Exception as e:
                logger.debug(f"Failed to load default_max_samples_feature_selection from config: {e}, using default=50000")
                max_samples = 50000
        else:
            max_samples = 50000
    
    results = []
    family_statuses = []  # Initialize family_statuses list
    
    try:
        # Load data
        df = safe_load_dataframe(data_path)
        
        # Validate target
        if target_column not in df.columns:
            logger.warning(f"Skipping {symbol}: Target '{target_column}' not found")
            return results, family_statuses
        
        # Drop NaN in target
        df = df.dropna(subset=[target_column])
        if df.empty:
            logger.warning(f"Skipping {symbol}: No valid data after dropping NaN")
            return results, family_statuses
        
        # Sample if too large - use deterministic seed based on symbol
        if len(df) > max_samples:
            # Generate stable seed from symbol name for deterministic sampling
            sample_seed = stable_seed_from([symbol, "data_sampling"])
            df = df.sample(n=max_samples, random_state=sample_seed)
        
        # LEAKAGE PREVENTION: Filter out leaking features (with registry validation)
        from TRAINING.utils.leakage_filtering import filter_features_for_target
        from TRAINING.utils.data_interval import detect_interval_from_dataframe
        
        # Detect data interval for horizon conversion
        detected_interval = detect_interval_from_dataframe(
            df, 
            timestamp_column='ts', 
            default=5,
            explicit_interval=explicit_interval,
            experiment_config=experiment_config
        )
        
        all_columns = df.columns.tolist()
        # Use target-aware filtering with registry validation
        safe_columns = filter_features_for_target(
            all_columns, 
            target_column, 
            verbose=False,
            use_registry=True,  # Enable registry validation
            data_interval_minutes=detected_interval
        )
        
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
            return results, family_statuses
        
        # Convert to numpy
        X_arr = X.to_numpy()
        y_arr = y.to_numpy()
        
        # CRITICAL: Use already-detected interval (detected above at line 773)
        # No need to detect again - use the same detected_interval from above
        if detected_interval != 5:
            logger.info(f"  Detected data interval: {detected_interval}m (was assuming 5m)")
        
        # Train each enabled model family with structured status tracking
        enabled_families = [f for f, cfg in model_families_config.items() if cfg.get('enabled', False)]
        
        for family_name, family_config in model_families_config.items():
            if not family_config.get('enabled', False):
                continue
            
            try:
                logger.info(f"  {symbol}: Training {family_name}...")
                model, importance, method, train_score = train_model_and_get_importance(
                    family_name, family_config, X_arr, y_arr, feature_names,
                    data_interval_minutes=detected_interval,
                    target_column=target_column,
                    symbol=symbol  # Pass symbol for deterministic seed generation
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
                    # Handle NaN scores gracefully (e.g., Boruta doesn't have a train score)
                    score_str = f"{train_score:.4f}" if not math.isnan(train_score) else "N/A"
                    logger.info(f"    ✅ {family_name}: score={score_str}, "
                              f"top feature={importance.idxmax()} ({importance.max():.2f})")
                    
                    # Save stability snapshot for this method (non-invasive hook)
                    # Only save if output_dir is available (optional feature)
                    if output_dir is not None:
                        try:
                            from TRAINING.stability.feature_importance import save_snapshot_from_series_hook
                            universe_id = symbol if symbol else "ALL"
                            save_snapshot_from_series_hook(
                                target_name=target_column if target_column else 'unknown',
                                method=method,  # "rfe", "boruta", "stability_selection", etc.
                                importance_series=importance,
                                universe_id=universe_id,
                                output_dir=output_dir,
                                auto_analyze=None,  # Load from config
                            )
                        except Exception as e:
                            logger.debug(f"Stability snapshot save failed for {method} (non-critical): {e}")
                    
                    # Check if model used a fallback (soft no-signal case)
                    fallback_reason = getattr(model, '_fallback_reason', None)
                    if fallback_reason:
                        # Soft no-signal fallback: not a failure, just "no strong preference"
                        family_statuses.append({
                            "status": "no_signal_fallback",
                            "family": family_name,
                            "symbol": symbol,
                            "score": float(train_score) if not math.isnan(train_score) else None,
                            "top_feature": importance.idxmax(),
                            "top_feature_score": float(importance.max()),
                            "error": fallback_reason,
                            "error_type": "NoSignalFallback"
                        })
                        logger.debug(f"    ℹ️  {family_name}: {fallback_reason} (not counted as failure)")
                    else:
                        # Normal success
                        family_statuses.append({
                            "status": "success",
                            "family": family_name,
                            "symbol": symbol,
                            "score": float(train_score) if not math.isnan(train_score) else None,
                            "top_feature": importance.idxmax(),
                            "top_feature_score": float(importance.max()),
                            "error": None,
                            "error_type": None
                        })
                else:
                    # Model returned but importance is None or all zeros (hard failure)
                    logger.warning(f"    ⚠️  {family_name}: Model trained but returned invalid importance (None or all zeros)")
                    family_statuses.append({
                        "status": "failed",
                        "family": family_name,
                        "symbol": symbol,
                        "score": None,
                        "top_feature": None,
                        "top_feature_score": None,
                        "error": "Invalid importance (None or all zeros)",
                        "error_type": "InvalidImportance"
                    })
                
            except Exception as e:
                # Capture exception details for debugging
                error_type = type(e).__name__
                error_msg = str(e)
                logger.error(f"    ❌ {symbol}: {family_name} FAILED: {error_type}: {error_msg}", exc_info=True)
                
                family_statuses.append({
                    "status": "failed",
                    "family": family_name,
                    "symbol": symbol,
                    "score": None,
                    "top_feature": None,
                    "top_feature_score": None,
                    "error": error_msg,
                    "error_type": error_type
                })
                continue
        
        # Log structured summary per symbol
        success_families = [s["family"] for s in family_statuses if s["status"] == "success"]
        failed_families = [s["family"] for s in family_statuses if s["status"] == "failed"]
        
        logger.info(f"✅ {symbol}: Completed {len(success_families)}/{len(enabled_families)} model families")
        if success_families:
            logger.info(f"   ✅ Success: {', '.join(success_families)}")
        if failed_families:
            logger.warning(f"   ❌ Failed: {', '.join(failed_families)}")
            # Log error types for failed families
            for status in family_statuses:
                if status["status"] == "failed":
                    logger.warning(f"      - {status['family']}: {status['error_type']}: {status['error']}")
        
    except Exception as e:
        logger.error(f"❌ {symbol}: Processing failed: {e}", exc_info=True)
        # Return empty results but preserve any statuses collected before failure
        return results, family_statuses
    
    return results, family_statuses


def aggregate_multi_model_importance(
    all_results: List[ImportanceResult],
    model_families_config: Dict[str, Dict[str, Any]],
    aggregation_config: Dict[str, Any],
    top_n: Optional[int] = None,
    all_family_statuses: Optional[List[Dict[str, Any]]] = None  # Optional: for logging excluded families
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
        logger.warning("⚠️  No results to aggregate - all model families may have failed or returned empty importance")
        return pd.DataFrame(), []
    
    # Group results by model family
    family_results = defaultdict(list)
    for result in all_results:
        family_results[result.model_family].append(result)
    
    # Log which families were excluded due to failures (if status info available)
    if all_family_statuses:
        enabled_families = set(f for f, cfg in model_families_config.items() if cfg.get('enabled', False))
        families_with_results = set(family_results.keys())
        families_without_results = enabled_families - families_with_results
        
        if families_without_results:
            logger.warning(f"⚠️  {len(families_without_results)} model families excluded from aggregation (no results): {', '.join(sorted(families_without_results))}")
            # Log per-symbol failure details if available
            for family in families_without_results:
                family_failures = [s for s in all_family_statuses if s.get('family') == family and s.get('status') == 'failed']
                if family_failures:
                    error_types = set(s.get('error_type') for s in family_failures if s.get('error_type'))
                    symbols_failed = [s.get('symbol') for s in family_failures]
                    logger.warning(f"   - {family}: Failed for {len(symbols_failed)} symbol(s) ({', '.join(set(symbols_failed))}) with error types: {', '.join(error_types) if error_types else 'Unknown'}")
        
        logger.info(f"✅ Aggregating {len(families_with_results)} model families with results: {', '.join(sorted(families_with_results))}")
    
    # Aggregate within each family
    family_scores = {}
    boruta_scores = None  # Store separately for gatekeeper role
    
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
        
        # CRITICAL: Boruta is NOT included in base consensus - it's a gatekeeper, not a scorer
        if family_name == 'boruta':
            boruta_scores = family_score  # Store for gatekeeper role only
            logger.info(f"🔒 {family_name}: Aggregated {len(results)} symbols (gatekeeper, excluded from base consensus)")
        else:
            family_scores[family_name] = family_score * weight
            logger.info(f"📊 {family_name}: Aggregated {len(results)} symbols, "
                       f"weight={weight}, top={family_score.idxmax()}")
    
    # Combine across families (EXCLUDING Boruta - it's a gatekeeper, not a scorer)
    if not family_scores:
        logger.warning("No model family results available (all families may have failed or been disabled)")
        return pd.DataFrame(), []
    
    combined_df = pd.DataFrame(family_scores)
    
    # Calculate BASE consensus score (from non-Boruta families only)
    # Keep this separate from final score so we can see Boruta's effect
    cross_model_method = aggregation_config.get('cross_model_method', 'weighted_mean')
    if cross_model_method == 'weighted_mean':
        consensus_score_base = combined_df.mean(axis=1)
    elif cross_model_method == 'median':
        consensus_score_base = combined_df.median(axis=1)
    elif cross_model_method == 'geometric_mean':
        # Geometric mean (good for multiplicative effects)
        consensus_score_base = np.exp(np.log(combined_df + 1e-10).mean(axis=1))
    else:
        consensus_score_base = combined_df.mean(axis=1)
    
    # BORUTA GATEKEEPER: Apply Boruta as statistical gate (bonus/penalty system)
    # Boruta is not just another importance scorer - it's a robustness check
    # It modifies consensus scores but doesn't contribute to base consensus
    
    # Check if Boruta is enabled in config (even if no results)
    boruta_enabled = model_families_config.get('boruta', {}).get('enabled', False)
    
    if boruta_scores is not None:
        boruta_bonus = aggregation_config.get('boruta_confirm_bonus', 0.2)  # Bonus for confirmed features
        boruta_penalty = aggregation_config.get('boruta_reject_penalty', -0.3)  # Penalty for rejected features
        
        # Boruta scores: 1.0=confirmed, 0.3=tentative, 0.0=rejected
        # (Note: rejected is 0.0, not -1.0, to ensure positive sum for validation)
        # Apply modifiers to consensus score
        confirmed_threshold = aggregation_config.get('boruta_confirmed_threshold', 0.9)  # Configurable threshold
        tentative_threshold = aggregation_config.get('boruta_tentative_threshold', 0.1)  # Updated: 0.1 to distinguish from 0.0 (rejected)
        
        confirmed_mask = boruta_scores >= confirmed_threshold  # Confirmed (score >= 0.9, typically = 1.0)
        rejected_mask = boruta_scores <= 0.0  # Rejected (score = 0.0, updated from < 0.0)
        tentative_mask = (boruta_scores > tentative_threshold) & (boruta_scores < confirmed_threshold)  # Tentative (between 0.1 and 0.9, typically = 0.3)
        
        # Calculate Boruta gate effect (bonus/penalty per feature)
        boruta_gate_effect = pd.Series(0.0, index=consensus_score_base.index)
        boruta_gate_effect[confirmed_mask] = boruta_bonus
        boruta_gate_effect[rejected_mask] = boruta_penalty
        # Tentative features get no modifier (neutral = 0.0)
        
        # Apply to base consensus to get final score
        consensus_score_final = consensus_score_base + boruta_gate_effect
        
        # Magnitude sanity check: warn if Boruta bonuses/penalties are too large relative to base consensus
        # Use explicit mathematical definition: ratio = max(|bonus|, |penalty|) / base_range
        base_min = consensus_score_base.min()
        base_max = consensus_score_base.max()
        base_range = max(base_max - base_min, 1e-9)  # Avoid division by zero
        
        # Calculate magnitude ratio (larger of bonus or penalty relative to base range)
        magnitude = max(abs(boruta_bonus), abs(boruta_penalty))
        magnitude_ratio = magnitude / base_range
        
        # Configurable threshold (default 0.5 = 50% of base range)
        magnitude_warning_threshold = aggregation_config.get('boruta_magnitude_warning_threshold', 0.5)
        
        if magnitude_ratio > magnitude_warning_threshold:
            logger.warning(
                "⚠️  Boruta gate magnitude ratio=%.3f exceeds threshold=%.3f "
                "(base_range=%.4f, base_min=%.4f, base_max=%.4f, confirm_bonus=%.3f, reject_penalty=%.3f). "
                "Consider reducing boruta_confirm_bonus/boruta_reject_penalty in config if Boruta dominates decisions.",
                magnitude_ratio,
                magnitude_warning_threshold,
                base_range,
                base_min,
                base_max,
                boruta_bonus,
                boruta_penalty
            )
        
        logger.info(f"🔒 Boruta gatekeeper: {confirmed_mask.sum()} confirmed (+{boruta_bonus}), "
                   f"{rejected_mask.sum()} rejected ({boruta_penalty}), "
                   f"{tentative_mask.sum()} tentative (neutral)")
        logger.debug(f"   Base consensus range: [{consensus_score_base.min():.3f}, {consensus_score_base.max():.3f}], "
                    f"std={consensus_score_base.std():.3f}, magnitude_ratio={magnitude_ratio:.3f}")
        
        # Calculate "Boruta changed ranking" metric: compare top-K sets before vs after gatekeeper
        # Use top_n if available, otherwise use a reasonable default (50) for comparison
        top_k_for_comparison = top_n if top_n is not None else min(50, len(consensus_score_base))
        if top_k_for_comparison > 0 and len(consensus_score_base) >= top_k_for_comparison:
            # Get top-K features from base consensus (without Boruta)
            top_base_features = set(
                consensus_score_base.sort_values(ascending=False).head(top_k_for_comparison).index
            )
            # Get top-K features from final consensus (with Boruta)
            top_final_features = set(
                consensus_score_final.sort_values(ascending=False).head(top_k_for_comparison).index
            )
            # Symmetric difference: features that changed in top-K set
            changed_features = len(top_base_features ^ top_final_features)
            logger.info(f"   Boruta ranking impact: {changed_features} features changed in top-{top_k_for_comparison} set "
                       f"(base vs final). Ratio: {changed_features/top_k_for_comparison:.1%}")
        
        # Store for summary_df
        boruta_gate_effect_series = boruta_gate_effect
        boruta_gate_scores_series = boruta_scores
        boruta_confirmed_mask = confirmed_mask
        boruta_rejected_mask = rejected_mask
        boruta_tentative_mask = tentative_mask
        
    elif boruta_enabled:
        # Boruta enabled but failed completely (no results from any symbol)
        logger.warning("🔒 Boruta gatekeeper disabled or unavailable for this target (no effect). "
                      "Boruta may have failed for all symbols or was disabled mid-run.")
        # Set defaults: no effect
        boruta_gate_effect_series = pd.Series(0.0, index=consensus_score_base.index)
        boruta_gate_scores_series = pd.Series(0.0, index=consensus_score_base.index)
        boruta_confirmed_mask = pd.Series(False, index=consensus_score_base.index)
        boruta_rejected_mask = pd.Series(False, index=consensus_score_base.index)
        boruta_tentative_mask = pd.Series(False, index=consensus_score_base.index)
        consensus_score_final = consensus_score_base.copy()
    else:
        # Boruta not enabled in config - explicit log for clarity
        logger.debug("🔒 Boruta gatekeeper: disabled via config (no effect on consensus).")
        boruta_gate_effect_series = pd.Series(0.0, index=consensus_score_base.index)
        boruta_gate_scores_series = pd.Series(0.0, index=consensus_score_base.index)
        boruta_confirmed_mask = pd.Series(False, index=consensus_score_base.index)
        boruta_rejected_mask = pd.Series(False, index=consensus_score_base.index)
        boruta_tentative_mask = pd.Series(False, index=consensus_score_base.index)
        consensus_score_final = consensus_score_base.copy()
    
    # Calculate consensus metrics
    n_models = combined_df.shape[1]
    frequency = (combined_df > 0).sum(axis=1)
    frequency_pct = (frequency / n_models) * 100
    
    # Standard deviation across models (lower = more consensus)
    consensus_std = combined_df.std(axis=1)
    
    # Create summary DataFrame with base and final consensus scores
    summary_df = pd.DataFrame({
        'feature': consensus_score_base.index,
        'consensus_score_base': consensus_score_base.values,  # Base consensus (without Boruta)
        'consensus_score': consensus_score_final.values,  # Final consensus (with Boruta gatekeeper effect)
        'boruta_gate_effect': boruta_gate_effect_series.values,  # Pure Boruta effect (final - base)
        'n_models_agree': frequency,
        'consensus_pct': frequency_pct,
        'std_across_models': consensus_std,
    })
    
    # Add per-family scores (excluding Boruta from per-family columns - it's in gatekeeper section)
    for family_name in family_scores.keys():
        summary_df[f'{family_name}_score'] = combined_df[family_name].values
    
    # Always add Boruta gatekeeper columns (even if disabled/failed - shows zeros/False)
    summary_df['boruta_gate_score'] = boruta_gate_scores_series.values  # Raw Boruta scores (1.0/0.3/0.0)
    summary_df['boruta_confirmed'] = boruta_confirmed_mask.values
    summary_df['boruta_rejected'] = boruta_rejected_mask.values
    summary_df['boruta_tentative'] = boruta_tentative_mask.values
    
    # Sort by final consensus score (with Boruta effect)
    summary_df = summary_df.sort_values('consensus_score', ascending=False).reset_index(drop=True)
    
    # Filter by minimum consensus if specified
    min_models = aggregation_config.get('require_min_models', 1)
    summary_df = summary_df[summary_df['n_models_agree'] >= min_models]
    
    # Select top N
    if top_n:
        summary_df = summary_df.head(top_n)
    
    selected_features = summary_df['feature'].tolist()
    
    return summary_df, selected_features


def compute_target_confidence(
    summary_df: pd.DataFrame,
    all_results: List[ImportanceResult],
    model_families_config: Dict[str, Dict[str, Any]],
    target_name: str,
    confidence_config: Optional[Dict[str, Any]] = None,
    top_k: Optional[int] = None
) -> Dict[str, Any]:
    """
    Compute target-level confidence metrics from multi-model feature selection results.
    
    Metrics computed:
    1. Boruta coverage (confirmed/tentative counts)
    2. Model coverage (successful vs available)
    3. Score strength (mean/max scores)
    4. Agreement ratio (features in top-K across multiple models)
    
    Args:
        summary_df: DataFrame with feature importance and Boruta status
        all_results: List of ImportanceResult from all model runs
        model_families_config: Config dict with enabled model families
        target_name: Target column name
        confidence_config: Optional config dict with confidence thresholds (from multi_model.yaml)
        top_k: Number of top features to consider for agreement (default: from config or 20)
    
    Returns:
        Dict with confidence metrics and bucket (HIGH/MEDIUM/LOW)
    """
    # Extract thresholds from config with defaults
    if confidence_config is None:
        confidence_config = {}
    
    high_cfg = confidence_config.get('high', {})
    medium_cfg = confidence_config.get('medium', {})
    low_reasons_cfg = confidence_config.get('low_reasons', {})
    agreement_cfg = confidence_config.get('agreement', {})
    
    # Default thresholds (matching current hardcoded values)
    high_boruta_min = high_cfg.get('boruta_confirmed_min', 5)
    high_agreement_min = high_cfg.get('agreement_ratio_min', 0.4)
    high_score_min = high_cfg.get('mean_score_min', 0.05)
    high_coverage_min = high_cfg.get('model_coverage_min', 0.7)
    
    medium_boruta_min = medium_cfg.get('boruta_confirmed_min', 1)
    medium_agreement_min = medium_cfg.get('agreement_ratio_min', 0.25)
    medium_score_min = medium_cfg.get('mean_score_min', 0.02)
    
    # Low reason thresholds
    boruta_zero_cfg = low_reasons_cfg.get('boruta_zero_confirmed', {})
    boruta_zero_confirmed_max = boruta_zero_cfg.get('boruta_confirmed_max', 0)
    boruta_zero_tentative_max = boruta_zero_cfg.get('boruta_tentative_max', 1)
    boruta_zero_score_max = boruta_zero_cfg.get('mean_score_max', 0.03)
    
    low_agreement_max = low_reasons_cfg.get('low_model_agreement', {}).get('agreement_ratio_max', 0.2)
    low_score_max = low_reasons_cfg.get('low_model_scores', {}).get('mean_score_max', 0.01)
    low_coverage_max = low_reasons_cfg.get('low_model_coverage', {}).get('model_coverage_max', 0.5)
    
    # Agreement top_k from config
    if top_k is None:
        top_k = agreement_cfg.get('top_k', 20)
    
    metrics = {
        'target_name': target_name,
        'boruta_confirmed_count': 0,
        'boruta_tentative_count': 0,
        'boruta_rejected_count': 0,
        'boruta_used': False,
        'n_models_available': 0,
        'n_models_successful': 0,
        'model_coverage_ratio': 0.0,
        'mean_score': 0.0,
        'max_score': 0.0,
        'mean_strong_score': 0.0,  # Tree ensembles + CatBoost + NN
        'agreement_ratio': 0.0,
        'score_tier': 'LOW',  # Orthogonal to confidence: signal strength
        'confidence': 'LOW',
        'low_confidence_reason': None
    }
    
    # 1. Boruta coverage
    if 'boruta_confirmed' in summary_df.columns:
        metrics['boruta_confirmed_count'] = int(summary_df['boruta_confirmed'].sum())
        metrics['boruta_tentative_count'] = int(summary_df['boruta_tentative'].sum())
        metrics['boruta_rejected_count'] = int(summary_df['boruta_rejected'].sum())
        metrics['boruta_used'] = (
            metrics['boruta_confirmed_count'] > 0 or
            metrics['boruta_tentative_count'] > 0 or
            metrics['boruta_rejected_count'] > 0
        )
    
    # 2. Model coverage
    enabled_families = [
        name for name, cfg in model_families_config.items()
        if cfg.get('enabled', False)
    ]
    metrics['n_models_available'] = len(enabled_families)
    
    # Count successful models (those with valid results)
    # Note: "no_signal_fallback" cases are counted as successful (they produced valid importance, just uniform)
    # Only hard failures (exceptions, InvalidImportance) are excluded
    successful_models = set(r.model_family for r in all_results if r.train_score is not None and not (isinstance(r.train_score, float) and (math.isnan(r.train_score) or math.isinf(r.train_score))))
    metrics['n_models_successful'] = len(successful_models)
    
    if metrics['n_models_available'] > 0:
        metrics['model_coverage_ratio'] = metrics['n_models_successful'] / metrics['n_models_available']
    
    # 3. Score strength
    valid_scores = [
        r.train_score for r in all_results
        if r.train_score is not None and not (isinstance(r.train_score, float) and (math.isnan(r.train_score) or math.isinf(r.train_score)))
    ]
    
    if valid_scores:
        metrics['mean_score'] = float(np.mean(valid_scores))
        metrics['max_score'] = float(np.max(valid_scores))
        
        # Strong models: tree ensembles, CatBoost, neural networks
        strong_model_families = {'lightgbm', 'xgboost', 'random_forest', 'catboost', 'neural_network'}
        strong_scores = [
            r.train_score for r in all_results
            if r.model_family in strong_model_families
            and r.train_score is not None
            and not (isinstance(r.train_score, float) and (math.isnan(r.train_score) or math.isinf(r.train_score)))
        ]
        if strong_scores:
            metrics['mean_strong_score'] = float(np.mean(strong_scores))
    
    # 3b. Score tier (orthogonal to confidence: pure signal strength)
    # Extract thresholds from config
    score_tier_cfg = confidence_config.get('score_tier', {})
    high_tier_cfg = score_tier_cfg.get('high', {})
    medium_tier_cfg = score_tier_cfg.get('medium', {})
    
    high_mean_strong_min = high_tier_cfg.get('mean_strong_score_min', 0.08)
    high_max_min = high_tier_cfg.get('max_score_min', 0.70)
    medium_mean_strong_min = medium_tier_cfg.get('mean_strong_score_min', 0.03)
    medium_max_min = medium_tier_cfg.get('max_score_min', 0.55)
    
    # HIGH if strong models show high scores OR max is very high
    if metrics['mean_strong_score'] >= high_mean_strong_min or metrics['max_score'] >= high_max_min:
        metrics['score_tier'] = 'HIGH'
    # MEDIUM if moderate scores
    elif metrics['mean_strong_score'] >= medium_mean_strong_min or metrics['max_score'] >= medium_max_min:
        metrics['score_tier'] = 'MEDIUM'
    # LOW otherwise
    else:
        metrics['score_tier'] = 'LOW'
    
    # 4. Agreement on top features
    if len(summary_df) > 0 and 'feature' in summary_df.columns:
        # Get top-K features by consensus score
        top_k_features = summary_df.nlargest(min(top_k, len(summary_df)), 'consensus_score')['feature'].tolist()
        
        # Count how many models have each feature in their top-K
        feature_model_count = defaultdict(int)
        
        for result in all_results:
            if result.importance_scores is None or len(result.importance_scores) == 0:
                continue
            
            # Get top-K features for this model
            model_top_k = result.importance_scores.nlargest(min(top_k, len(result.importance_scores))).index.tolist()
            
            # Count overlap with overall top-K
            for feature in top_k_features:
                if feature in model_top_k:
                    feature_model_count[feature] += 1
        
        # Agreement ratio: fraction of top-K features that appear in >= 2 models
        features_in_multiple_models = sum(1 for count in feature_model_count.values() if count >= 2)
        metrics['agreement_ratio'] = features_in_multiple_models / len(top_k_features) if top_k_features else 0.0
    
    # 5. Confidence bucket assignment (using config thresholds)
    # HIGH confidence (all conditions must be met)
    if (metrics['boruta_confirmed_count'] >= high_boruta_min and
        metrics['agreement_ratio'] >= high_agreement_min and
        metrics['mean_score'] >= high_score_min and
        metrics['model_coverage_ratio'] >= high_coverage_min):
        metrics['confidence'] = 'HIGH'
    
    # MEDIUM confidence (any one condition is sufficient)
    elif (metrics['boruta_confirmed_count'] >= medium_boruta_min or
          metrics['agreement_ratio'] >= medium_agreement_min or
          metrics['mean_score'] >= medium_score_min):
        metrics['confidence'] = 'MEDIUM'
    
    # LOW confidence (fallback)
    else:
        metrics['confidence'] = 'LOW'
        
        # Determine reason using config thresholds
        if (metrics['boruta_used'] and
            metrics['boruta_confirmed_count'] <= boruta_zero_confirmed_max and
            metrics['boruta_tentative_count'] <= boruta_zero_tentative_max and
            metrics['mean_score'] < boruta_zero_score_max):
            metrics['low_confidence_reason'] = 'boruta_zero_confirmed'
        elif metrics['agreement_ratio'] < low_agreement_max:
            metrics['low_confidence_reason'] = 'low_model_agreement'
        elif metrics['mean_score'] < low_score_max:
            metrics['low_confidence_reason'] = 'low_model_scores'
        elif metrics['model_coverage_ratio'] < low_coverage_max:
            metrics['low_confidence_reason'] = 'low_model_coverage'
        else:
            metrics['low_confidence_reason'] = 'multiple_weak_signals'
    
    return metrics


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
    
    # 2. Detailed summary CSV (includes all columns including Boruta gatekeeper)
    summary_df.to_csv(output_dir / "feature_importance_multi_model.csv", index=False)
    logger.info(f"✅ Saved detailed multi-model summary to feature_importance_multi_model.csv")
    
    # 2b. Explicit debug view: Boruta gatekeeper effect analysis
    # This is a stable, named file for quick inspection of Boruta's impact
    debug_columns = [
        'feature',
        'consensus_score_base',  # Base consensus (model families only)
        'consensus_score',  # Final consensus (with Boruta effect)
        'boruta_gate_effect',  # Pure Boruta effect (final - base)
        'boruta_gate_score',  # Raw Boruta scores (1.0/0.3/0.0)
        'boruta_confirmed',
        'boruta_rejected',
        'boruta_tentative',
        'n_models_agree',
        'consensus_pct'
    ]
    # Only include columns that exist in summary_df
    available_debug_columns = [col for col in debug_columns if col in summary_df.columns]
    if available_debug_columns:
        debug_df = summary_df[available_debug_columns].copy()
        debug_df = debug_df.sort_values('consensus_score', ascending=False)  # Sort by final score
        debug_df.to_csv(output_dir / "feature_importance_with_boruta_debug.csv", index=False)
        logger.info(f"✅ Saved Boruta gatekeeper debug view to feature_importance_with_boruta_debug.csv")
    
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
    
    # 6. Family status tracking JSON (for debugging broken models)
    if 'family_statuses' in metadata and metadata['family_statuses']:
        family_statuses = metadata['family_statuses']
        # Create summary by family
        status_summary = {}
        for status in family_statuses:
            family = status.get('family')
            if family not in status_summary:
                status_summary[family] = {
                    'total_runs': 0,
                'success': 0,
                'failed': 0,
                'no_signal_fallback': 0,  # Soft fallbacks (not counted as failures)
                'symbols_success': [],
                'symbols_failed': [],
                'error_types': set(),
                'errors': []
                }
            summary = status_summary[family]
            summary['total_runs'] += 1
            status_value = status.get('status')
            if status_value == 'success':
                summary['success'] += 1
                summary['symbols_success'].append(status.get('symbol'))
            elif status_value == 'no_signal_fallback':
                # Soft fallback: counted as success for coverage, but tracked separately
                summary['success'] += 1  # Count as success for model_coverage_ratio
                summary['no_signal_fallback'] = summary.get('no_signal_fallback', 0) + 1
                summary['symbols_success'].append(status.get('symbol'))
                if status.get('error_type'):
                    summary['error_types'].add(status.get('error_type'))
            else:
                # Hard failure (exceptions, InvalidImportance, etc.)
                summary['failed'] += 1
                summary['symbols_failed'].append(status.get('symbol'))
                if status.get('error_type'):
                    summary['error_types'].add(status.get('error_type'))
                if status.get('error'):
                    summary['errors'].append({
                        'symbol': status.get('symbol'),
                        'error_type': status.get('error_type'),
                        'error': status.get('error')
                    })
        
        # Convert sets to lists for JSON serialization
        for family_summary in status_summary.values():
            family_summary['error_types'] = list(family_summary['error_types'])
        
        # Save detailed status file
        with open(output_dir / "model_family_status.json", "w") as f:
            json.dump({
                'summary': status_summary,
                'detailed': family_statuses
            }, f, indent=2)
        logger.info(f"✅ Saved model family status tracking to model_family_status.json")
        
        # Log summary (only hard failures, not soft fallbacks)
        failed_families = [f for f, s in status_summary.items() if s['failed'] > 0]
        if failed_families:
            logger.warning(f"⚠️  {len(failed_families)} model families had hard failures: {', '.join(failed_families)}")
        
        # Log soft fallbacks separately (informational, not warnings)
        fallback_families = [f for f, s in status_summary.items() if s.get('no_signal_fallback', 0) > 0]
        if fallback_families:
            logger.info(f"ℹ️  {len(fallback_families)} model families used no-signal fallbacks (not failures): {', '.join(fallback_families)}")
    
    # 6. Target confidence metrics (if model_families_config available in metadata)
    try:
        model_families_config = metadata.get('model_families_config')
        config = metadata.get('config', {})
        
        if model_families_config is None:
            # Try to extract from metadata config if nested
            model_families_config = config.get('model_families', {})
        
        # Extract confidence config from nested config
        confidence_config = config.get('confidence', {})
        
        if model_families_config:
            target_name = metadata.get('target_column', 'unknown_target')
            confidence_metrics = compute_target_confidence(
                summary_df=summary_df,
                all_results=all_results,
                model_families_config=model_families_config,
                target_name=target_name,
                confidence_config=confidence_config,
                top_k=None  # Will use config or default
            )
            
            with open(output_dir / "target_confidence.json", "w") as f:
                json.dump(confidence_metrics, f, indent=2)
            
            # Log confidence summary
            confidence = confidence_metrics['confidence']
            reason = confidence_metrics.get('low_confidence_reason', '')
            if confidence == 'LOW':
                logger.warning(f"⚠️  Target {target_name}: confidence={confidence} ({reason})")
            elif confidence == 'MEDIUM':
                logger.info(f"ℹ️  Target {target_name}: confidence={confidence}")
            else:
                logger.info(f"✅ Target {target_name}: confidence={confidence}")
            
            logger.info(f"✅ Saved target confidence metrics to target_confidence.json")
    except Exception as e:
        logger.warning(f"Failed to compute target confidence metrics: {e}")
        logger.debug("Confidence computation requires model_families_config in metadata", exc_info=True)


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
    all_family_statuses = []  # Collect status info for debugging
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
                        # Convert None back to NaN for train_score (from checkpoint deserialization)
                        if r_dict.get('train_score') is None:
                            r_dict['train_score'] = math.nan
                        all_results.append(ImportanceResult(**r_dict))
            continue
        elif not args.resume:
            continue
        
        logger.info(f"\n[{i}/{len(labeled_files)}] Processing {symbol}...")
        try:
            results, family_statuses = process_single_symbol(
                symbol, path, args.target_column,
                config['model_families'],
                config['sampling']['max_samples_per_symbol']
            )
            all_results.extend(results)
            all_family_statuses.extend(family_statuses)
            
            # Save checkpoint after each symbol
            # Convert results to dict for serialization (handle pd.Series and NaN)
            results_dict = []
            for r in results:
                r_dict = asdict(r)
                # Convert pd.Series to dict
                if isinstance(r_dict.get('importance_scores'), pd.Series):
                    r_dict['importance_scores'] = r_dict['importance_scores'].to_dict()
                # Convert NaN to None for JSON serialization (checkpoint can't serialize NaN)
                if 'train_score' in r_dict and math.isnan(r_dict['train_score']):
                    r_dict['train_score'] = None
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
        args.top_n,
        all_family_statuses=all_family_statuses  # Pass status info for logging excluded families
    )
    
    if summary_df.empty:
        logger.error("❌ No features selected")
        return 1
    
    # Save results
    metadata = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'target_column': args.target_column,
        'family_statuses': all_family_statuses,  # Include for debugging
        'top_n': args.top_n,
        'n_symbols': len(labeled_files),
        'enabled_families': enabled_families,
        'config': config,
        'model_families_config': config.get('model_families', {}),  # Explicit for confidence computation
        'family_statuses': all_family_statuses  # Include for debugging
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
    logger.info(f"  • {args.output_dir}/feature_importance_with_boruta_debug.csv")
    logger.info(f"  • {args.output_dir}/model_agreement_matrix.csv")
    logger.info(f"  • {args.output_dir}/target_confidence.json")
    logger.info(f"  • {args.output_dir}/multi_model_metadata.json")
    logger.info(f"  • {args.output_dir}/model_family_status.json (family status tracking)")
    logger.info(f"  • {args.output_dir}/importance_<family>.csv (per-family)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

