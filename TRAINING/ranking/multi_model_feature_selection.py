# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

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
from scipy.stats import spearmanr
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
from TRAINING.common.determinism import init_determinism_from_config, seed_for, stable_seed_from, is_strict_mode

# Set global determinism immediately (reads from config, respects REPRO_MODE env var)
BASE_SEED = init_determinism_from_config()

from CONFIG.config_loader import load_model_config
import yaml
import time
from contextlib import contextmanager

# Import checkpoint utility (after path is set)
from TRAINING.orchestration.utils.checkpoint import CheckpointManager
# Setup logging with journald support (after path is set)
from TRAINING.orchestration.utils.logging_setup import setup_logging
logger = setup_logging(
    script_name="multi_model_feature_selection",
    level=logging.INFO,
    use_journald=True
)

# Timed context manager for performance diagnostics
@contextmanager
def timed(name: str, **kwargs):
    """Context manager to time expensive operations with metadata."""
    t0 = time.perf_counter()
    metadata_str = " ".join(f"{k}={v}" for k, v in kwargs.items())
    logger.info(f"⏱️ START {name} {metadata_str}")
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        logger.info(f"⏱️ END   {name}: {dt:.2f}s ({dt/60:.2f} minutes) {metadata_str}")

# Suppress warnings from SHAP/sklearn
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='X does not have valid feature names')

# Import logging config utilities for backend verbosity
try:
    from CONFIG.logging_config_utils import get_backend_logging_config
    _LOGGING_CONFIG_AVAILABLE = True
except ImportError:
    _LOGGING_CONFIG_AVAILABLE = False


# Import shared config cleaner utility
from TRAINING.common.utils.config_cleaner import clean_config_for_estimator as _clean_config_for_estimator

# Import threading utilities for smart thread management
try:
    from TRAINING.common.threads import plan_for_family, thread_guard, default_threads
    _THREADING_UTILITIES_AVAILABLE = True
except ImportError:
    _THREADING_UTILITIES_AVAILABLE = False
    logger.warning("Threading utilities not available; will use manual thread management")


# Import from modular components
from TRAINING.ranking.multi_model_feature_selection.types import (
    ModelFamilyConfig,
    ImportanceResult
)


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
    
    # Defensive: ensure n_features > 0 to avoid division issues
    if n_features <= 0:
        logger.error(f"{family}: n_features={n_features} is invalid, using default n_features=1")
        n_features = 1
        importance = np.array([1.0], dtype=float)
        return importance, f"{family}:fallback_invalid_n_features"
    
    # Fallback if truly no signal (all zeros)
    if not np.any(importance > 0):
        # No signal: treat as "no strong preference" instead of failure
        importance = np.full(n_features, uniform_importance, dtype=float)
        fallback_reason = f"{family}:fallback_uniform_no_signal"
        
        # Always ensure sum > 0 (even if normalize_after_fallback is False)
        s = float(importance.sum())
        if s <= 0:
            # Defensive: if uniform_importance was 0 or negative, use 1/n
            importance = np.full(n_features, 1.0 / n_features, dtype=float)
            logger.warning(f"{family}: uniform_importance={uniform_importance} resulted in sum={s}, using 1/n normalization")
        elif normalize_after_fallback:
            # Normalize to sum to 1.0
            importance = importance / s
    else:
        fallback_reason = None
        # For non-zero importance, check sum and normalize if needed
        s = float(importance.sum())
        if s <= 0:
            # Edge case: importance has some positive values but sum is still <= 0 (shouldn't happen, but defensive)
            importance = np.full(n_features, 1.0 / n_features, dtype=float)
            fallback_reason = f"{family}:fallback_negative_sum"
            logger.warning(f"{family}: Importance had positive values but sum={s} <= 0, using uniform fallback")
        elif normalize_after_fallback:
            # Normalize existing signal
            importance = importance / s
    
    # Final check: ensure sum > 0 (defensive, should always be true now)
    s = float(importance.sum())
    if s <= 0:
        # Last resort: force positive sum
        importance = np.full(n_features, 1.0 / n_features, dtype=float)
        if fallback_reason is None:
            fallback_reason = f"{family}:fallback_final_check"
        logger.warning(f"{family}: Importance sum was {s} in final check, forcing uniform distribution")
    
    # Guarantee: sum must be > 0 (defensive check before assertion)
    final_sum = float(importance.sum())
    if final_sum <= 0 or not np.isfinite(final_sum):
        # Last resort: force positive sum
        importance = np.full(n_features, 1.0 / n_features, dtype=float)
        if fallback_reason is None:
            fallback_reason = f"{family}:fallback_assertion_fix"
        logger.error(f"{family}: Importance sum was {final_sum} after all normalization, forcing uniform distribution")
        final_sum = 1.0  # After uniform distribution, sum should be 1.0
    
    # Final assertion (should always pass now)
    assert final_sum > 0 and np.isfinite(final_sum), \
        f"Importance sum should be positive and finite after normalization (family={family}, sum={final_sum}, n_features={n_features})"
    
    return importance, fallback_reason


def compute_per_model_reproducibility(
    symbol: str,
    target_column: str,
    model_family: str,
    current_score: float,
    current_importance: pd.Series,
    previous_data: Optional[Dict[str, Any]] = None,
    top_k: int = 50
) -> Dict[str, Any]:
    """
    Compute per-model reproducibility statistics.
    
    Args:
        symbol: Symbol name
        target_column: Target column name
        model_family: Model family name
        current_score: Current validation score
        current_importance: Current importance Series
        previous_data: Previous run data (dict with 'score' and 'importance' keys)
        top_k: Number of top features for Jaccard calculation
    
    Returns:
        Dict with reproducibility stats: delta_score, jaccard_top_k, importance_corr, status
    """
    if previous_data is None:
        return {
            "delta_score": None,
            "jaccard_top_k": None,
            "importance_corr": None,
            "status": "no_previous_run"
        }
    
    prev_score = previous_data.get('score')
    prev_importance = previous_data.get('importance')
    
    # Compute delta_score
    delta_score = None
    if prev_score is not None and not math.isnan(current_score) and not math.isnan(prev_score):
        delta_score = abs(current_score - prev_score)
    
    # Compute Jaccard@K
    jaccard_top_k = None
    if prev_importance is not None and isinstance(prev_importance, pd.Series):
        try:
            # Get top K features from both runs
            current_top_k = set(current_importance.nlargest(top_k).index)
            prev_top_k = set(prev_importance.nlargest(top_k).index)
            
            if current_top_k or prev_top_k:
                intersection = len(current_top_k & prev_top_k)
                union = len(current_top_k | prev_top_k)
                jaccard_top_k = intersection / union if union > 0 else 0.0
        except Exception as e:
            logger.debug(f"    {symbol}:{model_family}: Jaccard calculation failed: {e}")
    
    # Compute importance correlation (Spearman)
    importance_corr = None
    if prev_importance is not None and isinstance(prev_importance, pd.Series):
        try:
            # Align features (use union of features from both runs)
            all_features = set(current_importance.index) | set(prev_importance.index)
            if len(all_features) > 1:
                current_aligned = current_importance.reindex(all_features, fill_value=0.0)
                prev_aligned = prev_importance.reindex(all_features, fill_value=0.0)
                
                # Compute Spearman correlation
                corr, p_value = spearmanr(current_aligned.values, prev_aligned.values)
                if not math.isnan(corr):
                    importance_corr = float(corr)
        except Exception as e:
            logger.debug(f"    {symbol}:{model_family}: Correlation calculation failed: {e}")
    
    # Determine status based on thresholds
    # Model family priorities: high-variance families get stricter thresholds
    high_variance_families = {'neural_network', 'lasso', 'stability_selection', 'boruta', 'rfe', 'xgboost'}
    
    if model_family in high_variance_families:
        delta_score_threshold = 0.01
        min_jaccard = 0.7
        min_corr = 0.7
    else:
        # More stable families (random_forest, lightgbm, catboost)
        delta_score_threshold = 0.01
        min_jaccard = 0.7
        min_corr = 0.7
    
    # Filter/scoring methods (mutual_information, univariate_selection) use same thresholds
    # but we'll be more lenient in logging
    
    status = "stable"
    if delta_score is not None and delta_score > delta_score_threshold:
        status = "unstable"
    elif jaccard_top_k is not None and jaccard_top_k < min_jaccard:
        status = "unstable"
    elif importance_corr is not None and importance_corr < min_corr:
        status = "unstable"
    elif (delta_score is not None and delta_score > delta_score_threshold * 0.7) or \
         (jaccard_top_k is not None and jaccard_top_k < min_jaccard * 1.1) or \
         (importance_corr is not None and importance_corr < min_corr * 1.1):
        status = "borderline"
    
    return {
        "delta_score": delta_score,
        "jaccard_top_k": jaccard_top_k,
        "importance_corr": importance_corr,
        "status": status
    }


def load_previous_model_results(
    output_dir: Optional[Path],
    symbol: str,
    target_column: str,
    model_family: str
) -> Optional[Dict[str, Any]]:
    """
    Load previous run results for a specific model family.
    
    Args:
        output_dir: Output directory (may contain previous run metadata)
        symbol: Symbol name
        target_column: Target column name
        model_family: Model family name
    
    Returns:
        Dict with 'score' and 'importance' keys, or None if not found
    """
    if output_dir is None:
        return None
    
    try:
        # Look for metadata JSON in output_dir (feature_selections/{target}/model_metadata.json)
        # output_dir might be feature_selections/{target}/ or a parent
        if output_dir.name == target_column or (output_dir.parent / target_column).exists():
            if output_dir.name != target_column:
                metadata_dir = output_dir.parent / target_column
            else:
                metadata_dir = output_dir
        else:
            metadata_dir = output_dir
        
        metadata_file = metadata_dir / "model_metadata.json"
        
        # Try current location first
        if not metadata_file.exists():
            # Try parent directories (for previous runs)
            for parent in [metadata_dir.parent, metadata_dir.parent.parent]:
                if parent.exists():
                    prev_metadata = parent / "model_metadata.json"
                    if prev_metadata.exists():
                        metadata_file = prev_metadata
                        break
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Look for this symbol/target/model combination
            key = f"{symbol}:{target_column}:{model_family}"
            if key in metadata:
                prev_data = metadata[key]
                # Convert importance back to Series if needed
                if 'importance' in prev_data and isinstance(prev_data['importance'], dict):
                    prev_data['importance'] = pd.Series(prev_data['importance'])
                return prev_data
    except Exception as e:
        logger.debug(f"Could not load previous model results for {symbol}:{model_family}: {e}")
    
    return None


def save_model_metadata(
    output_dir: Optional[Path],
    symbol: str,
    target_column: str,
    model_family: str,
    score: float,
    importance: pd.Series,
    reproducibility: Dict[str, Any]
):
    """
    Save model metadata including reproducibility stats.
    
    Args:
        output_dir: Output directory
        symbol: Symbol name
        target_column: Target column name
        model_family: Model family name
        score: Validation score
        importance: Importance Series
        reproducibility: Reproducibility stats dict
    """
    if output_dir is None:
        return
    
    try:
        # Determine metadata directory (feature_selections/{target}/)
        if output_dir.name == target_column or (output_dir.parent / target_column).exists():
            if output_dir.name != target_column:
                metadata_dir = output_dir.parent / target_column
            else:
                metadata_dir = output_dir
        else:
            metadata_dir = output_dir
        
        metadata_file = metadata_dir / "model_metadata.json"
        
        # Load existing metadata
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        # Store this model's data
        key = f"{symbol}:{target_column}:{model_family}"
        metadata[key] = {
            "score": float(score) if not math.isnan(score) else None,
            "importance": importance.to_dict(),  # Convert Series to dict for JSON
            "reproducibility": reproducibility
        }
        
        # Save metadata
        metadata_dir.mkdir(parents=True, exist_ok=True)
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        logger.debug(f"Could not save model metadata for {symbol}:{model_family}: {e}")


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
    
    Uses centralized config loader if available, otherwise falls back to manual path resolution.
    Checks new location first (CONFIG/ranking/features/multi_model.yaml),
    then old location (CONFIG/feature_selection/multi_model.yaml),
    then falls back to legacy location (CONFIG/multi_model_feature_selection.yaml).
    """
    if config_path is None:
        # Try using centralized config loader first
        try:
            from CONFIG.config_loader import get_config_path
            config_path = get_config_path("feature_selection_multi_model")
            if config_path.exists():
                logger.debug(f"Using centralized config loader: {config_path}")
            else:
                # Fallback to manual resolution
                config_path = None
        except (ImportError, AttributeError):
            # Config loader not available, use manual resolution
            config_path = None
        
        if config_path is None:
            # Manual path resolution (fallback)
            # Try newest location first (ranking/features/)
            newest_path = _REPO_ROOT / "CONFIG" / "ranking" / "features" / "multi_model.yaml"
            # Then old location (feature_selection/)
            old_path = _REPO_ROOT / "CONFIG" / "feature_selection" / "multi_model.yaml"
            # Finally legacy location (root) - but this file was deleted, so skip
            # legacy_path = _REPO_ROOT / "CONFIG" / "multi_model_feature_selection.yaml"
            
            if newest_path.exists():
                config_path = newest_path
                logger.debug(f"Using new config location: {config_path}")
            elif old_path.exists():
                config_path = old_path
                logger.debug(f"Using old config location: {config_path} (consider migrating to ranking/features/)")
            else:
                logger.warning(f"Config not found in new ({newest_path}) or old ({old_path}) locations, using defaults")
                return get_default_config()
    
    if not config_path.exists():
        logger.warning(f"Config not found: {config_path}, using defaults")
        return get_default_config()
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Inject global defaults from defaults.yaml (SST)
    # This centralizes common settings like seed, n_jobs, etc.
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


# Import from modular components (keeping original implementations for now due to complexity)
# from TRAINING.ranking.multi_model_feature_selection.importance_extractors import (
#     safe_load_dataframe,
#     extract_native_importance,
#     extract_shap_importance,
#     extract_permutation_importance
# )

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

# Module-level function for CatBoost importance computation (must be picklable for multiprocessing)
def _compute_catboost_importance_worker(model_data, X_data, feature_names_data, result_queue):
    """
    Worker process to compute CatBoost importance.
    
    This must be a module-level function (not nested) to be picklable for multiprocessing.
    CRITICAL: CatBoost get_feature_importance() requires Pool objects when data parameter is provided.
    """
    try:
        import numpy as np
        from catboost import Pool
        
        # CRITICAL: CatBoost requires Pool objects for get_feature_importance(data=...)
        # Convert numpy array to Pool if needed
        if isinstance(X_data, np.ndarray):
            # Get categorical features if available
            cat_features = []
            if hasattr(model_data, 'cat_features'):
                cat_features = model_data.cat_features
            elif hasattr(model_data, 'base_model') and hasattr(model_data.base_model, 'get_cat_feature_indices'):
                try:
                    cat_features = model_data.base_model.get_cat_feature_indices()
                except Exception:
                    cat_features = []
            
            importance_data = Pool(data=X_data, cat_features=cat_features if cat_features else None)
        else:
            importance_data = X_data
        
        if hasattr(model_data, 'base_model'):
            importance_raw = model_data.base_model.get_feature_importance(data=importance_data, type='PredictionValuesChange')
        else:
            importance_raw = model_data.get_feature_importance(data=importance_data, type='PredictionValuesChange')
        result_queue.put(('success', pd.Series(importance_raw, index=feature_names_data)))
    except Exception as e:
        result_queue.put(('error', str(e)))


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
    symbol: Optional[str] = None,  # Symbol name for deterministic seed generation
    X_train: Optional[np.ndarray] = None,  # Optional: Pre-split training data (for CV-based normalization)
    X_test: Optional[np.ndarray] = None,  # Optional: Pre-split test data (for CV-based normalization)
    y_train: Optional[np.ndarray] = None,  # Optional: Pre-split training target
    y_test: Optional[np.ndarray] = None  # Optional: Pre-split test target
) -> Tuple[Any, pd.Series, str]:
    """
    Train a single model family and extract importance
    
    FIX #2: For proper CV-based normalization, pass X_train/X_test separately.
    When provided, normalization (imputation/scaling) will fit only on X_train.
    If not provided, falls back to full-dataset normalization (leakage risk).
    
    PERFORMANCE AUDIT: This function is tracked for call counts and timing.
    """
    
    # Generate deterministic seed for this model/symbol/target combination
    seed_parts = [model_family]
    if symbol:
        seed_parts.append(symbol)
    if target_column:
        seed_parts.append(target_column)
    model_seed = stable_seed_from(seed_parts)
    
    # PERFORMANCE AUDIT: Track train_model_and_get_importance calls
    import time
    train_start_time = time.time()
    try:
        from TRAINING.common.utils.performance_audit import get_auditor
        auditor = get_auditor()
        if auditor.enabled:
            fingerprint_kwargs = {
                'model_family': model_family,
                'data_shape': X.shape,
                'n_features': len(feature_names),
                'target': target_column,
                'symbol': symbol
            }
            fingerprint = auditor._compute_fingerprint('train_model_and_get_importance', **fingerprint_kwargs)
    except Exception:
        auditor = None
        fingerprint = None
    
    # Validate target before training
    try:
        from TRAINING.ranking.utils.target_validation import validate_target
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
    
    # CRITICAL: Strict mode assertions for determinism
    if is_strict_mode():
        import os
        # Assert threading is single-thread
        omp_threads = os.environ.get('OMP_NUM_THREADS')
        if omp_threads != '1':
            logger.warning(f"Strict mode: OMP_NUM_THREADS={omp_threads}, expected '1'. Determinism may be compromised.")
        
        # Assert model params use single-thread AND deterministic flags AND randomness knobs
        if model_family in ['lightgbm', 'lgbm']:
            if model_config.get('num_threads', 1) != 1:
                logger.warning(f"Strict mode: LightGBM num_threads={model_config.get('num_threads')}, expected 1")
            if not model_config.get('deterministic', False):
                logger.warning(f"Strict mode: LightGBM deterministic={model_config.get('deterministic')}, expected True")
            # Check randomness knobs
            if 'seed' not in model_config and 'random_state' not in model_config:
                logger.warning(f"Strict mode: LightGBM missing seed/random_state")
        elif model_family in ['xgboost', 'xgb']:
            if model_config.get('nthread', 1) != 1:
                logger.warning(f"Strict mode: XGBoost nthread={model_config.get('nthread')}, expected 1")
            if 'seed' not in model_config and 'random_state' not in model_config:
                logger.warning(f"Strict mode: XGBoost missing seed/random_state")
        elif model_family in ['randomforest', 'random_forest']:
            if model_config.get('n_jobs', 1) != 1:
                logger.warning(f"Strict mode: RandomForest n_jobs={model_config.get('n_jobs')}, expected 1")
            if 'random_state' not in model_config:
                logger.warning(f"Strict mode: RandomForest missing random_state")
        elif model_family in ['catboost', 'cat']:
            if model_config.get('thread_count', 1) != 1:
                logger.warning(f"Strict mode: CatBoost thread_count={model_config.get('thread_count')}, expected 1")
            if 'random_seed' not in model_config and 'random_state' not in model_config:
                logger.warning(f"Strict mode: CatBoost missing random_seed/random_state")
    
    # Load cv_n_jobs for parallelization (same logic as model_evaluation.py)
    cv_n_jobs = 1  # Default to single-threaded
    try:
        from CONFIG.config_loader import get_cfg
        cv_n_jobs = int(get_cfg("training.cv_n_jobs", default=1, config_name="intelligent_training_config"))
    except Exception:
        # Fallback: try to get from multi_model_config if available
        try:
            cv_config = model_config.get('cross_validation', {})
            if cv_config is None:
                cv_config = {}
            cv_n_jobs = cv_config.get('n_jobs', 1)
        except Exception:
            cv_n_jobs = 1
    
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
            
            # GPU settings (will fallback to CPU if GPU not available)
            gpu_params = {}
            
            # STRICT MODE: Force CPU for determinism
            if is_strict_mode():
                logger.debug("  Strict mode: forcing CPU for LightGBM (GPU disabled for determinism)")
                # Skip GPU detection entirely - gpu_params stays empty (CPU)
            else:
                try:
                    from CONFIG.config_loader import get_cfg
                    # SST: All values from config, no hardcoded defaults
                    test_enabled = get_cfg('gpu.lightgbm.test_enabled', default=True, config_name='gpu_config')
                    test_n_estimators = get_cfg('gpu.lightgbm.test_n_estimators', default=1, config_name='gpu_config')
                    test_samples = get_cfg('gpu.lightgbm.test_samples', default=10, config_name='gpu_config')
                    test_features = get_cfg('gpu.lightgbm.test_features', default=5, config_name='gpu_config')
                    gpu_device_id = get_cfg('gpu.lightgbm.gpu_device_id', default=0, config_name='gpu_config')
                    gpu_platform_id = get_cfg('gpu.lightgbm.gpu_platform_id', default=0, config_name='gpu_config')
                    try_cuda_first = get_cfg('gpu.lightgbm.try_cuda_first', default=True, config_name='gpu_config')
                    preferred_device = get_cfg('gpu.lightgbm.device', default='cuda', config_name='gpu_config')
                    
                    if test_enabled and try_cuda_first:
                        # Try CUDA first (fastest)
                        try:
                            test_model = LGBMRegressor(device='cuda', n_estimators=test_n_estimators, gpu_device_id=gpu_device_id, verbose=-1)
                            test_model.fit(np.random.rand(test_samples, test_features), np.random.rand(test_samples))
                            gpu_params = {'device': 'cuda', 'gpu_device_id': gpu_device_id}
                        except Exception:
                            try:
                                # Try OpenCL
                                test_model = LGBMRegressor(device='gpu', n_estimators=test_n_estimators, gpu_platform_id=gpu_platform_id, gpu_device_id=gpu_device_id, verbose=-1)
                                test_model.fit(np.random.rand(test_samples, test_features), np.random.rand(test_samples))
                                gpu_params = {'device': 'gpu', 'gpu_platform_id': gpu_platform_id, 'gpu_device_id': gpu_device_id}
                            except Exception:
                                pass  # Fallback to CPU silently
                    elif test_enabled and preferred_device in ['cuda', 'gpu']:
                        # Use preferred device directly
                        try:
                            if preferred_device == 'cuda':
                                test_model = LGBMRegressor(device='cuda', n_estimators=test_n_estimators, gpu_device_id=gpu_device_id, verbose=-1)
                                gpu_params = {'device': 'cuda', 'gpu_device_id': gpu_device_id}
                            else:
                                test_model = LGBMRegressor(device='gpu', n_estimators=test_n_estimators, gpu_platform_id=gpu_platform_id, gpu_device_id=gpu_device_id, verbose=-1)
                                gpu_params = {'device': 'gpu', 'gpu_platform_id': gpu_platform_id, 'gpu_device_id': gpu_device_id}
                            test_model.fit(np.random.rand(test_samples, test_features), np.random.rand(test_samples))
                        except Exception:
                            pass  # Fallback to CPU silently
                    elif preferred_device in ['cuda', 'gpu']:
                        # Skip test, use preferred device from config
                        if preferred_device == 'cuda':
                            gpu_params = {'device': 'cuda', 'gpu_device_id': gpu_device_id}
                        else:
                            gpu_params = {'device': 'gpu', 'gpu_platform_id': gpu_platform_id, 'gpu_device_id': gpu_device_id}
                except Exception:
                    pass  # Fallback to CPU silently
            
            # Clean config: remove params that don't apply to sklearn wrapper
            # Remove early stopping (requires eval_set) and other semantic params
            # Also remove device if present (we set this from GPU config)
            lgb_config = model_config.copy()
            lgb_config.pop('boosting_type', None)
            lgb_config.pop('device', None)
            lgb_config.pop('gpu_device_id', None)
            lgb_config.pop('gpu_platform_id', None)
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
            
            # Remove 'verbose' from lgb_config if present (to avoid double argument error)
            # verbose will be set explicitly if needed, or use default from config_cleaner
            lgb_config.pop('verbose', None)
            
            # CRITICAL: Force deterministic mode for reproducibility
            lgb_config['deterministic'] = True
            lgb_config['force_row_wise'] = True  # Required for deterministic=True
            
            # Add GPU params if available (will override any device in config)
            lgb_config.update(gpu_params)
            
            # Use threading utilities for smart thread management
            if _THREADING_UTILITIES_AVAILABLE:
                # Detect if GPU is being used
                using_gpu = gpu_params and 'device' in gpu_params and gpu_params['device'] in ['cuda', 'gpu']
                # Get thread plan based on family and GPU usage
                plan = plan_for_family('LightGBM', total_threads=default_threads())
                # Set num_threads from plan (OMP threads for LightGBM)
                lgb_config['num_threads'] = plan['OMP']
                # Use thread_guard context manager for safe thread control
                with thread_guard(omp=plan['OMP'], mkl=plan['MKL']):
                    # Instantiate with cleaned config + explicit params
                    model = est_cls(**lgb_config, **extra)
                    model.fit(X, y)
            else:
                # Fallback: manual thread management
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
            
            # GPU settings (will fallback to CPU if GPU not available)
            gpu_params = {}
            try:
                from CONFIG.config_loader import get_cfg
                # SST: All values from config, no hardcoded defaults
                xgb_device = get_cfg('gpu.xgboost.device', default='cpu', config_name='gpu_config')
                xgb_tree_method = get_cfg('gpu.xgboost.tree_method', default='hist', config_name='gpu_config')
                # Note: gpu_id removed in XGBoost 3.1+, use device='cuda:0' format if needed
                test_enabled = get_cfg('gpu.xgboost.test_enabled', default=True, config_name='gpu_config')
                test_n_estimators = get_cfg('gpu.xgboost.test_n_estimators', default=1, config_name='gpu_config')
                test_samples = get_cfg('gpu.xgboost.test_samples', default=10, config_name='gpu_config')
                test_features = get_cfg('gpu.xgboost.test_features', default=5, config_name='gpu_config')
                
                if xgb_device == 'cuda':
                    if test_enabled:
                        # XGBoost 3.1+ uses device='cuda' with tree_method='hist' (gpu_id removed)
                        try:
                            test_model = xgb.XGBRegressor(tree_method='hist', device='cuda', n_estimators=test_n_estimators, verbosity=0)
                            test_model.fit(np.random.rand(test_samples, test_features), np.random.rand(test_samples))
                            gpu_params = {'tree_method': xgb_tree_method, 'device': 'cuda'}
                        except Exception:
                            # Try legacy API: tree_method='gpu_hist' (for XGBoost < 2.0)
                            try:
                                test_model = xgb.XGBRegressor(tree_method='gpu_hist', n_estimators=test_n_estimators, verbosity=0)
                                test_model.fit(np.random.rand(test_samples, test_features), np.random.rand(test_samples))
                                gpu_params = {'tree_method': 'gpu_hist'}  # Legacy API doesn't use device parameter
                            except Exception:
                                pass  # Fallback to CPU silently
                    else:
                        # Skip test, use config values directly
                        gpu_params = {'tree_method': xgb_tree_method, 'device': 'cuda'}
            except Exception:
                pass  # Fallback to CPU silently
            
            # Remove early stopping params (requires eval_set) - feature selection doesn't need it
            # XGBoost 2.x requires eval_set if early_stopping_rounds is set, so we must remove it
            # Also remove tree_method and device if present (we set these from GPU config)
            xgb_config = model_config.copy()
            xgb_config.pop('early_stopping_rounds', None)
            xgb_config.pop('early_stopping_round', None)  # Alternative name
            xgb_config.pop('callbacks', None)
            xgb_config.pop('eval_set', None)  # Remove if present
            xgb_config.pop('eval_metric', None)  # Often paired with early stopping
            xgb_config.pop('tree_method', None)
            xgb_config.pop('device', None)
            # Remove gpu_id if present (removed in XGBoost 3.1+, use device='cuda:0' format if needed)
            xgb_config.pop('gpu_id', None)
            
            # Determine estimator class
            est_cls = xgb.XGBClassifier if (is_binary or is_multiclass) else xgb.XGBRegressor
            
            # Clean config using systematic helper (removes duplicates and unknown params)
            extra = {"random_state": model_seed}  # FIX: XGBoost uses random_state, not seed
            xgb_config = _clean_config_for_estimator(est_cls, xgb_config, extra, "xgboost")
            
            # Add GPU params if available (will override any tree_method/device in config)
            xgb_config.update(gpu_params)
            
            # Use threading utilities for smart thread management
            if _THREADING_UTILITIES_AVAILABLE:
                # Detect if GPU is being used
                using_gpu = gpu_params and 'device' in gpu_params and gpu_params['device'] == 'cuda'
                # Get thread plan based on family and GPU usage
                plan = plan_for_family('XGBoost', total_threads=default_threads())
                # Set n_jobs from plan (OMP threads for XGBoost)
                xgb_config['n_jobs'] = plan['OMP']
                # Use thread_guard context manager for safe thread control
                with thread_guard(omp=plan['OMP'], mkl=plan['MKL']):
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
        extra = {"random_state": model_seed}  # FIX: sklearn uses random_state, not seed
        rf_config = _clean_config_for_estimator(est_cls, model_config, extra, "random_forest")
        
        # Use threading utilities for smart thread management
        if _THREADING_UTILITIES_AVAILABLE:
            # Get thread plan based on family
            plan = plan_for_family('RandomForest', total_threads=default_threads())
            # Set n_jobs from plan (OMP threads for RandomForest)
            rf_config['n_jobs'] = plan['OMP']
            # Use thread_guard context manager for safe thread control
            with thread_guard(omp=plan['OMP'], mkl=plan['MKL']):
                # Instantiate with cleaned config + explicit params
                model = est_cls(**rf_config, **extra)
                model.fit(X, y)
        else:
            # Fallback: manual thread management
            # Instantiate with cleaned config + explicit params
            model = est_cls(**rf_config, **extra)
            model.fit(X, y)
        train_score = model.score(X, y)
    
    elif model_family == 'neural_network':
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        
        # Load look-ahead bias fix config
        try:
            from TRAINING.ranking.utils.lookahead_bias_config import get_lookahead_bias_fix_config
            fix_config = get_lookahead_bias_fix_config()
            normalize_inside_cv = fix_config.get('normalize_inside_cv', False)
        except Exception:
            normalize_inside_cv = False
        
        # FIX #2: CV-based normalization
        # If X_train/X_test are provided, normalize only on training data to prevent leakage
        # Otherwise, fall back to full-dataset normalization (acceptable for feature selection, not model training)
        if X_train is not None and X_test is not None and normalize_inside_cv:
            # Fit imputer and scaler on training data only
            imputer = SimpleImputer(strategy='median')
            X_train_imputed = imputer.fit_transform(X_train)
            X_test_imputed = imputer.transform(X_test)  # Transform test using training statistics
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_imputed)
            X_test_scaled = scaler.transform(X_test_imputed)  # Transform test using training statistics
            
            # Use training data for model fitting
            X_scaled = X_train_scaled
            X_for_scoring = X_test_scaled  # For evaluation if needed
            y_for_training = y_train if y_train is not None else y
            y_for_scoring = y_test if y_test is not None else y
            
            logger.debug(f"  Neural Network: Using CV-based normalization (fit on train, transform test)")
        else:
            # Fallback: Full-dataset normalization (for feature selection context)
            if normalize_inside_cv and X_train is None:
                logger.warning(
                    f"normalize_inside_cv=True but no train/test split provided. "
                    f"Using full-dataset normalization (acceptable for feature selection, not model training). "
                    f"To fix: Pass X_train, X_test, y_train, y_test parameters."
                )
            
            # Handle NaN values (neural networks can't handle them)
            imputer = SimpleImputer(strategy='median')
            X_imputed = imputer.fit_transform(X)  # ⚠️ LEAK: Fits on full dataset
            
            # Scale for neural networks
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_imputed)  # ⚠️ LEAK: Fits on full dataset
            X_for_scoring = X_scaled
            y_for_training = y
            y_for_scoring = y
        
        # Clean config using systematic helper (removes duplicates and unknown params)
        # MLPRegressor doesn't accept n_jobs, num_threads, or threads
        extra = {"random_state": model_seed}  # FIX: sklearn uses random_state, not seed
        nn_config = _clean_config_for_estimator(MLPRegressor, model_config, extra, "neural_network")
        
        # Instantiate with cleaned config + explicit params
        model = MLPRegressor(**nn_config, **extra)
        try:
            # Use threading utilities for smart thread management
            if _THREADING_UTILITIES_AVAILABLE:
                # Get thread plan based on family (neural networks are GPU families, so OMP=1, MKL=1)
                plan = plan_for_family('MLP', total_threads=default_threads())
                # Use thread_guard context manager for safe thread control
                with thread_guard(omp=plan['OMP'], mkl=plan['MKL']):
                    model.fit(X_scaled, y_for_training)
                    train_score = model.score(X_scaled, y_for_training)
            else:
                # Fallback: manual thread management
                model.fit(X_scaled, y_for_training)
                train_score = model.score(X_scaled, y_for_training)
        except (ValueError, TypeError) as e:
            error_str = str(e).lower()
            if any(kw in error_str for kw in ['least populated class', 'too few', 'invalid classes']):
                logger.debug(f"    Neural Network: Target too imbalanced")
                return None, pd.Series(0.0, index=feature_names), importance_method, 0.0
            raise
        
        # Use scaled data for importance (use full dataset for importance extraction in feature selection context)
        # For CV-based normalization, we still use full X_scaled for importance to get complete feature rankings
        if X_train is not None and X_test is not None and normalize_inside_cv:
            # Concatenate train and test for importance extraction (feature selection needs full dataset)
            # NOTE: np is already imported at module level, don't import locally
            X = np.vstack([X_train_scaled, X_test_scaled])
        else:
            X = X_scaled
    
    elif model_family == 'catboost':
        try:
            import catboost as cb
            from catboost import Pool
            # Determine task type
            unique_vals = np.unique(y[~np.isnan(y)])
            is_binary = len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0})
            is_multiclass = len(unique_vals) <= 10 and all(
                isinstance(v, (int, np.integer)) or (isinstance(v, float) and v.is_integer())
                for v in unique_vals
            )
            
            # GPU settings (will fallback to CPU if GPU not available)
            gpu_params = {}
            try:
                from CONFIG.config_loader import get_cfg
                # SST: All values from config, no hardcoded defaults
                # FIX: Rename to catboost_task_type to avoid potential variable name collision
                catboost_task_type = get_cfg('gpu.catboost.task_type', default='CPU', config_name='gpu_config')
                devices = get_cfg('gpu.catboost.devices', default='0', config_name='gpu_config')
                thread_count = get_cfg('gpu.catboost.thread_count', default=8, config_name='gpu_config')
                test_enabled = get_cfg('gpu.catboost.test_enabled', default=True, config_name='gpu_config')
                test_iterations = get_cfg('gpu.catboost.test_iterations', default=1, config_name='gpu_config')
                test_samples = get_cfg('gpu.catboost.test_samples', default=10, config_name='gpu_config')
                test_features = get_cfg('gpu.catboost.test_features', default=5, config_name='gpu_config')
                
                if catboost_task_type == 'GPU':
                    if test_enabled:
                        # Try GPU (CatBoost uses task_type='GPU' or devices parameter)
                        try:
                            test_model = cb.CatBoostRegressor(task_type='GPU', devices=devices, iterations=test_iterations, verbose=False)
                            test_model.fit(np.random.rand(test_samples, test_features), np.random.rand(test_samples))
                            gpu_params = {'task_type': 'GPU', 'devices': devices}
                        except Exception:
                            gpu_params = {}  # Fallback to CPU silently
                    else:
                        # Skip test, use config values directly
                        gpu_params = {'task_type': 'GPU', 'devices': devices}
                else:
                    gpu_params = {}  # Use CPU (no GPU params)
            except Exception:
                gpu_params = {}  # Fallback to CPU silently
            
            # Remove task-specific params (CatBoost uses thread_count, not n_jobs)
            # Also remove task_type, devices, and thread_count if present (we set these from GPU config)
            cb_config = model_config.copy()
            cb_config.pop('verbose', None)
            cb_config.pop('loss_function', None)
            cb_config.pop('n_jobs', None)
            cb_config.pop('task_type', None)
            cb_config.pop('devices', None)
            cb_config.pop('thread_count', None)  # Remove if present, we'll set from GPU config when using GPU
            
            # CRITICAL: CatBoost only accepts ONE of iterations/n_estimators/num_boost_round/num_trees
            # Prefer 'iterations' (CatBoost's native param), remove ALL synonyms
            iteration_synonyms = ['n_estimators', 'num_boost_round', 'num_trees']
            has_iterations = 'iterations' in cb_config
            
            # Remove all synonyms (we'll keep 'iterations' if present, or let CatBoost use default)
            for synonym in iteration_synonyms:
                if synonym in cb_config:
                    logger.debug(f"    catboost: Removing {synonym} (preferring 'iterations' if present)")
                    cb_config.pop(synonym, None)
            
            # If we removed synonyms but 'iterations' is not present, that's fine - CatBoost will use default
            # But if both were present, we now only have 'iterations' which is what we want
            
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
            
            # Add GPU params BEFORE cleaning (so they're preserved)
            # This ensures task_type and devices are not removed by config cleaner
            if gpu_params:
                cb_config.update(gpu_params)
            
            # Get CatBoost verbose level from backend config
            if _LOGGING_CONFIG_AVAILABLE:
                try:
                    catboost_backend_cfg = get_backend_logging_config('catboost')
                    verbose_level = catboost_backend_cfg.native_verbosity
                except Exception:
                    verbose_level = 1  # Default to info level
            else:
                verbose_level = 1  # Default to info level
            
            # Clean config using systematic helper (removes duplicates and unknown params)
            # Note: task_type and devices should be valid CatBoost params, so they won't be removed
            extra = {
                "random_seed": model_seed,
                "verbose": verbose_level,  # Use backend config instead of hardcoded False
                "loss_function": loss_fn
            }
            cb_config = _clean_config_for_estimator(est_cls, cb_config, extra, "catboost")
            
            # Double-check: ensure no iteration synonyms remain after cleaning
            for synonym in iteration_synonyms:
                if synonym in cb_config:
                    logger.warning(f"    catboost: {synonym} still present after cleaning, removing it")
                    cb_config.pop(synonym, None)
            
            # Re-add GPU params after cleaning (in case they were removed)
            # CRITICAL: CatBoost REQUIRES task_type='GPU' to actually use GPU (devices alone is ignored)
            if gpu_params:
                cb_config.update(gpu_params)
                # Use threading utilities for smart thread management
                if _THREADING_UTILITIES_AVAILABLE:
                    # Get thread plan based on family and GPU usage
                    plan = plan_for_family('CatBoost', total_threads=default_threads())
                    # Set thread_count from plan (OMP threads for CatBoost)
                    if 'thread_count' not in cb_config:
                        cb_config['thread_count'] = plan['OMP']
                else:
                    # Fallback: use thread_count variable already in scope (read at line 1272)
                    if gpu_params.get('task_type') == 'GPU' and 'thread_count' not in cb_config:
                        cb_config['thread_count'] = thread_count
                # Explicit verification that task_type is set
                if cb_config.get('task_type') != 'GPU':
                    logger.warning(f"    catboost: GPU params updated but task_type is '{cb_config.get('task_type')}', expected 'GPU'")
                else:
                    logger.debug(f"    catboost: GPU verified: task_type={cb_config.get('task_type')}, devices={cb_config.get('devices')}, thread_count={cb_config.get('thread_count', 'not set')}")
            elif gpu_params is None or (isinstance(gpu_params, dict) and not gpu_params):
                # GPU was requested but params are empty - log for debugging
                logger.debug(f"    catboost: No GPU params to add (gpu_params={gpu_params})")
            
            # CatBoost Performance Diagnostics and Optimizations
            # Check for common issues that cause slow training (>20min for 50k samples)
            warnings_issued = []
            
            # 1. Check for excessive depth (exponential complexity: 2^d)
            depth = cb_config.get('depth', 6)  # Default is 6
            if depth > 8:
                warnings_issued.append(f"⚠️  CatBoost depth={depth} is high (exponential complexity 2^{depth}). Consider depth ≤ 8 for faster training.")
            
            # 2. Check for text-like features and high cardinality categoricals
            # Convert X to DataFrame temporarily to check dtypes if feature_names available
            if feature_names and len(feature_names) == X.shape[1]:
                try:
                    # Create temporary DataFrame to check dtypes
                    X_df = pd.DataFrame(X, columns=feature_names)
                    
                    # Check for object/string dtype columns (potential text features)
                    object_cols = X_df.select_dtypes(include=['object', 'string']).columns.tolist()
                    if object_cols:
                        if 'text_features' not in cb_config or not cb_config.get('text_features'):
                            warnings_issued.append(
                                f"⚠️  CatBoost: Detected {len(object_cols)} text/object columns: {object_cols[:5]}{'...' if len(object_cols) > 5 else ''}. "
                                f"Add text_features=['col_name'] to config to avoid treating them as high-cardinality categoricals."
                            )
                    
                    # Check for high cardinality categoricals (potential ID columns)
                    # Only flag for DROP when multiple ID signals agree (categorical + high unique ratio + ID-like name)
                    # Numeric columns with high cardinality are normal (continuous features) - just warn, don't suggest dropping
                    cat_features_list = cb_config.get('cat_features', [])
                    if isinstance(cat_features_list, (list, tuple)):
                        cat_features_set = set(cat_features_list)
                    else:
                        cat_features_set = set()
                    
                    high_cardinality_features = []
                    for col in feature_names:
                        if col in X_df.columns:
                            try:
                                unique_count = X_df[col].nunique()
                                unique_ratio = unique_count / len(X_df) if len(X_df) > 0 else 0
                                
                                # Check if column is treated as categorical
                                is_categorical = (
                                    col in cat_features_set or
                                    X_df[col].dtype.name in ['object', 'category', 'string'] or
                                    str(X_df[col].dtype).startswith('category')
                                )
                                
                                # Check if it's numeric (float/int) - high cardinality is normal for continuous features
                                is_numeric = pd.api.types.is_numeric_dtype(X_df[col])
                                
                                # ID-like name patterns
                                id_patterns = ['_id', '_ID', 'id_', 'ID_', 'user_', 'User_', 'ip_', 'IP_', 'row_', 'Row_',
                                              'uuid', 'UUID', 'tx_', 'order_', 'session_', 'hash_', '_key', '_Key']
                                has_id_name = any(pattern in col for pattern in id_patterns)
                                
                                # Check if values mostly occur once (median count per value <= 2)
                                value_counts = X_df[col].value_counts()
                                median_count = value_counts.median() if len(value_counts) > 0 else float('inf')
                                
                                # Only suggest DROP when multiple ID signals agree:
                                # 1. Treated as categorical (not numeric)
                                # 2. High unique ratio (>0.2 or >0.5 for strict)
                                # 3. Values mostly unique (median count <= 2) OR unique_ratio > 0.8
                                # 4. ID-like name OR near-perfect uniqueness
                                should_drop = (
                                    is_categorical and  # Must be categorical (not numeric)
                                    unique_ratio > 0.2 and  # High unique ratio
                                    (median_count <= 2 or unique_ratio > 0.8) and  # Mostly unique values
                                    (has_id_name or unique_ratio > 0.95)  # ID-like name OR near-perfect uniqueness
                                )
                                
                                if should_drop:
                                    high_cardinality_features.append((col, unique_count, unique_ratio, is_categorical))
                                elif is_numeric and unique_ratio > 0.8 and unique_count > 1000:
                                    # Numeric column with high cardinality - this is normal for continuous features
                                    # Just log a debug message, don't warn (this is expected behavior)
                                    logger.debug(f"    catboost: Column '{col}' is numeric with high cardinality ({unique_count} unique, {unique_ratio:.1%} unique ratio) - this is normal for continuous features")
                            except Exception:
                                pass  # Skip if can't compute unique count
                    
                    if high_cardinality_features:
                        id_cols = [col for col, _, _, _ in high_cardinality_features[:5]]
                        warnings_issued.append(
                            f"⚠️  CatBoost: Detected {len(high_cardinality_features)} high-cardinality ID-like CATEGORICAL columns: {id_cols}{'...' if len(high_cardinality_features) > 5 else ''}. "
                            f"These are treated as categorical with high unique ratios and ID-like names. Consider dropping or encoding differently (they don't generalize and slow training)."
                        )
                except Exception as e:
                    # If DataFrame conversion fails, skip diagnostics (non-critical)
                    logger.debug(f"    catboost: Diagnostics skipped (non-critical): {e}")
            
            # 3. Automatic metric_period injection (reduces evaluation overhead)
            if 'metric_period' not in cb_config:
                # Set default metric_period from model_config or use 50 as default
                cb_config['metric_period'] = model_config.get('metric_period', 50)
                logger.debug(f"    catboost: Added metric_period={cb_config['metric_period']} to reduce evaluation overhead")
            
            # 4. Automatic early stopping injection (od_type and od_wait) to prevent long training times
            # Early stopping will stop training if validation doesn't improve, dramatically reducing training time
            if 'od_type' not in cb_config:
                cb_config['od_type'] = model_config.get('od_type', 'Iter')
                logger.debug(f"    catboost: Added od_type={cb_config['od_type']} for early stopping")
            if 'od_wait' not in cb_config:
                cb_config['od_wait'] = model_config.get('od_wait', 20)
                logger.debug(f"    catboost: Added od_wait={cb_config['od_wait']} for early stopping")
            
            # 5. CRITICAL: Cap iterations for feature selection to prevent 2-3 hour training times
            # For feature selection, we don't need full training - just enough for importance ranking
            # Match target ranking approach: use 300 iterations (same as target ranking config)
            # Early stopping will further reduce actual iterations if validation doesn't improve
            max_iterations_feature_selection = 300  # Match target ranking (was 2000, too high)
            current_iterations = cb_config.get('iterations', 300)
            if current_iterations > max_iterations_feature_selection:
                logger.info(f"    catboost: Capping iterations from {current_iterations} to {max_iterations_feature_selection} for feature selection (matching target ranking)")
                cb_config['iterations'] = max_iterations_feature_selection
                cb_config['use_best_model'] = True  # Use best model from early stopping
            
            # Also ensure depth is reasonable (exponential complexity: 2^d)
            if cb_config.get('depth', 6) > 8:
                logger.info(f"    catboost: Reducing depth from {cb_config.get('depth')} to 6 for faster feature selection")
                cb_config['depth'] = 6
            
            # Log warnings if any
            if warnings_issued:
                logger.warning(f"    catboost: Performance Warnings:")
                for warning in warnings_issued:
                    logger.warning(f"      {warning}")
            
            # Log final config for debugging (only if GPU was requested)
            if gpu_params and gpu_params.get('task_type') == 'GPU':
                logger.debug(f"    catboost: Final config (sample): task_type={cb_config.get('task_type')}, devices={cb_config.get('devices')}, thread_count={cb_config.get('thread_count', 'not set')}")
            
            # CRITICAL: Explicitly set GPU params right before model instantiation to ensure they're not dropped
            # This prevents cases where GPU params might be lost in config processing
            if gpu_params and gpu_params.get('task_type') == 'GPU':
                cb_config.update({
                    "task_type": "GPU",
                    "devices": gpu_params.get('devices', '0'),
                })
                logger.debug(f"    catboost: Explicitly set GPU params before instantiation: task_type={cb_config.get('task_type')}, devices={cb_config.get('devices')}")
            
            # Instantiate with cleaned config + explicit params
            base_model = est_cls(**cb_config, **extra)
            
            # FIX: When GPU mode is enabled, CatBoost requires Pool objects instead of numpy arrays
            # Create a wrapper class that converts numpy arrays to Pool objects in fit() method
            use_gpu = gpu_params and gpu_params.get('task_type') == 'GPU'
            
            if use_gpu:
                # Create a wrapper class that handles Pool conversion for GPU mode
                # FIX: Make sklearn-compatible by implementing get_params/set_params
                class CatBoostGPUWrapper:
                    """Wrapper for CatBoost models that converts numpy arrays to Pool objects when GPU is enabled."""
                    def __init__(self, base_model=None, cat_features=None, use_gpu=True, _model_class=None, **kwargs):
                        # If base_model is provided, use it; otherwise create from kwargs (for sklearn cloning)
                        if base_model is not None:
                            self.base_model = base_model
                            # Store the model class for sklearn cloning
                            self._model_class = type(base_model)
                        else:
                            # Recreate base model from kwargs (for sklearn clone)
                            # Determine model class from loss_function or use stored class
                            if _model_class is not None:
                                model_class = _model_class
                            else:
                                # Infer from loss_function in kwargs
                                loss_fn = kwargs.get('loss_function', 'RMSE')
                                if loss_fn in ['Logloss', 'MultiClass']:
                                    model_class = cb.CatBoostClassifier
                                else:
                                    model_class = cb.CatBoostRegressor
                            self.base_model = model_class(**kwargs)
                            self._model_class = model_class
                        # FIX: For sklearn clone validation, ensure cat_features is set exactly as passed
                        # If None, use empty list; if already a list, use it directly; otherwise convert
                        if cat_features is None:
                            self.cat_features = []
                        elif isinstance(cat_features, list):
                            # Already a list - use it directly (sklearn expects this for clone validation)
                            self.cat_features = cat_features
                        else:
                            # Convert to list if it's not already
                            self.cat_features = list(cat_features)
                        self.use_gpu = use_gpu
                    
                    def get_params(self, deep=True):
                        """Get parameters for sklearn compatibility."""
                        # Get base model params and add wrapper-specific params
                        params = self.base_model.get_params(deep=deep)
                        # FIX: Return cat_features as-is (it's already a list from __init__)
                        # Sklearn's clone validation requires exact round-trip: get_params() -> __init__(**params) -> get_params()
                        params['cat_features'] = self.cat_features
                        params['use_gpu'] = self.use_gpu
                        params['_model_class'] = self._model_class
                        # Remove base_model from params (it's not a constructor arg)
                        params.pop('base_model', None)
                        return params
                    
                    def set_params(self, **params):
                        """Set parameters for sklearn compatibility."""
                        # Extract wrapper-specific params
                        cat_features = params.pop('cat_features', None)
                        use_gpu = params.pop('use_gpu', None)
                        model_class = params.pop('_model_class', None)
                        if cat_features is not None:
                            # FIX: Set exactly as passed (sklearn clone validation requires this)
                            if isinstance(cat_features, list):
                                self.cat_features = cat_features
                            else:
                                self.cat_features = list(cat_features) if cat_features else []
                        if use_gpu is not None:
                            self.use_gpu = use_gpu
                        if model_class is not None:
                            self._model_class = model_class
                        # Update base model params
                        self.base_model.set_params(**params)
                        return self
                    
                    def fit(self, X, y=None, eval_set=None, **kwargs):
                        """Convert numpy arrays to Pool objects when GPU is enabled."""
                        # Handle eval_set if provided (for early stopping)
                        eval_set_pools = None
                        if eval_set is not None:
                            eval_set_pools = []
                            for X_eval, y_eval in eval_set:
                                if isinstance(X_eval, np.ndarray):
                                    eval_pool = Pool(data=X_eval, label=y_eval, cat_features=self.cat_features)
                                    eval_set_pools.append(eval_pool)
                                elif isinstance(X_eval, Pool):
                                    eval_set_pools.append(X_eval)
                                else:
                                    # Fallback: try to create Pool from other types
                                    eval_pool = Pool(data=X_eval, label=y_eval, cat_features=self.cat_features)
                                    eval_set_pools.append(eval_pool)
                        
                        # Convert X and y to Pool objects for GPU mode
                        if isinstance(X, np.ndarray):
                            train_pool = Pool(data=X, label=y, cat_features=self.cat_features)
                            if eval_set_pools:
                                return self.base_model.fit(train_pool, eval_set=eval_set_pools, **kwargs)
                            else:
                                return self.base_model.fit(train_pool, **kwargs)
                        elif isinstance(X, Pool):
                            # Already a Pool object
                            if eval_set_pools:
                                return self.base_model.fit(X, eval_set=eval_set_pools, **kwargs)
                            else:
                                return self.base_model.fit(X, y, **kwargs)
                        else:
                            # Fallback: try direct fit (for other data types)
                            if eval_set_pools:
                                return self.base_model.fit(X, y, eval_set=eval_set_pools, **kwargs)
                            else:
                                return self.base_model.fit(X, y, **kwargs)
                    
                    def predict(self, X, **kwargs):
                        """Delegate predict to base model."""
                        if isinstance(X, np.ndarray) and self.use_gpu:
                            # Convert to Pool for consistency, though predict may work with arrays
                            test_pool = Pool(data=X, cat_features=self.cat_features)
                            return self.base_model.predict(test_pool, **kwargs)
                        return self.base_model.predict(X, **kwargs)
                    
                    def score(self, X, y, **kwargs):
                        """Delegate score to base model."""
                        if isinstance(X, np.ndarray) and self.use_gpu:
                            test_pool = Pool(data=X, label=y, cat_features=self.cat_features)
                            return self.base_model.score(test_pool, **kwargs)
                        return self.base_model.score(X, y, **kwargs)
                    
                    def __getattr__(self, name):
                        """Delegate all other attributes to base model."""
                        return getattr(self.base_model, name)
                
                # Get categorical features from config if specified
                cat_features = cb_config.get('cat_features', [])
                if isinstance(cat_features, list) and len(cat_features) > 0:
                    # If cat_features are column names, convert to indices
                    if feature_names and isinstance(cat_features[0], str):
                        cat_feature_indices = [feature_names.index(f) for f in cat_features if f in feature_names]
                    else:
                        cat_feature_indices = cat_features
                else:
                    cat_feature_indices = []
                
                model = CatBoostGPUWrapper(base_model=base_model, cat_features=cat_feature_indices, use_gpu=use_gpu)
            else:
                # CPU mode: use model directly (no Pool conversion needed)
                model = base_model
            
            # Verify GPU was actually set (post-instantiation check)
            if gpu_params and gpu_params.get('task_type') == 'GPU':
                # Check if model has task_type attribute (CatBoost models expose this)
                if hasattr(model, 'base_model'):
                    # Wrapper model - check base model
                    if hasattr(model.base_model, 'get_params'):
                        model_params = model.base_model.get_params()
                        if model_params.get('task_type') != 'GPU':
                            logger.warning(f"    catboost: Model instantiated but task_type is '{model_params.get('task_type')}', expected 'GPU'")
                        else:
                            logger.debug(f"    catboost: Model confirmed using GPU (task_type={model_params.get('task_type')})")
                elif hasattr(model, 'get_params'):
                    model_params = model.get_params()
                    if model_params.get('task_type') != 'GPU':
                        logger.warning(f"    catboost: Model instantiated but task_type is '{model_params.get('task_type')}', expected 'GPU'")
                    else:
                        logger.debug(f"    catboost: Model confirmed using GPU (task_type={model_params.get('task_type')})")
            
            # CRITICAL: CatBoost dtype fix - ensure all features are numeric float32/float64
            # CatBoost can treat object dtypes as text/categorical and memorize patterns
            # This causes fake performance (perfect scores) with poor generalization
            try:
                # Convert X to DataFrame to check/fix dtypes
                if isinstance(X, np.ndarray):
                    X_df = pd.DataFrame(X, columns=feature_names)
                else:
                    X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
                    if X_df.shape[1] != len(feature_names):
                        X_df.columns = feature_names[:X_df.shape[1]]
                
                # Hard-cast all numeric columns to float32 (CatBoost expects numeric, not object)
                for col in X_df.columns:
                    if pd.api.types.is_numeric_dtype(X_df[col]):
                        # Convert to float32 explicitly (prevents object dtype from NaN/mixed types)
                        X_df[col] = X_df[col].astype('float32')
                    elif X_df[col].dtype.name in ['object', 'string', 'category']:
                        # Try to convert to numeric, drop if fails
                        try:
                            X_df[col] = pd.to_numeric(X_df[col], errors='coerce').astype('float32')
                        except Exception:
                            logger.warning(f"    catboost: Dropping non-numeric column {col} (dtype={X_df[col].dtype})")
                            X_df = X_df.drop(columns=[col])
                            feature_names = [f for f in feature_names if f != col]
                
                # Update feature_names to match X_df
                feature_names = [f for f in feature_names if f in X_df.columns]
                
                # FIX: CRITICAL - Verify no object columns reach CatBoost (fail fast if they do)
                # This prevents CatBoost from treating numeric features as text/categorical
                object_cols_remaining = [c for c in X_df.columns if X_df[c].dtype.name in ['object', 'string', 'category']]
                if object_cols_remaining:
                    raise TypeError(f"CRITICAL: Object columns reached CatBoost: {object_cols_remaining[:10]}. "
                                   f"This will cause fake performance. Fix dtype enforcement upstream.")
                
                X_catboost = X_df.values.astype('float32')
                
                # FIX: Also verify X_catboost has no object dtype (double-check)
                if X_catboost.dtype.name in ['object', 'string']:
                    raise TypeError(f"CRITICAL: X_catboost has object dtype. This will cause CatBoost to treat features as text.")
                
                # Explicitly tell CatBoost there are no categorical features (unless specified)
                if 'cat_features' not in cb_config:
                    cb_config['cat_features'] = []  # No categoricals by default
                
                # Add cross_val_score for parallelization (matching target ranking behavior)
                # Determine task type and scoring
                unique_vals = np.unique(y[~np.isnan(y)])
                is_binary = len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0})
                is_multiclass = len(unique_vals) <= 10 and all(
                    isinstance(v, (int, np.integer)) or (isinstance(v, float) and v.is_integer())
                    for v in unique_vals
                )
                
                # PRE-TRAINING DATA QUALITY CHECKS (quick verification only)
                # Note: Extensive checks skipped - feature pruning already filtered low-quality features
                logger.debug(f"    CatBoost: Pre-training data quality checks")
                
                # Check 1: Constant features (should already be filtered by pruning)
                constant_features = []
                for i, feat_name in enumerate(feature_names):
                    if i < X_catboost.shape[1]:
                        feat_values = X_catboost[:, i]
                        if len(np.unique(feat_values[~np.isnan(feat_values)])) <= 1:
                            constant_features.append(feat_name)
                if constant_features:
                    logger.warning(f"    CatBoost: Found {len(constant_features)} constant features (should have been filtered by pruning): {constant_features[:5]}")
                
                # Check 2: Perfect separability (could explain 100% accuracy on small dataset)
                # Quick check: if any single feature perfectly separates classes, log warning
                if is_binary:
                    perfect_separators = []
                    for i, feat_name in enumerate(feature_names):
                        if i < X_catboost.shape[1]:
                            feat_values = X_catboost[:, i]
                            # Check if feature values perfectly separate classes
                            unique_vals_feat = np.unique(feat_values[~np.isnan(feat_values)])
                            if len(unique_vals_feat) == 2:  # Binary feature
                                # Check if this binary feature perfectly separates classes
                                class_0_mask = y == 0
                                class_1_mask = y == 1
                                if np.all(feat_values[class_0_mask] == unique_vals_feat[0]) and np.all(feat_values[class_1_mask] == unique_vals_feat[1]):
                                    perfect_separators.append(feat_name)
                    if perfect_separators:
                        logger.warning(f"    CatBoost: Found {len(perfect_separators)} features that perfectly separate classes (may explain 100% accuracy): {perfect_separators[:5]}")
                
                # Check 3: Dataset size vs feature count (high feature-to-sample ratio can cause overfitting)
                n_samples, n_features = X_catboost.shape
                feature_to_sample_ratio = n_features / n_samples if n_samples > 0 else 0
                if feature_to_sample_ratio > 0.2:
                    logger.warning(f"    CatBoost: High feature-to-sample ratio ({feature_to_sample_ratio:.3f}, {n_features} features / {n_samples} samples) - risk of overfitting")
                
                logger.debug(f"    CatBoost: Pre-training checks completed (dataset: {n_samples} samples, {n_features} features)")
                
                # Set up CV splitter and scoring
                from sklearn.model_selection import cross_val_score
                from TRAINING.ranking.utils.purged_time_series_split import PurgedTimeSeriesSplit
                from TRAINING.ranking.utils.leakage_filtering import _extract_horizon, _load_leakage_config
                
                # Calculate purge_overlap from target horizon
                leakage_config = _load_leakage_config()
                target_horizon_minutes = _extract_horizon(target_column, leakage_config) if target_column else None
                purge_buffer_bars = model_config.get('purge_buffer_bars', 5)
                if target_horizon_minutes is not None:
                    target_horizon_bars = target_horizon_minutes // data_interval_minutes
                    purge_overlap = target_horizon_bars + purge_buffer_bars
                else:
                    purge_overlap = 17  # Conservative default
                
                n_splits = model_config.get('n_splits', 3)
                tscv = PurgedTimeSeriesSplit(n_splits=n_splits, purge_overlap=purge_overlap)
                
                # Determine scoring metric
                if is_binary or is_multiclass:
                    from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score
                    scoring = 'roc_auc' if is_binary else 'accuracy'
                else:
                    scoring = 'r2'
                
                # CRITICAL: Add performance timing to identify bottlenecks
                import time
                total_start_time = time.time()
                cv_start_time = None
                cv_elapsed = None
                importance_elapsed = 0.0  # Initialize early for timing breakdown (used at line 2126)
                
                # PERFORMANCE FIX: Use manual CV loop with early stopping per fold for CatBoost
                # This keeps CV for rigor and stability analysis while making it efficient
                # cross_val_score() doesn't support early stopping per fold, causing 3-hour training times
                # Manual loop with early stopping reduces time from 3 hours to <30 minutes
                
                # Get early stopping rounds from config (default: 50)
                early_stopping_rounds = cb_config.get('od_wait', 20)
                try:
                    from CONFIG.config_loader import get_cfg
                    early_stopping_rounds = int(get_cfg("preprocessing.validation.early_stopping_rounds", default=early_stopping_rounds, config_name="preprocessing_config"))
                except Exception:
                    pass
                
                # Manual CV loop with early stopping per fold (for stability analysis)
                cv_start_time = time.time()
                cv_scores = []
                # PERFORMANCE FIX: Removed fold_importances - we compute importance only once after final fit
                # This avoids calling expensive PredictionValuesChange 3× (once per fold)
                importance_series = None  # Will be computed from final fit only
                
                logger.info(f"    CatBoost: Running CV with early stopping per fold (n_splits={n_splits}, early_stopping_rounds={early_stopping_rounds})")
                logger.info(f"    CatBoost: CV provides stability signal via scores (importance computed once after final fit to avoid 3× overhead)")
                
                from sklearn.base import clone
                with timed("catboost_cv", n_splits=n_splits, n_features=len(feature_names), n_samples=len(X_catboost)):
                    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_catboost, y)):
                        try:
                            X_train_fold, X_val_fold = X_catboost[train_idx], X_catboost[val_idx]
                            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                            
                            # Clone model for this fold
                            fold_model = clone(model)
                            
                            # Fit with early stopping using eval_set
                            fold_model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)])
                            
                            # Evaluate on validation set
                            if scoring == 'r2':
                                from sklearn.metrics import r2_score
                                y_pred = fold_model.predict(X_val_fold)
                                score = r2_score(y_val_fold, y_pred)
                            elif scoring == 'roc_auc':
                                from sklearn.metrics import roc_auc_score
                                y_proba = fold_model.predict_proba(X_val_fold)[:, 1] if hasattr(fold_model, 'predict_proba') else fold_model.predict(X_val_fold)
                                if len(np.unique(y_val_fold)) == 2:
                                    score = roc_auc_score(y_val_fold, y_proba)
                                else:
                                    score = np.nan
                            elif scoring == 'accuracy':
                                from sklearn.metrics import accuracy_score
                                y_pred = fold_model.predict(X_val_fold)
                                score = accuracy_score(y_val_fold, y_pred)
                            else:
                                score = np.nan
                            
                            cv_scores.append(score)
                            
                            # PERFORMANCE FIX: Skip expensive PredictionValuesChange during CV folds
                            # PredictionValuesChange is extremely slow (40-80 min per call with 531 features)
                            # We compute it only once after final fit to avoid 3× overhead
                            # CV scores provide stability signal without expensive importance computation
                            
                            # Log fold progress
                            actual_iterations = None
                            if hasattr(fold_model, 'base_model'):
                                actual_iterations = getattr(fold_model.base_model, 'tree_count_', None) or getattr(fold_model.base_model, 'iteration_count_', None)
                            elif hasattr(fold_model, 'tree_count_') or hasattr(fold_model, 'iteration_count_'):
                                actual_iterations = getattr(fold_model, 'tree_count_', None) or getattr(fold_model, 'iteration_count_', None)
                            
                            max_iterations = cb_config.get('iterations', 300)
                            logger.debug(f"    CatBoost: Fold {fold_idx + 1}/{n_splits} completed (iterations: {actual_iterations if actual_iterations else 'unknown'}/{max_iterations}, score: {score:.4f})")
                            
                        except Exception as e:
                            logger.debug(f"    CatBoost: Fold {fold_idx + 1} failed: {e}")
                            cv_scores.append(np.nan)
                
                cv_elapsed = time.time() - cv_start_time
                cv_scores = np.array(cv_scores)
                valid_cv_scores = cv_scores[~np.isnan(cv_scores)]
                
                # PERFORMANCE FIX: Removed per-fold importance aggregation
                # PredictionValuesChange is computed only once after final fit (avoids 3× overhead)
                # CV scores provide stability signal without expensive importance computation
                logger.info(f"    CatBoost: CV completed in {cv_elapsed/60:.2f} minutes ({len(valid_cv_scores)}/{n_splits} valid folds)")
                if len(valid_cv_scores) > 0:
                    logger.info(f"    CatBoost: CV scores - Mean: {np.nanmean(valid_cv_scores):.4f}, Std: {np.nanstd(valid_cv_scores):.4f}")
                logger.info(f"    CatBoost: Importance will be computed once after final fit (not during CV to avoid 3× overhead)")
                
                # Final fit on full dataset for importance (if needed for comparison)
                # Note: We already have fold-aggregated importance, but keep final fit for consistency
                from sklearn.model_selection import train_test_split
                X_train_fit, X_val_fit, y_train_fit, y_val_fit = train_test_split(
                    X_catboost, y, test_size=0.2, random_state=model_seed
                )
                
                # Early stopping should already be configured in cb_config (added above during config processing)
                # Verify it's set on the model (should be set from config during instantiation)
                if hasattr(model, 'base_model'):
                    model_params = model.base_model.get_params()
                elif hasattr(model, 'get_params'):
                    model_params = model.get_params()
                else:
                    model_params = {}
                
                od_type_set = model_params.get('od_type')
                od_wait_set = model_params.get('od_wait')
                if od_type_set and od_wait_set:
                    logger.debug(f"    CatBoost: Early stopping configured (od_type={od_type_set}, od_wait={od_wait_set})")
                else:
                    # Fallback: set directly if not already configured (shouldn't happen if config is correct)
                    logger.debug(f"    CatBoost: Setting early stopping params directly (od_type='Iter', od_wait=20)")
                    if hasattr(model, 'base_model'):
                        model.base_model.set_params(od_type='Iter', od_wait=20)
                    elif hasattr(model, 'set_params'):
                        model.set_params(od_type='Iter', od_wait=20)
                
                # CRITICAL: Verify GPU is actually configured before training
                # The log line "✅ Using GPU" may be a capability check, not actual execution mode
                if use_gpu:
                    # Get actual model params (from base_model if wrapped)
                    if hasattr(model, 'base_model'):
                        actual_params = model.base_model.get_params()
                    elif hasattr(model, 'get_params'):
                        actual_params = model.get_params()
                    else:
                        actual_params = {}
                    
                    task_type = actual_params.get('task_type')
                    if task_type != 'GPU':
                        raise RuntimeError(
                            f"CatBoost not configured for GPU! Expected task_type='GPU', got task_type='{task_type}'. "
                            f"Model params: {list(actual_params.keys())[:10]}. "
                            f"To verify GPU usage, run 'nvidia-smi -l 1' in another terminal during training."
                        )
                    else:
                        logger.info(f"    CatBoost: Verified GPU configuration (task_type={task_type}, devices={actual_params.get('devices', 'not set')})")
                        # Add verbose logging to confirm GPU usage during training
                        # Note: Only set 'verbose' parameter - 'logging_level' conflicts with it
                        if hasattr(model, 'base_model'):
                            model.base_model.set_params(verbose=1)
                        elif hasattr(model, 'set_params'):
                            model.set_params(verbose=1)
                
                # CRITICAL: Add timeout mechanism and duration logging for CatBoost training
                # This helps identify if "hours" is actually CatBoost or cumulative overhead
                import signal
                
                fit_start_time = time.time()
                max_training_time_seconds = 30 * 60  # 30 minutes hard limit
                training_timed_out = False
                
                def timeout_handler(signum, frame):
                    nonlocal training_timed_out
                    training_timed_out = True
                    raise TimeoutError(f"CatBoost training exceeded {max_training_time_seconds/60:.0f} minute timeout")
                
                # Set up timeout (Unix only - Windows will skip this)
                try:
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(int(max_training_time_seconds))
                except (AttributeError, OSError):
                    # Windows doesn't support SIGALRM - log warning but continue
                    logger.warning(f"    CatBoost: Timeout not available on this platform (Windows), using soft timeout check")
                
                try:
                    # Use threading utilities for smart thread management
                    if _THREADING_UTILITIES_AVAILABLE:
                        # Get thread plan based on family and GPU usage
                        plan = plan_for_family('CatBoost', total_threads=default_threads())
                        # Use thread_guard context manager for safe thread control
                        with thread_guard(omp=plan['OMP'], mkl=plan['MKL']):
                            with timed("catboost_fit", n_features=len(feature_names), n_samples=len(X_train_fit)):
                                # Fit with early stopping using eval_set
                                model.fit(X_train_fit, y_train_fit, eval_set=[(X_val_fit, y_val_fit)])
                                train_score = model.score(X_train_fit, y_train_fit)
                                val_score = model.score(X_val_fit, y_val_fit)
                    else:
                        # Fallback: manual thread management
                        with timed("catboost_fit", n_features=len(feature_names), n_samples=len(X_train_fit)):
                            # Fit with early stopping using eval_set
                            model.fit(X_train_fit, y_train_fit, eval_set=[(X_val_fit, y_val_fit)])
                            train_score = model.score(X_train_fit, y_train_fit)
                            val_score = model.score(X_val_fit, y_val_fit)
                    
                    # Check soft timeout (for Windows or if SIGALRM didn't fire)
                    fit_elapsed = time.time() - fit_start_time
                    if fit_elapsed > max_training_time_seconds:
                        logger.error(f"    CatBoost: Training took {fit_elapsed/60:.1f} minutes (exceeded {max_training_time_seconds/60:.0f} min limit)")
                        raise TimeoutError(f"CatBoost training exceeded {max_training_time_seconds/60:.0f} minute timeout")
                    
                    # Log actual iteration count vs max (to verify early stopping worked)
                    if hasattr(model, 'base_model'):
                        actual_iterations = getattr(model.base_model, 'tree_count_', None) or getattr(model.base_model, 'iteration_count_', None)
                        # Try to get best iteration from early stopping
                        best_iteration = getattr(model.base_model, 'best_iteration_', None)
                    elif hasattr(model, 'tree_count_') or hasattr(model, 'iteration_count_'):
                        actual_iterations = getattr(model, 'tree_count_', None) or getattr(model, 'iteration_count_', None)
                        best_iteration = getattr(model, 'best_iteration_', None)
                    else:
                        actual_iterations = None
                        best_iteration = None
                    
                    max_iterations = cb_config.get('iterations', 300)
                    fit_elapsed_minutes = fit_elapsed / 60
                    
                    # Log train/val scores and overfitting indicators
                    train_val_gap = train_score - val_score if 'val_score' in locals() else None
                    logger.info(f"    CatBoost: Final fit completed in {fit_elapsed_minutes:.2f} minutes "
                              f"(iterations: {actual_iterations if actual_iterations else 'unknown'}/{max_iterations}, "
                              f"best_iteration: {best_iteration if best_iteration else 'N/A'}, "
                              f"early_stopping: {'triggered' if actual_iterations and actual_iterations < max_iterations else 'not triggered'})")
                    val_score_str = f"{val_score:.4f}" if 'val_score' in locals() else 'N/A'
                    train_val_gap_str = f"{train_val_gap:.4f}" if train_val_gap is not None else 'N/A'
                    logger.info(f"    CatBoost: Scores - Train: {train_score:.4f}, Val: {val_score_str}, "
                              f"Gap: {train_val_gap_str} "
                              f"{'(OVERFITTING)' if train_val_gap and train_val_gap > 0.3 else ''}")
                    
                    # Log total time breakdown (with importance time)
                    total_elapsed = time.time() - total_start_time
                    total_elapsed_minutes = total_elapsed / 60
                    cv_time_str = f"{cv_elapsed/60:.2f}" if cv_elapsed else "0.00"
                    fit_time_str = f"{fit_elapsed_minutes:.2f}"
                    importance_time_str = f"{importance_elapsed/60:.2f}"
                    remaining_time = total_elapsed - (cv_elapsed if cv_elapsed else 0) - fit_elapsed - importance_elapsed
                    remaining_time_str = f"{remaining_time/60:.2f}" if remaining_time > 0 else "0.00"
                    logger.info(f"    CatBoost: Total time breakdown - CV: {cv_time_str} min, Fit: {fit_time_str} min, Importance: {importance_time_str} min, Other: {remaining_time_str} min, Total: {total_elapsed_minutes:.2f} min")
                    
                except TimeoutError:
                    logger.error(f"    CatBoost: Training timeout after {max_training_time_seconds/60:.0f} minutes - aborting")
                    # Return early with empty importance
                    return None, pd.Series(0.0, index=feature_names), importance_method, 0.0
                finally:
                    # Cancel timeout
                    try:
                        signal.alarm(0)
                    except (AttributeError, OSError):
                        pass  # Windows doesn't support SIGALRM
                
                # OVERFITTING DETECTION: Policy-based gating for expensive importance computation
                # Tree-based models can easily overfit to 100% training accuracy
                # This indicates the model is memorizing data and will take a long time to compute importance
                
                # Load importance policy from config (SST)
                try:
                    from CONFIG.config_loader import get_cfg
                    importance_config = get_cfg('safety.leakage_detection.feature_importance', default={}, config_name='safety_config')
                    skip_on_overfit = importance_config.get('importance_skip_on_overfit', True)
                    fallback_method = importance_config.get('importance_fallback', 'gain')
                except Exception:
                    importance_config = {}
                    skip_on_overfit = True
                    fallback_method = 'gain'
                
                # Unified overfitting detection
                from TRAINING.ranking.utils.overfitting_detection import should_skip_expensive_importance
                
                cv_mean = np.nanmean(cv_scores) if cv_scores is not None and len(cv_scores) > 0 else None
                should_skip, skip_reason, skip_metadata = should_skip_expensive_importance(
                    train_score=train_score,
                    cv_score=cv_mean,
                    val_score=val_score if 'val_score' in locals() else None,
                    n_features=len(feature_names),
                    config=importance_config
                )
                
                # Log decision with actual values
                cv_mean_str = f"{cv_mean:.4f}" if cv_mean is not None else 'N/A'
                val_score_str = f"{val_score:.4f}" if 'val_score' in locals() and val_score is not None else 'N/A'
                logger.info(f"    CatBoost: Overfitting check - train={train_score:.4f}, cv={cv_mean_str}, "
                          f"val={val_score_str}, "
                          f"decision={'SKIP' if should_skip else 'RUN'}, reason={skip_reason}")
                
                if should_skip and skip_on_overfit:
                    logger.warning(f"    CatBoost: ⚠️  OVERFITTING DETECTED - {skip_reason}")
                    logger.warning(f"    CatBoost: Skipping expensive PredictionValuesChange, using fallback: {fallback_method}")
                    
                    # Use deterministic fallback importance
                    if fallback_method == 'gain':
                        # Use CatBoost native FeatureImportance (cheap)
                        try:
                            if hasattr(model, 'base_model'):
                                importance_raw = model.base_model.get_feature_importance(type='FeatureImportance')
                            else:
                                importance_raw = model.get_feature_importance(type='FeatureImportance')
                            importance = pd.Series(importance_raw, index=feature_names)
                            logger.info(f"    CatBoost: Using fallback FeatureImportance (gain-based)")
                        except Exception as e:
                            logger.warning(f"    CatBoost: Fallback FeatureImportance failed: {e}, using zeros")
                            importance = pd.Series(0.0, index=feature_names)
                    elif fallback_method == 'split':
                        # Use split-based importance (if available)
                        try:
                            if hasattr(model, 'base_model'):
                                importance_raw = model.base_model.get_feature_importance(type='Split')
                            else:
                                importance_raw = model.get_feature_importance(type='Split')
                            importance = pd.Series(importance_raw, index=feature_names)
                            logger.info(f"    CatBoost: Using fallback Split importance")
                        except Exception as e:
                            logger.warning(f"    CatBoost: Fallback Split importance failed: {e}, using zeros")
                            importance = pd.Series(0.0, index=feature_names)
                    else:  # 'none'
                        importance = pd.Series(0.0, index=feature_names)
                        logger.info(f"    CatBoost: Using zero importance (fallback=none)")
                    
                    # Track that we used fallback
                    importance_elapsed = 0.0  # Fallback is fast
                    return model, importance, importance_method, train_score
                
                # Log warnings for overfitting (but don't skip if policy doesn't require it)
                if 'val_score' in locals() and val_score is not None:
                    train_val_gap = train_score - val_score
                    if train_val_gap > 0.3:
                        logger.warning(f"    CatBoost: ⚠️  OVERFITTING WARNING - Large train/val gap: {train_val_gap:.4f} (train={train_score:.4f}, val={val_score:.4f})")
                        logger.warning(f"    CatBoost: Model is overfitting. Consider: reduce depth, increase regularization, reduce iterations")
                    elif train_val_gap > 0.1:
                        logger.info(f"    CatBoost: Moderate train/val gap: {train_val_gap:.4f} (train={train_score:.4f}, val={val_score:.4f}) - monitor for overfitting")
                
                if cv_mean is not None:
                    cv_train_gap = train_score - cv_mean
                    if cv_train_gap > 0.3:
                        logger.warning(f"    CatBoost: ⚠️  OVERFITTING WARNING - Large CV/Train gap: {cv_train_gap:.4f} (train={train_score:.4f}, CV={cv_mean:.4f})")
                        logger.warning(f"    CatBoost: CV score is much lower than training score - model is overfitting to training data")
                    elif cv_train_gap > 0.1:
                        logger.info(f"    CatBoost: Moderate CV/Train gap: {cv_train_gap:.4f} (train={train_score:.4f}, CV={cv_mean:.4f})")
                
                # PERFORMANCE FIX: Compute PredictionValuesChange only once after final fit
                # This is the ONLY place we compute expensive PVC (avoids 3× overhead from CV folds)
                # CV scores provide stability signal; importance is for explanation/reporting only
                # WARNING: Using final-fit importance for feature selection would cause leakage-by-selection
                # This importance is for explanation/reporting, not for selection decisions
                
                # CACHING: PVC results should be cached keyed by featureset fingerprint
                # Cache key: (model_family, params_hash, featureset_hash, dataset_fingerprint, importance_kind)
                # TODO: Implement cache lookup here if caching infrastructure exists
                # Example cache key computation:
                #   from TRAINING.common.utils.config_hashing import compute_config_hash
                #   cache_key_data = {
                #       'model_family': 'catboost',
                #       'params_hash': compute_config_hash(model_params),
                #       'featureset_hash': compute_config_hash({'features': sorted(feature_names)}),
                #       'dataset_fingerprint': compute_config_hash({'shape': X_train_fit.shape, 'target': target_column}),
                #       'importance_kind': 'PredictionValuesChange'
                #   }
                #   cache_key = compute_config_hash(cache_key_data)
                
                # Load timeout from config
                max_importance_time_seconds = importance_config.get('importance_max_wall_minutes', 30) * 60
                
                importance_start_time = time.time()
                importance_elapsed = 0.0
                
                # PERFORMANCE AUDIT: Track CatBoost importance computation
                try:
                    from TRAINING.common.utils.performance_audit import get_auditor
                    auditor = get_auditor()
                    if auditor.enabled:
                        fingerprint_kwargs = {
                            'data_shape': X_train_fit.shape,
                            'n_features': len(feature_names),
                            'importance_type': 'PredictionValuesChange'
                        }
                        fingerprint = auditor._compute_fingerprint('catboost.get_feature_importance', **fingerprint_kwargs)
                except Exception:
                    auditor = None
                    fingerprint = None
                
                # Get model params for logging
                if hasattr(model, 'base_model'):
                    model_params = model.base_model.get_params()
                else:
                    model_params = model.get_params()
                trees = model_params.get('iterations', 'unknown')
                depth = model_params.get('depth', 'unknown')
                
                try:
                    with timed("catboost_pvc", 
                               n_features=len(feature_names), 
                               n_samples=X_train_fit.shape[0],
                               trees=trees,
                               depth=depth):
                        
                        # Process-based timeout for PVC
                        import multiprocessing
                        
                        # Use module-level function (picklable for multiprocessing)
                        # Create process and queue
                        result_queue = multiprocessing.Queue()
                        p = multiprocessing.Process(
                            target=_compute_catboost_importance_worker,
                            args=(model, X_train_fit, feature_names, result_queue)
                        )
                        p.start()
                        p.join(timeout=max_importance_time_seconds)
                        
                        if p.is_alive():
                            # Timeout - kill process
                            p.terminate()
                            p.join(timeout=5)
                            if p.is_alive():
                                p.kill()
                            logger.warning(f"    CatBoost: PVC_TIMEOUT after {max_importance_time_seconds/60:.0f} minutes")
                            raise TimeoutError(f"Importance computation exceeded {max_importance_time_seconds/60:.0f} minute timeout")
                        
                        # Get result
                        if not result_queue.empty():
                            status, result = result_queue.get()
                            if status == 'success':
                                importance = result
                            else:
                                raise Exception(f"Importance computation failed: {result}")
                        else:
                            raise Exception("Importance computation process returned no result")
                    
                    importance_elapsed = time.time() - importance_start_time
                    logger.info(f"    CatBoost: Importance computation completed in {importance_elapsed/60:.2f} minutes (computed once after final fit)")
                    
                    # Track call
                    if auditor and auditor.enabled:
                        auditor.track_call(
                            func_name='catboost.get_feature_importance',
                            duration=importance_elapsed,
                            rows=X_train_fit.shape[0],
                            cols=len(feature_names),
                            stage='feature_selection',
                            cache_hit=False,
                            input_fingerprint=fingerprint
                        )
                        
                except TimeoutError:
                    logger.warning(f"    CatBoost: Importance computation timed out, using fallback: {fallback_method}")
                    importance_elapsed = time.time() - importance_start_time
                    
                    # Use fallback importance
                    if fallback_method == 'gain':
                        try:
                            if hasattr(model, 'base_model'):
                                importance_raw = model.base_model.get_feature_importance(type='FeatureImportance')
                            else:
                                importance_raw = model.get_feature_importance(type='FeatureImportance')
                            importance = pd.Series(importance_raw, index=feature_names)
                        except Exception as e:
                            logger.warning(f"    CatBoost: Fallback FeatureImportance failed: {e}, using zeros")
                            importance = pd.Series(0.0, index=feature_names)
                    else:
                        importance = pd.Series(0.0, index=feature_names)
                        
                except Exception as e:
                    logger.warning(f"    CatBoost: Failed to extract importance from final model: {e}")
                    importance = pd.Series(0.0, index=feature_names)
                    importance_elapsed = time.time() - importance_start_time
                
                # PERFORMANCE AUDIT: Track train_model_and_get_importance completion
                train_duration = time.time() - train_start_time
                if auditor and auditor.enabled:
                    auditor.track_call(
                        func_name='train_model_and_get_importance',
                        duration=train_duration,
                        rows=X.shape[0],
                        cols=len(feature_names),
                        stage='feature_selection',
                        cache_hit=False,
                        input_fingerprint=fingerprint,
                        model_family=model_family
                    )
                
                # Return model, final-fit importance, method, and train_score
                # Note: Importance is computed once after final fit (not during CV to avoid 3× overhead)
                # CV scores provide stability signal; importance is for explanation/reporting only
                return model, importance, importance_method, train_score
                
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
        from TRAINING.common.utils.sklearn_safe import make_sklearn_dense_X
        
        # Lasso doesn't handle NaNs - use sklearn-safe conversion
        X_dense, feature_names_dense = make_sklearn_dense_X(X, feature_names)
        
        # Clean config using systematic helper (removes duplicates and unknown params)
        # Note: Lasso is deterministic and doesn't accept random_state parameter
        extra = {}  # FIX: Lasso has no random component, remove seed
        lasso_config = _clean_config_for_estimator(Lasso, model_config, extra, "lasso")
        
        # Instantiate with cleaned config + explicit params
        model = Lasso(**lasso_config, **extra)
        # Use threading utilities for smart thread management
        if _THREADING_UTILITIES_AVAILABLE:
            # Get thread plan based on family (linear models use MKL threads)
            plan = plan_for_family('Lasso', total_threads=default_threads())
            # Use thread_guard context manager for safe thread control
            with thread_guard(omp=plan['OMP'], mkl=plan['MKL']):
                model.fit(X_dense, y)
                train_score = model.score(X_dense, y)
        else:
            # Fallback: manual thread management
            model.fit(X_dense, y)
            train_score = model.score(X_dense, y)
        
        # Update feature_names to match dense array
        feature_names = feature_names_dense
    
    elif model_family == 'ridge':
        # Ridge Regression/Classification - Linear model with L2 regularization
        from sklearn.linear_model import Ridge, RidgeClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from TRAINING.common.utils.sklearn_safe import make_sklearn_dense_X
        
        # Ridge doesn't handle NaNs - use sklearn-safe conversion
        X_dense, feature_names_dense = make_sklearn_dense_X(X, feature_names)
        
        # Determine task type
        unique_vals = np.unique(y[~np.isnan(y)])
        is_binary = len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0})
        is_multiclass = len(unique_vals) <= 10 and all(
            isinstance(v, (int, np.integer)) or (isinstance(v, float) and v.is_integer())
            for v in unique_vals
        )
        
        # CRITICAL: Use correct estimator based on task type
        # For classification: RidgeClassifier (not Ridge regression)
        # For regression: Ridge
        if is_binary or is_multiclass:
            est_cls = RidgeClassifier
        else:
            est_cls = Ridge
        
        # Clean config using systematic helper
        extra = {}  # FIX: Ridge/RidgeClassifier is deterministic, no seed parameter
        ridge_config = _clean_config_for_estimator(est_cls, model_config, extra, "ridge")
        
        # CRITICAL: Ridge requires scaling for proper convergence
        # Pipeline ensures scaling happens within each CV fold (no leakage)
        steps = [
            ('scaler', StandardScaler()),  # Required for Ridge convergence
            ('model', est_cls(**ridge_config, **extra))
        ]
        pipeline = Pipeline(steps)
        
        # Fit on full data for importance extraction (CV is done elsewhere)
        try:
            pipeline.fit(X_dense, y)
            train_score = pipeline.score(X_dense, y)
        except Exception as e:
            logger.warning(f"    ridge: Failed to fit: {e}")
            # Use uniform fallback
            importance_values, fallback_reason = normalize_importance(
                raw_importance=None,
                n_features=len(feature_names_dense),
                family="ridge",
                feature_names=feature_names_dense
            )
            model = type('DummyModel', (), {'importance': importance_values, '_fallback_reason': fallback_reason})()
            feature_names = feature_names_dense
        else:
            # Extract coefficients from the fitted model
            model = pipeline.named_steps['model']
            # FIX: Handle both 1D (binary) and 2D (multiclass) coef_ shapes
            coef = model.coef_
            if len(coef.shape) > 1:
                # Multiclass: use max absolute coefficient across classes
                importance_values = np.abs(coef).max(axis=0)
            else:
                # Binary or regression: use absolute coefficients
                importance_values = np.abs(coef)
            
            # Update feature_names to match dense array
            feature_names = feature_names_dense
            
            # Validate importance is not all zeros
            if np.all(importance_values == 0) or np.sum(importance_values) == 0:
                logger.warning(f"    ridge: All coefficients are zero (over-regularized or no signal). Marking as invalid.")
                # FIX: Don't use uniform fallback - it injects randomness into consensus
                # Instead, raise exception to mark model as invalid (will be caught and marked as failed)
                raise ValueError("ridge: All coefficients are zero (over-regularized or no signal). Model invalid.")
            else:
                # Normalize importance to sum to 1.0 for consistency
                total = np.sum(importance_values)
                if total > 0:
                    importance_values = importance_values / total
                model = type('DummyModel', (), {'importance': importance_values})()
    
    elif model_family == 'elastic_net':
        # Elastic Net Regression/Classification - Linear model with L1+L2 regularization
        from sklearn.linear_model import ElasticNet, LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from TRAINING.common.utils.sklearn_safe import make_sklearn_dense_X
        
        # Elastic Net doesn't handle NaNs - use sklearn-safe conversion
        X_dense, feature_names_dense = make_sklearn_dense_X(X, feature_names)
        
        # Determine task type
        unique_vals = np.unique(y[~np.isnan(y)])
        is_binary = len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0})
        is_multiclass = len(unique_vals) <= 10 and all(
            isinstance(v, (int, np.integer)) or (isinstance(v, float) and v.is_integer())
            for v in unique_vals
        )
        
        # CRITICAL: Use correct estimator based on task type
        # For classification: LogisticRegression with penalty='elasticnet' and solver='saga'
        # For regression: ElasticNet
        if is_binary or is_multiclass:
            # LogisticRegression with elasticnet penalty
            est_cls = LogisticRegression
            # ElasticNet requires solver='saga' for penalty='elasticnet'
            elastic_net_config = model_config.copy()
            elastic_net_config['penalty'] = 'elasticnet'
            elastic_net_config['solver'] = 'saga'  # Required for elasticnet penalty
            # l1_ratio maps to ElasticNet's l1_ratio (0 = pure L2, 1 = pure L1)
            if 'l1_ratio' not in elastic_net_config:
                elastic_net_config['l1_ratio'] = model_config.get('l1_ratio', 0.5)
            # alpha maps to C (inverse regularization strength)
            if 'alpha' in elastic_net_config:
                # Convert alpha to C (C = 1/alpha for consistency with sklearn)
                alpha = elastic_net_config.pop('alpha')
                elastic_net_config['C'] = 1.0 / alpha if alpha > 0 else 1.0
            elif 'C' not in elastic_net_config:
                elastic_net_config['C'] = 1.0  # Default C=1.0
        else:
            # ElasticNet regression
            est_cls = ElasticNet
            elastic_net_config = model_config.copy()
        
        # Clean config using systematic helper
        extra = {"random_state": model_seed}  # FIX: ElasticNet uses random_state for coordinate descent

        # FAIL-FAST: Set reasonable max_iter limit to avoid long-running fits
        # Default max_iter is 1000, but saga solver can be very slow
        # Use a lower limit (500) to fail faster if it's not converging or going to zero
        original_max_iter = elastic_net_config.get('max_iter', 1000)
        if 'max_iter' not in elastic_net_config:
            elastic_net_config['max_iter'] = 500  # Reduced from default 1000 for fail-fast
        elif elastic_net_config.get('max_iter', 1000) > 500:
            # Cap at 500 for fail-fast behavior
            logger.debug(f"    elastic_net: Capping max_iter at 500 for fail-fast (was {elastic_net_config['max_iter']})")
            elastic_net_config['max_iter'] = 500
        
        elastic_net_config_clean = _clean_config_for_estimator(est_cls, elastic_net_config, extra, "elastic_net")
        
        # CRITICAL: Elastic Net requires scaling for proper convergence
        # Pipeline ensures scaling happens within each CV fold (no leakage)
        steps = [
            ('scaler', StandardScaler()),  # Required for ElasticNet convergence
            ('model', est_cls(**elastic_net_config_clean, **extra))
        ]
        pipeline = Pipeline(steps)
        
        # FAIL-FAST: Quick pre-check with very small max_iter to detect obvious failures early
        # This catches over-regularization cases that would zero out quickly
        if original_max_iter > 50:
            try:
                quick_config = elastic_net_config_clean.copy()
                quick_config['max_iter'] = 50  # Very quick check
                quick_steps = [
                    ('scaler', StandardScaler()),
                    ('model', est_cls(**quick_config, **extra))
                ]
                quick_pipeline = Pipeline(quick_steps)
                quick_pipeline.fit(X_dense, y)
                quick_model = quick_pipeline.named_steps['model']
                quick_coef = quick_model.coef_
                if len(quick_coef.shape) > 1:
                    quick_importance = np.abs(quick_coef).max(axis=0)
                else:
                    quick_importance = np.abs(quick_coef)
                
                # If quick check shows all zeros, fail immediately without full fit
                if np.all(quick_importance == 0) or np.sum(quick_importance) == 0:
                    raise ValueError("elastic_net: All coefficients are zero (over-regularized or no signal). Model invalid.")
            except ValueError:
                # Re-raise ValueError (our fail-fast signal)
                raise
            except Exception:
                # Other exceptions from quick check - continue with full fit
                pass
        
        # Add cross_val_score for parallelization (matching target ranking behavior)
        # Set up CV splitter and scoring
        from sklearn.model_selection import cross_val_score
        from TRAINING.ranking.utils.purged_time_series_split import PurgedTimeSeriesSplit
        from TRAINING.ranking.utils.leakage_filtering import _extract_horizon, _load_leakage_config
        
        # Calculate purge_overlap from target horizon
        leakage_config = _load_leakage_config()
        target_horizon_minutes = _extract_horizon(target_column, leakage_config) if target_column else None
        purge_buffer_bars = elastic_net_config_clean.get('purge_buffer_bars', 5)
        if target_horizon_minutes is not None:
            target_horizon_bars = target_horizon_minutes // data_interval_minutes
            purge_overlap = target_horizon_bars + purge_buffer_bars
        else:
            purge_overlap = 17  # Conservative default
        
        n_splits = elastic_net_config_clean.get('n_splits', 3)
        tscv = PurgedTimeSeriesSplit(n_splits=n_splits, purge_overlap=purge_overlap)
        
        # Determine task type and scoring
        unique_vals = np.unique(y[~np.isnan(y)])
        is_binary = len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0})
        is_multiclass = len(unique_vals) <= 10 and all(
            isinstance(v, (int, np.integer)) or (isinstance(v, float) and v.is_integer())
            for v in unique_vals
        )
        scoring = 'roc_auc' if is_binary else ('accuracy' if is_multiclass else 'r2')
        
        # Run cross_val_score for parallelization (if cv_n_jobs > 1)
        if cv_n_jobs > 1:
            try:
                cv_scores = cross_val_score(pipeline, X_dense, y, cv=tscv, scoring=scoring, n_jobs=cv_n_jobs, error_score=np.nan)
                logger.debug(f"    elastic_net: CV scores (mean={np.nanmean(cv_scores):.4f}, std={np.nanstd(cv_scores):.4f})")
            except Exception as e:
                logger.debug(f"    elastic_net: CV failed (continuing with single fit): {e}")
        
        # Fit on full data for importance extraction (CV is done elsewhere)
        try:
            pipeline.fit(X_dense, y)
            train_score = pipeline.score(X_dense, y)
        except Exception as e:
            logger.warning(f"    elastic_net: Failed to fit: {e}")
            # Use uniform fallback
            importance_values, fallback_reason = normalize_importance(
                raw_importance=None,
                n_features=len(feature_names_dense),
                family="elastic_net",
                feature_names=feature_names_dense
            )
            model = type('DummyModel', (), {'importance': importance_values, '_fallback_reason': fallback_reason})()
            feature_names = feature_names_dense
        else:
            # FAIL-FAST: Check coefficients immediately after fit
            # Extract coefficients from the fitted model
            model = pipeline.named_steps['model']
            # FIX: Handle both 1D (binary) and 2D (multiclass) coef_ shapes
            coef = model.coef_
            if len(coef.shape) > 1:
                # Multiclass: use max absolute coefficient across classes
                importance_values = np.abs(coef).max(axis=0)
            else:
                # Binary or regression: use absolute coefficients
                importance_values = np.abs(coef)
            
            # Update feature_names to match dense array
            feature_names = feature_names_dense
            
            # FAIL-FAST: Validate importance immediately and raise exception right away
            # This prevents wasting time on downstream processing
            if np.all(importance_values == 0) or np.sum(importance_values) == 0:
                # Raise exception immediately - don't log warning first, just fail fast
                raise ValueError("elastic_net: All coefficients are zero (over-regularized or no signal). Model invalid.")
            else:
                # Normalize importance to sum to 1.0 for consistency
                total = np.sum(importance_values)
                if total > 0:
                    importance_values = importance_values / total
                model = type('DummyModel', (), {'importance': importance_values})()
    
    elif model_family == 'logistic_regression':
        # Standalone Logistic Regression for classification (binary/multiclass only)
        # Task-type filtering in process_single_symbol() prevents this from running on regression
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from TRAINING.common.utils.sklearn_safe import make_sklearn_dense_X
        
        # LogisticRegression doesn't handle NaNs - use sklearn-safe conversion
        X_dense, feature_names_dense = make_sklearn_dense_X(X, feature_names)
        
        # Determine task type (should always be classification due to task-type filtering)
        unique_vals = np.unique(y[~np.isnan(y)])
        is_binary = len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0})
        is_multiclass = len(unique_vals) <= 10 and all(
            isinstance(v, (int, np.integer)) or (isinstance(v, float) and v.is_integer())
            for v in unique_vals
        )
        
        if not (is_binary or is_multiclass):
            # This shouldn't happen due to task-type filtering, but handle gracefully
            logger.warning(f"    logistic_regression: Target appears to be regression, returning empty importance")
            importance_values = np.zeros(len(feature_names_dense))
            model = type('DummyModel', (), {'coef_': importance_values, 'predict': lambda x: np.zeros(len(x))})()
            feature_names = feature_names_dense
        else:
            # Build LogisticRegression config
            lr_config = model_config.copy()
            extra = {"random_state": model_seed}
            
            # Clean config for LogisticRegression
            lr_config_clean = _clean_config_for_estimator(LogisticRegression, lr_config, extra, "logistic_regression")
            
            # Pipeline with scaling for proper convergence
            steps = [
                ('scaler', StandardScaler()),
                ('model', LogisticRegression(**lr_config_clean, **extra))
            ]
            pipeline = Pipeline(steps)
            
            try:
                pipeline.fit(X_dense, y)
                train_score = pipeline.score(X_dense, y)
                
                # Extract coefficients
                model = pipeline.named_steps['model']
                coef = model.coef_
                if len(coef.shape) > 1:
                    # Multiclass: use max absolute coefficient across classes
                    importance_values = np.abs(coef).max(axis=0)
                else:
                    # Binary: use absolute coefficients
                    importance_values = np.abs(coef).ravel()
                
                # Update feature_names to match dense array
                feature_names = feature_names_dense
                
                # Normalize importance to sum to 1.0
                total = np.sum(importance_values)
                if total > 0:
                    importance_values = importance_values / total
                
                # Wrap model for consistent API
                model = pipeline  # Use full pipeline for predict()
                
            except Exception as e:
                logger.warning(f"    logistic_regression: Failed to fit: {e}")
                importance_values = np.zeros(len(feature_names_dense))
                model = type('DummyModel', (), {'coef_': importance_values, 'predict': lambda x: np.zeros(len(x))})()
                feature_names = feature_names_dense
    
    elif model_family == 'ftrl_proximal':
        # FTRL-Proximal approximation using SGDClassifier with elasticnet penalty
        # True FTRL is online learning; we approximate with mini-batch SGD
        # Task-type filtering ensures this only runs on binary classification
        from sklearn.linear_model import SGDClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from TRAINING.common.utils.sklearn_safe import make_sklearn_dense_X
        
        # SGDClassifier doesn't handle NaNs - use sklearn-safe conversion
        X_dense, feature_names_dense = make_sklearn_dense_X(X, feature_names)
        
        # Determine task type (should always be binary due to task-type filtering)
        unique_vals = np.unique(y[~np.isnan(y)])
        is_binary = len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0})
        
        if not is_binary:
            logger.warning(f"    ftrl_proximal: Target is not binary, returning empty importance")
            importance_values = np.zeros(len(feature_names_dense))
            model = type('DummyModel', (), {'coef_': importance_values, 'predict': lambda x: np.zeros(len(x))})()
            feature_names = feature_names_dense
        else:
            # Build FTRL-like config using SGDClassifier
            # FTRL uses L1+L2 regularization (elasticnet) with adaptive learning rate
            ftrl_config = {
                'loss': 'log_loss',  # Logistic regression loss
                'penalty': 'elasticnet',  # L1 + L2 regularization (FTRL-like)
                'alpha': model_config.get('alpha', 0.1),  # Regularization strength
                'l1_ratio': model_config.get('l1_ratio', 0.5),  # L1 vs L2 balance
                'max_iter': model_config.get('max_iter', 1000),
                'tol': 1e-4,
                'learning_rate': 'adaptive',  # FTRL uses adaptive learning
                'eta0': 0.1,  # Initial learning rate
                'early_stopping': True,
                'validation_fraction': 0.1,
                'n_iter_no_change': 5,
            }
            extra = {"random_state": model_seed}
            
            # Clean config for SGDClassifier
            ftrl_config_clean = _clean_config_for_estimator(SGDClassifier, ftrl_config, extra, "ftrl_proximal")
            
            # Pipeline with scaling
            steps = [
                ('scaler', StandardScaler()),
                ('model', SGDClassifier(**ftrl_config_clean, **extra))
            ]
            pipeline = Pipeline(steps)
            
            try:
                pipeline.fit(X_dense, y)
                train_score = pipeline.score(X_dense, y)
                
                # Extract coefficients
                model = pipeline.named_steps['model']
                coef = model.coef_
                importance_values = np.abs(coef).ravel()
                
                # Update feature_names to match dense array
                feature_names = feature_names_dense
                
                # Normalize importance to sum to 1.0
                total = np.sum(importance_values)
                if total > 0:
                    importance_values = importance_values / total
                
                # Wrap model for consistent API
                model = pipeline  # Use full pipeline for predict()
                
            except Exception as e:
                logger.warning(f"    ftrl_proximal: Failed to fit: {e}")
                importance_values = np.zeros(len(feature_names_dense))
                model = type('DummyModel', (), {'coef_': importance_values, 'predict': lambda x: np.zeros(len(x))})()
                feature_names = feature_names_dense
    
    elif model_family == 'ngboost':
        # NGBoost - Probabilistic gradient boosting with uncertainty estimation
        # Task-type filtering ensures this only runs on regression and binary
        try:
            from ngboost import NGBRegressor, NGBClassifier
            from ngboost.distns import Normal, Bernoulli
            NGBOOST_AVAILABLE = True
        except ImportError:
            NGBOOST_AVAILABLE = False
            logger.warning("    ngboost: NGBoost not installed. Install with: pip install ngboost")
        
        if not NGBOOST_AVAILABLE:
            importance_values = np.zeros(len(feature_names))
            model = type('DummyModel', (), {'feature_importances_': importance_values, 'predict': lambda x: np.zeros(len(x))})()
        else:
            from sklearn.preprocessing import StandardScaler
            from TRAINING.common.utils.sklearn_safe import make_sklearn_dense_X
            
            # NGBoost doesn't handle NaNs - use sklearn-safe conversion
            X_dense, feature_names_dense = make_sklearn_dense_X(X, feature_names)
            
            # Scale features for better convergence
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_dense)
            
            # Determine task type
            unique_vals = np.unique(y[~np.isnan(y)])
            is_binary = len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0})
            
            # Build NGBoost config
            ngb_config = {
                'n_estimators': model_config.get('n_estimators', 200),
                'learning_rate': model_config.get('learning_rate', 0.05),
                'minibatch_frac': model_config.get('minibatch_frac', 0.8),
                'verbose': False,
            }
            
            try:
                if is_binary:
                    # Binary classification with Bernoulli distribution
                    ngb_model = NGBClassifier(
                        Dist=Bernoulli,
                        random_state=model_seed,
                        **ngb_config
                    )
                    ngb_model.fit(X_scaled, y.astype(int))
                    train_score = ngb_model.score(X_scaled, y.astype(int))
                else:
                    # Regression with Normal distribution
                    ngb_model = NGBRegressor(
                        Dist=Normal,
                        random_state=model_seed,
                        **ngb_config
                    )
                    ngb_model.fit(X_scaled, y)
                    train_score = ngb_model.score(X_scaled, y)
                
                # Extract feature importance from base learner (DecisionTreeRegressor)
                # NGBoost uses an ensemble of trees, so we aggregate importance
                importance_values = ngb_model.feature_importances_
                
                # Update feature_names to match dense array
                feature_names = feature_names_dense
                
                # Normalize importance to sum to 1.0
                total = np.sum(importance_values)
                if total > 0:
                    importance_values = importance_values / total
                
                # Wrap model for consistent API (include scaler)
                class NGBoostWrapper:
                    def __init__(self, ngb_model, scaler):
                        self.ngb_model = ngb_model
                        self.scaler = scaler
                        self.feature_importances_ = ngb_model.feature_importances_
                    
                    def predict(self, X):
                        X_scaled = self.scaler.transform(X)
                        return self.ngb_model.predict(X_scaled)
                    
                    def predict_proba(self, X):
                        if hasattr(self.ngb_model, 'predict_proba'):
                            X_scaled = self.scaler.transform(X)
                            return self.ngb_model.predict_proba(X_scaled)
                        return None
                
                model = NGBoostWrapper(ngb_model, scaler)
                
            except Exception as e:
                logger.warning(f"    ngboost: Failed to fit: {e}")
                importance_values = np.zeros(len(feature_names_dense))
                model = type('DummyModel', (), {'feature_importances_': importance_values, 'predict': lambda x: np.zeros(len(x))})()
                feature_names = feature_names_dense
    
    elif model_family == 'mutual_information':
        # Mutual information doesn't train a model, just calculates information
        from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
        from TRAINING.common.utils.sklearn_safe import make_sklearn_dense_X
        
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
        from TRAINING.common.utils.sklearn_safe import make_sklearn_dense_X
        
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
        
        # CRITICAL: F-statistics can be negative (e.g., negative correlations)
        # We need to handle signed scores properly instead of falling back to uniform
        # Strategy: Use absolute values for ranking, but preserve sign information if needed
        
        # Option 1: Use absolute values (recommended for feature selection)
        # This treats negative correlations as potentially useful (just weaker signal)
        abs_scores = np.abs(scores)
        
        # Normalize scores (F-statistics can be very large)
        # Use normalize_importance for robust handling of edge cases (all zeros, etc.)
        raw_importance = abs_scores.copy()
        max_score = np.max(abs_scores)
        if max_score > 0:
            # Normalize by max for initial scaling
            raw_importance = abs_scores / max_score
        # else: raw_importance stays as all zeros, normalize_importance will handle fallback
        
        # Use normalize_importance to handle edge cases (all zeros, NaN, etc.) consistently
        importance_values, fallback_reason = normalize_importance(
            raw_importance=raw_importance,
            n_features=len(feature_names),
            family="univariate_selection",
            feature_names=feature_names
        )
        
        # Log if we had to use absolute values (indicates negative scores)
        if np.any(scores < 0):
            n_negative = np.sum(scores < 0)
            logger.debug(f"    univariate_selection: {n_negative}/{len(scores)} features had negative F-statistics, using absolute values for ranking")
        
        if fallback_reason:
            logger.debug(f"    univariate_selection: {fallback_reason}")
        
        class DummyModel:
            def __init__(self, importance, fallback_reason=None):
                self.importance = importance
                self._fallback_reason = fallback_reason
        
        model = DummyModel(importance_values, fallback_reason=fallback_reason)
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
        
        # FIX: Use default based on feature count or top_k if available
        # Default to 20% of features, but at least 1
        default_n_features = max(1, int(0.2 * len(feature_names)))
        n_features_to_select = min(model_config.get('n_features_to_select', default_n_features), len(feature_names))
        step = model_config.get('step', 5)
        
        # Use config for RFE's internal estimator (load from preprocessing config if not in model_config)
        try:
            from CONFIG.config_loader import get_cfg
            rfe_cfg = get_cfg("preprocessing.multi_model_feature_selection.rfe", default={}, config_name="preprocessing_config")
            estimator_n_estimators = model_config.get('estimator_n_estimators', rfe_cfg.get('estimator_n_estimators', 100))
            estimator_max_depth = model_config.get('estimator_max_depth', rfe_cfg.get('estimator_max_depth', 10))
            estimator_n_jobs = model_config.get('estimator_n_jobs', rfe_cfg.get('estimator_n_jobs', 1))
            # Get seed from SST (determinism system) - no hardcoded defaults
            estimator_seed = model_config.get('seed')
            if estimator_seed is None:
                estimator_seed = stable_seed_from(['rfe', target_column if target_column else 'default', symbol if symbol else 'all'])
        except Exception as e:
            logger.debug(f"Failed to load RFE config: {e}, using model_config defaults")
            estimator_n_estimators = model_config.get('estimator_n_estimators', 100)
            estimator_max_depth = model_config.get('estimator_max_depth', 10)
            estimator_n_jobs = model_config.get('estimator_n_jobs', 1)
            # Get seed from SST (determinism system) - no hardcoded defaults
            estimator_seed = model_config.get('seed')
            if estimator_seed is None:
                estimator_seed = stable_seed_from(['rfe', target_column if target_column else 'default', symbol if symbol else 'all'])
        
        if is_binary or is_multiclass:
            estimator = RandomForestClassifier(
                n_estimators=estimator_n_estimators,
                max_depth=estimator_max_depth,
                random_state=estimator_seed,
                n_jobs=estimator_n_jobs
            )
        else:
            estimator = RandomForestRegressor(
                n_estimators=estimator_n_estimators,
                max_depth=estimator_max_depth,
                random_state=estimator_seed,
                n_jobs=estimator_n_jobs
            )
        
        selector = RFE(estimator, n_features_to_select=n_features_to_select, step=step)
        # Use threading utilities for smart thread management
        if _THREADING_UTILITIES_AVAILABLE:
            # Get thread plan based on family (RFE uses RandomForest estimator)
            plan = plan_for_family('RandomForest', total_threads=default_threads())
            # Set n_jobs on estimator from plan (OMP threads for RandomForest)
            estimator.set_params(n_jobs=plan['OMP'])
            # Use thread_guard context manager for safe thread control
            with thread_guard(omp=plan['OMP'], mkl=plan['MKL']):
                selector.fit(X, y)
        else:
            # Fallback: manual thread management
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
        
        # Conditional execution gate (SST: all thresholds from config)
        boruta_should_run = True
        skip_reason = None
        
        try:
            from CONFIG.config_loader import get_cfg
            boruta_cfg = get_cfg("preprocessing.multi_model_feature_selection.boruta", default={}, config_name="preprocessing_config")
            
            # Check if Boruta is enabled
            boruta_enabled = family_config.get('enabled', boruta_cfg.get('enabled', True))
            if not boruta_enabled:
                boruta_should_run = False
                skip_reason = "disabled in config"
            
            # Check dataset size thresholds (SST: from config)
            if boruta_should_run:
                max_features_threshold = boruta_cfg.get('max_features_threshold', 200)
                max_samples_threshold = boruta_cfg.get('max_samples_threshold', 20000)
                
                n_features = len(feature_names) if feature_names else X.shape[1] if X is not None else 0
                n_samples = len(y) if y is not None else X.shape[0] if X is not None else 0
                
                if n_features > max_features_threshold:
                    boruta_should_run = False
                    skip_reason = f"too many features ({n_features} > {max_features_threshold})"
                elif n_samples > max_samples_threshold:
                    # Check if subsampling is enabled (will be checked later)
                    subsample_enabled = boruta_cfg.get('subsample_large_datasets', {}).get('enabled', True)
                    if not subsample_enabled:
                        boruta_should_run = False
                        skip_reason = f"too many samples ({n_samples} > {max_samples_threshold}) and subsampling disabled"
            
            # Time budget check (will be done during fit, but log here if we're skipping)
            # Note: Actual time budget enforcement happens during fit
            
        except Exception as e:
            logger.debug(f"Failed to load Boruta conditional execution config: {e}, proceeding with Boruta")
            # If config load fails, proceed with Boruta (graceful degradation)
        
        if not boruta_should_run:
            logger.info(f"    boruta: SKIPPED - {skip_reason}")
            # Return empty importance (quality-safe: skip when inappropriate)
            importance = pd.Series(0.0, index=feature_names)
            return None, importance, "boruta_skipped", 0.0
        
        try:
            from boruta import BorutaPy
            from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
            from TRAINING.common.utils.sklearn_safe import make_sklearn_dense_X
            from sklearn.model_selection import train_test_split
            
            # Boruta doesn't support NaN - use sklearn-safe conversion
            X_dense, feature_names_dense = make_sklearn_dense_X(X, feature_names)
            
            # Determine task type (needed for subsampling)
            unique_vals = np.unique(y[~np.isnan(y)])
            is_binary = len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0})
            is_multiclass = len(unique_vals) <= 10 and all(
                isinstance(v, (int, np.integer)) or (isinstance(v, float) and v.is_integer())
                for v in unique_vals
            )
            
            # Subsample large datasets for Boruta (SST: all parameters from config)
            subsample_applied = False
            try:
                subsample_cfg = boruta_cfg.get('subsample_large_datasets', {})
                subsample_enabled = subsample_cfg.get('enabled', True)
                subsample_threshold = subsample_cfg.get('threshold', 10000)
                subsample_max_samples = subsample_cfg.get('max_samples', 10000)
                
                n_samples = len(y) if y is not None else X_dense.shape[0] if X_dense is not None else 0
                if subsample_enabled and n_samples > subsample_threshold:
                    # Use stratified sampling for classification, random for regression
                    if is_binary or is_multiclass:
                        X_dense, _, y, _ = train_test_split(
                            X_dense, y,
                            train_size=min(subsample_max_samples, n_samples),
                            stratify=y,
                            random_state=boruta_seed
                        )
                    else:
                        # For regression, use random sampling
                        indices = np.random.RandomState(boruta_seed).choice(
                            n_samples, size=min(subsample_max_samples, n_samples), replace=False
                        )
                        X_dense = X_dense[indices]
                        y = y[indices]
                    
                    subsample_applied = True
                    logger.info(f"    boruta: Applied subsampling: {n_samples} → {len(y)} samples (threshold: {subsample_threshold}, max: {subsample_max_samples})")
            except Exception as e:
                logger.debug(f"Failed to apply Boruta subsampling: {e}, using full dataset")
                # Continue with full dataset if subsampling fails
            
            # Use ExtraTrees (more random, better for stability testing) with Boruta-optimized hyperparams
            # More trees + shallower depth = stable importance signals, not best predictive performance
            # Load from preprocessing config if not in model_config
            # CRITICAL: seed comes from SST (determinism system), not hardcoded
            try:
                from CONFIG.config_loader import get_cfg
                boruta_cfg = get_cfg("preprocessing.multi_model_feature_selection.boruta", default={}, config_name="preprocessing_config")
                boruta_n_estimators = model_config.get('n_estimators', boruta_cfg.get('n_estimators', 300))  # Updated default: 300
                boruta_max_depth = model_config.get('max_depth', boruta_cfg.get('max_depth', 6))
                # Get seed from SST (determinism config) - no hardcoded defaults
                boruta_seed = model_config.get('seed') or boruta_cfg.get('seed')
                if boruta_seed is None:
                    # Fallback to determinism system if not in config
                    boruta_seed = stable_seed_from(['boruta', target_column if target_column else 'default', symbol if symbol else 'all'])
                boruta_max_iter_base = model_config.get('max_iter', boruta_cfg.get('max_iter', 50))  # Updated default: 50
                boruta_n_jobs = model_config.get('n_jobs', boruta_cfg.get('n_jobs', 1))
                boruta_verbose = model_config.get('verbose', boruta_cfg.get('verbose', 0))
                
                # Adaptive max_iter based on dataset size (SST: all thresholds from config)
                adaptive_max_iter_enabled = boruta_cfg.get('adaptive_max_iter', {}).get('enabled', True)
                if adaptive_max_iter_enabled:
                    n_samples = len(y) if y is not None else X.shape[0] if X is not None else 0
                    adaptive_cfg = boruta_cfg.get('adaptive_max_iter', {})
                    small_threshold = adaptive_cfg.get('small_dataset_threshold', 5000)
                    small_max_iter = adaptive_cfg.get('small_dataset_max_iter', 30)
                    medium_threshold = adaptive_cfg.get('medium_dataset_threshold', 20000)
                    medium_max_iter = adaptive_cfg.get('medium_dataset_max_iter', 50)
                    large_max_iter = adaptive_cfg.get('large_dataset_max_iter', 75)
                    
                    if n_samples < small_threshold:
                        boruta_max_iter = small_max_iter
                        logger.debug(f"    boruta: Adaptive max_iter: {n_samples} samples < {small_threshold} → max_iter={small_max_iter}")
                    elif n_samples < medium_threshold:
                        boruta_max_iter = medium_max_iter
                        logger.debug(f"    boruta: Adaptive max_iter: {small_threshold} <= {n_samples} < {medium_threshold} → max_iter={medium_max_iter}")
                    else:
                        boruta_max_iter = large_max_iter
                        logger.debug(f"    boruta: Adaptive max_iter: {n_samples} >= {medium_threshold} → max_iter={large_max_iter}")
                else:
                    boruta_max_iter = boruta_max_iter_base
            except Exception as e:
                logger.debug(f"Failed to load Boruta config: {e}, using model_config defaults")
                boruta_n_estimators = model_config.get('n_estimators', 300)  # Updated default: 300
                boruta_max_depth = model_config.get('max_depth', 6)
                # Get seed from SST (determinism system) - no hardcoded defaults
                boruta_seed = model_config.get('seed')
                if boruta_seed is None:
                    boruta_seed = stable_seed_from(['boruta', target_column if target_column else 'default', symbol if symbol else 'all'])
                boruta_max_iter = model_config.get('max_iter', 50)  # Updated default: 50
                boruta_n_jobs = model_config.get('n_jobs', 1)
                boruta_verbose = model_config.get('verbose', 0)
                # Adaptive max_iter disabled if config load fails (use base value)
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
                
                # Clean config to prevent double seed argument
                from TRAINING.common.utils.config_cleaner import clean_config_for_estimator
                et_config = {'n_estimators': boruta_n_estimators, 'max_depth': boruta_max_depth, 'n_jobs': boruta_n_jobs, 'class_weight': class_weight_value}
                et_config_clean = clean_config_for_estimator(ExtraTreesClassifier, et_config, extra_kwargs={'random_state': boruta_seed}, family_name='boruta_et')
                base_estimator = ExtraTreesClassifier(**et_config_clean, random_state=boruta_seed)
            else:
                # Clean config to prevent double seed argument
                from TRAINING.common.utils.config_cleaner import clean_config_for_estimator
                et_config = {'n_estimators': boruta_n_estimators, 'max_depth': boruta_max_depth, 'n_jobs': boruta_n_jobs}
                et_config_clean = clean_config_for_estimator(ExtraTreesRegressor, et_config, extra_kwargs={'random_state': boruta_seed}, family_name='boruta_et')
                base_estimator = ExtraTreesRegressor(**et_config_clean, random_state=boruta_seed)
            
            # Early stopping configuration (SST: from config)
            early_stopping_enabled = True
            early_stopping_stable_iterations = 5
            try:
                early_stopping_cfg = boruta_cfg.get('early_stopping', {})
                early_stopping_enabled = early_stopping_cfg.get('enabled', True)
                early_stopping_stable_iterations = early_stopping_cfg.get('stable_iterations', 5)
            except Exception:
                pass  # Use defaults if config not available
            
            boruta = BorutaPy(
                base_estimator,
                n_estimators='auto',
                verbose=boruta_verbose,
                random_state=boruta_seed,
                max_iter=boruta_max_iter,
                perc=model_config.get('perc', 95)  # Higher threshold = more conservative (needs to beat shadow more decisively)
            )
            
            # Note: Early stopping detection happens after fit (check n_iter_ vs max_iter)
            # Note: make_sklearn_dense_X imputes NaNs but doesn't filter rows, so y matches X_dense length
            # Add detailed timing for Boruta iterations (SST: thresholds from config)
            import time
            import signal
            boruta_fit_start = time.time()
            timing_log_enabled = True
            timing_log_threshold_seconds = 1.0
            boruta_max_time_seconds = None  # Time budget from config (SST)
            try:
                from CONFIG.config_loader import get_cfg
                timing_log_enabled = get_cfg('preprocessing.multi_model_feature_selection.timing.enabled', default=True, config_name='preprocessing_config')
                timing_log_threshold_seconds = get_cfg('preprocessing.multi_model_feature_selection.timing.log_threshold_seconds', default=1.0, config_name='preprocessing_config')
                # Load time budget from config (SST)
                boruta_cfg_time = get_cfg('preprocessing.multi_model_feature_selection.boruta', default={}, config_name='preprocessing_config')
                max_time_minutes = boruta_cfg_time.get('max_time_minutes', 10)  # Default 10 minutes
                boruta_max_time_seconds = max_time_minutes * 60
            except Exception:
                pass  # Use defaults if config not available
            
            # Time-budget wrapper for Boruta fit (SST: budget from config)
            boruta_timed_out = False
            boruta_budget_hit = False
            
            def boruta_timeout_handler(signum, frame):
                nonlocal boruta_timed_out, boruta_budget_hit
                boruta_timed_out = True
                boruta_budget_hit = True
                raise TimeoutError(f"Boruta training exceeded {boruta_max_time_seconds/60:.0f} minute time budget")
            
            # Set up timeout if budget is configured (Unix only - Windows will use soft check)
            timeout_set = False
            if boruta_max_time_seconds is not None:
                try:
                    signal.signal(signal.SIGALRM, boruta_timeout_handler)
                    signal.alarm(int(boruta_max_time_seconds))
                    timeout_set = True
                except (AttributeError, OSError):
                    # Windows doesn't support SIGALRM - will use soft timeout check
                    logger.debug(f"    boruta: Timeout signal not available on this platform, using soft timeout check")
            
            try:
                # Use threading utilities for smart thread management
                if _THREADING_UTILITIES_AVAILABLE:
                    # Get thread plan based on family (Boruta uses ExtraTrees estimator)
                    plan = plan_for_family('RandomForest', total_threads=default_threads())
                    # Set n_jobs on ExtraTrees from plan (OMP threads for tree models)
                    et_config['n_jobs'] = plan['OMP']
                    # Use thread_guard context manager for safe thread control
                    with thread_guard(omp=plan['OMP'], mkl=plan['MKL']):
                        boruta.fit(X_dense, y)
                else:
                    # Fallback: manual thread management
                    boruta.fit(X_dense, y)
                
                # Cancel timeout if it was set
                if timeout_set:
                    try:
                        signal.alarm(0)
                    except (AttributeError, OSError):
                        pass
                
                # Soft timeout check (for Windows or if signal didn't fire)
                boruta_fit_elapsed = time.time() - boruta_fit_start
                if boruta_max_time_seconds is not None and boruta_fit_elapsed > boruta_max_time_seconds:
                    boruta_budget_hit = True
                    logger.warning(f"    boruta: ⚠️  BORUTA_BUDGET_HIT - Training took {boruta_fit_elapsed/60:.1f} minutes (exceeded {boruta_max_time_seconds/60:.0f} min budget)")
                    # Continue with current results (quality-safe: use what we have)
                    
            except TimeoutError as te:
                boruta_fit_elapsed = time.time() - boruta_fit_start
                boruta_budget_hit = True
                logger.warning(f"    boruta: ⚠️  BORUTA_BUDGET_HIT - Training exceeded {boruta_max_time_seconds/60:.0f} minute budget after {boruta_fit_elapsed/60:.1f} minutes")
                # Cancel timeout
                if timeout_set:
                    try:
                        signal.alarm(0)
                    except (AttributeError, OSError):
                        pass
                # Check if Boruta has partial results we can use
                if hasattr(boruta, 'ranking_') and boruta.ranking_ is not None:
                    logger.info(f"    boruta: Using partial results from interrupted fit")
                    # Continue with partial results (quality-safe: better than nothing)
                else:
                    # No partial results - return empty importance (quality-safe: skip when budget exceeded)
                    logger.warning(f"    boruta: No partial results available, returning empty importance")
                    importance = pd.Series(0.0, index=feature_names)
                    return None, importance, "boruta_timeout", 0.0
            
            # Log detailed Boruta timing
            boruta_fit_elapsed = time.time() - boruta_fit_start
            if timing_log_enabled and boruta_fit_elapsed >= timing_log_threshold_seconds:
                # Check iteration count (BorutaPy exposes n_iter_ after fitting)
                n_iterations = getattr(boruta, 'n_iter_', None)
                if n_iterations is not None:
                    avg_time_per_iter = boruta_fit_elapsed / n_iterations if n_iterations > 0 else 0
                    early_convergence = n_iterations < boruta_max_iter
                    convergence_msg = " (early convergence)" if early_convergence else ""
                    logger.info(f"    boruta: Fit completed in {boruta_fit_elapsed:.2f}s - "
                              f"{n_iterations}/{boruta_max_iter} iterations{convergence_msg}, "
                              f"avg {avg_time_per_iter:.2f}s/iteration")
                else:
                    logger.info(f"    boruta: Fit completed in {boruta_fit_elapsed:.2f}s")
            
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
        except TimeoutError as te:
            # TimeoutError already handled above, but catch here for safety
            logger.warning(f"Boruta timeout: {te}")
            return None, pd.Series(0.0, index=feature_names), "boruta_timeout", 0.0
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
        from TRAINING.ranking.utils.purged_time_series_split import PurgedTimeSeriesSplit
        from TRAINING.ranking.utils.leakage_filtering import _extract_horizon, _load_leakage_config
        
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
        # Get seed from SST (determinism system) - no hardcoded defaults
        stability_seed = model_config.get('seed')
        if stability_seed is None:
            stability_seed = stable_seed_from(['stability_selection', target_column if target_column else 'default', symbol if symbol else 'all'])
        stability_cs = model_config.get('Cs', 10)  # Number of C values for LogisticRegressionCV
        stability_max_iter = model_config.get('max_iter', 1000)  # Max iterations for LassoCV/LogisticRegressionCV
        stability_n_jobs = model_config.get('n_jobs', 1)  # Parallel jobs
        
        stability_scores = np.zeros(X.shape[1])
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_boot, y_boot = X[indices], y[indices]
            
            try:
                # Clean config to prevent double seed argument
                from TRAINING.common.utils.config_cleaner import clean_config_for_estimator
                if is_binary or is_multiclass:
                    lr_config = {'Cs': stability_cs, 'cv': purged_cv, 'max_iter': stability_max_iter, 'n_jobs': stability_n_jobs}
                    lr_config_clean = clean_config_for_estimator(LogisticRegressionCV, lr_config, extra_kwargs={'random_state': stability_seed}, family_name='stability_selection')
                    model = LogisticRegressionCV(**lr_config_clean, random_state=stability_seed)
                else:
                    lasso_config = {'cv': purged_cv, 'max_iter': stability_max_iter, 'n_jobs': stability_n_jobs}
                    lasso_config_clean = clean_config_for_estimator(LassoCV, lasso_config, extra_kwargs={'random_state': stability_seed}, family_name='stability_selection')
                    model = LassoCV(**lasso_config_clean, random_state=stability_seed)
                
                # Use threading utilities for smart thread management
                if _THREADING_UTILITIES_AVAILABLE:
                    # Get thread plan based on family (linear models use MKL threads)
                    plan = plan_for_family('Lasso', total_threads=default_threads())
                    # Set n_jobs from plan if model supports it
                    if hasattr(model, 'set_params') and 'n_jobs' in model.get_params():
                        model.set_params(n_jobs=plan['OMP'])
                    # Use thread_guard context manager for safe thread control
                    with thread_guard(omp=plan['OMP'], mkl=plan['MKL']):
                        model.fit(X_boot, y_boot)
                else:
                    # Fallback: manual thread management
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
    import time
    importance_start_time = time.time() if model_family == 'catboost' else None
    try:
        if importance_method == 'native':
            if model_family == 'catboost':
                logger.info(f"    CatBoost: Starting feature importance computation")
            importance = extract_native_importance(model, feature_names)
            if model_family == 'catboost' and importance_start_time:
                importance_elapsed = time.time() - importance_start_time
                logger.info(f"    CatBoost: Feature importance computation completed in {importance_elapsed/60:.2f} minutes")
                
                # Log top 10 features by importance
                if isinstance(importance, pd.Series) and len(importance) > 0:
                    top_10 = importance.nlargest(10)
                    logger.info(f"    CatBoost: Top 10 features by importance:")
                    for idx, (feat, imp) in enumerate(top_10.items(), 1):
                        logger.info(f"      {idx:2d}. {feat}: {imp:.6f} ({imp/importance.sum()*100:.2f}%)")
                    
                    # Check for importance concentration (potential leakage indicator)
                    top_5_sum = top_10.head(5).sum()
                    total_importance = importance.sum()
                    if total_importance > 0:
                        top_5_pct = (top_5_sum / total_importance) * 100
                        if top_5_pct > 50:
                            logger.warning(f"    CatBoost: Top 5 features account for {top_5_pct:.1f}% of importance - potential overfitting or leakage")
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
    output_dir: Optional[Path] = None,  # Optional output directory for stability snapshots
    selected_features: Optional[List[str]] = None,  # FIX: Use pruned feature list from shared harness
    run_identity: Optional[Any] = None,  # RunIdentity for snapshot storage
) -> Tuple[List[ImportanceResult], List[Dict[str, Any]]]:
    """
    Process a single symbol with multiple model families.
    
    Args:
        symbol: Symbol name
        data_path: Path to symbol data file
        target_column: Target column name
        model_families_config: Model families configuration
        max_samples: Maximum samples per symbol
        explicit_interval: Explicit data interval
        experiment_config: Experiment configuration
        output_dir: Output directory for snapshots
        selected_features: Optional pruned feature list from shared harness (ensures consistency)
    """
    
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
        from TRAINING.ranking.utils.leakage_filtering import filter_features_for_target
        from TRAINING.ranking.utils.data_interval import detect_interval_from_dataframe
        
        # Detect data interval for horizon conversion
        detected_interval = detect_interval_from_dataframe(
            df, 
            timestamp_column='ts', 
            default=5,
            explicit_interval=explicit_interval,
            experiment_config=experiment_config
        )
        
        # Track registry filtering stats for metadata
        registry_stats = {
            'features_before_registry': 0,
            'features_after_registry': 0,
            'features_rejected_by_registry': 0
        }
        
        # FIX: Use pruned feature list from shared harness if available (ensures consistency)
        # This prevents features like "adjusted" from "coming back" after pruning
        # CRITICAL: Re-validate with STRICT registry filtering even if selected_features provided
        # Shared harness uses permissive ranking mode, but we need strict mode for feature selection
        if selected_features is not None and len(selected_features) > 0:
            # Track stats before filtering
            registry_stats['features_before_registry'] = len(selected_features)
            
            # Re-validate selected_features with STRICT registry filtering
            # This ensures features that passed permissive ranking mode also pass strict training mode
            from TRAINING.ranking.utils.leakage_filtering import filter_features_for_target
            validated_features = filter_features_for_target(
                selected_features,
                target_column,
                verbose=False,
                use_registry=True,
                data_interval_minutes=detected_interval,
                for_ranking=False  # CRITICAL: Use strict mode (same as training)
            )
            
            # Track stats after filtering
            registry_stats['features_after_registry'] = len(validated_features)
            registry_stats['features_rejected_by_registry'] = len(selected_features) - len(validated_features)
            
            # Only keep features that exist in the dataframe AND pass strict registry validation
            available_features = [f for f in validated_features if f in df.columns]
            
            if len(available_features) < len(selected_features):
                rejected = set(selected_features) - set(available_features)
                logger.debug(f"  {symbol}: Registry strict mode rejected {len(rejected)} features from shared harness: {list(rejected)[:5]}")
            
            # Apply runtime quarantine (dominance quarantine confirmed features)
            if output_dir:
                try:
                    from TRAINING.ranking.utils.dominance_quarantine import load_confirmed_quarantine
                    runtime_quarantine = load_confirmed_quarantine(
                        output_dir=output_dir,
                        target=target_column,
                        view="SYMBOL_SPECIFIC",  # process_single_symbol is always SYMBOL_SPECIFIC
                        symbol=symbol
                    )
                    if runtime_quarantine:
                        available_features = [f for f in available_features if f not in runtime_quarantine]
                        logger.info(f"  🔒 {symbol}: Applied runtime quarantine: Removed {len(runtime_quarantine)} confirmed leaky features ({len(available_features)} remaining)")
                except Exception as e:
                    logger.debug(f"Could not load runtime quarantine for {symbol}: {e}")
            
            # Keep only validated features + target + required ID columns (ts, symbol, etc.)
            required_cols = ['ts', 'symbol'] if 'ts' in df.columns else []
            keep_cols = available_features + [target_column] + [c for c in required_cols if c in df.columns]
            df = df[keep_cols]
            logger.debug(f"  {symbol}: Using {len(available_features)} features (validated with strict registry filtering)")
        else:
            # Fallback: rebuild feature list (original behavior)
            # CRITICAL: Sort columns for deterministic ordering
            all_columns = sorted(df.columns.tolist())
            # Track stats before filtering
            registry_stats['features_before_registry'] = len([c for c in all_columns if c != target_column])
            
            # CRITICAL FIX: Use STRICT registry filtering (not permissive ranking mode)
            # This ensures features selected here will also pass training-time registry validation
            # If we use permissive mode here, we'll select features that get rejected at training time
            safe_columns = filter_features_for_target(
                all_columns, 
                target_column, 
                verbose=False,
                use_registry=True,  # Enable registry validation
                data_interval_minutes=detected_interval,
                for_ranking=False  # CRITICAL: Use strict mode (same as training), not permissive ranking mode
            )
            
            # Track stats after filtering
            registry_stats['features_after_registry'] = len([c for c in safe_columns if c != target_column])
            registry_stats['features_rejected_by_registry'] = registry_stats['features_before_registry'] - registry_stats['features_after_registry']
            
            # Keep only safe features + target
            safe_columns_with_target = [c for c in safe_columns if c != target_column] + [target_column]
            df = df[safe_columns_with_target]
            
            logger.debug(f"  {symbol}: Registry filtering (strict mode): {registry_stats['features_before_registry']} → {registry_stats['features_after_registry']} features ({registry_stats['features_rejected_by_registry']} rejected)")
        
        # Prepare features (target already in safe list, so exclude it explicitly)
        X = df.drop(columns=[target_column], errors='ignore')
        
        # FIX: Enforce numeric dtypes BEFORE any model training (prevents CatBoost object column errors)
        # This is critical - CatBoost treating numeric columns as object/text causes fake performance
        import pandas as pd
        import numpy as np
        
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=X.columns if hasattr(X, 'columns') else [f'f{i}' for i in range(X.shape[1])])
        else:
            X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        
        # Hard-cast all numeric columns to float32 (prevents object dtype from NaN/mixed types)
        object_cols = []
        for col in X_df.columns:
            if X_df[col].dtype.name in ['object', 'string', 'category']:
                # Try to convert to numeric, drop if fails
                try:
                    X_df[col] = pd.to_numeric(X_df[col], errors='coerce').astype('float32')
                    logger.debug(f"  {symbol}: Converted object column {col} to float32")
                except Exception:
                    object_cols.append(col)
            elif pd.api.types.is_numeric_dtype(X_df[col]):
                # Explicitly cast to float32 (prevents object dtype)
                X_df[col] = X_df[col].astype('float32')
        
        # Drop columns that couldn't be converted
        if object_cols:
            logger.warning(f"  {symbol}: Dropping {len(object_cols)} non-numeric columns: {object_cols[:5]}")
            X_df = X_df.drop(columns=object_cols)
        
        # Verify all columns are numeric
        still_bad = [c for c in X_df.columns if not np.issubdtype(X_df[c].dtype, np.number)]
        if still_bad:
            raise TypeError(f"  {symbol}: Non-numeric columns remain after conversion: {still_bad[:10]}")
        
        # FIX: Replace inf/-inf with nan before fail-fast (prevents phantom issues)
        # Some models (e.g., Ridge) may fail on inf values
        X_df = X_df.replace([np.inf, -np.inf], np.nan)
        # Drop columns that are all nan/inf
        X_df = X_df.dropna(axis=1, how='all')
        
        # FIX: Initialize feature_names from X_df.columns before any filtering
        # This prevents UnboundLocalError when feature_names is used before assignment
        # In fallback path, feature_names may not exist from shared harness, so derive from dataframe
        # CRITICAL: Sort feature_names for deterministic ordering
        # DataFrame column order may not be deterministic across runs
        feature_names = sorted(X_df.columns.tolist())
        # Reorder DataFrame columns to match sorted feature_names
        X_df = X_df[feature_names]
        
        # Update X
        X_arr = X_df.values.astype('float32')  # Already float32 from conversion above
        
        y = df[target_column]
        y_arr = y.to_numpy() if hasattr(y, 'to_numpy') else np.array(y)
        
        if not feature_names:
            logger.warning(f"Skipping {symbol}: No features after filtering")
            return results, family_statuses
        
        # CRITICAL: Use already-detected interval (detected above at line 773)
        # No need to detect again - use the same detected_interval from above
        if detected_interval != 5:
            logger.info(f"  Detected data interval: {detected_interval}m (was assuming 5m)")
        
        # Infer task type from target values (for task-type filtering)
        unique_vals = np.unique(y_arr[~np.isnan(y_arr)])
        is_binary = len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0})
        is_multiclass = len(unique_vals) <= 10 and len(unique_vals) > 2 and all(
            isinstance(v, (int, np.integer)) or (isinstance(v, float) and v.is_integer())
            for v in unique_vals
        )
        if is_binary:
            inferred_task_type = "binary"
        elif is_multiclass:
            inferred_task_type = "multiclass"
        else:
            inferred_task_type = "regression"
        
        # Train each enabled model family with structured status tracking
        # Filter by task type compatibility BEFORE training (prevents garbage importance scores)
        from TRAINING.training_strategies.utils import is_family_compatible
        enabled_families = []
        for family_name, family_cfg in model_families_config.items():
            if not family_cfg.get('enabled', False):
                continue
            compatible, skip_reason = is_family_compatible(family_name, inferred_task_type)
            if compatible:
                enabled_families.append(family_name)
            else:
                logger.info(f"  {symbol}: ⏭️ Skipping {family_name} for feature selection: {skip_reason}")
                family_statuses.append({
                    'family': family_name,
                    'status': 'skipped',
                    'skip_reason': skip_reason,
                    'symbol': symbol
                })
        
        # FIX: Create fallback identity using SST factory if run_identity wasn't passed
        # This ensures ALL model families can save snapshots (not just xgboost)
        # Mirrors TARGET_RANKING pattern from main.py:238-249
        effective_run_identity = run_identity
        if effective_run_identity is None:
            try:
                from TRAINING.common.utils.fingerprinting import create_stage_identity
                effective_run_identity = create_stage_identity(
                    stage="FEATURE_SELECTION",
                    symbols=[symbol] if symbol else [],
                    experiment_config=experiment_config,
                )
                logger.debug(f"  {symbol}: Created fallback FEATURE_SELECTION identity with train_seed={effective_run_identity.train_seed}")
            except Exception as e:
                logger.debug(f"  {symbol}: Failed to create fallback identity: {e}")
        
        # Log reproducibility info for this symbol
        try:
            from TRAINING.common.determinism import BASE_SEED
            base_seed = BASE_SEED if BASE_SEED is not None else 42
            logger.debug(f"  {symbol}: Reproducibility - base_seed={base_seed}, n_features={len(feature_names)}, n_samples={len(X_arr)}, detected_interval={detected_interval}m")
        except Exception:
            logger.debug(f"  {symbol}: Reproducibility - base_seed=N/A (determinism system unavailable)")
        
        # Track per-model reproducibility
        per_model_reproducibility = []
        
        # Iterate only over compatible, enabled families (filtered above)
        for family_name in enabled_families:
            family_config = model_families_config[family_name]
            
            try:
                logger.info(f"  {symbol}: Training {family_name}...")
                model, importance, method, train_score = train_model_and_get_importance(
                    family_name, family_config, X_arr, y_arr, feature_names,
                    data_interval_minutes=detected_interval,
                    target_column=target_column,
                    symbol=symbol  # Pass symbol for deterministic seed generation
                )
                
                # Compute prediction fingerprint for determinism tracking (SST)
                model_prediction_fp = None
                if model is not None:
                    try:
                        from TRAINING.common.utils.prediction_hashing import compute_prediction_fingerprint_for_model
                        from TRAINING.common.utils.fingerprinting import get_identity_mode
                        strict_mode = get_identity_mode() == "strict"
                        
                        # Get predictions from trained model
                        y_pred = model.predict(X_arr)
                        
                        # Determine task type from target values
                        unique_vals = np.unique(y_arr[~np.isnan(y_arr)])
                        is_binary = len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0})
                        task_type = "BINARY_CLASSIFICATION" if is_binary else "REGRESSION"
                        
                        # Get probabilities for classification if available
                        y_proba = None
                        if is_binary and hasattr(model, 'predict_proba'):
                            try:
                                y_proba = model.predict_proba(X_arr)
                            except Exception:
                                pass
                        
                        model_prediction_fp = compute_prediction_fingerprint_for_model(
                            preds=y_pred,
                            proba=y_proba,
                            model=model,
                            task_type=task_type,
                            X=X_arr,
                            strict_mode=strict_mode,
                        )
                        if model_prediction_fp:
                            logger.debug(f"    {family_name}: prediction_fingerprint={model_prediction_fp.get('prediction_hash', '')[:12]}...")
                    except Exception as fp_e:
                        logger.debug(f"    {family_name}: prediction fingerprint failed: {fp_e}")
                
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
                    
                    # Compute per-model reproducibility
                    previous_data = load_previous_model_results(output_dir, symbol, target_column, family_name)
                    repro_stats = compute_per_model_reproducibility(
                        symbol=symbol,
                        target_column=target_column,
                        model_family=family_name,
                        current_score=train_score,
                        current_importance=importance,
                        previous_data=previous_data,
                        top_k=50
                    )
                    
                    # Store reproducibility in metadata
                    save_model_metadata(
                        output_dir=output_dir,
                        symbol=symbol,
                        target_column=target_column,
                        model_family=family_name,
                        score=train_score,
                        importance=importance,
                        reproducibility=repro_stats
                    )
                    
                    # Compact logging based on status
                    if repro_stats['status'] == 'no_previous_run':
                        # First run - no comparison
                        pass
                    elif repro_stats['status'] == 'stable':
                        # OK/stable - one compact line
                        delta_str = f"Δscore={repro_stats['delta_score']:.3f}" if repro_stats['delta_score'] is not None else "Δscore=N/A"
                        jaccard_str = f"Jaccard@50={repro_stats['jaccard_top_k']:.2f}" if repro_stats['jaccard_top_k'] is not None else "Jaccard@50=N/A"
                        corr_str = f"corr={repro_stats['importance_corr']:.2f}" if repro_stats['importance_corr'] is not None else "corr=N/A"
                        logger.info(f"  {symbol}: {family_name} reproducibility: {delta_str}, {jaccard_str}, {corr_str} [OK]")
                    elif repro_stats['status'] == 'borderline':
                        # Borderline - info level
                        delta_str = f"Δscore={repro_stats['delta_score']:.3f}" if repro_stats['delta_score'] is not None else "Δscore=N/A"
                        jaccard_str = f"Jaccard@50={repro_stats['jaccard_top_k']:.2f}" if repro_stats['jaccard_top_k'] is not None else "Jaccard@50=N/A"
                        corr_str = f"corr={repro_stats['importance_corr']:.2f}" if repro_stats['importance_corr'] is not None else "corr=N/A"
                        logger.info(f"  {symbol}: {family_name} reproducibility: {delta_str}, {jaccard_str}, {corr_str} [BORDERLINE]")
                    else:  # unstable
                        # Unstable - WARNING level
                        delta_str = f"Δscore={repro_stats['delta_score']:.3f}" if repro_stats['delta_score'] is not None else "Δscore=N/A"
                        jaccard_str = f"Jaccard@50={repro_stats['jaccard_top_k']:.2f}" if repro_stats['jaccard_top_k'] is not None else "Jaccard@50=N/A"
                        corr_str = f"corr={repro_stats['importance_corr']:.2f}" if repro_stats['importance_corr'] is not None else "corr=N/A"
                        logger.warning(f"  {symbol}: {family_name} reproducibility: {delta_str}, {jaccard_str}, {corr_str} [UNSTABLE]")
                    
                    # Store for symbol-level summary
                    per_model_reproducibility.append({
                        "family": family_name,
                        "status": repro_stats['status'],
                        "delta_score": repro_stats['delta_score'],
                        "jaccard_top_k": repro_stats['jaccard_top_k'],
                        "importance_corr": repro_stats['importance_corr']
                    })
                    
                    # Save stability snapshot for this model family (non-invasive hook)
                    # CRITICAL: Use model_family as method name, not importance_method
                    # This ensures stability is computed per-model-family (comparing same family across runs)
                    # Only save if output_dir is available (optional feature)
                    if output_dir is not None:
                        try:
                            from TRAINING.stability.feature_importance import save_snapshot_from_series_hook
                            import hashlib
                            
                            # FIX: Use feature_universe_fingerprint instead of symbol for universe_sig
                            # Symbol is useful for INDIVIDUAL, but universe fingerprint is the real guard
                            # against comparing different candidate sets (pruner/sanitizer differences)
                            # Compute fingerprint from sorted feature names (stable across runs)
                            sorted_features = sorted(feature_names)
                            feature_universe_str = "|".join(sorted_features)
                            feature_universe_fingerprint = hashlib.sha256(feature_universe_str.encode()).hexdigest()[:16]
                            
                            # For INDIVIDUAL mode, include symbol in universe_sig for clarity
                            # But use fingerprint as the primary identifier
                            if symbol:
                                universe_sig = f"{symbol}:{feature_universe_fingerprint}"
                            else:
                                universe_sig = f"ALL:{feature_universe_fingerprint}"
                            
                            # FIX: Use model_family (e.g., "lightgbm", "ridge", "elastic_net") as method name
                            # NOT importance_method (e.g., "native", "shap") - stability should be per-family
                            # Compute identity for this model family
                            # FIX: Use effective_run_identity (includes fallback) instead of run_identity
                            family_identity = None
                            partial_identity_dict = None  # Fallback: extract signatures from partial identity
                            
                            if effective_run_identity is not None:
                                # Always extract partial identity signatures as fallback
                                partial_identity_dict = {
                                    "dataset_signature": getattr(effective_run_identity, 'dataset_signature', None),
                                    "split_signature": getattr(effective_run_identity, 'split_signature', None),
                                    "target_signature": getattr(effective_run_identity, 'target_signature', None),
                                    "routing_signature": getattr(effective_run_identity, 'routing_signature', None),
                                    "train_seed": getattr(effective_run_identity, 'train_seed', None),
                                }
                                
                                try:
                                    from TRAINING.common.utils.fingerprinting import (
                                        RunIdentity, compute_hparams_fingerprint,
                                        compute_feature_fingerprint_from_specs
                                    )
                                    # Hparams for this family
                                    hparams_signature = compute_hparams_fingerprint(
                                        model_family=family_name,
                                        params={},  # Default params used
                                    )
                                    # Feature signature from importance series (registry-resolved)
                                    from TRAINING.common.utils.fingerprinting import resolve_feature_specs_from_registry
                                    feature_specs = resolve_feature_specs_from_registry(list(importance.index))
                                    feature_signature = compute_feature_fingerprint_from_specs(feature_specs)
                                    
                                    # Add computed signatures to fallback dict
                                    partial_identity_dict["hparams_signature"] = hparams_signature
                                    partial_identity_dict["feature_signature"] = feature_signature
                                    
                                    # Create updated partial and finalize
                                    updated_partial = RunIdentity(
                                        dataset_signature=effective_run_identity.dataset_signature if hasattr(effective_run_identity, 'dataset_signature') else "",
                                        split_signature=effective_run_identity.split_signature if hasattr(effective_run_identity, 'split_signature') else "",
                                        target_signature=effective_run_identity.target_signature if hasattr(effective_run_identity, 'target_signature') else "",
                                        feature_signature=None,
                                        hparams_signature=hparams_signature or "",
                                        routing_signature=effective_run_identity.routing_signature if hasattr(effective_run_identity, 'routing_signature') else "",
                                        routing_payload=effective_run_identity.routing_payload if hasattr(effective_run_identity, 'routing_payload') else None,
                                        train_seed=effective_run_identity.train_seed if hasattr(effective_run_identity, 'train_seed') else None,
                                        is_final=False,
                                    )
                                    family_identity = updated_partial.finalize(feature_signature)
                                except Exception as e:
                                    # FIX: Log at WARNING level so failures are visible
                                    logger.warning(
                                        f"Failed to compute family identity for {family_name}: {e}. "
                                        f"Using partial identity signatures as fallback."
                                    )
                            
                            # FIX: If identity not finalized but we have partial signatures, pass them
                            effective_identity = family_identity if family_identity else partial_identity_dict
                            
                            # Compute feature_fingerprint_input for per-family snapshots
                            family_candidate_features = list(importance.index) if importance is not None else []
                            family_feature_input_hash = None
                            if family_candidate_features:
                                import hashlib
                                import json as json_mod
                                sorted_features = sorted(family_candidate_features)
                                family_feature_input_hash = hashlib.sha256(json_mod.dumps(sorted_features).encode()).hexdigest()
                            
                            family_inputs = {
                                "candidate_features": family_candidate_features,
                                "feature_fingerprint_input": family_feature_input_hash,
                            }
                            
                            save_snapshot_from_series_hook(
                                target=target_column if target_column else 'unknown',
                                method=family_name,  # Use model_family, not importance_method
                                importance_series=importance,
                                universe_sig=universe_sig,  # FIX: Use feature_universe_fingerprint (not just symbol)
                                output_dir=output_dir,
                                auto_analyze=None,  # Load from config
                                run_identity=effective_identity,  # Pass finalized identity or partial dict fallback
                                allow_legacy=(family_identity is None and partial_identity_dict is None),
                                prediction_fingerprint=model_prediction_fp,  # SST: prediction hash for determinism
                                view="SYMBOL_SPECIFIC",  # process_single_symbol is always SYMBOL_SPECIFIC
                                symbol=symbol,  # Pass symbol for proper scoping
                                inputs=family_inputs,  # Pass inputs with feature_fingerprint_input
                                stage="FEATURE_SELECTION",  # Explicit stage for proper path scoping
                            )
                        except Exception as e:
                            logger.debug(f"Stability snapshot save failed for {family_name} (non-critical): {e}")
                    
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
                
                # Check if this is an expected failure (over-regularization, no signal, etc.)
                # These are common and expected, so log as WARNING without traceback
                is_expected_failure = (
                    "all coefficients are zero" in error_msg.lower() or
                    "over-regularized" in error_msg.lower() or
                    "no signal" in error_msg.lower() or
                    "model invalid" in error_msg.lower()
                )
                
                if is_expected_failure:
                    logger.warning(f"    ⚠️  {symbol}: {family_name} failed (expected): {error_msg}")
                else:
                    # Unexpected errors get full ERROR logging with traceback
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
        
        # Log reproducibility summary per symbol
        if per_model_reproducibility:
            stable_count = sum(1 for r in per_model_reproducibility if r['status'] == 'stable')
            borderline_count = sum(1 for r in per_model_reproducibility if r['status'] == 'borderline')
            unstable_count = sum(1 for r in per_model_reproducibility if r['status'] == 'unstable')
            no_prev_count = sum(1 for r in per_model_reproducibility if r['status'] == 'no_previous_run')
            
            if unstable_count > 0 or borderline_count > 0:
                unstable_families = [r['family'] for r in per_model_reproducibility if r['status'] == 'unstable']
                borderline_families = [r['family'] for r in per_model_reproducibility if r['status'] == 'borderline']
                
                summary_parts = []
                if stable_count > 0:
                    summary_parts.append(f"{stable_count} stable")
                if borderline_count > 0:
                    summary_parts.append(f"{borderline_count} borderline")
                if unstable_count > 0:
                    summary_parts.append(f"{unstable_count} unstable")
                if no_prev_count > 0:
                    summary_parts.append(f"{no_prev_count} no_previous_run")
                
                summary_str = ", ".join(summary_parts)
                
                if unstable_count > 0:
                    logger.warning(f"  {symbol}: reproducibility summary: {summary_str} -> ⚠️ check model_families: {', '.join(unstable_families)}")
                else:
                    logger.info(f"  {symbol}: reproducibility summary: {summary_str}")
        
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
            # FIX: Log skip reasons for failed models in consensus summary (makes debugging easier)
            failed_models_with_reasons = []
            for family in families_without_results:
                family_failures = [s for s in all_family_statuses if s.get('family') == family and s.get('status') == 'failed']
                if family_failures:
                    # Extract error messages and create concise skip reasons
                    error_messages = [s.get('error', '') for s in family_failures if s.get('error')]
                    error_types = set(s.get('error_type') for s in family_failures if s.get('error_type'))
                    symbols_failed = [s.get('symbol') for s in family_failures]
                    
                    # Create concise skip reason (e.g., 'ridge:zero_coefs', 'elastic_net:singular')
                    skip_reason = None
                    if error_messages:
                        # Extract key phrase from error message
                        first_error = error_messages[0].lower()
                        if 'zero' in first_error and 'coefficient' in first_error:
                            skip_reason = f"{family}:zero_coefs"
                        elif 'singular' in first_error:
                            skip_reason = f"{family}:singular"
                        elif 'invalid' in first_error:
                            skip_reason = f"{family}:invalid"
                        else:
                            # Use error_type if available, otherwise generic
                            skip_reason = f"{family}:{list(error_types)[0].lower() if error_types else 'unknown'}"
                    else:
                        skip_reason = f"{family}:{list(error_types)[0].lower() if error_types else 'unknown'}"
                    
                    if skip_reason:
                        failed_models_with_reasons.append(skip_reason)
                    
                    logger.warning(f"   - {family}: Failed for {len(symbols_failed)} symbol(s) ({', '.join(set(symbols_failed))}) with error types: {', '.join(error_types) if error_types else 'Unknown'}")
            
            # Log concise skip reasons for consensus summary
            if failed_models_with_reasons:
                logger.info(f"📋 Failed models (excluded from consensus): {', '.join(failed_models_with_reasons)}")
        
        logger.info(f"✅ Aggregating {len(families_with_results)} model families with results: {', '.join(sorted(families_with_results))}")
    
    # Aggregate within each family
    family_scores = {}
    boruta_scores = None  # Store separately for gatekeeper role
    
    for family_name, results in family_results.items():
        # Combine importances across symbols for this family
        # CRITICAL: Check if we have any importance scores before concatenation
        importance_series_list = [r.importance_scores for r in results if hasattr(r, 'importance_scores') and r.importance_scores is not None]
        
        if not importance_series_list:
            logger.warning(f"⚠️  {family_name}: No importance scores available (all results have None or missing importance_scores)")
            continue  # Skip this family
        
        importances_df = pd.concat(
            importance_series_list,
            axis=1,
            sort=False
        ).fillna(0)
        
        # CRITICAL: Check if importances_df is empty after concatenation
        if importances_df.empty:
            logger.warning(f"⚠️  {family_name}: Empty importance DataFrame after concatenation (no features available)")
            continue  # Skip this family
        
        # Aggregate across symbols (mean by default)
        method = aggregation_config.get('per_symbol_method', 'mean')
        if method == 'mean':
            family_score = importances_df.mean(axis=1)
        elif method == 'median':
            family_score = importances_df.median(axis=1)
        else:
            family_score = importances_df.mean(axis=1)
        
        # CRITICAL: Check if family_score is empty after aggregation
        if len(family_score) == 0 or family_score.empty:
            logger.warning(f"⚠️  {family_name}: Empty family_score after aggregation (no features with importance scores)")
            continue  # Skip this family
        
        # Apply family weight
        weight = model_families_config[family_name].get('weight', 1.0)
        
        # CRITICAL: Boruta is NOT included in base consensus - it's a gatekeeper, not a scorer
        # FIX: Count unique symbols, not number of results (CROSS_SECTIONAL has 1 result with symbol="ALL")
        unique_symbols = set(r.symbol for r in results if hasattr(r, 'symbol') and r.symbol)
        n_symbols = len(unique_symbols)
        symbol_str = f"{n_symbols} symbol{'s' if n_symbols != 1 else ''}"
        if n_symbols == 1 and results and hasattr(results[0], 'symbol') and results[0].symbol == "ALL":
            symbol_str = "all symbols (CROSS_SECTIONAL)"
        
        # CRITICAL: Safe idxmax() call - family_score is guaranteed non-empty at this point
        # but add defensive check anyway for robustness
        top_feature = family_score.idxmax() if len(family_score) > 0 else "N/A"
        
        if family_name == 'boruta':
            boruta_scores = family_score  # Store for gatekeeper role only
            logger.info(f"🔒 {family_name}: Aggregated {symbol_str} (gatekeeper, excluded from base consensus)")
        else:
            family_scores[family_name] = family_score * weight
            logger.info(f"📊 {family_name}: Aggregated {symbol_str}, "
                       f"weight={weight}, top={top_feature}")
    
    # Combine across families (EXCLUDING Boruta - it's a gatekeeper, not a scorer)
    if not family_scores:
        logger.warning("No model family results available (all families may have failed or been disabled)")
        return pd.DataFrame(), []
    
    combined_df = pd.DataFrame(family_scores)
    
    # CRITICAL: Check if combined_df is empty before creating consensus scores
    if combined_df.empty:
        logger.warning("Empty combined DataFrame after aggregating family scores - no features available for consensus")
        return pd.DataFrame(), []
    
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
        # GRACEFUL DEGRADATION: Log warning and continue without Boruta gatekeeper
        # Collect error information for diagnostics
        boruta_failures = []
        if all_family_statuses:
            boruta_failures = [s for s in all_family_statuses 
                             if s.get('family') == 'boruta' and s.get('status') == 'failed']
        
        error_summary = "Unknown error"
        if boruta_failures:
            error_messages = [s.get('error', '') for s in boruta_failures if s.get('error')]
            error_types = set(s.get('error_type') for s in boruta_failures if s.get('error_type'))
            symbols_failed = [s.get('symbol') for s in boruta_failures]
            
            if error_messages:
                # Use first error message as summary
                error_summary = error_messages[0]
            elif error_types:
                error_summary = f"Error types: {', '.join(error_types)}"
            
            error_details = (
                f"Boruta gatekeeper FAILED for {len(symbols_failed)} symbol(s): {', '.join(set(symbols_failed))}. "
                f"Error: {error_summary}"
            )
        else:
            # Check if this is CROSS_SECTIONAL view (symbol="ALL")
            is_cross_sectional = all_family_statuses and any(
                s.get('symbol') == 'ALL' for s in all_family_statuses
            )
            if is_cross_sectional:
                error_details = (
                    "Boruta gatekeeper enabled in config but no results produced and no failure status recorded. "
                    "This likely indicates Boruta failed silently in the shared harness (CROSS_SECTIONAL view). "
                    "Check harness logs for Boruta errors."
                )
            else:
                error_details = (
                    "Boruta gatekeeper enabled in config but no results produced and no failure status recorded. "
                    "This may indicate Boruta was silently skipped or failed without proper error tracking."
                )
        
        # GRACEFUL DEGRADATION: Log warning and continue without Boruta
        logger.warning(
            f"⚠️  Boruta gatekeeper is enabled but failed. {error_details} "
            f"Continuing without Boruta gatekeeper (features will not receive Boruta bonuses/penalties)."
        )
        # Continue without Boruta (graceful degradation)
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
    # CRITICAL: Use stable sort with tie-breaker for deterministic ordering
    # Round consensus_score to 12 decimals for ordering stability (float jitter protection)
    summary_df['_consensus_score_rounded'] = summary_df['consensus_score'].round(12)
    summary_df = summary_df.sort_values(
        ['_consensus_score_rounded', 'feature'], 
        ascending=[False, True],
        kind='mergesort'  # CRITICAL: Stable sort for ties
    ).reset_index(drop=True)
    # Drop temporary rounded column (keep original consensus_score)
    summary_df = summary_df.drop(columns=['_consensus_score_rounded'])
    
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
    target: str,
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
        target: Target column name
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
    high_score_min = high_cfg.get('auc_min', 0.05)
    high_coverage_min = high_cfg.get('model_coverage_min', 0.7)
    
    medium_boruta_min = medium_cfg.get('boruta_confirmed_min', 1)
    medium_agreement_min = medium_cfg.get('agreement_ratio_min', 0.25)
    medium_score_min = medium_cfg.get('auc_min', 0.02)
    
    # Low reason thresholds
    boruta_zero_cfg = low_reasons_cfg.get('boruta_zero_confirmed', {})
    boruta_zero_confirmed_max = boruta_zero_cfg.get('boruta_confirmed_max', 0)
    boruta_zero_tentative_max = boruta_zero_cfg.get('boruta_tentative_max', 1)
    boruta_zero_score_max = boruta_zero_cfg.get('auc_max', 0.03)
    
    low_agreement_max = low_reasons_cfg.get('low_model_agreement', {}).get('agreement_ratio_max', 0.2)
    low_score_max = low_reasons_cfg.get('low_model_scores', {}).get('auc_max', 0.01)
    low_coverage_max = low_reasons_cfg.get('low_model_coverage', {}).get('model_coverage_max', 0.5)
    
    # Agreement top_k from config
    if top_k is None:
        top_k = agreement_cfg.get('top_k', 20)
    
    metrics = {
        'target': target,
        'boruta_confirmed_count': 0,
        'boruta_tentative_count': 0,
        'boruta_rejected_count': 0,
        'boruta_used': False,
        'n_models_available': 0,
        'n_models_successful': 0,
        'model_coverage_ratio': 0.0,
        'auc': 0.0,
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
        metrics['auc'] = float(np.mean(valid_scores))
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
        metrics['auc'] >= high_score_min and
        metrics['model_coverage_ratio'] >= high_coverage_min):
        metrics['confidence'] = 'HIGH'
    
    # MEDIUM confidence (any one condition is sufficient)
    elif (metrics['boruta_confirmed_count'] >= medium_boruta_min or
          metrics['agreement_ratio'] >= medium_agreement_min or
          metrics['auc'] >= medium_score_min):
        metrics['confidence'] = 'MEDIUM'
    
    # LOW confidence (fallback)
    else:
        metrics['confidence'] = 'LOW'
        
        # Determine reason using config thresholds
        if (metrics['boruta_used'] and
            metrics['boruta_confirmed_count'] <= boruta_zero_confirmed_max and
            metrics['boruta_tentative_count'] <= boruta_zero_tentative_max and
            metrics['auc'] < boruta_zero_score_max):
            metrics['low_confidence_reason'] = 'boruta_zero_confirmed'
        elif metrics['agreement_ratio'] < low_agreement_max:
            metrics['low_confidence_reason'] = 'low_model_agreement'
        elif metrics['auc'] < low_score_max:
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
    metadata: Dict[str, Any],
    universe_sig: Optional[str] = None,  # Phase A: optional for backward compat
):
    """Save multi-model feature selection results.
    
    Target-first structure (with OutputLayout when universe_sig provided):
    targets/<target>/reproducibility/{view}/universe={universe_sig}/feature_importances/
      {model}_importances.csv
      feature_importance_multi_model.csv
      feature_importance_with_boruta_debug.csv
      model_agreement_matrix.csv
    targets/<target>/reproducibility/{view}/universe={universe_sig}/selected_features.txt
    
    Falls back to legacy structure without universe scoping when universe_sig not provided.
    
    ROUTING NOTE: These are SCOPE-LEVEL SUMMARY artifacts, not cohort artifacts.
    They use OutputLayout.repro_dir() / feature_importance_dir() directly.
    Do NOT use _save_to_cohort for these outputs - the cohort firewall is for
    per-cohort metrics/run data under cohort=cs_* or cohort=sy_* directories.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find base run directory and target name
    base_output_dir = output_dir
    target = None
    
    # Try to extract target from metadata
    if metadata and 'target' in metadata:
        target = metadata['target']
    elif metadata and 'target_column' in metadata:
        target = metadata['target_column']
    
    # If not in metadata, try to extract from output_dir path
    if not target:
        # output_dir is typically: REPRODUCIBILITY/FEATURE_SELECTION/CROSS_SECTIONAL/{target}/
        parts = output_dir.parts
        if 'FEATURE_SELECTION' in parts:
            idx = parts.index('FEATURE_SELECTION')
            if idx + 2 < len(parts):
                target = parts[idx + 2]
    
    # Also try to extract universe_sig from metadata if not passed explicitly
    # Validate extracted sig before using - invalid values are ignored
    if not universe_sig and metadata:
        extracted_sig = metadata.get('universe_sig')
        if extracted_sig:
            try:
                from TRAINING.orchestration.utils.cohort_metadata import validate_universe_sig
                validate_universe_sig(extracted_sig)
                universe_sig = extracted_sig
            except ValueError as e:
                logger.warning(
                    "Invalid universe_sig from metadata; ignoring and falling back to legacy paths. "
                    f"error={e} metadata_keys={list(metadata.keys())}"
                )
                universe_sig = None  # Don't use invalid value
    
    # Phase A: Use OutputLayout if universe_sig provided (new canonical path)
    # Otherwise fall back to legacy path resolution with warning
    use_output_layout = bool(universe_sig)
    if not use_output_layout:
        logger.warning(
            f"universe_sig not provided for {target} multi-model results, "
            f"falling back to legacy path resolution. Pass universe_sig for canonical paths."
        )
    
    # Walk up to find base run directory
    for _ in range(10):
        # Only stop if we find a run directory (has targets/, globals/, or cache/)
        # Don't stop at RESULTS/ - continue to find actual run directory
        if (base_output_dir / "targets").exists() or (base_output_dir / "globals").exists() or (base_output_dir / "cache").exists():
            break
        if not base_output_dir.parent.exists():
            break
        base_output_dir = base_output_dir.parent
    
    # Set up target-first structure if we found target and base directory (view/symbol-scoped)
    target_importances_dir = None
    target_selected_features_path = None
    repro_dir = None  # Track for later use
    
    if target and base_output_dir.exists():
        try:
            target_clean = target.replace('/', '_').replace('\\', '_')
            
            # Extract view and symbol from metadata (view is REQUIRED)
            view = metadata.get('view') if metadata else None
            symbol = metadata.get('symbol') if metadata else None
            
            # Validate view is provided
            if view is None:
                raise ValueError(
                    f"view must be provided in metadata for save_multi_model_results. "
                    f"Metadata keys: {list(metadata.keys()) if metadata else 'None'}"
                )
            if view not in ("CROSS_SECTIONAL", "SYMBOL_SPECIFIC"):
                raise ValueError(f"Invalid view in metadata: {view}. Must be 'CROSS_SECTIONAL' or 'SYMBOL_SPECIFIC'")
            if view == "SYMBOL_SPECIFIC" and symbol is None:
                raise ValueError("symbol required in metadata when view='SYMBOL_SPECIFIC'")
            
            if use_output_layout and universe_sig:
                # Canonical path via OutputLayout (non-cohort write)
                from TRAINING.orchestration.utils.output_layout import OutputLayout
                layout = OutputLayout(
                    output_root=base_output_dir,
                    target=target_clean,
                    view=view,
                    universe_sig=universe_sig,
                    symbol=symbol if view == "SYMBOL_SPECIFIC" else None,
                    stage="FEATURE_SELECTION",  # Explicit stage for proper path scoping
                )
                repro_dir = layout.repro_dir()
                target_importances_dir = layout.feature_importance_dir()
                target_selected_features_path = repro_dir / "selected_features.txt"
            else:
                # Legacy path resolution with stage
                from TRAINING.orchestration.utils.target_first_paths import (
                    run_root, target_repro_dir, target_repro_file_path, ensure_target_structure
                )
                run_root_dir = run_root(base_output_dir)
                ensure_target_structure(run_root_dir, target_clean)
                repro_dir = target_repro_dir(run_root_dir, target_clean, view=view, symbol=symbol, stage="FEATURE_SELECTION")
                target_importances_dir = repro_dir / "feature_importances"
                target_selected_features_path = target_repro_file_path(run_root_dir, target_clean, "selected_features.txt", view=view, symbol=symbol, stage="FEATURE_SELECTION")
            
            target_importances_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.debug(f"Failed to set up target-first structure: {e}")
    
    # Write to target-first structure only
    # 1. Selected features list
    if target_selected_features_path:
        try:
            target_selected_features_path.parent.mkdir(parents=True, exist_ok=True)
            with open(target_selected_features_path, "w") as f:
                for feature in selected_features:
                    f.write(f"{feature}\n")
            logger.info(f"✅ Saved {len(selected_features)} features to {target_selected_features_path}")
        except Exception as e:
            logger.warning(f"Failed to write selected features to target-first location: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
    else:
        logger.warning(f"Target selected features path not available")
    
    # 2. Detailed summary CSV (includes all columns including Boruta gatekeeper)
    if target_importances_dir:
        try:
            target_importances_dir.mkdir(parents=True, exist_ok=True)
            target_csv_path = target_importances_dir / "feature_importance_multi_model.csv"
            summary_df.to_csv(target_csv_path, index=False)
            logger.info(f"✅ Saved detailed multi-model summary to {target_csv_path}")
        except Exception as e:
            logger.warning(f"Failed to write feature importance summary to target-first location: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
    else:
        logger.warning(f"Target importances directory not available")
    
    # 2b. Explicit debug view: Boruta gatekeeper effect analysis → feature_importances/
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
        
        # Write to target-first structure only
        if target_importances_dir:
            try:
                target_debug_path = target_importances_dir / "feature_importance_with_boruta_debug.csv"
                debug_df.to_csv(target_debug_path, index=False)
                logger.info(f"✅ Saved Boruta gatekeeper debug view to {target_debug_path}")
            except Exception as e:
                logger.warning(f"Failed to write Boruta debug view to target-first location: {e}")
                import traceback
                logger.debug(f"Traceback: {traceback.format_exc()}")
    
    # 3. Per-model-family breakdowns → feature_importances/ (matching target ranking naming)
    for family_name in summary_df.columns:
        if family_name.endswith('_score') and family_name not in ['consensus_score']:
            family_df = summary_df[['feature', family_name]].copy()
            family_df = family_df.sort_values(family_name, ascending=False)
            # Match target ranking naming: {model}_importances.csv
            model_name = family_name.replace('_score', '')
            
            # Write to target-first structure only
            if target_importances_dir:
                try:
                    target_family_csv = target_importances_dir / f"{model_name}_importances.csv"
                    family_df.to_csv(target_family_csv, index=False)
                    logger.debug(f"✅ Saved {model_name} importances to target-first location: {target_family_csv}")
                except Exception as e:
                    logger.warning(f"Failed to write {model_name} importances to target-first location: {e}")
    
    # 4. Model agreement matrix → feature_importances/
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
    
    # Write to target-first structure only
    if target_importances_dir:
        try:
            target_agreement_path = target_importances_dir / "model_agreement_matrix.csv"
            agreement_matrix.to_csv(target_agreement_path)
            logger.debug(f"Also saved model agreement matrix to target-first location: {target_agreement_path}")
        except Exception as e:
            logger.debug(f"Failed to write model agreement matrix to target-first location: {e}")
    
    # 5. Metadata JSON → target level (matching TARGET_RANKING, metadata goes in cohort/ folder from reproducibility tracker)
    # For now, save a summary at target level for quick access (detailed metadata is in cohort/)
    metadata['n_selected_features'] = len(selected_features)
    metadata['n_total_results'] = len(all_results)
    metadata['model_families_used'] = list(set(r.model_family for r in all_results))

    # Save summary metadata at target level (detailed metadata is in cohort/ from reproducibility tracker)
    # This matches TARGET_RANKING structure where summary files are at target level
    # Write only to target-first structure (no legacy root-level writes) - view/symbol-scoped
    if target and base_output_dir.exists():
        try:
            target_clean = target.replace('/', '_').replace('\\', '_')
            # Extract view and symbol from metadata (view is REQUIRED)
            view = metadata.get('view') if metadata else None
            symbol = metadata.get('symbol') if metadata else None
            
            # Validate view is provided
            if view is None:
                raise ValueError(
                    f"view must be provided in metadata for feature_selection_summary. "
                    f"Metadata keys: {list(metadata.keys()) if metadata else 'None'}"
                )
            if view not in ("CROSS_SECTIONAL", "SYMBOL_SPECIFIC"):
                raise ValueError(f"Invalid view in metadata: {view}. Must be 'CROSS_SECTIONAL' or 'SYMBOL_SPECIFIC'")
            if view == "SYMBOL_SPECIFIC" and symbol is None:
                raise ValueError("symbol required in metadata when view='SYMBOL_SPECIFIC'")
            
            if use_output_layout and universe_sig:
                # Canonical path via OutputLayout (non-cohort write)
                from TRAINING.orchestration.utils.output_layout import OutputLayout
                layout = OutputLayout(
                    output_root=base_output_dir,
                    target=target_clean,
                    view=view,
                    universe_sig=universe_sig,
                    symbol=symbol if view == "SYMBOL_SPECIFIC" else None,
                    stage="FEATURE_SELECTION",  # Explicit stage for proper path scoping
                )
                target_summary_path = layout.repro_dir() / "feature_selection_summary.json"
            else:
                # Legacy path resolution with stage
                from TRAINING.orchestration.utils.target_first_paths import run_root, target_repro_file_path
                run_root_dir = run_root(base_output_dir)
                target_summary_path = target_repro_file_path(run_root_dir, target_clean, "feature_selection_summary.json", view=view, symbol=symbol, stage="FEATURE_SELECTION")
            
            target_summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(target_summary_path, "w") as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"✅ Saved feature selection summary to {target_summary_path}")
        except Exception as e:
            logger.warning(f"Failed to write feature selection summary to target-first location: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
    
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
        
        # Save detailed status file → target-first structure only (matching TARGET_RANKING structure) - view/symbol-scoped
        if target and base_output_dir.exists():
            try:
                from TRAINING.orchestration.utils.target_first_paths import run_root, target_repro_file_path
                target_clean = target.replace('/', '_').replace('\\', '_')
                run_root_dir = run_root(base_output_dir)
                # Extract view and symbol from metadata (view is REQUIRED)
                view = metadata.get('view') if metadata else None
                symbol = metadata.get('symbol') if metadata else None
                
                # Validate view is provided
                if view is None:
                    raise ValueError(
                        f"view must be provided in metadata for model_family_status. "
                        f"Metadata keys: {list(metadata.keys()) if metadata else 'None'}"
                    )
                if view not in ("CROSS_SECTIONAL", "SYMBOL_SPECIFIC"):
                    raise ValueError(f"Invalid view in metadata: {view}. Must be 'CROSS_SECTIONAL' or 'SYMBOL_SPECIFIC'")
                if view == "SYMBOL_SPECIFIC" and symbol is None:
                    raise ValueError("symbol required in metadata when view='SYMBOL_SPECIFIC'")
                
                # Use view/symbol-scoped path helper with explicit stage
                status_path = target_repro_file_path(run_root_dir, target_clean, "model_family_status.json", view=view, symbol=symbol, stage="FEATURE_SELECTION")
                status_path.parent.mkdir(parents=True, exist_ok=True)
                with open(status_path, "w") as f:
                    json.dump({
                        'summary': status_summary,
                        'detailed': family_statuses
                    }, f, indent=2)
                logger.info(f"✅ Saved model family status tracking to {status_path}")
            except Exception as e:
                logger.warning(f"Failed to write model family status to target-first location: {e}")
                import traceback
                logger.debug(f"Traceback: {traceback.format_exc()}")
        
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
            target = metadata.get('target_column', 'unknown_target')
            confidence_metrics = compute_target_confidence(
                summary_df=summary_df,
                all_results=all_results,
                model_families_config=model_families_config,
                target=target,
                confidence_config=confidence_config,
                top_k=None  # Will use config or default
            )
            
            # Save target confidence at target-first structure only (matching TARGET_RANKING structure) - view/symbol-scoped
            if target and base_output_dir.exists():
                try:
                    from TRAINING.orchestration.utils.target_first_paths import run_root, target_repro_file_path
                    target_clean = target.replace('/', '_').replace('\\', '_')
                    run_root_dir = run_root(base_output_dir)
                    # Extract view and symbol from metadata (view is REQUIRED)
                    view = metadata.get('view') if metadata else None
                    symbol = metadata.get('symbol') if metadata else None
                    
                    # Validate view is provided
                    if view is None:
                        raise ValueError(
                            f"view must be provided in metadata for target_confidence. "
                            f"Metadata keys: {list(metadata.keys()) if metadata else 'None'}"
                        )
                    if view not in ("CROSS_SECTIONAL", "SYMBOL_SPECIFIC"):
                        raise ValueError(f"Invalid view in metadata: {view}. Must be 'CROSS_SECTIONAL' or 'SYMBOL_SPECIFIC'")
                    if view == "SYMBOL_SPECIFIC" and symbol is None:
                        raise ValueError("symbol required in metadata when view='SYMBOL_SPECIFIC'")
                    
                    # Use view/symbol-scoped path helper with explicit stage
                    confidence_path = target_repro_file_path(run_root_dir, target_clean, "target_confidence.json", view=view, symbol=symbol, stage="FEATURE_SELECTION")
                    confidence_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(confidence_path, "w") as f:
                        json.dump(confidence_metrics, f, indent=2)
                    
                    # Log confidence summary
                    confidence = confidence_metrics['confidence']
                    reason = confidence_metrics.get('low_confidence_reason', '')
                    if confidence == 'LOW':
                        logger.warning(f"⚠️  Target {target}: confidence={confidence} ({reason})")
                    elif confidence == 'MEDIUM':
                        logger.info(f"ℹ️  Target {target}: confidence={confidence}")
                    else:
                        logger.info(f"✅ Target {target}: confidence={confidence}")
                    
                    logger.info(f"✅ Saved target confidence metrics to {confidence_path}")
                except Exception as e:
                    logger.warning(f"Failed to write target confidence to target-first location: {e}")
                    import traceback
                    logger.debug(f"Traceback: {traceback.format_exc()}")
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
                       default=_REPO_ROOT / "RESULTS/features/multi_model",
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
    logger.info(f"  • {args.output_dir}/artifacts/selected_features.txt")
    logger.info(f"  • {args.output_dir}/feature_importances/feature_importance_multi_model.csv")
    logger.info(f"  • {args.output_dir}/feature_importances/feature_importance_with_boruta_debug.csv")
    logger.info(f"  • {args.output_dir}/feature_importances/model_agreement_matrix.csv")
    logger.info(f"  • {args.output_dir}/feature_importances/<model>_importances.csv (per-model)")
    logger.info(f"  • {args.output_dir}/metadata/target_confidence.json")
    logger.info(f"  • {args.output_dir}/metadata/multi_model_metadata.json")
    logger.info(f"  • {args.output_dir}/metadata/model_family_status.json (family status tracking)")
    
    # Generate metrics rollups after all feature selections complete
    try:
        from TRAINING.orchestration.utils.reproducibility_tracker import ReproducibilityTracker
        from datetime import datetime
        
        # Find the REPRODUCIBILITY directory
        repro_dir = args.output_dir / "REPRODUCIBILITY"
        if not repro_dir.exists() and args.output_dir.parent.exists():
            repro_dir = args.output_dir.parent / "REPRODUCIBILITY"
        
        if repro_dir.exists():
            # Use output_dir parent as base (where RESULTS/runs/ typically is)
            base_dir = args.output_dir.parent if (args.output_dir / "REPRODUCIBILITY").exists() else args.output_dir
            tracker = ReproducibilityTracker(output_dir=base_dir)
            # Generate run_id from output_dir name or timestamp
            run_id = args.output_dir.name if args.output_dir.name else datetime.now().strftime("%Y%m%d_%H%M%S")
            tracker.generate_metrics_rollups(stage="FEATURE_SELECTION", run_id=run_id)
            logger.debug("✅ Generated metrics rollups for FEATURE_SELECTION")
    except Exception as e:
        logger.debug(f"Failed to generate metrics rollups: {e}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

