# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

"""
Target Predictability Ranking

Uses multiple model families to evaluate which of your 63 targets are most predictable.
This helps prioritize compute: train models on high-predictability targets first.

Methodology:
1. For each target, train multiple model families on sample data
2. Calculate predictability scores:
   - Model R² scores (cross-validated)
   - Feature importance magnitude (mean absolute SHAP/importance)
   - Consistency across models (low std = high confidence)
3. Rank targets by composite predictability score
4. Output ranked list with recommendations

Usage:
  # Rank all enabled targets
  python SCRIPTS/rank_target_predictability.py
  
  # Test on specific symbols first
  python SCRIPTS/rank_target_predictability.py --symbols AAPL,MSFT,GOOGL
  
  # Rank specific targets
  python SCRIPTS/rank_target_predictability.py --targets peak_60m,valley_60m,swing_high_15m
"""


import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass
import yaml
import json
from collections import defaultdict
import warnings

# Add project root FIRST (before any scripts.* imports)
# TRAINING/ranking/rank_target_predictability.py -> parents[2] = repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Add CONFIG directory to path for centralized config loading
_CONFIG_DIR = _REPO_ROOT / "CONFIG"
if str(_CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(_CONFIG_DIR))

# Try to import config loader
_CONFIG_AVAILABLE = False
try:
    from config_loader import get_cfg, get_safety_config
    _CONFIG_AVAILABLE = True
except ImportError:
    pass  # Logger not yet initialized, will be set up below

def _get_importance_top_fraction() -> float:
    """Get the top fraction for importance analysis from config."""
    if _CONFIG_AVAILABLE:
        try:
            # Load from feature_selection/multi_model.yaml
            fraction = float(get_cfg("aggregation.importance_top_fraction", default=0.10, config_name="multi_model"))
            return fraction
        except Exception:
            return 0.10  # FALLBACK_DEFAULT_OK
    return 0.10  # FALLBACK_DEFAULT_OK

# Import logging config utilities
try:
    from CONFIG.logging_config_utils import get_module_logging_config, get_backend_logging_config
    _LOGGING_CONFIG_AVAILABLE = True
except ImportError:
    _LOGGING_CONFIG_AVAILABLE = False
    # Fallback: create a simple config-like object
    class _DummyLoggingConfig:
        def __init__(self):
            self.gpu_detail = False
            self.cv_detail = False
            self.edu_hints = False
            self.detail = False

# Import checkpoint utility (after path is set)
from TRAINING.orchestration.utils.checkpoint import CheckpointManager
from TRAINING.ranking.predictability.data_loading import get_model_config

# Import unified task type system
from TRAINING.common.utils.task_types import (
    TaskType, TargetConfig, ModelConfig, 
    is_compatible, create_model_configs_from_yaml
)
from TRAINING.common.utils.task_metrics import evaluate_by_task, compute_composite_score
from TRAINING.ranking.utils.target_validation import validate_target, check_cv_compatibility

# Suppress expected warnings (harmless)
warnings.filterwarnings('ignore', message='X does not have valid feature names')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn')
warnings.filterwarnings('ignore', message='invalid value encountered in divide')
warnings.filterwarnings('ignore', message='invalid value encountered in true_divide')

# Setup logging with journald support
from TRAINING.orchestration.utils.logging_setup import setup_logging
logger = setup_logging(
    script_name="rank_target_predictability",
    level=logging.INFO,
    use_journald=True
)

# Leakage detection and feature analysis for target predictability

# Import from modular components
from TRAINING.ranking.predictability.leakage_detection.feature_analysis import (
    find_near_copy_features,
    is_calendar_feature as _is_calendar_feature,
    detect_leaking_features as _detect_leaking_features
)
from TRAINING.ranking.predictability.leakage_detection.reporting import (
    save_feature_importances as _save_feature_importances,
    log_suspicious_features as _log_suspicious_features
)

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
        min_corr: For regression, minimum absolute correlation (default: 99.9%)
    
    Returns:
        List of feature names that are near-copies of the target
    """
    # Load thresholds from config if not provided
    if min_match is None or min_corr is None:
        if _CONFIG_AVAILABLE:
            try:
                safety_cfg = get_safety_config()
                # safety_config.yaml has a top-level 'safety' key
                safety_section = safety_cfg.get('safety', {})
                leakage_cfg = safety_section.get('leakage_detection', {})
                pre_scan_cfg = leakage_cfg.get('pre_scan', {})
                if min_match is None:
                    min_match = float(pre_scan_cfg.get('min_match', 0.999))
                if min_corr is None:
                    min_corr = float(pre_scan_cfg.get('min_corr', 0.999))
                min_valid_pairs = int(pre_scan_cfg.get('min_valid_pairs', 10))
            except Exception:
                if min_match is None:
                    min_match = 0.999
                if min_corr is None:
                    min_corr = 0.999
                min_valid_pairs = 10
        else:
            if min_match is None:
                min_match = 0.999
            if min_corr is None:
                min_corr = 0.999
            min_valid_pairs = 10
    else:
        min_valid_pairs = 10  # Default if config not available
    
    leaky = []
    y_arr = y.to_numpy()
    
    for col in X.columns:
        try:
            x = X[col].to_numpy()
            
            # Ignore rows where either is NaN
            mask = ~np.isnan(x) & ~np.isnan(y_arr)
            if mask.sum() < min_valid_pairs:
                continue
            
            x_valid = x[mask]
            y_valid = y_arr[mask]
            
            # Binary classification: check if feature matches target (or inverse)
            if task_type == TaskType.BINARY_CLASSIFICATION:
                # Check if feature is exactly (or almost) the target
                same = (np.abs(x_valid - y_valid) < tol).mean()
                # Check if feature is exactly (or almost) 1 - target (inverted)
                inv_same = (np.abs(x_valid - (1 - y_valid)) < tol).mean()
                
                if same >= min_match or inv_same >= min_match:
                    leaky.append(col)
                    logger.error(
                        f"  🚨 PRE-TRAINING LEAK: {col} is a near-copy of target "
                        f"(match: {same:.1%}, inverse: {inv_same:.1%}, threshold: {min_match:.1%})"
                    )
            
            # Regression: check correlation
            elif task_type == TaskType.REGRESSION:
                try:
                    corr = np.corrcoef(x_valid, y_valid)[0, 1]
                    if not np.isnan(corr) and abs(corr) >= min_corr:
                        leaky.append(col)
                        logger.error(
                            f"  🚨 PRE-TRAINING LEAK: {col} has {abs(corr):.4f} correlation with target "
                            f"(threshold: {min_corr:.4f})"
                        )
                except Exception:
                    pass
            
            # Multiclass: check if feature matches target exactly
            elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
                same = (np.abs(x_valid - y_valid) < tol).mean()
                if same >= min_match:
                    leaky.append(col)
                    logger.error(
                        f"  🚨 PRE-TRAINING LEAK: {col} is a near-copy of target "
                        f"(match: {same:.1%}, threshold: {min_match:.1%})"
                    )
        except Exception as e:
            # Skip features that cause errors (e.g., non-numeric)
            continue
    
    return leaky


def _is_calendar_feature(feature_name: str) -> bool:
    """
    Check if a feature is a calendar/time-index feature (not suspicious by default).
    
    Calendar features are time-index features that are not "future info" but may
    dominate importance due to seasonality. These should be excluded from leak
    suspicion or handled separately.
    """
    calendar_patterns = [
        'trading_day_of_quarter', 'trading_day_of_month', 'trading_day_of_week',
        'wd_', 'month_', 'quarter_', 'day_of_', 'dow_', 'dom_', 'doy_',
        'monthly_seasonality', 'quarterly_seasonality', 'weekly_seasonality',
        'hour_', 'minute_', 'time_of_day', 'is_weekend', 'is_month_end',
        'is_quarter_end', 'is_year_end'
    ]
    feature_lower = feature_name.lower()
    return any(pattern in feature_lower for pattern in calendar_patterns)


def _detect_leaking_features(
    feature_names: List[str],
    importances: np.ndarray,
    model_name: str,
    threshold: float = 0.50,
    force_report: bool = False  # If True, always report top features even if no leak detected
) -> List[Tuple[str, float]]:
    """
    Detect features with suspiciously high importance (likely data leakage).
    
    Excludes calendar/time-index features from leak suspicion (they may dominate
    due to seasonality, which is legitimate, not leakage).
    
    Returns list of (feature_name, importance) tuples for suspicious features.
    """
    if len(feature_names) != len(importances):
        logger.warning(f"  Feature count mismatch: {len(feature_names)} names vs {len(importances)} importances")
        return []
    
    # Normalize importances to sum to 1
    total_importance = np.sum(importances)
    if total_importance == 0:
        logger.warning(f"  Total importance is zero for {model_name}")
        return []
    
    normalized_importance = importances / total_importance
    
    # Create sorted list of (feature, importance) pairs
    feature_imp_pairs = list(zip(feature_names, normalized_importance))
    feature_imp_pairs.sort(key=lambda x: x[1], reverse=True)
    
    # Find features with importance above threshold (excluding calendar features)
    suspicious = []
    for feat, imp in feature_imp_pairs:
        if imp >= threshold:
            # Exclude calendar features from leak detection
            if _is_calendar_feature(feat):
                logger.debug(
                    f"  📅 Calendar feature {feat} has {imp:.1%} importance (threshold: {threshold:.1%}) - "
                    f"likely seasonal, not leakage. Excluding from leak detection."
                )
                continue
            
            suspicious.append((feat, float(imp)))
            logger.error(
                f"  🚨 LEAK DETECTED: {feat} has {imp:.1%} importance in {model_name} "
                f"(threshold: {threshold:.1%}) - likely data leakage!"
            )
    
    # Also check if top feature dominates (even if below threshold)
    # But exclude calendar features from this check too
    if len(normalized_importance) > 0:
        top_feature, top_importance = feature_imp_pairs[0]
        
        # Skip dominance check if top feature is a calendar feature
        if not _is_calendar_feature(top_feature):
            # If top feature has >30% and is much larger than second, flag it
            if top_importance >= 0.30 and len(feature_imp_pairs) > 1:
                second_importance = feature_imp_pairs[1][1]
                if top_importance > second_importance * 3:  # 3x larger than second
                    if (top_feature, top_importance) not in suspicious:
                        suspicious.append((top_feature, float(top_importance)))
                        logger.warning(
                            f"  ⚠️  SUSPICIOUS: {top_feature} has {top_importance:.1%} importance "
                            f"(3x larger than next feature: {feature_imp_pairs[1][0]}={second_importance:.1%}) - investigate for leakage"
                        )
        else:
            # Calendar feature is top - log as info, not warning
            if len(feature_imp_pairs) > 1:
                second_importance = feature_imp_pairs[1][1]
                logger.info(
                    f"  📅 Calendar feature {top_feature} has {top_importance:.1%} importance "
                    f"(next: {feature_imp_pairs[1][0]}={second_importance:.1%}) - likely seasonal pattern, not leakage"
                )
    
    # CRITICAL: If we suspect a leak (force_report=True) or found suspicious features,
    # always print top 10 features to help identify the leak
    if force_report or suspicious:
        logger.info(f"  📊 TOP 10 FEATURES BY IMPORTANCE ({model_name}):")
        logger.info(f"  {'='*70}")
        for i, (feat, imp) in enumerate(feature_imp_pairs[:10], 1):
            marker = "🚨" if (feat, imp) in suspicious else "  "
            logger.info(f"    {marker} {i:2d}. {feat:50s} = {imp:.2%}")
        
        # Also check cumulative importance of top features
        top_5_importance = sum(imp for _, imp in feature_imp_pairs[:5])
        top_10_importance = sum(imp for _, imp in feature_imp_pairs[:10])
        logger.info(f"  📈 Cumulative: Top 5 = {top_5_importance:.1%}, Top 10 = {top_10_importance:.1%}")
        if top_5_importance > 0.80:
            logger.warning(f"  ⚠️  WARNING: Top 5 features account for {top_5_importance:.1%} of importance - likely leakage!")
        
        # Provide actionable next steps
        logger.info(f"  💡 NEXT STEPS:")
        logger.info(f"     1. Review the top features above - they likely contain future information")
        logger.info(f"     2. Check feature importance CSV for full analysis")
        logger.info(f"     3. Add leaking features to CONFIG/excluded_features.yaml")
        logger.info(f"     4. Restart Python process and re-run to apply new filters")
    
    return suspicious


def train_and_evaluate_models(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    task_type: TaskType,
    model_families: List[str] = None,
    multi_model_config: Dict[str, Any] = None,
    target_column: str = None,  # For leak reporting and horizon extraction
    data_interval_minutes: int = 5,  # Data bar interval (default: 5-minute bars)
    time_vals: Optional[np.ndarray] = None,  # Timestamps for each sample (for fold timestamp tracking)
    explicit_interval: Optional[Union[int, str]] = None,  # Explicit interval from config (for consistency)
    experiment_config: Optional[Any] = None  # Optional ExperimentConfig (for data.bar_interval)
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float], float, Dict[str, List[Tuple[str, float]]], Dict[str, Dict[str, float]], List[Dict[str, Any]]]:
    """
    Train multiple models and return task-aware metrics + importance magnitude
    
    Args:
        X: Feature matrix
        y: Target array
        feature_names: List of feature names
        task_type: TaskType enum (REGRESSION, BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION)
        model_families: List of model family names to use
        multi_model_config: Multi-model config dict
    
    Returns:
        model_metrics: Dict of {model_name: {metric_name: value}} per model (full metrics)
        model_scores: Dict of {model_name: primary_score} per model (backward compat)
        mean_importance: Mean absolute feature importance
        all_suspicious_features: Dict of {model_name: [(feature, importance), ...]}
        all_feature_importances: Dict of {model_name: {feature: importance}}
        fold_timestamps: List of {fold_idx, train_start, train_end, test_start, test_end} per fold
        perfect_correlation_models: Set of model names that triggered perfect correlation warnings
    
    Always returns 7 values, even on error (returns empty dicts, 0.0, empty list, and empty set)
    """
    # Get logging config for this module (at function start)
    if _LOGGING_CONFIG_AVAILABLE:
        log_cfg = get_module_logging_config('rank_target_predictability')
        lgbm_backend_cfg = get_backend_logging_config('lightgbm')
    else:
        log_cfg = _DummyLoggingConfig()
        lgbm_backend_cfg = type('obj', (object,), {'native_verbosity': -1, 'show_sparse_warnings': True})()
    
    # Initialize return values (ensures we always return 6 values)
    model_metrics = {}
    model_scores = {}
    importance_magnitudes = []
    all_suspicious_features = {}  # {model_name: [(feature, importance), ...]}
    all_feature_importances = {}  # {model_name: {feature: importance}} for detailed export
    fold_timestamps = []  # List of fold timestamp info
    
    try:
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler
        import lightgbm as lgb
        from TRAINING.ranking.utils.purged_time_series_split import PurgedTimeSeriesSplit
        from TRAINING.ranking.utils.leakage_filtering import _extract_horizon, _load_leakage_config
        from TRAINING.ranking.utils.feature_pruning import quick_importance_prune
    except Exception as e:
        logger.warning(f"Failed to import required libraries: {e}")
        return {}, {}, 0.0, {}, {}, []
    
    # Helper function for CV with early stopping (for gradient boosting models)
    def cross_val_score_with_early_stopping(model, X, y, cv, scoring, early_stopping_rounds=50, n_jobs=1):
        """
        Cross-validation with early stopping support for gradient boosting models.
        
        cross_val_score doesn't support early stopping callbacks, so we need a manual loop.
        This prevents overfitting by stopping when validation performance plateaus.
        """
        scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            try:
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Clone model for this fold
                from sklearn.base import clone
                fold_model = clone(model)
                
                # Train with early stopping
                # Check if model supports early stopping (LightGBM/XGBoost)
                supports_eval_set = hasattr(fold_model, 'fit') and 'eval_set' in fold_model.fit.__code__.co_varnames
                supports_early_stopping = hasattr(fold_model, 'fit') and 'early_stopping_rounds' in fold_model.fit.__code__.co_varnames
                
                if supports_eval_set:
                    # LightGBM style: uses callbacks
                    # Check by module name for reliability (str(type()) can be fragile)
                    model_module = type(fold_model).__module__
                    if 'lightgbm' in model_module.lower():
                        fold_model.fit(
                            X_train, y_train,
                            eval_set=[(X_val, y_val)],
                            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
                        )
                    # XGBoost style: early_stopping_rounds is set in constructor (XGBoost 2.0+)
                    # Don't pass it to fit() - it's already in the model
                    elif 'xgboost' in model_module.lower():
                        import xgboost as xgb
                        # XGBoost 2.0+ has early_stopping_rounds in constructor, not fit()
                        # Check if model already has it set, otherwise use eval_set only
                        fold_model.fit(
                            X_train, y_train,
                            eval_set=[(X_val, y_val)],
                            verbose=False
                        )
                    else:
                        # Fallback: try eval_set without callbacks
                        fold_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
                else:
                    # Standard fit for models without early stopping
                    fold_model.fit(X_train, y_train)
                
                # Evaluate on validation set
                if scoring == 'r2':
                    from sklearn.metrics import r2_score
                    y_pred = fold_model.predict(X_val)
                    score = r2_score(y_val, y_pred)
                elif scoring == 'roc_auc':
                    from sklearn.metrics import roc_auc_score
                    y_proba = fold_model.predict_proba(X_val)[:, 1] if hasattr(fold_model, 'predict_proba') else fold_model.predict(X_val)
                    if len(np.unique(y_val)) == 2:
                        score = roc_auc_score(y_val, y_proba)
                    else:
                        score = np.nan
                elif scoring == 'accuracy':
                    from sklearn.metrics import accuracy_score
                    y_pred = fold_model.predict(X_val)
                    score = accuracy_score(y_val, y_pred)
                else:
                    # Fallback to default scorer
                    from sklearn.metrics import get_scorer
                    scorer = get_scorer(scoring)
                    score = scorer(fold_model, X_val, y_val)
                
                scores.append(score)
            except Exception as e:
                logger.debug(f"  Fold {fold_idx + 1} failed: {e}")
                scores.append(np.nan)
        
        return np.array(scores)
    
    # ARCHITECTURAL IMPROVEMENT: Pre-prune low-importance features before expensive training
    # This reduces noise and prevents "Curse of Dimensionality" issues
    # Drop features with < 0.01% cumulative importance using a fast LightGBM model
    original_feature_count = len(feature_names)
    if original_feature_count > 100:  # Only prune if we have many features
        logger.info(f"  Pre-pruning features: {original_feature_count} features")
        
        # Determine task type string for pruning
        if task_type == TaskType.REGRESSION:
            task_str = 'regression'
        elif task_type == TaskType.BINARY_CLASSIFICATION:
            task_str = 'classification'
        else:
            task_str = 'classification'
        
        try:
            # Generate deterministic seed for feature pruning based on target
            from TRAINING.common.determinism import stable_seed_from
            # Use target_column if available, otherwise use default
            target_for_seed = target_column if target_column else 'pruning'
            prune_seed = stable_seed_from([target_for_seed, 'feature_pruning'])
            
            # Load feature pruning config
            if _CONFIG_AVAILABLE:
                try:
                    cumulative_threshold = get_cfg("preprocessing.feature_pruning.cumulative_threshold", default=0.0001, config_name="preprocessing_config")
                    min_features = get_cfg("preprocessing.feature_pruning.min_features", default=50, config_name="preprocessing_config")
                    n_estimators = get_cfg("preprocessing.feature_pruning.n_estimators", default=50, config_name="preprocessing_config")
                except Exception:
                    cumulative_threshold = 0.0001
                    min_features = 50
                    n_estimators = 50
            else:
                cumulative_threshold = 0.0001
                min_features = 50
                n_estimators = 50
            
            X_pruned, feature_names_pruned, pruning_stats = quick_importance_prune(
                X, y, feature_names,
                cumulative_threshold=cumulative_threshold,
                min_features=min_features,
                task_type=task_str,
                n_estimators=n_estimators,
                seed=prune_seed
            )
            
            if pruning_stats.get('dropped_count', 0) > 0:
                logger.info(f"  ✅ Pruned: {original_feature_count} → {len(feature_names_pruned)} features "
                          f"(dropped {pruning_stats['dropped_count']} low-importance features)")
                X = X_pruned
                feature_names = feature_names_pruned
            else:
                logger.info(f"  No features pruned (all above threshold)")
        except Exception as e:
            logger.warning(f"  Feature pruning failed: {e}, using all features")
            # Continue with original features
    
    # Get CV config
    cv_config = multi_model_config.get('cross_validation', {}) if multi_model_config else {}
    folds = cv_config.get('folds', 3)
    cv_n_jobs = cv_config.get('n_jobs', 1)
    
    # CRITICAL: Use PurgedTimeSeriesSplit to prevent temporal leakage
    # Standard K-Fold shuffles data randomly, which destroys time patterns
    # TimeSeriesSplit respects time order but doesn't prevent overlap leakage
    # PurgedTimeSeriesSplit enforces a gap between train/test = target horizon
    
    # Calculate purge_overlap based on target horizon
    # Extract target horizon (in minutes) from target column name
    leakage_config = _load_leakage_config()
    target_horizon_minutes = _extract_horizon(target_column, leakage_config) if target_column else None
    
    # Auto-detect data interval from timestamps if available, otherwise use parameter
    # CRITICAL: Using wrong interval causes data leakage (e.g., 1m data with 5m assumption leaks 4 minutes)
    if time_vals is not None and len(time_vals) > 1:
        try:
            # Convert to pandas Timestamp if needed
            # Handle both numeric (nanoseconds) and datetime timestamps
            if isinstance(time_vals[0], (int, float, np.integer, np.floating)):
                # Handle numeric timestamps (nanoseconds or Unix timestamp)
                time_series = pd.to_datetime(time_vals, unit='ns')
            elif isinstance(time_vals, np.ndarray) and time_vals.dtype.kind == 'M':
                # Already datetime64 array
                time_series = pd.Series(time_vals)
            else:
                time_series = pd.Series(time_vals)
            
            # Ensure time_series is datetime type for proper diff calculation
            if not pd.api.types.is_datetime64_any_dtype(time_series):
                time_series = pd.to_datetime(time_series)
            
            # CRITICAL: For panel data, multiple rows share the same timestamp
            # Calculate diff on UNIQUE timestamps, not all rows (otherwise median will be 0)
            unique_times = time_series.unique()
            unique_times_sorted = pd.Series(unique_times).sort_values()
            
            # Calculate median time difference between unique timestamps
            time_diffs = unique_times_sorted.diff().dropna()
            # time_diffs should be TimedeltaIndex when time_series is datetime
            if isinstance(time_diffs, pd.TimedeltaIndex) and len(time_diffs) > 0:
                median_diff_minutes = abs(time_diffs.median().total_seconds()) / 60.0
            elif len(time_diffs) > 0:
                # Fallback: if diff didn't produce Timedeltas, calculate manually
                median_diff = time_diffs.median()
                if isinstance(median_diff, pd.Timedelta):
                    median_diff_minutes = abs(median_diff.total_seconds()) / 60.0
                elif isinstance(median_diff, (int, float, np.integer, np.floating)):
                    # Assume nanoseconds if numeric (use abs to handle unsorted timestamps)
                    median_diff_minutes = abs(float(median_diff)) / 1e9 / 60.0
                else:
                    raise ValueError(f"Unexpected median_diff type: {type(median_diff)}")
            else:
                # No differences (all timestamps identical) - use default
                median_diff_minutes = data_interval_minutes
                logger.warning(f"  All timestamps identical, cannot detect interval, using parameter: {data_interval_minutes}m")
            
            # Round to common intervals (1m, 5m, 15m, 30m, 60m)
            common_intervals = [1, 5, 15, 30, 60]
            detected_interval = min(common_intervals, key=lambda x: abs(x - median_diff_minutes))
            
            # Only use auto-detection if it's close to a common interval (within 20% tolerance)
            if abs(median_diff_minutes - detected_interval) / detected_interval < 0.2:
                data_interval_minutes = detected_interval
                logger.info(f"  Auto-detected data interval: {median_diff_minutes:.1f}m → {data_interval_minutes}m (from timestamps)")
            else:
                # Fall back to parameter if detection is unclear
                logger.warning(f"  Auto-detection unclear ({median_diff_minutes:.1f}m), using parameter: {data_interval_minutes}m")
        except Exception as e:
            logger.warning(f"  Failed to auto-detect interval from timestamps: {e}, using parameter: {data_interval_minutes}m")
    else:
        # Use parameter value (default: 5)
        logger.info(f"  Using data interval from parameter: {data_interval_minutes}m")
    
    # Convert horizon from minutes to number of bars
    # Load purge settings from config
    if _CONFIG_AVAILABLE:
        try:
            purge_buffer_bars = int(get_cfg("pipeline.leakage.purge_buffer_bars", default=5, config_name="pipeline_config"))
        except Exception:
            purge_buffer_bars = 5
    else:
        purge_buffer_bars = 5  # Safety buffer (5 bars = 25 minutes)
    
    # ARCHITECTURAL FIX: Use time-based purging instead of row-count based
    # This prevents leakage when data interval doesn't match assumptions
    if target_horizon_minutes is not None:
        # Create Timedelta for purge window (target horizon + safety buffer)
        purge_buffer_minutes = purge_buffer_bars * data_interval_minutes
        purge_time = pd.Timedelta(minutes=target_horizon_minutes + purge_buffer_minutes)
        logger.info(f"  Target horizon: {target_horizon_minutes}m, purge_time: {purge_time}")
    else:
        # Fallback: use config value or conservative default (60m + 25m buffer = 85m)
        if _CONFIG_AVAILABLE:
            try:
                purge_time_minutes = int(get_cfg("pipeline.leakage.purge_time_minutes", default=85, config_name="pipeline_config"))
                purge_time = pd.Timedelta(minutes=purge_time_minutes)
            except Exception:
                purge_time = pd.Timedelta(minutes=85)
        else:
            purge_time = pd.Timedelta(minutes=85)
        logger.warning(f"  Could not extract target horizon from '{target_column}', using default purge_time={purge_time}")
    
    # Create purged time series split with time-based purging
    # CRITICAL: Validate time_vals alignment and sorting before using time-based purging
    if time_vals is not None and len(time_vals) == len(X):
        # Ensure time_vals is sorted (required for binary search in PurgedTimeSeriesSplit)
        time_series = pd.Series(time_vals) if not isinstance(time_vals, pd.Series) else time_vals
        if not time_series.is_monotonic_increasing:
            logger.warning("⚠️  time_vals is not sorted! Sorting X, y, and time_vals together")
            sort_idx = np.argsort(time_vals)
            X = X[sort_idx]
            y = y[sort_idx]
            time_vals = time_series.iloc[sort_idx].values if isinstance(time_series, pd.Series) else time_series[sort_idx]
            logger.info(f"  Sorted data by timestamp (preserving alignment)")
        
        tscv = PurgedTimeSeriesSplit(
            n_splits=folds, 
            purge_overlap_time=purge_time,
            time_column_values=time_vals
        )
        if log_cfg.cv_detail:
            logger.info(f"  Using PurgedTimeSeriesSplit (TIME-BASED): {folds} folds, purge_time={purge_time}")
    else:
        # CRITICAL: Row-count based purging is INVALID for panel data (multiple symbols per timestamp)
        # With 50 symbols, 1 bar = 50 rows. Using row counts causes catastrophic leakage.
        # We MUST fail loudly rather than silently producing invalid results.
        raise ValueError(
            f"CRITICAL: time_vals is required for panel data (cross-sectional). "
            f"Row-count based purging is INVALID when multiple symbols share the same timestamp. "
            f"With {len(X)} samples, row-count purging would cause 100% data leakage. "
            f"Please ensure cross_sectional_data.py returns time_vals."
        )
    
    # Capture fold timestamps if time_vals is provided
    if time_vals is not None and len(time_vals) == len(X):
        try:
            for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X, y)):
                train_times = time_vals[train_idx]
                test_times = time_vals[test_idx]
                fold_timestamps.append({
                    'fold_idx': fold_idx + 1,
                    'train_start': pd.Timestamp(train_times.min()) if len(train_times) > 0 else None,
                    'train_end': pd.Timestamp(train_times.max()) if len(train_times) > 0 else None,
                    'test_start': pd.Timestamp(test_times.min()) if len(test_times) > 0 else None,
                    'test_end': pd.Timestamp(test_times.max()) if len(test_times) > 0 else None,
                    'train_samples': len(train_idx),
                    'test_samples': len(test_idx)
                })
            if log_cfg.cv_detail:
                logger.info(f"  Captured timestamps for {len(fold_timestamps)} folds")
        except Exception as e:
            logger.warning(f"  Failed to capture fold timestamps: {e}")
            fold_timestamps = []
    
    if model_families is None:
        # Load from multi-model config if available
        if multi_model_config:
            model_families_dict = multi_model_config.get('model_families', {})
            if model_families_dict is None or not isinstance(model_families_dict, dict):
                logger.warning("model_families in config is None or not a dict. Using defaults.")
                model_families = ['lightgbm', 'random_forest', 'neural_network']
            else:
                model_families = [
                    name for name, config in model_families_dict.items()
                    if config is not None and isinstance(config, dict) and config.get('enabled', False)
                ]
            logger.debug(f"Using {len(model_families)} models from config: {', '.join(model_families)}")
        else:
            model_families = ['lightgbm', 'random_forest', 'neural_network']
    
    # Create ModelConfig objects for this task type
    model_configs = create_model_configs_from_yaml(multi_model_config, task_type) if multi_model_config else []
    # Filter to only enabled model families
    model_configs = [mc for mc in model_configs if mc.name in model_families]
    
    # Note: model_metrics, model_scores, importance_magnitudes already initialized at function start
    
    # Determine task characteristics
    unique_vals = np.unique(y[~np.isnan(y)])
    is_binary = task_type == TaskType.BINARY_CLASSIFICATION
    is_multiclass = task_type == TaskType.MULTICLASS_CLASSIFICATION
    is_classification = is_binary or is_multiclass
    
    # Select scoring metric based on task type
    if task_type == TaskType.REGRESSION:
        scoring = 'r2'
    elif task_type == TaskType.BINARY_CLASSIFICATION:
        scoring = 'roc_auc'
    else:  # MULTICLASS_CLASSIFICATION
        scoring = 'accuracy'
    
    # Helper function to detect perfect correlation (data leakage)
    # Track which models had perfect correlation warnings (for auto-fixer)
    _perfect_correlation_models = set()
    
    # Load thresholds from config
    if _CONFIG_AVAILABLE:
        try:
            safety_cfg = get_safety_config()
            # safety_config.yaml has a top-level 'safety' key
            safety_section = safety_cfg.get('safety', {})
            leakage_cfg = safety_section.get('leakage_detection', {})
            _correlation_threshold = float(leakage_cfg.get('auto_fix_thresholds', {}).get('perfect_correlation', 0.999))
            _suspicious_score_threshold = float(leakage_cfg.get('model_alerts', {}).get('suspicious_score', 0.99))
        except Exception:
            _correlation_threshold = 0.999  # FALLBACK_DEFAULT_OK
            _suspicious_score_threshold = 0.99  # FALLBACK_DEFAULT_OK
    else:
        # Load from safety config
        if _CONFIG_AVAILABLE:
            try:
                safety_cfg = get_safety_config()
                # safety_config.yaml has a top-level 'safety' key
                safety_section = safety_cfg.get('safety', {})
                leakage_cfg = safety_section.get('leakage_detection', {})
                _correlation_threshold = float(leakage_cfg.get('auto_fix_thresholds', {}).get('perfect_correlation', 0.999))
                _suspicious_score_threshold = float(leakage_cfg.get('model_alerts', {}).get('suspicious_score', 0.99))
            except Exception:
                _correlation_threshold = 0.999  # FALLBACK_DEFAULT_OK
                _suspicious_score_threshold = 0.99  # FALLBACK_DEFAULT_OK
        else:
            _correlation_threshold = 0.999  # FALLBACK_DEFAULT_OK
            _suspicious_score_threshold = 0.99  # FALLBACK_DEFAULT_OK
    
    # NOTE: Removed _critical_leakage_detected flag - training accuracy alone is not
    # a reliable leakage signal for tree-based models. Real defense: schema filters + pre-scan.
    
    def _check_for_perfect_correlation(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> bool:
        """
        Check if predictions are perfectly correlated with targets.
        
        NOTE: High training accuracy alone is NOT a reliable signal for leakage, especially
        for tree-based models (Random Forest, LightGBM) which can overfit to 100% training
        accuracy through memorization even without leakage.
        
        This function now only logs a warning for debugging purposes. Real leakage defense
        comes from:
        - Explicit feature filters (schema, pattern-based exclusions)
        - Pre-training near-copy scan
        - Time-purged cross-validation
        
        Returns True if perfect correlation detected (for tracking), but does NOT trigger
        early exit or mark target as LEAKAGE_DETECTED.
        """
        try:
            # Tree-based models can easily overfit to 100% training accuracy
            tree_models = {'random_forest', 'lightgbm', 'xgboost', 'catboost'}
            is_tree_model = model_name.lower() in tree_models
            
            # For classification, check if predictions match exactly
            if task_type in {TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION}:
                if len(y_true) == len(y_pred):
                    accuracy = np.mean(y_true == y_pred)
                    if accuracy >= _correlation_threshold:  # Configurable threshold (default: 99.9%)
                        metric_name = "training accuracy"
                        
                        if is_tree_model:
                            # Tree models: This is likely just overfitting, not leakage
                            logger.warning(
                                f"  ⚠️  {model_name} reached {accuracy:.1%} {metric_name} "
                                f"(threshold: {_correlation_threshold:.1%}). "
                                f"This may just be overfitting - tree ensembles can memorize training data. "
                                f"Check CV metrics instead. Real leakage defense: schema filters + pre-scan."
                            )
                        else:
                            # Non-tree models: Still suspicious but less likely to be false positive
                            logger.warning(
                                f"  ⚠️  {model_name} reached {accuracy:.1%} {metric_name} "
                                f"(threshold: {_correlation_threshold:.1%}). "
                                f"High training accuracy detected - investigate if CV metrics are also suspiciously high."
                            )
                        
                        _perfect_correlation_models.add(model_name)  # Track for debugging/auto-fixer
                        return True  # Return True for tracking, but don't trigger early exit
            
            # For regression, check correlation
            elif task_type == TaskType.REGRESSION:
                if len(y_true) == len(y_pred):
                    corr = np.corrcoef(y_true, y_pred)[0, 1]
                    if not np.isnan(corr) and abs(corr) >= _correlation_threshold:
                        if is_tree_model:
                            logger.warning(
                                f"  ⚠️  {model_name} has correlation {corr:.4f} "
                                f"(threshold: {_correlation_threshold:.4f}). "
                                f"This may just be overfitting - check CV metrics instead."
                            )
                        else:
                            logger.warning(
                                f"  ⚠️  {model_name} has correlation {corr:.4f} "
                                f"(threshold: {_correlation_threshold:.4f}). "
                                f"High correlation detected - investigate if CV metrics are also suspiciously high."
                            )
                        
                        _perfect_correlation_models.add(model_name)  # Track for debugging
                        return True  # Return True for tracking, but don't trigger early exit
        except Exception:
            pass
        return False
    
    # Helper function to compute and store full task-aware metrics
    def _compute_and_store_metrics(model_name: str, model, X: np.ndarray, y: np.ndarray,
                                   primary_score: float, task_type: TaskType):
        """
        Compute full task-aware metrics and store in both model_metrics and model_scores.
        
        Args:
            model_name: Name of the model
            model: Fitted model
            X: Feature matrix (for predictions)
            y: True target values
            primary_score: Primary score from CV (R², AUC, or accuracy)
            task_type: TaskType enum
        """
        # Defensive check: ensure model_scores and model_metrics are dicts
        nonlocal model_scores, model_metrics
        if model_scores is None or not isinstance(model_scores, dict):
            logger.warning(f"model_scores is None or not a dict in _compute_and_store_metrics, reinitializing")
            model_scores = {}
        if model_metrics is None or not isinstance(model_metrics, dict):
            logger.warning(f"model_metrics is None or not a dict in _compute_and_store_metrics, reinitializing")
            model_metrics = {}
        
        # Store primary score for backward compatibility
        model_scores[model_name] = primary_score
        
        # Compute full task-aware metrics
        try:
            # Calculate training accuracy/correlation BEFORE checking for perfect correlation
            # This is needed for auto-fixer to detect high training scores
            training_accuracy = None
            if task_type in {TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION}:
                if hasattr(model, 'predict_proba'):
                    if task_type == TaskType.BINARY_CLASSIFICATION:
                        y_proba = model.predict_proba(X)[:, 1]
                        y_pred_train = (y_proba >= 0.5).astype(int)
                    else:
                        y_proba = model.predict_proba(X)
                        y_pred_train = y_proba.argmax(axis=1)
                else:
                    y_pred_train = model.predict(X)
                if len(y) == len(y_pred_train):
                    training_accuracy = np.mean(y == y_pred_train)
            elif task_type == TaskType.REGRESSION:
                y_pred_train = model.predict(X)
                if len(y) == len(y_pred_train):
                    corr = np.corrcoef(y, y_pred_train)[0, 1]
                    if not np.isnan(corr):
                        training_accuracy = abs(corr)  # Store absolute correlation for regression
            
            if task_type == TaskType.REGRESSION:
                y_pred = model.predict(X)
                # Check for perfect correlation (leakage) - this sets _critical_leakage_detected flag
                if _check_for_perfect_correlation(y, y_pred, model_name):
                    logger.error(f"  CRITICAL: {model_name} shows signs of data leakage! Check feature filtering.")
                    # Early exit: don't compute more metrics, return immediately
                    return
                full_metrics = evaluate_by_task(task_type, y, y_pred, return_ic=True)
            elif task_type == TaskType.BINARY_CLASSIFICATION:
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X)[:, 1]  # Probability of class 1
                    y_pred = (y_proba >= 0.5).astype(int)
                else:
                    # Fallback for models without predict_proba
                    y_pred = model.predict(X)
                    y_proba = np.clip(y_pred, 0, 1)  # Assume predictions are probabilities
                # Check for perfect correlation (for debugging/tracking only - not a leakage signal)
                _check_for_perfect_correlation(y, y_pred, model_name)
                full_metrics = evaluate_by_task(task_type, y, y_proba)
            else:  # MULTICLASS_CLASSIFICATION
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X)
                    y_pred = y_proba.argmax(axis=1)
                else:
                    # Fallback: one-hot encode predictions
                    y_pred = model.predict(X)
                    n_classes = len(np.unique(y[~np.isnan(y)]))
                    y_proba = np.eye(n_classes)[y_pred.astype(int)]
                # Check for perfect correlation (for debugging/tracking only - not a leakage signal)
                _check_for_perfect_correlation(y, y_pred, model_name)
                full_metrics = evaluate_by_task(task_type, y, y_proba)
            
            # Store full metrics
            model_metrics[model_name] = full_metrics
            
            # Also store training accuracy/correlation for auto-fixer detection
            # This is the in-sample training score (not CV), which is what triggers leakage warnings
            if training_accuracy is not None:
                if task_type == TaskType.REGRESSION:
                    # For regression, store as 'training_r2' (even though it's correlation, auto-fixer checks 'r2')
                    # The actual CV R² is already in full_metrics['r2']
                    model_metrics[model_name]['training_r2'] = training_accuracy
                else:
                    # For classification, store as 'training_accuracy' (auto-fixer checks 'accuracy')
                    # The actual CV accuracy is already in full_metrics['accuracy']
                    model_metrics[model_name]['training_accuracy'] = training_accuracy
        except Exception as e:
            logger.warning(f"Failed to compute full metrics for {model_name}: {e}")
            # Fallback to primary score only
            if task_type == TaskType.REGRESSION:
                model_metrics[model_name] = {'r2': primary_score}
            elif task_type == TaskType.BINARY_CLASSIFICATION:
                model_metrics[model_name] = {'roc_auc': primary_score}
            else:
                model_metrics[model_name] = {'accuracy': primary_score}
    
    # Helper function to update both model_scores and model_metrics
    # NOTE: This is now mainly for backward compat - full metrics computed after training
    def _update_model_score(model_name: str, score: float):
        """Update model_scores (backward compat) - full metrics computed separately"""
        model_scores[model_name] = score
    
    # Check for degenerate target BEFORE training models
    # A target is degenerate if it has < 2 unique values or one class has < 2 samples
    unique_vals = np.unique(y[~np.isnan(y)])
    if len(unique_vals) < 2:
        logger.debug(f"    Skipping: Target has only {len(unique_vals)} unique value(s)")
        return {}, {}, 0.0, {}, {}, [], set()  # model_metrics, model_scores, mean_importance, suspicious_features, feature_importances, fold_timestamps, perfect_correlation_models
    
    # For classification, check class balance
    if is_binary or is_multiclass:
        class_counts = np.bincount(y[~np.isnan(y)].astype(int))
        min_class_count = class_counts[class_counts > 0].min()
        if min_class_count < 2:
            logger.debug(f"    Skipping: Smallest class has only {min_class_count} sample(s)")
            return {}, {}, 0.0, {}, {}, [], set()  # model_metrics, model_scores, mean_importance, suspicious_features, feature_importances, fold_timestamps, perfect_correlation_models
    
    # LightGBM
    if 'lightgbm' in model_families:
        try:
            # GPU settings (will fallback to CPU if GPU not available)
            gpu_params = {}
            try:
                # Try CUDA first (fastest)
                # DESIGN_CONSTANT_OK: n_estimators=1 for diagnostic leakage detection only, not production behavior
                test_model = lgb.LGBMRegressor(device='cuda', n_estimators=1, verbose=lgbm_backend_cfg.native_verbosity)
                test_model.fit(np.random.rand(10, 5), np.random.rand(10))
                gpu_params = {'device': 'cuda', 'gpu_device_id': 0}
                if log_cfg.gpu_detail:
                    logger.info("  Using GPU (CUDA) for LightGBM")
            except:
                try:
                    # Try OpenCL
                    # DESIGN_CONSTANT_OK: n_estimators=1 for diagnostic leakage detection only, not production behavior
                    test_model = lgb.LGBMRegressor(device='gpu', n_estimators=1, verbose=lgbm_backend_cfg.native_verbosity)
                    test_model.fit(np.random.rand(10, 5), np.random.rand(10))
                    gpu_params = {'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0}
                    if log_cfg.gpu_detail:
                        logger.info("  Using GPU (OpenCL) for LightGBM")
                except:
                    if log_cfg.gpu_detail:
                        logger.info("  Using CPU for LightGBM")
            
            # Get config values
            lgb_config = get_model_config('lightgbm', multi_model_config)
            # Defensive check: ensure config is a dict
            if not isinstance(lgb_config, dict):
                lgb_config = {}
            # Remove objective and device from config (we set these explicitly)
            lgb_config_clean = {k: v for k, v in lgb_config.items() if k not in ['device', 'objective', 'metric']}
            
            # Set verbose level from backend config
            # Note: verbose is a model constructor parameter, not fit() parameter
            verbose_level = lgbm_backend_cfg.native_verbosity
            
            if is_binary:
                model = lgb.LGBMClassifier(
                    objective='binary',
                    verbose=verbose_level,  # Enable verbose for GPU verification
                    **lgb_config_clean,
                    **gpu_params
                )
            elif is_multiclass:
                n_classes = len(unique_vals)
                model = lgb.LGBMClassifier(
                    objective='multiclass',
                    num_class=n_classes,
                    verbose=verbose_level,  # Enable verbose for GPU verification
                    **lgb_config_clean,
                    **gpu_params
                )
            else:
                model = lgb.LGBMRegressor(
                    objective='regression',
                    verbose=verbose_level,  # Enable verbose for GPU verification
                    **lgb_config_clean,
                    **gpu_params
                )
            
            # CRITICAL FIX: Use manual CV loop with early stopping for gradient boosting
            # Get early stopping rounds from config (default: 50)
            early_stopping_rounds = lgb_config.get('early_stopping_rounds', 50) if isinstance(lgb_config, dict) else 50
            
            if log_cfg.cv_detail:
                logger.info(f"  Using CV with early stopping (rounds={early_stopping_rounds}) for LightGBM")
            scores = cross_val_score_with_early_stopping(
                model, X, y, cv=tscv, scoring=scoring, 
                early_stopping_rounds=early_stopping_rounds, n_jobs=1  # n_jobs=1 for early stopping compatibility
            )
            valid_scores = scores[~np.isnan(scores)]
            primary_score = valid_scores.mean() if len(valid_scores) > 0 else np.nan
            
            # Train once on full data (with early stopping on a validation split) to get importance
            # CRITICAL: Use time-aware split (last 20% as validation) - don't shuffle time series data
            # Guard against empty arrays
            if len(X) < 10:
                logger.warning(f"  ⚠️  Too few samples ({len(X)}) for train/val split, fitting on all data")
                split_idx = len(X)
            else:
                split_idx = int(len(X) * 0.8)
                split_idx = max(1, split_idx)  # Ensure at least 1 sample in validation
            
            if split_idx < len(X):
                X_train_final, X_val_final = X[:split_idx], X[split_idx:]
                y_train_final, y_val_final = y[:split_idx], y[split_idx:]
            else:
                # Fallback: use all data if too small
                X_train_final, X_val_final = X, X
                y_train_final, y_val_final = y, y
            # Log GPU usage if available (controlled by config)
            if 'device' in gpu_params and log_cfg.gpu_detail:
                logger.info(f"  🚀 Training LightGBM on {gpu_params['device'].upper()} (device_id={gpu_params.get('gpu_device_id', 0)})")
                logger.info(f"  📊 Dataset size: {len(X_train_final)} samples, {X_train_final.shape[1]} features")
                if log_cfg.edu_hints:
                    logger.info(f"  💡 Note: GPU is most efficient for large datasets (>100k samples)")
            
            model.fit(
                X_train_final, y_train_final,
                eval_set=[(X_val_final, y_val_final)],
                callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
            )
            
            # Verify GPU was actually used (only if gpu_detail enabled)
            if 'device' in gpu_params and log_cfg.gpu_detail:
                # Check model parameters to see what device was actually used
                try:
                    model_params = model.get_params()
                    actual_device = model_params.get('device', 'unknown')
                    if actual_device != 'cpu':
                        logger.info(f"  ✅ LightGBM confirmed using {actual_device.upper()}")
                    else:
                        logger.warning(f"  ⚠️  LightGBM fell back to CPU despite GPU params")
                        logger.warning(f"     This can happen if dataset is too small or GPU not properly configured")
                except:
                    logger.debug("  Could not verify device from model params")
            
            # CRITICAL: Check for suspiciously high scores (likely leakage)
            has_leak = False
            if not np.isnan(primary_score) and primary_score >= _suspicious_score_threshold:
                # Use task-appropriate metric name
                if task_type == TaskType.REGRESSION:
                    metric_name = "R²"
                elif task_type == TaskType.BINARY_CLASSIFICATION:
                    metric_name = "ROC-AUC"
                else:
                    metric_name = "Accuracy"
                logger.error(f"  🚨 LEAKAGE ALERT: lightgbm {metric_name}={primary_score:.4f} >= 0.99 - likely data leakage!")
                logger.error(f"    Features: {len(feature_names)} features")
                logger.error(f"    Analyzing feature importances to identify leaks...")
                has_leak = True
            
            # LEAK DETECTION: Analyze feature importance for suspicious patterns
            importances = model.feature_importances_
            # Load importance threshold from config
            if _CONFIG_AVAILABLE:
                try:
                    safety_cfg = get_safety_config()
                    # safety_config.yaml has a top-level 'safety' key
                    safety_section = safety_cfg.get('safety', {})
                    leakage_cfg = safety_section.get('leakage_detection', {})
                    importance_threshold = float(leakage_cfg.get('importance', {}).get('single_feature_threshold', 0.50))
                except Exception:
                    importance_threshold = 0.50
            else:
                importance_threshold = 0.50
            suspicious_features = _detect_leaking_features(
                feature_names, importances, model_name='lightgbm',
                threshold=importance_threshold,
                force_report=has_leak  # Always report top features if score indicates leak
            )
            if suspicious_features:
                all_suspicious_features['lightgbm'] = suspicious_features
            
            # Store all feature importances for detailed export
            # CRITICAL: Align importance to feature_names order to ensure fingerprint match
            importance_series = pd.Series(importances, index=feature_names[:len(importances)] if len(importances) <= len(feature_names) else feature_names)
            # Reindex to match exact feature_names order (fills missing with 0.0)
            importance_series = importance_series.reindex(feature_names, fill_value=0.0)
            all_feature_importances['lightgbm'] = importance_series.to_dict()
            
            # Compute and store full task-aware metrics
            _compute_and_store_metrics('lightgbm', model, X, y, primary_score, task_type)
            
            # Use percentage of total importance in top 10% features (0-1 scale, interpretable)
            total_importance = np.sum(importances)
            if total_importance > 0:
                top_fraction = _get_importance_top_fraction()
                top_k = max(1, int(len(importances) * top_fraction))
                top_importance_sum = np.sum(np.sort(importances)[-top_k:])
                # Normalize to 0-1: what % of total importance is in top 10%?
                importance_ratio = top_importance_sum / total_importance
            else:
                importance_ratio = 0.0
            importance_magnitudes.append(importance_ratio)
            
        except Exception as e:
            logger.warning(f"LightGBM failed: {e}")
    
    # Random Forest
    if 'random_forest' in model_families:
        try:
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            
            # Get config values
            rf_config = get_model_config('random_forest', multi_model_config)
            
            if is_binary or is_multiclass:
                model = RandomForestClassifier(**rf_config)
            else:
                model = RandomForestRegressor(**rf_config)
            
            scores = cross_val_score(model, X, y, cv=tscv, scoring=scoring, n_jobs=cv_n_jobs, error_score=np.nan)
            valid_scores = scores[~np.isnan(scores)]
            primary_score = valid_scores.mean() if len(valid_scores) > 0 else np.nan
            
            # ⚠️ IMPORTANCE BIAS WARNING: This fits on the full dataset (in-sample)
            # Deep trees/GBMs can memorize noise, making feature importance biased.
            # TODO: Future enhancement - use permutation importance calculated on CV test folds
            # For now, this is acceptable but be aware that importance may be inflated
            model.fit(X, y)
            
            # Check for suspicious scores
            has_leak = not np.isnan(primary_score) and primary_score >= _suspicious_score_threshold
            
            # LEAK DETECTION: Analyze feature importance
            importances = model.feature_importances_
            suspicious_features = _detect_leaking_features(
                feature_names, importances, model_name='random_forest', 
                threshold=0.50, force_report=has_leak
            )
            if suspicious_features:
                all_suspicious_features['random_forest'] = suspicious_features
            
            # Store all feature importances for detailed export
            # CRITICAL: Align importance to feature_names order to ensure fingerprint match
            importance_series = pd.Series(importances, index=feature_names[:len(importances)] if len(importances) <= len(feature_names) else feature_names)
            # Reindex to match exact feature_names order (fills missing with 0.0)
            importance_series = importance_series.reindex(feature_names, fill_value=0.0)
            all_feature_importances['random_forest'] = importance_series.to_dict()
            
            # Compute and store full task-aware metrics
            _compute_and_store_metrics('random_forest', model, X, y, primary_score, task_type)
            
            # Use percentage of total importance in top 10% features (0-1 scale, interpretable)
            total_importance = np.sum(importances)
            if total_importance > 0:
                top_fraction = _get_importance_top_fraction()
                top_k = max(1, int(len(importances) * top_fraction))
                top_importance_sum = np.sum(np.sort(importances)[-top_k:])
                # Normalize to 0-1: what % of total importance is in top 10%?
                importance_ratio = top_importance_sum / total_importance
            else:
                importance_ratio = 0.0
            importance_magnitudes.append(importance_ratio)
            
        except Exception as e:
            logger.warning(f"RandomForest failed: {e}")
    
    # Neural Network
    if 'neural_network' in model_families:
        try:
            from sklearn.neural_network import MLPRegressor, MLPClassifier
            from sklearn.impute import SimpleImputer
            from sklearn.exceptions import ConvergenceWarning
            from sklearn.compose import TransformedTargetRegressor
            from sklearn.pipeline import Pipeline
            
            # Get config values
            nn_config = get_model_config('neural_network', multi_model_config)
            
            if is_binary or is_multiclass:
                # For classification: Pipeline handles imputation and scaling within CV folds
                steps = [
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                    ('model', MLPClassifier(**nn_config))
                ]
                pipeline = Pipeline(steps)
                model = pipeline
                y_for_training = y
            else:
                # For regression: Pipeline for features + TransformedTargetRegressor for target
                # This ensures no data leakage - all scaling/imputation happens within CV folds
                feature_steps = [
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                    ('model', MLPRegressor(**nn_config))
                ]
                feature_pipeline = Pipeline(feature_steps)
                model = TransformedTargetRegressor(
                    regressor=feature_pipeline,
                    transformer=StandardScaler()
                )
                y_for_training = y
            
            # Neural networks need special handling for degenerate targets
            # Suppress convergence warnings (they're noisy and we handle failures gracefully)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=ConvergenceWarning)
                try:
                    # Pipeline ensures imputation/scaling happens within each CV fold (no leakage)
                    scores = cross_val_score(model, X, y_for_training, cv=tscv, scoring=scoring, n_jobs=cv_n_jobs, error_score=np.nan)
                    valid_scores = scores[~np.isnan(scores)]
                    primary_score = valid_scores.mean() if len(valid_scores) > 0 else np.nan
                except ValueError as e:
                    if "least populated class" in str(e) or "too few" in str(e):
                        logger.debug(f"    Neural Network: Target too imbalanced for CV")
                        primary_score = np.nan
                        model_metrics['neural_network'] = {'roc_auc': np.nan} if task_type == TaskType.BINARY_CLASSIFICATION else {'r2': np.nan} if task_type == TaskType.REGRESSION else {'accuracy': np.nan}
                        model_scores['neural_network'] = np.nan
                    else:
                        raise
            
            # Fit on raw data (Pipeline handles preprocessing internally)
            # ⚠️ IMPORTANCE BIAS WARNING: This fits on the full dataset (in-sample)
            # See comment above for details
            if not np.isnan(primary_score):
                model.fit(X, y_for_training)
                
                # Compute and store full task-aware metrics (Pipeline handles preprocessing)
                _compute_and_store_metrics('neural_network', model, X, y_for_training, primary_score, task_type)
            
            baseline_score = model.score(X, y_for_training)
            
            perm_scores = []
            for i in range(min(10, X.shape[1])):  # Sample 10 features
                X_perm = X.copy()
                np.random.shuffle(X_perm[:, i])
                perm_score = model.score(X_perm, y_for_training)
                perm_scores.append(abs(baseline_score - perm_score))
            
            importance_magnitudes.append(np.mean(perm_scores))
            
        except Exception as e:
            logger.warning(f"NeuralNetwork failed: {e}")
    
    # XGBoost
    if 'xgboost' in model_families:
        try:
            import xgboost as xgb
            
            # Get config values
            xgb_config = get_model_config('xgboost', multi_model_config)
            # Defensive check: ensure config is a dict
            if not isinstance(xgb_config, dict):
                xgb_config = {}
            # Remove task-specific parameters (we set these explicitly based on task type)
            # CRITICAL: Extract early_stopping_rounds from config - it goes in constructor for XGBoost 2.0+
            early_stopping_rounds = xgb_config.get('early_stopping_rounds', None)
            xgb_config_clean = {k: v for k, v in xgb_config.items() 
                              if k not in ['objective', 'eval_metric', 'early_stopping_rounds']}
            
            # XGBoost 2.0+ requires early_stopping_rounds in constructor, not fit()
            if early_stopping_rounds is not None:
                xgb_config_clean['early_stopping_rounds'] = early_stopping_rounds
            
            if is_binary:
                model = xgb.XGBClassifier(
                    objective='binary:logistic',
                    **xgb_config_clean
                )
            elif is_multiclass:
                n_classes = len(unique_vals)
                model = xgb.XGBClassifier(
                    objective='multi:softprob',
                    num_class=n_classes,
                    **xgb_config_clean
                )
            else:
                model = xgb.XGBRegressor(
                    objective='reg:squarederror',
                    **xgb_config_clean
                )
            
            # CRITICAL FIX: Use manual CV loop with early stopping for gradient boosting
            # Get early stopping rounds from config (default: 50)
            # NOTE: For XGBoost 2.0+, early_stopping_rounds is set in constructor above, not passed to fit()
            early_stopping_rounds = xgb_config.get('early_stopping_rounds', 50) if isinstance(xgb_config, dict) else 50
            
            logger.info(f"  Using CV with early stopping (rounds={early_stopping_rounds}) for XGBoost")
            try:
                # XGBoost uses same early stopping interface as LightGBM
                scores = cross_val_score_with_early_stopping(
                    model, X, y, cv=tscv, scoring=scoring,
                    early_stopping_rounds=early_stopping_rounds, n_jobs=1
                )
                valid_scores = scores[~np.isnan(scores)]
                primary_score = valid_scores.mean() if len(valid_scores) > 0 else np.nan
            except ValueError as e:
                if "Invalid classes" in str(e) or "Expected" in str(e):
                    logger.debug(f"    XGBoost: Target degenerate in some CV folds")
                    primary_score = np.nan
                    model_metrics['xgboost'] = {'roc_auc': np.nan} if task_type == TaskType.BINARY_CLASSIFICATION else {'r2': np.nan} if task_type == TaskType.REGRESSION else {'accuracy': np.nan}
                    model_scores['xgboost'] = np.nan
                else:
                    raise
            
            # Train once on full data (with early stopping) to get importance and full metrics
            # CRITICAL: Use time-aware split (last 20% as validation) - don't shuffle time series data
            if not np.isnan(primary_score):
                # Guard against empty arrays
                if len(X) < 10:
                    logger.warning(f"  ⚠️  Too few samples ({len(X)}) for train/val split, fitting on all data")
                    split_idx = len(X)
                else:
                    split_idx = int(len(X) * 0.8)
                    split_idx = max(1, split_idx)  # Ensure at least 1 sample in validation
                
                if split_idx < len(X):
                    X_train_final, X_val_final = X[:split_idx], X[split_idx:]
                    y_train_final, y_val_final = y[:split_idx], y[split_idx:]
                else:
                    # Fallback: use all data if too small
                    X_train_final, X_val_final = X, X
                    y_train_final, y_val_final = y, y
                # XGBoost 2.0+: early_stopping_rounds is set in constructor, not passed to fit()
                # The model already has it configured from the constructor above
                model.fit(
                    X_train_final, y_train_final,
                    eval_set=[(X_val_final, y_val_final)],
                    verbose=False
                )
                
                # Check for suspicious scores
                has_leak = primary_score >= _suspicious_score_threshold
                
                # Compute and store full task-aware metrics
                _compute_and_store_metrics('xgboost', model, X, y, primary_score, task_type)
                
                # LEAK DETECTION: Analyze feature importance
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    suspicious_features = _detect_leaking_features(
                        feature_names, importances, model_name='xgboost', 
                        threshold=0.50, force_report=has_leak
                    )
                    if suspicious_features:
                        all_suspicious_features['xgboost'] = suspicious_features
                    
                    # Store all feature importances for detailed export
                    # CRITICAL: Align importance to feature_names order to ensure fingerprint match
                    importance_series = pd.Series(importances, index=feature_names[:len(importances)] if len(importances) <= len(feature_names) else feature_names)
                    # Reindex to match exact feature_names order (fills missing with 0.0)
                    importance_series = importance_series.reindex(feature_names, fill_value=0.0)
                    all_feature_importances['xgboost'] = importance_series.to_dict()
            if hasattr(model, 'feature_importances_'):
                # Use percentage of total importance in top 10% features (0-1 scale, interpretable)
                importances = model.feature_importances_
                total_importance = np.sum(importances)
                if total_importance > 0:
                    top_fraction = _get_importance_top_fraction()
                    top_k = max(1, int(len(importances) * top_fraction))
                    top_importance_sum = np.sum(np.sort(importances)[-top_k:])
                    # Normalize to 0-1: what % of total importance is in top fraction?
                    importance_ratio = top_importance_sum / total_importance
                else:
                    importance_ratio = 0.0
                importance_magnitudes.append(importance_ratio)
        except Exception as e:
            logger.warning(f"XGBoost failed: {e}")
    
    # CatBoost
    if 'catboost' in model_families:
        try:
            import catboost as cb
            from TRAINING.ranking.utils.target_utils import is_classification_target, is_binary_classification_target
            
            # Get config values
            cb_config = get_model_config('catboost', multi_model_config)
            # Defensive check: ensure config is a dict
            if not isinstance(cb_config, dict):
                cb_config = {}
            
            # Build params dict (copy to avoid mutating original)
            params = dict(cb_config)
            
            # Auto-detect target type and set loss_function if not specified
            if "loss_function" not in params:
                if is_classification_target(y):
                    if is_binary_classification_target(y):
                        params["loss_function"] = "Logloss"
                    else:
                        params["loss_function"] = "MultiClass"
                else:
                    params["loss_function"] = "RMSE"
            # If loss_function is specified in config, respect it (YAML in charge)
            
            # Choose model class based on target type
            if is_classification_target(y):
                model = cb.CatBoostClassifier(**params)
            else:
                model = cb.CatBoostRegressor(**params)
            
            try:
                scores = cross_val_score(model, X, y, cv=tscv, scoring=scoring, n_jobs=cv_n_jobs, error_score=np.nan)
                valid_scores = scores[~np.isnan(scores)]
                primary_score = valid_scores.mean() if len(valid_scores) > 0 else np.nan
            except (ValueError, TypeError) as e:
                if "Invalid classes" in str(e) or "Expected" in str(e):
                    logger.debug(f"    CatBoost: Target degenerate in some CV folds")
                    primary_score = np.nan
                    model_metrics['catboost'] = {'roc_auc': np.nan} if task_type == TaskType.BINARY_CLASSIFICATION else {'r2': np.nan} if task_type == TaskType.REGRESSION else {'accuracy': np.nan}
                    model_scores['catboost'] = np.nan
                else:
                    raise
            
            if not np.isnan(primary_score):
                model.fit(X, y)

                # Compute and store full task-aware metrics
                _compute_and_store_metrics('catboost', model, X, y, primary_score, task_type)
                
                # PERFORMANCE AUDIT: Track CatBoost importance computation
                import time
                importance_start_time = time.time()
                try:
                    from TRAINING.common.utils.performance_audit import get_auditor
                    auditor = get_auditor()
                    if auditor.enabled:
                        fingerprint_kwargs = {
                            'data_shape': X.shape,
                            'n_features': X.shape[1] if len(X.shape) > 1 else None,
                            'importance_type': 'PredictionValuesChange',
                            'stage': 'leakage_detection'
                        }
                        fingerprint = auditor._compute_fingerprint('catboost.get_feature_importance', **fingerprint_kwargs)
                except Exception:
                    auditor = None
                    fingerprint = None
                
                # CatBoost requires training dataset to compute feature importance
                importance = model.get_feature_importance(data=X, type='PredictionValuesChange')
                
                # Track call
                if auditor and auditor.enabled:
                    importance_elapsed = time.time() - importance_start_time
                    auditor.track_call(
                        func_name='catboost.get_feature_importance',
                        duration=importance_elapsed,
                        rows=X.shape[0],
                        cols=X.shape[1] if len(X.shape) > 1 else None,
                        stage='leakage_detection',
                        cache_hit=False,
                        input_fingerprint=fingerprint
                    )
            else:
                importance = np.array([])
            if len(importance) > 0:
                total_importance = np.sum(importance)
                if total_importance > 0:
                    top_k = max(1, int(len(importance) * 0.1))
                    top_importance_sum = np.sum(np.sort(importance)[-top_k:])
                    importance_ratio = top_importance_sum / total_importance
                else:
                    importance_ratio = 0.0
            else:
                importance_ratio = 0.0
            importance_magnitudes.append(importance_ratio)
        except ImportError:
            logger.warning("CatBoost not available (pip install catboost)")
        except Exception as e:
            logger.warning(f"CatBoost failed: {e}")
    
    # Lasso
    if 'lasso' in model_families:
        try:
            from sklearn.linear_model import Lasso
            from sklearn.pipeline import Pipeline
            from TRAINING.common.utils.sklearn_safe import make_sklearn_dense_X
            
            # Get config values
            lasso_config = get_model_config('lasso', multi_model_config)
            
            # Use sklearn-safe conversion (handles NaNs, dtypes, infs)
            X_dense, feature_names_dense = make_sklearn_dense_X(X, feature_names)
            
            # CRITICAL FIX: Pipeline ensures scaling happens within each CV fold (no leakage)
            # Lasso requires scaling for proper convergence (features must be on similar scales)
            # Note: X_dense is already imputed by make_sklearn_dense_X, so we only need scaler
            steps = [
                ('scaler', StandardScaler()),  # Required for Lasso convergence
                ('model', Lasso(**lasso_config))
            ]
            pipeline = Pipeline(steps)
            
            scores = cross_val_score(pipeline, X_dense, y, cv=tscv, scoring=scoring, n_jobs=cv_n_jobs, error_score=np.nan)
            valid_scores = scores[~np.isnan(scores)]
            primary_score = valid_scores.mean() if len(valid_scores) > 0 else np.nan
            
            # ⚠️ IMPORTANCE BIAS WARNING: This fits on the full dataset (in-sample)
            # See comment above for details
            pipeline.fit(X_dense, y)
            
            # Compute and store full task-aware metrics (Lasso is regression-only)
            if not np.isnan(primary_score) and task_type == TaskType.REGRESSION:
                _compute_and_store_metrics('lasso', pipeline, X_dense, y, primary_score, task_type)
            
            # Extract coefficients from the fitted model
            model = pipeline.named_steps['model']
            importance = np.abs(model.coef_)
            
            # Update feature_names to match dense array
            feature_names = feature_names_dense
            if len(importance) > 0:
                total_importance = np.sum(importance)
                if total_importance > 0:
                    top_k = max(1, int(len(importance) * 0.1))
                    top_importance_sum = np.sum(np.sort(importance)[-top_k:])
                    importance_ratio = top_importance_sum / total_importance
                else:
                    importance_ratio = 0.0
            else:
                importance_ratio = 0.0
            importance_magnitudes.append(importance_ratio)
        except Exception as e:
            logger.warning(f"Lasso failed: {e}")
    
    # Mutual Information
    if 'mutual_information' in model_families:
        try:
            from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
            from TRAINING.common.utils.sklearn_safe import make_sklearn_dense_X
            
            # Mutual information doesn't handle NaN - use sklearn-safe conversion
            X_dense, feature_names_dense = make_sklearn_dense_X(X, feature_names)
            
            # Get config values
            mi_config = get_model_config('mutual_information', multi_model_config)
            
            # Get seed from SST (determinism system) - no hardcoded defaults
            mi_seed = mi_config.get('seed')
            if mi_seed is None:
                from TRAINING.common.determinism import stable_seed_from
                mi_seed = stable_seed_from(['mutual_information', target_column if target_column else 'default'])
            
            # Suppress warnings for zero-variance features
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                if is_binary or is_multiclass:
                    importance = mutual_info_classif(X_dense, y, 
                                                    random_state=mi_seed,
                                                    discrete_features=mi_config.get('discrete_features', 'auto'))
                else:
                    importance = mutual_info_regression(X_dense, y, 
                                                       random_state=mi_seed,
                                                       discrete_features=mi_config.get('discrete_features', 'auto'))
            
            # Update feature_names to match dense array
            feature_names = feature_names_dense
            
            # Handle NaN/inf
            importance = np.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Mutual information doesn't have R², so we use a proxy based on max MI
            # Normalize to 0-1 scale for importance
            if len(importance) > 0 and np.max(importance) > 0:
                importance_normalized = importance / np.max(importance)
                total_importance = np.sum(importance_normalized)
                if total_importance > 0:
                    top_fraction = _get_importance_top_fraction()
                    top_k = max(1, int(len(importance_normalized) * top_fraction))
                    top_importance_sum = np.sum(np.sort(importance_normalized)[-top_k:])
                    importance_ratio = top_importance_sum / total_importance
                else:
                    importance_ratio = 0.0
            else:
                importance_ratio = 0.0
            
            # For mutual information, we can't compute R² directly
            # Use a proxy: higher MI concentration = better predictability
            # Scale to approximate R² range (0-0.3 for good targets)
            model_scores['mutual_information'] = min(0.3, importance_ratio * 0.3)
            importance_magnitudes.append(importance_ratio)
        except Exception as e:
            logger.warning(f"Mutual Information failed: {e}")
    
    # Univariate Selection
    if 'univariate_selection' in model_families:
        try:
            from sklearn.feature_selection import f_regression, f_classif
            from TRAINING.common.utils.sklearn_safe import make_sklearn_dense_X
            
            # F-tests don't handle NaN - use sklearn-safe conversion
            X_dense, feature_names_dense = make_sklearn_dense_X(X, feature_names)
            
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
            
            # Normalize F-statistics
            if len(scores) > 0 and np.max(scores) > 0:
                importance = scores / np.max(scores)
                total_importance = np.sum(importance)
                if total_importance > 0:
                    top_k = max(1, int(len(importance) * 0.1))
                    top_importance_sum = np.sum(np.sort(importance)[-top_k:])
                    importance_ratio = top_importance_sum / total_importance
                else:
                    importance_ratio = 0.0
            else:
                importance_ratio = 0.0
            
            # F-statistics don't have R², use proxy
            model_scores['univariate_selection'] = min(0.3, importance_ratio * 0.3)
            importance_magnitudes.append(importance_ratio)
        except Exception as e:
            logger.warning(f"Univariate Selection failed: {e}")
    
    # RFE
    if 'rfe' in model_families:
        try:
            from sklearn.feature_selection import RFE
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            from sklearn.impute import SimpleImputer
            
            # RFE uses RandomForest which handles NaN, but let's impute for consistency
            imputer = SimpleImputer(strategy='median')
            X_imputed = imputer.fit_transform(X)
            
            # Get config values
            rfe_config = get_model_config('rfe', multi_model_config)
            # FIX: Use .get() with default to prevent KeyError, and clamp to [1, n_features]
            default_n_features = max(1, int(0.2 * X_imputed.shape[1]))
            n_features_to_select = rfe_config.get('n_features_to_select', default_n_features)
            # FIX: Clamp to [1, n_features] to prevent edge-case crashes on small feature sets
            n_features_to_select = max(1, min(n_features_to_select, X_imputed.shape[1]))
            step = rfe_config.get('step', 5)
            
            # Use random_forest config for RFE estimator
            rf_config = get_model_config('random_forest', multi_model_config)
            
            if is_binary or is_multiclass:
                estimator = RandomForestClassifier(**rf_config)
            else:
                estimator = RandomForestRegressor(**rf_config)
            
            selector = RFE(estimator, n_features_to_select=n_features_to_select, step=step)
            selector.fit(X_imputed, y)
            
            # Get R² using cross-validation on selected features (proper validation)
            selected_features = selector.support_
            if np.any(selected_features):
                X_selected = X_imputed[:, selected_features]
                # Quick RF for scoring (use smaller config)
                quick_rf_config = get_model_config('random_forest', multi_model_config).copy()
                # Use smaller model for quick scoring
                quick_rf_config['n_estimators'] = 50
                quick_rf_config['max_depth'] = 8
                
                if is_binary or is_multiclass:
                    quick_rf = RandomForestClassifier(**quick_rf_config)
                else:
                    quick_rf = RandomForestRegressor(**quick_rf_config)
                
                # Use cross-validation for proper validation (not training score)
                scores = cross_val_score(quick_rf, X_selected, y, cv=tscv, scoring=scoring, n_jobs=cv_n_jobs, error_score=np.nan)
                valid_scores = scores[~np.isnan(scores)]
                model_scores['rfe'] = valid_scores.mean() if len(valid_scores) > 0 else np.nan
            else:
                model_scores['rfe'] = np.nan
            
            # Convert ranking to importance
            ranking = selector.ranking_
            importance = 1.0 / (ranking + 1e-6)
            if len(importance) > 0:
                total_importance = np.sum(importance)
                if total_importance > 0:
                    top_k = max(1, int(len(importance) * 0.1))
                    top_importance_sum = np.sum(np.sort(importance)[-top_k:])
                    importance_ratio = top_importance_sum / total_importance
                else:
                    importance_ratio = 0.0
            else:
                importance_ratio = 0.0
            importance_magnitudes.append(importance_ratio)
        except Exception as e:
            logger.warning(f"RFE failed: {e}")
    
    # Boruta
    if 'boruta' in model_families:
        try:
            from boruta import BorutaPy
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            from TRAINING.common.utils.sklearn_safe import make_sklearn_dense_X
            
            # Boruta doesn't support NaN - use sklearn-safe conversion
            X_dense, feature_names_dense = make_sklearn_dense_X(X, feature_names)
            
            # Get config values
            boruta_config = get_model_config('boruta', multi_model_config)
            
            # Use random_forest config for Boruta estimator
            rf_config = get_model_config('random_forest', multi_model_config)
            
            # Get seed from SST (determinism system) - no hardcoded defaults
            boruta_seed = boruta_config.get('seed')
            if boruta_seed is None:
                from TRAINING.common.determinism import stable_seed_from
                boruta_seed = stable_seed_from(['boruta', target_column if target_column else 'default'])
            
            # Remove seed from rf_config to prevent double argument error
            rf_config_clean = rf_config.copy()
            rf_config_clean.pop('seed', None)
            
            if is_binary or is_multiclass:
                rf = RandomForestClassifier(**rf_config_clean, random_state=boruta_seed)
            else:
                rf = RandomForestRegressor(**rf_config_clean, random_state=boruta_seed)
            
            boruta = BorutaPy(rf, n_estimators='auto', verbose=0,
                            random_state=boruta_seed,
                            max_iter=boruta_config.get('max_iter', 100))
            boruta.fit(X_dense, y)
            
            # Get R² using cross-validation on selected features (proper validation)
            selected_features = boruta.support_
            if np.any(selected_features):
                X_selected = X_dense[:, selected_features]
                # Quick RF for scoring (use smaller config)
                quick_rf_config = get_model_config('random_forest', multi_model_config).copy()
                # Use smaller model for quick scoring
                quick_rf_config['n_estimators'] = 50
                quick_rf_config['max_depth'] = 8
                
                if is_binary or is_multiclass:
                    quick_rf = RandomForestClassifier(**quick_rf_config)
                else:
                    quick_rf = RandomForestRegressor(**quick_rf_config)
                
                # Use cross-validation for proper validation (not training score)
                scores = cross_val_score(quick_rf, X_selected, y, cv=tscv, scoring=scoring, n_jobs=cv_n_jobs, error_score=np.nan)
                valid_scores = scores[~np.isnan(scores)]
                model_scores['boruta'] = valid_scores.mean() if len(valid_scores) > 0 else np.nan
            else:
                model_scores['boruta'] = np.nan
            
            # Update feature_names to match dense array
            feature_names = feature_names_dense
            
            # Convert to importance
            ranking = boruta.ranking_
            selected = boruta.support_
            importance = np.where(selected, 1.0, np.where(ranking == 2, 0.5, 0.1))
            if len(importance) > 0:
                total_importance = np.sum(importance)
                if total_importance > 0:
                    top_k = max(1, int(len(importance) * 0.1))
                    top_importance_sum = np.sum(np.sort(importance)[-top_k:])
                    importance_ratio = top_importance_sum / total_importance
                else:
                    importance_ratio = 0.0
            else:
                importance_ratio = 0.0
            importance_magnitudes.append(importance_ratio)
        except ImportError:
            logger.warning("Boruta not available (pip install Boruta)")
        except Exception as e:
            logger.warning(f"Boruta failed: {e}")
    
    # Stability Selection
    if 'stability_selection' in model_families:
        try:
            from sklearn.linear_model import LassoCV, LogisticRegressionCV
            from TRAINING.common.utils.sklearn_safe import make_sklearn_dense_X
            
            # Stability selection uses Lasso/LogisticRegression which don't handle NaN
            X_dense, feature_names_dense = make_sklearn_dense_X(X, feature_names)
            
            # Get config values
            stability_config = get_model_config('stability_selection', multi_model_config)
            n_bootstrap = stability_config.get('n_bootstrap', 50)
            # Get seed from SST (determinism system) - no hardcoded defaults
            seed = stability_config.get('seed')
            if seed is None:
                from TRAINING.common.determinism import stable_seed_from
                seed = stable_seed_from(['stability_selection', target_column if target_column else 'default'])
            stability_cv = stability_config.get('cv', 3)
            stability_n_jobs = stability_config.get('n_jobs', 1)
            stability_cs = stability_config.get('Cs', 10)
            stability_scores = np.zeros(X_dense.shape[1])
            bootstrap_r2_scores = []
            
            # Use lasso config for stability selection models
            lasso_config = get_model_config('lasso', multi_model_config)
            
            for _ in range(n_bootstrap):
                indices = np.random.choice(len(X_dense), size=len(X_dense), replace=True)
                X_boot, y_boot = X_dense[indices], y[indices]
                
                try:
                    # Use TimeSeriesSplit for internal CV (even though bootstrap breaks temporal order,
                    # this maintains consistency with the rest of the codebase)
                    # Clean config to prevent double seed argument
                    from TRAINING.common.utils.config_cleaner import clean_config_for_estimator
                    if is_binary or is_multiclass:
                        lr_config = {'Cs': stability_cs, 'cv': tscv, 'max_iter': lasso_config.get('max_iter', 1000), 'n_jobs': stability_n_jobs}
                        lr_config_clean = clean_config_for_estimator(LogisticRegressionCV, lr_config, extra_kwargs={'random_state': seed}, family_name='stability_selection')
                        model = LogisticRegressionCV(**lr_config_clean, random_state=seed)
                    else:
                        lasso_config_clean_dict = {'cv': tscv, 'max_iter': lasso_config.get('max_iter', 1000), 'n_jobs': stability_n_jobs}
                        lasso_config_clean = clean_config_for_estimator(LassoCV, lasso_config_clean_dict, extra_kwargs={'random_state': seed}, family_name='stability_selection')
                        model = LassoCV(**lasso_config_clean, random_state=seed)
                    
                    model.fit(X_boot, y_boot)
                    coef = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
                    stability_scores += (np.abs(coef) > 1e-6).astype(int)
                    
                    # Get R² using cross-validation (proper validation, not training score)
                    # Note: Bootstrap samples break temporal order, but we still use TimeSeriesSplit
                    # for consistency (it won't help here, but maintains the pattern)
                    # Use a quick model for CV scoring
                    if is_binary or is_multiclass:
                        lr_cv_config = {'Cs': [1.0], 'cv': tscv, 'max_iter': lasso_config.get('max_iter', 1000), 'n_jobs': 1}
                        lr_cv_config_clean = clean_config_for_estimator(LogisticRegressionCV, lr_cv_config, extra_kwargs={'random_state': seed}, family_name='stability_selection')
                        cv_model = LogisticRegressionCV(**lr_cv_config_clean, random_state=seed)
                    else:
                        lasso_cv_config = {'cv': tscv, 'max_iter': lasso_config.get('max_iter', 1000), 'n_jobs': 1}
                        lasso_cv_config_clean = clean_config_for_estimator(LassoCV, lasso_cv_config, extra_kwargs={'random_state': seed}, family_name='stability_selection')
                        cv_model = LassoCV(**lasso_cv_config_clean, random_state=seed)
                    cv_scores = cross_val_score(cv_model, X_boot, y_boot, cv=tscv, scoring=scoring, n_jobs=1, error_score=np.nan)
                    valid_cv_scores = cv_scores[~np.isnan(cv_scores)]
                    if len(valid_cv_scores) > 0:
                        bootstrap_r2_scores.append(valid_cv_scores.mean())
                except:
                    continue
            
            # Update feature_names to match dense array
            feature_names = feature_names_dense
            
            # Average R² across bootstraps
            if bootstrap_r2_scores:
                model_scores['stability_selection'] = np.mean(bootstrap_r2_scores)
            else:
                model_scores['stability_selection'] = np.nan
            
            # Normalize stability scores to importance
            importance = stability_scores / n_bootstrap
            if len(importance) > 0:
                total_importance = np.sum(importance)
                if total_importance > 0:
                    top_k = max(1, int(len(importance) * 0.1))
                    top_importance_sum = np.sum(np.sort(importance)[-top_k:])
                    importance_ratio = top_importance_sum / total_importance
                else:
                    importance_ratio = 0.0
            else:
                importance_ratio = 0.0
            importance_magnitudes.append(importance_ratio)
        except Exception as e:
            logger.warning(f"Stability Selection failed: {e}")
    
    # Histogram Gradient Boosting
    if 'histogram_gradient_boosting' in model_families:
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
            
            # Get config values
            hgb_config = get_model_config('histogram_gradient_boosting', multi_model_config)
            # Defensive check: ensure config is a dict
            if not isinstance(hgb_config, dict):
                hgb_config = {}
            # Remove task-specific parameters (loss is set automatically by classifier/regressor)
            hgb_config_clean = {k: v for k, v in hgb_config.items() if k != 'loss'}
            
            if is_binary or is_multiclass:
                model = HistGradientBoostingClassifier(**hgb_config_clean)
            else:
                model = HistGradientBoostingRegressor(**hgb_config_clean)
            
            scores = cross_val_score(model, X, y, cv=tscv, scoring=scoring, n_jobs=cv_n_jobs, error_score=np.nan)
            valid_scores = scores[~np.isnan(scores)]
            primary_score = valid_scores.mean() if len(valid_scores) > 0 else np.nan
            
            # Train once to get importance and full metrics
            model.fit(X, y)
            
            # Compute and store full task-aware metrics
            if not np.isnan(primary_score):
                _compute_and_store_metrics('histogram_gradient_boosting', model, X, y, primary_score, task_type)
            if hasattr(model, 'feature_importances_'):
                # Use percentage of total importance in top 10% features (0-1 scale, interpretable)
                importances = model.feature_importances_
                total_importance = np.sum(importances)
                if total_importance > 0:
                    top_fraction = _get_importance_top_fraction()
                    top_k = max(1, int(len(importances) * top_fraction))
                    top_importance_sum = np.sum(np.sort(importances)[-top_k:])
                    # Normalize to 0-1: what % of total importance is in top fraction?
                    importance_ratio = top_importance_sum / total_importance
                else:
                    importance_ratio = 0.0
                importance_magnitudes.append(importance_ratio)
        except Exception as e:
            logger.warning(f"Histogram Gradient Boosting failed: {e}")
    
    mean_importance = np.mean(importance_magnitudes) if importance_magnitudes else 0.0
    
    # model_scores already contains primary scores (backward compatible)
    # model_metrics contains full metrics dict
    # all_suspicious_features contains leak detection results (aggregated across all models)
    # all_feature_importances contains detailed per-feature importances for export
    return model_metrics, model_scores, mean_importance, all_suspicious_features, all_feature_importances, fold_timestamps, _perfect_correlation_models


def _save_feature_importances(
    target_column: str,
    symbol: str,
    feature_importances: Dict[str, Dict[str, float]],
    output_dir: Path = None,
    view: str = "CROSS_SECTIONAL",
    universe_sig: Optional[str] = None  # PATCH 4: Required for proper scoping
) -> None:
    """
    Save detailed per-model, per-feature importance scores to CSV files.
    
    Creates structure (with universe_sig):
    targets/{target}/reproducibility/{view}/universe={sig}/(symbol={sym})/feature_importances/
      lightgbm_importances.csv
      xgboost_importances.csv
      ...
    
    Args:
        target_column: Name of the target being evaluated
        symbol: Symbol being evaluated
        feature_importances: Dict of {model_name: {feature: importance}}
        output_dir: Base output directory (defaults to results/)
        view: CROSS_SECTIONAL or SYMBOL_SPECIFIC
        universe_sig: Universe signature from SST (required for proper scoping)
    """
    # PATCH 4: Require universe_sig for proper scoping
    if not universe_sig:
        logger.error(
            f"SCOPE BUG: universe_sig not provided for {target_column} feature importances. "
            f"Cannot create view-scoped paths. Feature importances will not be written."
        )
        return  # Don't write to unscoped location
    
    if output_dir is None:
        output_dir = _REPO_ROOT / "results"
    
    # Find base run directory for target-first structure
    base_output_dir = output_dir
    for _ in range(10):
        if base_output_dir.name == "RESULTS" or (base_output_dir / "targets").exists():
            break
        if not base_output_dir.parent.exists():
            break
        base_output_dir = base_output_dir.parent
    
    target_clean = target_column.replace('/', '_').replace('\\', '_')
    
    # PATCH 4: Use OutputLayout for properly scoped paths
    try:
        from TRAINING.orchestration.utils.output_layout import OutputLayout
        from TRAINING.orchestration.utils.target_first_paths import ensure_target_structure
        
        ensure_target_structure(base_output_dir, target_clean)
        
        # Only pass symbol if view is SYMBOL_SPECIFIC
        symbol_for_layout = symbol if view == "SYMBOL_SPECIFIC" else None
        
        layout = OutputLayout(
            output_root=base_output_dir,
            target=target_clean,
            view=view,
            universe_sig=universe_sig,
            symbol=symbol_for_layout,
        )
        importances_dir = layout.feature_importance_dir()
        importances_dir.mkdir(parents=True, exist_ok=True)
        
        # Save per-model CSV files
        for model_name, importances in feature_importances.items():
            if not importances:
                continue
            
            # Create DataFrame sorted by importance
            df = pd.DataFrame([
                {'feature': feat, 'importance': imp}
                for feat, imp in importances.items()
            ])
            df = df.sort_values('importance', ascending=False)
            
            # Normalize to percentages
            total = df['importance'].sum()
            if total > 0:
                df['importance_pct'] = (df['importance'] / total * 100).round(2)
                df['cumulative_pct'] = df['importance_pct'].cumsum().round(2)
            else:
                df['importance_pct'] = 0.0
                df['cumulative_pct'] = 0.0
            
            # Reorder columns
            df = df[['feature', 'importance', 'importance_pct', 'cumulative_pct']]
            
            # Save to properly scoped location
            csv_file = importances_dir / f"{model_name}_importances.csv"
            df.to_csv(csv_file, index=False)
        
        logger.info(f"  💾 Saved feature importances to: {importances_dir}")
    except Exception as e:
        logger.warning(f"Failed to save feature importances to target-first structure: {e}")


def _log_suspicious_features(
    target_column: str,
    symbol: str,
    suspicious_features: Dict[str, List[Tuple[str, float]]]
) -> None:
    """
    Log suspicious features to a file for later analysis.
    
    Args:
        target_column: Name of the target being evaluated
        symbol: Symbol being evaluated
        suspicious_features: Dict of {model_name: [(feature, importance), ...]}
    """
    leak_report_file = _REPO_ROOT / "results" / "leak_detection_report.txt"
    leak_report_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(leak_report_file, 'a') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Target: {target_column} | Symbol: {symbol}\n")
        f.write(f"{'='*80}\n")
        
        for model_name, features in suspicious_features.items():
            if features:
                f.write(f"\n{model_name.upper()} - Suspicious Features:\n")
                f.write(f"{'-'*80}\n")
                for feat, imp in sorted(features, key=lambda x: x[1], reverse=True):
                    f.write(f"  {feat:50s} | Importance: {imp:.1%}\n")
                f.write("\n")
    
    logger.info(f"  Leak detection report saved to: {leak_report_file}")


def _triage_high_r2(
    X: np.ndarray,
    y: np.ndarray,
    time_vals: Optional[np.ndarray] = None,
    symbols: Optional[np.ndarray] = None,
    target: str = ""
) -> Dict[str, Any]:
    """
    Run triage checks for high R² scores to validate if they indicate real leakage.
    
    Args:
        X: Feature matrix
        y: Target vector
        time_vals: Optional timestamps for holdout block test
        symbols: Optional symbol array for panel data (shuffle within symbol)
        target: Target name (for logging)
    
    Returns:
        Dict with triage results:
        - permutation_r2: R² after shuffling y (should be ~0/negative if no leakage)
        - holdout_r2: R² on last 20% holdout block (should be lower if no leakage)
        - shift_correlation: Correlation between y and ret_oc (should be < 0.9 if target is properly shifted)
        - passed: True if all triage checks pass (no leakage), False otherwise
    """
    try:
        from sklearn.metrics import r2_score
        from sklearn.linear_model import Ridge
    except ImportError:
        logger.warning("sklearn not available for triage checks")
        return {"permutation_r2": None, "holdout_r2": None, "shift_correlation": None, "passed": False}
    
    triage_results = {
        "permutation_r2": None,
        "holdout_r2": None,
        "shift_correlation": None,
        "passed": False
    }
    
    try:
        # Check 1: Permutation test - shuffle y within symbol, R² should collapse
        if symbols is not None and len(np.unique(symbols)) > 1:
            # Panel data: shuffle within each symbol
            y_shuffled = y.copy()
            for sym in np.unique(symbols):
                sym_mask = symbols == sym
                y_shuffled[sym_mask] = np.random.permutation(y[sym_mask])
        else:
            # Single symbol: shuffle globally
            y_shuffled = np.random.permutation(y)
        
        # Fit simple model on shuffled y
        model_perm = Ridge(alpha=1.0)
        model_perm.fit(X, y_shuffled)
        y_pred_perm = model_perm.predict(X)
        triage_results["permutation_r2"] = float(r2_score(y_shuffled, y_pred_perm))
        
        # Check 2: Single holdout block - last 20% of time
        if time_vals is not None and len(time_vals) > 10:
            time_quantile = np.quantile(time_vals, 0.8)
            train_idx = time_vals < time_quantile
            test_idx = ~train_idx
            
            if np.sum(train_idx) > 5 and np.sum(test_idx) > 5:
                model_holdout = Ridge(alpha=1.0)
                model_holdout.fit(X[train_idx], y[train_idx])
                y_pred_holdout = model_holdout.predict(X[test_idx])
                triage_results["holdout_r2"] = float(r2_score(y[test_idx], y_pred_holdout))
        
        # Check 3: Shift sanity - correlate y with ret_oc (if available in feature names)
        # Look for ret_oc or similar "current bar return" features
        if hasattr(X, 'columns') or (isinstance(X, pd.DataFrame)):
            # X is DataFrame - check for ret_oc column
            if 'ret_oc' in X.columns:
                ret_oc = X['ret_oc'].values
                triage_results["shift_correlation"] = float(np.corrcoef(y, ret_oc)[0, 1])
        else:
            # X is numpy array - try to find ret_oc in feature_names if available
            # (This is a limitation - we'd need feature_names passed in)
            pass
        
        # Determine if triage passed (all checks suggest no leakage)
        passed = True
        if triage_results["permutation_r2"] is not None and triage_results["permutation_r2"] > 0.1:
            passed = False  # Permutation R² should be ~0/negative
        if triage_results["holdout_r2"] is not None and triage_results["holdout_r2"] > 0.4:
            passed = False  # Holdout R² should be lower
        if triage_results["shift_correlation"] is not None and abs(triage_results["shift_correlation"]) > 0.9:
            passed = False  # Shift correlation should be < 0.9
        
        triage_results["passed"] = passed
        
    except Exception as e:
        logger.debug(f"Triage checks failed: {e}")
        # If triage fails, assume leakage (fail-safe)
        triage_results["passed"] = False
    
    return triage_results


def detect_leakage(
    auc: float,
    composite_score: float,
    mean_importance: float,
    target: str = "",
    model_scores: Dict[str, float] = None,
    task_type: TaskType = TaskType.REGRESSION,
    X: Optional[np.ndarray] = None,  # NEW: Optional data for triage checks
    y: Optional[np.ndarray] = None,  # NEW: Optional target for triage checks
    time_vals: Optional[np.ndarray] = None,  # NEW: Optional timestamps for triage checks
    symbols: Optional[np.ndarray] = None  # NEW: Optional symbols for triage checks
) -> str:
    """
    Detect potential data leakage based on suspicious patterns.
    
    Returns:
        "OK" - No signs of leakage
        "HIGH_R2" - R² > threshold (suspiciously high)
        "INCONSISTENT" - Composite score too high for R² (possible leakage)
        "SUSPICIOUS" - Multiple warning signs
    """
    flags = []
    
    # Load thresholds from config
    small_panel_cfg = {}
    if _CONFIG_AVAILABLE:
        try:
            safety_cfg = get_safety_config()
            # safety_config.yaml has a top-level 'safety' key
            safety_section = safety_cfg.get('safety', {})
            leakage_cfg = safety_section.get('leakage_detection', {})
            warning_cfg = leakage_cfg.get('warning_thresholds', {})
            small_panel_cfg = leakage_cfg.get('small_panel', {})
        except Exception:
            warning_cfg = {}
            small_panel_cfg = {}
    else:
        warning_cfg = {}
        small_panel_cfg = {}
    
    # Compute n_symbols from symbols array if available
    n_symbols = None
    if symbols is not None and len(symbols) > 0:
        n_symbols = len(np.unique(symbols))
    
    # Determine threshold based on task type and target name
    if task_type == TaskType.REGRESSION:
        is_forward_return = target.startswith('fwd_ret_')
        if is_forward_return:
            # For forward returns: R² > 0.50 is suspicious
            reg_cfg = warning_cfg.get('regression', {}).get('forward_return', {})
            high_threshold = float(reg_cfg.get('high', 0.50))
            very_high_threshold = float(reg_cfg.get('very_high', 0.60))
            metric_name = "R²"
        else:
            # For barrier targets: R² > 0.70 is suspicious
            reg_cfg = warning_cfg.get('regression', {}).get('barrier', {})
            high_threshold = float(reg_cfg.get('high', 0.70))
            very_high_threshold = float(reg_cfg.get('very_high', 0.80))
            metric_name = "R²"
    elif task_type == TaskType.BINARY_CLASSIFICATION:
        # ROC-AUC > 0.95 is suspicious (near-perfect classification)
        class_cfg = warning_cfg.get('classification', {})
        high_threshold = float(class_cfg.get('high', 0.90))
        very_high_threshold = float(class_cfg.get('very_high', 0.95))
        metric_name = "ROC-AUC"
    else:  # MULTICLASS_CLASSIFICATION
        # Accuracy > 0.95 is suspicious
        class_cfg = warning_cfg.get('classification', {})
        high_threshold = float(class_cfg.get('high', 0.90))
        very_high_threshold = float(class_cfg.get('very_high', 0.95))
        metric_name = "Accuracy"
    
    # Check 1: Suspiciously high mean score (with triage for R²)
    if auc > very_high_threshold:
        flags.append("HIGH_SCORE")
        # Run triage checks if data is available and this is R²
        triage_results = None
        if metric_name == "R²" and X is not None and y is not None:
            triage_results = _triage_high_r2(X, y, time_vals=time_vals, symbols=symbols, target=target)
            if triage_results["passed"]:
                logger.info(
                    f"R²={auc:.3f} > {very_high_threshold:.2f}, but triage passed: "
                    f"permutation={triage_results.get('permutation_r2', 'N/A'):.3f}, "
                    f"holdout={triage_results.get('holdout_r2', 'N/A'):.3f}, "
                    f"shift_corr={triage_results.get('shift_correlation', 'N/A'):.3f}"
                )
            else:
                logger.warning(
                    f"LEAKAGE WARNING: {metric_name}={auc:.3f} > {very_high_threshold:.2f} "
                    f"(extremely high - likely leakage). Triage: "
                    f"permutation={triage_results.get('permutation_r2', 'N/A'):.3f}, "
                    f"holdout={triage_results.get('holdout_r2', 'N/A'):.3f}, "
                    f"shift_corr={triage_results.get('shift_correlation', 'N/A'):.3f}"
                )
        else:
            logger.warning(
                f"LEAKAGE WARNING: {metric_name}={auc:.3f} > {very_high_threshold:.2f} "
                f"(extremely high - likely leakage)"
            )
    elif auc > high_threshold:
        flags.append("HIGH_SCORE")
        # Run triage checks if data is available and this is R²
        triage_results = None
        if metric_name == "R²" and X is not None and y is not None:
            triage_results = _triage_high_r2(X, y, time_vals=time_vals, symbols=symbols, target=target)
            if triage_results["passed"]:
                logger.info(
                    f"R²={auc:.3f} > {high_threshold:.2f}, but triage passed: "
                    f"permutation={triage_results.get('permutation_r2', 'N/A'):.3f}, "
                    f"holdout={triage_results.get('holdout_r2', 'N/A'):.3f}, "
                    f"shift_corr={triage_results.get('shift_correlation', 'N/A'):.3f}"
                )
            else:
                logger.warning(
                    f"LEAKAGE WARNING: {metric_name}={auc:.3f} > {high_threshold:.2f} "
                    f"(suspiciously high - investigate). Triage: "
                    f"permutation={triage_results.get('permutation_r2', 'N/A'):.3f}, "
                    f"holdout={triage_results.get('holdout_r2', 'N/A'):.3f}, "
                    f"shift_corr={triage_results.get('shift_correlation', 'N/A'):.3f}"
                )
        else:
            logger.warning(
                f"LEAKAGE WARNING: {metric_name}={auc:.3f} > {high_threshold:.2f} "
                f"(suspiciously high - investigate)"
            )
    
    # Check 2: Individual model scores too high (even if mean is lower)
    if model_scores:
        high_model_count = sum(1 for score in model_scores.values() 
                              if not np.isnan(score) and score > high_threshold)
        if high_model_count >= 3:  # 3+ models with high scores
            flags.append("HIGH_SCORE")
            logger.warning(
                f"LEAKAGE WARNING: {high_model_count} models have {metric_name} > {high_threshold:.2f} "
                f"(models: {[k for k, v in model_scores.items() if not np.isnan(v) and v > high_threshold]})"
            )
    
    # Check 3: Composite score inconsistent with mean score
    # If composite is very high (> 0.5) but score is low (< 0.2 for regression, < 0.6 for classification), something's wrong
    score_low_threshold = 0.2 if task_type == TaskType.REGRESSION else 0.6
    if composite_score > 0.5 and auc < score_low_threshold:
        flags.append("INCONSISTENT")
        logger.warning(
            f"LEAKAGE WARNING: Composite={composite_score:.3f} but {metric_name}={auc:.3f} "
            f"(inconsistent - possible leakage)"
        )
    
    # Check 4: Very high importance with low score (might indicate leaked features)
    score_very_low_threshold = 0.1 if task_type == TaskType.REGRESSION else 0.5
    if mean_importance > 0.7 and auc < score_very_low_threshold:
        flags.append("INCONSISTENT")
        logger.warning(
            f"LEAKAGE WARNING: Importance={mean_importance:.2f} but {metric_name}={auc:.3f} "
            f"(high importance with low {metric_name} - check for leaked features)"
        )
    
    # Determine final leakage status
    if len(flags) > 1:
        leakage_status = "SUSPICIOUS"
    elif len(flags) == 1:
        leakage_status = flags[0]
    else:
        leakage_status = "OK"
    
    # Small-panel leniency: downgrade BLOCKED to SUSPECT for small panels
    if (small_panel_cfg.get('enabled', False) and 
        leakage_status in ["HIGH_SCORE", "SUSPICIOUS"] and 
        n_symbols is not None):
        min_symbols_threshold = small_panel_cfg.get('min_symbols_threshold', 10)
        downgrade_enabled = small_panel_cfg.get('downgrade_block_to_suspect', True)
        log_warning = small_panel_cfg.get('log_warning', True)
        
        if n_symbols < min_symbols_threshold and downgrade_enabled:
            if log_warning:
                logger.warning(
                    f"🔒 Small panel detected (n_symbols={n_symbols} < {min_symbols_threshold}), "
                    f"downgrading leakage severity from {leakage_status} to SUSPECT. "
                    f"This allows dominance quarantine to attempt recovery before blocking."
                )
            # Downgrade to SUSPECT (allows training to proceed, but with warning)
            leakage_status = "SUSPECT"
    
    return leakage_status


