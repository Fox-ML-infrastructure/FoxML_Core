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
import json
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
from TRAINING.utils.checkpoint import CheckpointManager

# Import unified task type system
from TRAINING.utils.task_types import (
    TaskType, TargetConfig, ModelConfig, 
    is_compatible, create_model_configs_from_yaml
)
from TRAINING.utils.task_metrics import evaluate_by_task, compute_composite_score
from TRAINING.utils.target_validation import validate_target, check_cv_compatibility

# Suppress expected warnings (harmless)
warnings.filterwarnings('ignore', message='X does not have valid feature names')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn')
warnings.filterwarnings('ignore', message='invalid value encountered in divide')
warnings.filterwarnings('ignore', message='invalid value encountered in true_divide')

# Setup logging with journald support
from TRAINING.utils.logging_setup import setup_logging
logger = setup_logging(
    script_name="rank_target_predictability",
    level=logging.INFO,
    use_journald=True
)



# Import dependencies
from TRAINING.ranking.predictability.scoring import TargetPredictabilityScore
from TRAINING.ranking.predictability.composite_score import calculate_composite_score
from TRAINING.ranking.predictability.data_loading import load_sample_data, prepare_features_and_target, get_model_config
from TRAINING.ranking.predictability.leakage_detection import detect_leakage, _save_feature_importances, _log_suspicious_features, find_near_copy_features, _detect_leaking_features


def _compute_suspicion_score(
    train_score: float,
    cv_score: Optional[float],
    feature_importances: Dict[str, float],
    task_type: str = 'classification'
) -> float:
    """
    Compute suspicion score for perfect train accuracy.
    
    Higher score = more suspicious (likely real leakage, not just overfitting).
    
    Signals that increase suspicion:
    - CV too good to be true (cv_mean >= 0.85)
    - Generalization gap too small with perfect train (gap < 0.05)
    - Single feature domination (top1_importance / sum >= 0.40)
    
    Signals that decrease suspicion:
    - CV is normal-ish (0.55-0.75)
    - Large gap (classic overfit)
    - Feature dominance not extreme
    """
    suspicion = 0.0
    
    # Signal 1: CV too good to be true
    if cv_score is not None:
        if cv_score >= 0.85:
            suspicion += 0.4  # High suspicion
        elif cv_score >= 0.75:
            suspicion += 0.2  # Medium suspicion
        elif cv_score < 0.55:
            suspicion -= 0.2  # Low suspicion (normal performance)
    
    # Signal 2: Generalization gap (small gap with perfect train = suspicious)
    if cv_score is not None:
        gap = train_score - cv_score
        if gap < 0.05 and train_score >= 0.99:
            suspicion += 0.3  # Very suspicious: perfect train but CV also high
        elif gap > 0.20:
            suspicion -= 0.2  # Large gap = classic overfit (less suspicious)
    
    # Signal 3: Feature dominance
    if feature_importances:
        importances = list(feature_importances.values())
        if importances:
            total_importance = sum(importances)
            if total_importance > 0:
                top1_importance = max(importances)
                dominance_ratio = top1_importance / total_importance
                if dominance_ratio >= 0.50:
                    suspicion += 0.3  # Single feature dominates
                elif dominance_ratio >= 0.40:
                    suspicion += 0.2  # High dominance
                elif dominance_ratio < 0.20:
                    suspicion -= 0.1  # Low dominance (less suspicious)
    
    # Clamp to [0, 1]
    return max(0.0, min(1.0, suspicion))


def _log_canonical_summary(
    target_name: str,
    target_column: str,
    symbols: List[str],
    time_vals: Optional[np.ndarray],
    interval: Optional[Union[int, str]],
    horizon: Optional[int],
    rows: int,
    features_safe: int,
    features_pruned: int,
    leak_scan_verdict: str,
    auto_fix_verdict: str,
    auto_fix_reason: Optional[str],
    cv_metric: str,
    composite: float,
    leakage_flag: str,
    cohort_path: Optional[str]
):
    """
    Log canonical run summary block (one block that can be screenshot for PR comments).
    
    This provides a stable anchor for reviewers to quickly understand:
    - What was evaluated
    - Data characteristics
    - Feature pipeline
    - Leakage status
    - Performance metrics
    - Reproducibility path
    """
    # Extract date range from time_vals if available
    date_range = "N/A"
    if time_vals is not None and len(time_vals) > 0:
        try:
            import pandas as pd
            if isinstance(time_vals[0], (int, float)):
                time_series = pd.to_datetime(time_vals, unit='ns')
            else:
                time_series = pd.Series(time_vals)
            if len(time_series) > 0:
                date_range = f"{time_series.min().strftime('%Y-%m-%d')} → {time_series.max().strftime('%Y-%m-%d')}"
        except Exception:
            pass
    
    # Format symbols (show first 5, then count)
    if len(symbols) <= 5:
        symbols_str = ', '.join(symbols)
    else:
        symbols_str = f"{', '.join(symbols[:5])}, ... ({len(symbols)} total)"
    
    # Format interval/horizon
    interval_str = f"{interval}" if interval else "auto"
    horizon_str = f"{horizon}m" if horizon else "N/A"
    
    # Format auto-fix info
    auto_fix_str = auto_fix_verdict
    if auto_fix_reason:
        auto_fix_str += f" (reason={auto_fix_reason})"
    
    logger.info("=" * 60)
    logger.info("TARGET_RANKING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"target: {target_column:<40} horizon: {horizon_str:<8} interval: {interval_str}")
    logger.info(f"symbols: {len(symbols)} ({symbols_str})")
    logger.info(f"date: {date_range}")
    logger.info(f"rows: {rows:<10} features: safe={features_safe} → pruned={features_pruned}")
    logger.info(f"leak_scan: {leak_scan_verdict:<6} auto_fix: {auto_fix_str}")
    logger.info(f"cv: {cv_metric:<25} composite: {composite:.3f}")
    if cohort_path:
        logger.info(f"repro: {cohort_path}")
    logger.info("=" * 60)

def _enforce_final_safety_gate(
    X: np.ndarray,
    feature_names: List[str],
    resolved_config: Any,
    interval_minutes: float,
    logger: logging.Logger
) -> Tuple[np.ndarray, List[str]]:
    """
    Final Gatekeeper: Enforce safety at the last possible moment.
    
    This runs AFTER all loading/merging/sanitization is done.
    It physically drops features that violate the purge limit from the dataframe.
    This is the "worry-free" auto-corrector that handles race conditions.
    
    Why this is needed:
    - Schema loader might add features after sanitization
    - Registry might allow features that violate purge
    - Ghost features might slip through multiple layers
    - This is the absolute last check before data touches the model
    
    Args:
        X: Feature matrix (numpy array)
        feature_names: List of feature names
        resolved_config: ResolvedConfig object with purge_minutes
        interval_minutes: Data interval in minutes
        logger: Logger instance
    
    Returns:
        (filtered_X, filtered_feature_names) tuple
    """
    if X is None or len(feature_names) == 0:
        return X, feature_names
    
    purge_limit = resolved_config.purge_minutes if resolved_config else None
    if purge_limit is None or purge_limit == 0:
        # No purge, no rules - allow all features
        return X, feature_names
    
    # Define maximum allowed lookback (with 1% safety buffer)
    # If purge is 100m, max lookback is ~99m
    safe_lookback_max = purge_limit * 0.99
    
    dropped_features = []
    dropped_indices = []
    
    # Get feature registry for lookback calculation
    registry = None
    try:
        from TRAINING.common.feature_registry import get_registry
        registry = get_registry()
    except Exception:
        pass
    
    # CRITICAL: Use the SAME lookback calculation as the audit system
    # Compute lookback for ALL features using the same logic as resolved_config
    from TRAINING.utils.resolved_config import compute_feature_lookback_max
    
    # Compute lookback for all features at once (same as audit)
    max_lookback_all, top_offenders_all = compute_feature_lookback_max(
        feature_names,
        interval_minutes=interval_minutes,
        max_lookback_cap_minutes=None  # Don't cap - we want the real value
    )
    
    # Build lookup dict from top_offenders (contains all features with lookback > 0)
    feature_lookback_dict = {}
    for feat_name, lookback_minutes in top_offenders_all:
        feature_lookback_dict[feat_name] = lookback_minutes
    
    # For features not in top_offenders, compute individually to catch any that exceed threshold
    # (top_offenders only contains top 10, but we need to check ALL features)
    for feat_name in feature_names:
        if feat_name not in feature_lookback_dict:
            # Compute lookback for this feature individually
            max_lookback, _ = compute_feature_lookback_max(
                [feat_name],
                interval_minutes=interval_minutes,
                max_lookback_cap_minutes=None
            )
            if max_lookback is not None:
                feature_lookback_dict[feat_name] = max_lookback
            else:
                feature_lookback_dict[feat_name] = 0.0  # Unknown feature - assume safe
    
    # Iterate through features in the FINAL dataframe
    for idx, feature_name in enumerate(feature_names):
        should_drop = False
        reason = None
        
        # Get lookback from our computed dict (same calculation as audit)
        lookback_minutes = feature_lookback_dict.get(feature_name, 0.0)
        
        # Check explicit 24h/daily naming (very aggressive - catch patterns that might not be in registry)
        is_daily_name = any(x in feature_name.lower() for x in ['_1d', '_24h', 'daily', 'day'])
        
        # The Logic: If it violates the purge, KILL IT
        # Use the SAME calculation as audit - if audit sees 1440m, we should too
        if is_daily_name:
            should_drop = True
            reason = "daily/24h naming pattern"
        elif lookback_minutes > safe_lookback_max:
            should_drop = True
            reason = f"lookback ({lookback_minutes:.1f}m) > safe_limit ({safe_lookback_max:.1f}m)"
        
        if should_drop:
            dropped_features.append((feature_name, reason))
            dropped_indices.append(idx)
    
    # Mutate the Dataframe (drop columns)
    if dropped_features:
        logger.warning(
            f"🛡️ FINAL GATEKEEPER: Dropping {len(dropped_features)} features that violate purge limit "
            f"({purge_limit:.1f}m, safe_lookback_max={safe_lookback_max:.1f}m)"
        )
        for feat_name, feat_reason in dropped_features[:10]:  # Show first 10
            logger.warning(f"   🗑️ {feat_name}: {feat_reason}")
        if len(dropped_features) > 10:
            logger.warning(f"   ... and {len(dropped_features) - 10} more")
        
        # Drop columns from X (numpy array)
        keep_indices = [i for i in range(X.shape[1]) if i not in dropped_indices]
        X = X[:, keep_indices]
        feature_names = [name for idx, name in enumerate(feature_names) if idx not in dropped_indices]
        
        logger.info(f"   ✅ After final gatekeeper: {X.shape[1]} features remaining")
    
    return X, feature_names


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
    experiment_config: Optional[Any] = None,  # Optional ExperimentConfig (for data.bar_interval)
    output_dir: Optional[Path] = None,  # Optional output directory for stability snapshots
    resolved_config: Optional[Any] = None  # NEW: ResolvedConfig with correct purge/embargo (post-pruning)
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
        from TRAINING.utils.purged_time_series_split import PurgedTimeSeriesSplit
        from TRAINING.utils.leakage_filtering import _extract_horizon, _load_leakage_config
        from TRAINING.utils.feature_pruning import quick_importance_prune
    except Exception as e:
        logger.warning(f"Failed to import required libraries: {e}")
        return {}, {}, 0.0, {}, {}, []
    
    # Helper function for CV with early stopping (for gradient boosting models)
    def cross_val_score_with_early_stopping(model, X, y, cv, scoring, early_stopping_rounds=None, n_jobs=1):
        # Load default early stopping rounds from config
        if early_stopping_rounds is None:
            if _CONFIG_AVAILABLE:
                try:
                    early_stopping_rounds = int(get_cfg("preprocessing.validation.early_stopping_rounds", default=50, config_name="preprocessing_config"))
                except Exception:
                    early_stopping_rounds = 50
            else:
                early_stopping_rounds = 50
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
    # Load feature count threshold from config
    try:
        from CONFIG.config_loader import get_cfg
        feature_count_threshold = int(get_cfg("safety.leakage_detection.model_evaluation.feature_count_pruning_threshold", default=100, config_name="safety_config"))
    except Exception:
        feature_count_threshold = 100
    if original_feature_count > feature_count_threshold:  # Only prune if we have many features
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
            target_name_for_seed = target_column if target_column else 'pruning'
            prune_seed = stable_seed_from([target_name_for_seed, 'feature_pruning'])
            
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
                random_state=prune_seed
            )
            
            if pruning_stats.get('dropped_count', 0) > 0:
                logger.info(f"  ✅ Pruned: {original_feature_count} → {len(feature_names_pruned)} features "
                          f"(dropped {pruning_stats['dropped_count']} low-importance features)")
                
                # Check for duplicates before assignment
                if len(feature_names_pruned) != len(set(feature_names_pruned)):
                    duplicates = [name for name in set(feature_names_pruned) if feature_names_pruned.count(name) > 1]
                    logger.error(f"  🚨 DUPLICATE COLUMN NAMES in pruned features: {duplicates}")
                    raise ValueError(f"Duplicate feature names after pruning: {duplicates}")
                
                feature_names_before_prune = feature_names.copy()
                X = X_pruned
                feature_names = feature_names_pruned
                
                # Log feature set transition
                from TRAINING.utils.cross_sectional_data import _log_feature_set
                _log_feature_set("PRUNER_SELECTED", feature_names, previous_names=feature_names_before_prune, logger_instance=logger)
            else:
                logger.info(f"  No features pruned (all above threshold)")
                from TRAINING.utils.cross_sectional_data import _log_feature_set
                _log_feature_set("PRUNER_SELECTED", feature_names, previous_names=None, logger_instance=logger)
            
            # CRITICAL: Recompute resolved_config with feature_lookback_max from PRUNED features
            # This prevents paying 1440m purge for features we don't even use
            from TRAINING.utils.resolved_config import compute_feature_lookback_max, create_resolved_config
            
            # Get n_symbols_available from mtf_data
            n_symbols_available = len(mtf_data) if 'mtf_data' in locals() else 1
            
            # Load ranking mode cap from config
            max_lookback_cap = None
            try:
                from CONFIG.config_loader import get_cfg
                max_lookback_cap = get_cfg("safety.leakage_detection.ranking_mode_max_lookback_minutes", default=None, config_name="safety_config")
                if max_lookback_cap is not None:
                    max_lookback_cap = float(max_lookback_cap)
            except Exception:
                pass
            
            # Compute feature lookback from PRUNED features
            computed_lookback, top_offenders = compute_feature_lookback_max(
                feature_names, data_interval_minutes, max_lookback_cap_minutes=max_lookback_cap
            )
            
            if computed_lookback is not None:
                feature_lookback_max_minutes = computed_lookback
                if top_offenders and top_offenders[0][1] > 240:  # Only log if > 4 hours
                    logger.info(f"  📊 Feature lookback (post-prune): max={computed_lookback:.1f}m")
                    logger.info(f"    Top lookback features: {', '.join([f'{f}({m:.0f}m)' for f, m in top_offenders[:5]])}")
            else:
                feature_lookback_max_minutes = None
            
            # Recompute resolved_config with actual pruned feature lookback
            # This overrides the baseline config created earlier
            if resolved_config is not None:
                # Override with post-prune config
                resolved_config = create_resolved_config(
                    requested_min_cs=resolved_config.requested_min_cs,
                    n_symbols_available=n_symbols_available,
                    max_cs_samples=resolved_config.max_cs_samples,
                    interval_minutes=resolved_config.interval_minutes,
                    horizon_minutes=resolved_config.horizon_minutes,
                    feature_lookback_max_minutes=feature_lookback_max_minutes,  # Now with actual pruned lookback
                    purge_buffer_bars=resolved_config.purge_buffer_bars,
                    default_purge_minutes=None,  # Loads from safety_config.yaml (SST)
                    features_safe=resolved_config.features_safe,
                    features_dropped_nan=resolved_config.features_dropped_nan,
                    features_final=len(feature_names),  # Updated count
                    view=resolved_config.view,
                    symbol=resolved_config.symbol,
                    feature_names=feature_names,  # Pruned features
                    recompute_lookback=False  # Already computed above
                )
                if log_cfg.cv_detail:
                    logger.info(f"  ✅ Resolved config (post-prune): purge={resolved_config.purge_minutes:.1f}m, embargo={resolved_config.embargo_minutes:.1f}m")
            
            # Save stability snapshot for quick pruning (non-invasive hook)
            # Only save if output_dir is available (optional feature)
            if 'full_importance_dict' in pruning_stats and output_dir is not None:
                try:
                    from TRAINING.stability.feature_importance import save_snapshot_hook
                    save_snapshot_hook(
                        target_name=target_column if target_column else 'unknown',
                        method="quick_pruner",
                        importance_dict=pruning_stats['full_importance_dict'],
                        universe_id="CROSS_SECTIONAL",
                        output_dir=output_dir,
                        auto_analyze=None,  # Load from config
                    )
                except Exception as e:
                    logger.debug(f"Stability snapshot save failed for quick_pruner (non-critical): {e}")
        except Exception as e:
            logger.warning(f"  Feature pruning failed: {e}, using all features")
            logger.exception("  Pruning exception details (non-critical):")  # Better error logging
            # Continue with original features (baseline resolved_config already assigned)
    
    # CRITICAL: Create resolved_config AFTER pruning (or if pruning skipped)
    # This ensures feature_lookback_max is computed from actual features used in training
    if resolved_config is None:
        from TRAINING.utils.resolved_config import compute_feature_lookback_max, create_resolved_config
        
        # Get n_symbols_available from cohort_context
        n_symbols_available = len(mtf_data) if 'mtf_data' in locals() else 1
        
        # Load ranking mode cap from config
        max_lookback_cap = None
        try:
            from CONFIG.config_loader import get_cfg
            max_lookback_cap = get_cfg("safety.leakage_detection.ranking_mode_max_lookback_minutes", default=None, config_name="safety_config")
            if max_lookback_cap is not None:
                max_lookback_cap = float(max_lookback_cap)
        except Exception:
            pass
        
        # Compute feature lookback from actual features (pruned or unpruned)
        computed_lookback, top_offenders = compute_feature_lookback_max(
            feature_names, data_interval_minutes, max_lookback_cap_minutes=max_lookback_cap
        )
        
        if computed_lookback is not None:
            feature_lookback_max_minutes = computed_lookback
            if top_offenders and top_offenders[0][1] > 240:  # Only log if > 4 hours
                logger.info(f"  📊 Feature lookback analysis: max={computed_lookback:.1f}m")
                logger.info(f"    Top lookback features: {', '.join([f'{f}({m:.0f}m)' for f, m in top_offenders[:5]])}")
        else:
            # Fallback: use conservative estimate if cannot compute
            if data_interval_minutes is not None and data_interval_minutes > 0:
                max_lookback_bars = 288  # 1 day of 5m bars
                feature_lookback_max_minutes = max_lookback_bars * data_interval_minutes
            else:
                feature_lookback_max_minutes = None
        
        # Extract horizon from target_column if available
        target_horizon_minutes = None
        if target_column:
            try:
                from TRAINING.utils.leakage_filtering import _extract_horizon, _load_leakage_config
                leakage_config = _load_leakage_config()
                target_horizon_minutes = _extract_horizon(target_column, leakage_config)
            except Exception:
                pass
        
        # Create resolved config with actual feature lookback
        resolved_config = create_resolved_config(
            requested_min_cs=1,  # Not used in train_and_evaluate_models context
            n_symbols_available=n_symbols_available,
            max_cs_samples=None,
            interval_minutes=data_interval_minutes,
            horizon_minutes=target_horizon_minutes,
            feature_lookback_max_minutes=feature_lookback_max_minutes,
            purge_buffer_bars=5,
            default_purge_minutes=None,  # Loads from safety_config.yaml (SST)
            features_safe=original_feature_count,
            features_dropped_nan=0,
            features_final=len(feature_names),
            view="CROSS_SECTIONAL",  # Default for train_and_evaluate_models
            symbol=None,
            feature_names=feature_names,
            recompute_lookback=False  # Already computed above
        )
        
        if log_cfg.cv_detail:
            logger.info(f"  ✅ Resolved config created: purge={resolved_config.purge_minutes:.1f}m, embargo={resolved_config.embargo_minutes:.1f}m")
    
    # Get CV config (with fallback if multi_model_config is None or cross_validation is None)
    if multi_model_config is None:
        cv_config = {}
        # Try to load from config if multi_model_config not provided
        try:
            from CONFIG.config_loader import get_cfg
            cv_folds = int(get_cfg("training.cv_folds", default=3, config_name="intelligent_training_config"))
            cv_n_jobs = int(get_cfg("training.cv_n_jobs", default=1, config_name="intelligent_training_config"))
        except Exception:
            cv_folds = 3
            cv_n_jobs = 1
    else:
        cv_config = multi_model_config.get('cross_validation', {})
        # Ensure cv_config is never None (handle case where key exists but value is None)
        if cv_config is None:
            cv_config = {}
        cv_folds = cv_config.get('cv_folds', 3)
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
            
            # Only use auto-detection if it's close to a common interval (load tolerance from config)
            try:
                from CONFIG.config_loader import get_cfg
                tolerance = float(get_cfg("safety.leakage_detection.model_evaluation.interval_detection_tolerance", default=0.2, config_name="safety_config"))
            except Exception:
                tolerance = 0.2
            if abs(median_diff_minutes - detected_interval) / detected_interval < tolerance:
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
    
    # ARCHITECTURAL FIX: Use resolved_config if provided (has correct purge/embargo post-pruning)
    # Otherwise compute here (fallback for legacy calls)
    if resolved_config is not None:
        purge_minutes_val = resolved_config.purge_minutes
        embargo_minutes_val = resolved_config.embargo_minutes
        feature_lookback_max_minutes = None  # Not needed here, already in resolved_config
        if log_cfg.cv_detail:
            logger.info(f"  Using purge/embargo from resolved_config: purge={purge_minutes_val:.1f}m, embargo={embargo_minutes_val:.1f}m")
    else:
        # Fallback: compute here (legacy path)
        from TRAINING.utils.resolved_config import derive_purge_embargo
        
        # Load purge settings from config
        if _CONFIG_AVAILABLE:
            try:
                purge_buffer_bars = int(get_cfg("pipeline.leakage.purge_buffer_bars", default=5, config_name="pipeline_config"))
            except Exception:
                purge_buffer_bars = 5
        else:
            purge_buffer_bars = 5  # Safety buffer (5 bars = 25 minutes)
        
        # Estimate feature lookback (conservative: 1 day = 288 bars for 5m data)
        feature_lookback_max_minutes = None
        if data_interval_minutes is not None and data_interval_minutes > 0:
            max_lookback_bars = 288  # 1 day of 5m bars
            feature_lookback_max_minutes = max_lookback_bars * data_interval_minutes
        
        # Use centralized derivation function
        purge_minutes_val, embargo_minutes_val = derive_purge_embargo(
            horizon_minutes=target_horizon_minutes,
            interval_minutes=data_interval_minutes,
            feature_lookback_max_minutes=feature_lookback_max_minutes,
            purge_buffer_bars=purge_buffer_bars,
            default_purge_minutes=85.0
        )
    
    purge_time = pd.Timedelta(minutes=purge_minutes_val)
    
    # Check for duplicate column names before training
    if len(feature_names) != len(set(feature_names)):
        duplicates = [name for name in set(feature_names) if feature_names.count(name) > 1]
        logger.error(f"  🚨 DUPLICATE COLUMN NAMES before training: {duplicates}")
        raise ValueError(f"Duplicate feature names before training: {duplicates}")
    
    # Log feature set before training
    from TRAINING.utils.cross_sectional_data import _log_feature_set
    _log_feature_set("MODEL_TRAIN_INPUT", feature_names, previous_names=None, logger_instance=logger)
    
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
            n_splits=cv_folds, 
            purge_overlap_time=purge_time,
            time_column_values=time_vals
        )
        if log_cfg.cv_detail:
            logger.info(f"  Using PurgedTimeSeriesSplit (TIME-BASED): {cv_folds} folds, purge_time={purge_time}")
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
                # Sort for deterministic order (ensures reproducible aggregations)
                model_families = sorted(model_families)
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
                        try:
                            from CONFIG.config_loader import get_cfg
                            binary_threshold = float(get_cfg("safety.leakage_detection.model_evaluation.binary_classification_threshold", default=0.5, config_name="safety_config"))
                        except Exception:
                            binary_threshold = 0.5
                        y_pred_train = (y_proba >= binary_threshold).astype(int)
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
                    # Load binary classification threshold from config
                    try:
                        from CONFIG.config_loader import get_cfg
                        binary_threshold = float(get_cfg("safety.leakage_detection.model_evaluation.binary_classification_threshold", default=0.5, config_name="safety_config"))
                    except Exception:
                        binary_threshold = 0.5
                    y_pred = (y_proba >= binary_threshold).astype(int)
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
            
            # Store full metrics (training metrics from evaluate_by_task)
            model_metrics[model_name] = full_metrics
            
            # CRITICAL: Overwrite training metrics with CV scores (primary_score is from CV)
            # This ensures model_metrics contains CV scores, not training scores
            if task_type == TaskType.REGRESSION:
                model_metrics[model_name]['r2'] = primary_score  # CV R²
            elif task_type == TaskType.BINARY_CLASSIFICATION:
                model_metrics[model_name]['roc_auc'] = primary_score  # CV AUC
            else:  # MULTICLASS_CLASSIFICATION
                model_metrics[model_name]['accuracy'] = primary_score  # CV accuracy
            
            # Also store training accuracy/correlation for auto-fixer detection
            # This is the in-sample training score (not CV), which is what triggers leakage warnings
            if training_accuracy is not None:
                if task_type == TaskType.REGRESSION:
                    model_metrics[model_name]['training_r2'] = training_accuracy
                else:
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
            # Remove objective, device, and verbose from config (we set these explicitly)
            # CRITICAL: Remove verbose to prevent double argument error
            lgb_config_clean = {k: v for k, v in lgb_config.items() if k not in ['device', 'objective', 'metric', 'verbose']}
            
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
            # CRITICAL: Use time-aware split (load ratio from config) - don't shuffle time series data
            # Guard against empty arrays
            try:
                from CONFIG.config_loader import get_cfg
                time_split_ratio = float(get_cfg("preprocessing.validation.time_aware_split_ratio", default=0.8, config_name="preprocessing_config"))
                min_samples_for_split = int(get_cfg("preprocessing.validation.min_samples_for_split", default=10, config_name="preprocessing_config"))
            except Exception:
                time_split_ratio = 0.8
                min_samples_for_split = 10
            
            if len(X) < min_samples_for_split:
                logger.warning(f"  ⚠️  Too few samples ({len(X)}) for train/val split, fitting on all data")
                split_idx = len(X)
            else:
                split_idx = int(len(X) * time_split_ratio)
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
            importance_dict = importance_series.to_dict()
            all_feature_importances['lightgbm'] = importance_dict
            
            # Log importance keys vs train input (now guaranteed to match order)
            importance_keys = list(importance_dict.keys())  # Use list to preserve order
            train_input_keys = feature_names  # Already a list
            if len(importance_keys) != len(train_input_keys):
                missing = set(train_input_keys) - set(importance_keys)
                logger.warning(f"  ⚠️  IMPORTANCE_KEYS mismatch: {len(importance_keys)} keys vs {len(train_input_keys)} train features")
                logger.warning(f"    Missing from importance: {sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}")
            elif importance_keys == train_input_keys:
                # Keys match AND order matches - safe to log fingerprint
                from TRAINING.utils.cross_sectional_data import _log_feature_set
                _log_feature_set("IMPORTANCE_KEYS", importance_keys, previous_names=feature_names, logger_instance=logger)
            
            # Compute and store full task-aware metrics
            _compute_and_store_metrics('lightgbm', model, X, y, primary_score, task_type)
            
            # Use percentage of total importance in top fraction features (0-1 scale, interpretable)
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
            importance_dict = importance_series.to_dict()
            all_feature_importances['random_forest'] = importance_dict
            
            # Log importance keys vs train input (only once per model, use random_forest as representative)
            # Now guaranteed to match order
            if 'random_forest' not in all_feature_importances or len(all_feature_importances) == 1:
                importance_keys = list(importance_dict.keys())  # Use list to preserve order
                train_input_keys = feature_names  # Already a list
                if len(importance_keys) != len(train_input_keys):
                    missing = set(train_input_keys) - set(importance_keys)
                    logger.warning(f"  ⚠️  IMPORTANCE_KEYS mismatch (random_forest): {len(importance_keys)} keys vs {len(train_input_keys)} train features")
                    logger.warning(f"    Missing from importance: {sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}")
                elif importance_keys == train_input_keys:
                    # Keys match AND order matches - safe to log fingerprint
                    from TRAINING.utils.cross_sectional_data import _log_feature_set
                    _log_feature_set("IMPORTANCE_KEYS", importance_keys, previous_names=feature_names, logger_instance=logger)
            
            # Compute and store full task-aware metrics
            _compute_and_store_metrics('random_forest', model, X, y, primary_score, task_type)
            
            # Use percentage of total importance in top fraction features (0-1 scale, interpretable)
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
                # Use deterministic seed for permutation
                from TRAINING.common.determinism import stable_seed_from
                perm_seed = stable_seed_from(['permutation', target_column if 'target_column' in locals() else 'default', f'feature_{i}'])
                np.random.seed(perm_seed)
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
            
            # GPU settings (will fallback to CPU if GPU not available)
            gpu_params = {}
            try:
                from CONFIG.config_loader import get_cfg
                gpu_cfg = get_cfg('gpu.xgboost', default={}, config_name='gpu_config')
                xgb_device = gpu_cfg.get('device', 'cpu')
                xgb_tree_method = gpu_cfg.get('tree_method', 'hist')
                xgb_gpu_id = gpu_cfg.get('gpu_id', 0)
                
                if xgb_device == 'cuda':
                    # Try CUDA GPU
                    test_model = xgb.XGBRegressor(tree_method='hist', device='cuda', n_estimators=1)
                    test_model.fit(np.random.rand(10, 5), np.random.rand(10))
                    gpu_params = {'tree_method': xgb_tree_method, 'device': 'cuda', 'gpu_id': xgb_gpu_id}
                    if log_cfg.gpu_detail:
                        logger.info("  Using GPU (CUDA) for XGBoost")
                else:
                    if log_cfg.gpu_detail:
                        logger.info("  Using CPU for XGBoost (device='cpu' in config)")
            except Exception as e:
                if log_cfg.gpu_detail:
                    logger.info(f"  Using CPU for XGBoost (GPU not available: {e})")
            
            # Get config values
            xgb_config = get_model_config('xgboost', multi_model_config)
            # Defensive check: ensure config is a dict
            if not isinstance(xgb_config, dict):
                xgb_config = {}
            # Remove task-specific parameters (we set these explicitly based on task type)
            # CRITICAL: Extract early_stopping_rounds from config - it goes in constructor for XGBoost 2.0+
            # Also remove tree_method and device if present (we set these from GPU config)
            early_stopping_rounds = xgb_config.get('early_stopping_rounds', None)
            xgb_config_clean = {k: v for k, v in xgb_config.items() 
                              if k not in ['objective', 'eval_metric', 'early_stopping_rounds', 'tree_method', 'device', 'gpu_id']}
            
            # XGBoost 2.0+ requires early_stopping_rounds in constructor, not fit()
            if early_stopping_rounds is not None:
                xgb_config_clean['early_stopping_rounds'] = early_stopping_rounds
            
            # Add GPU params if available (will override any tree_method/device in config)
            xgb_config_clean.update(gpu_params)
            
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
            
            # Log GPU usage if available (controlled by config)
            if 'device' in gpu_params and gpu_params.get('device') == 'cuda' and log_cfg.gpu_detail:
                logger.info(f"  🚀 Training XGBoost on CUDA (gpu_id={gpu_params.get('gpu_id', 0)})")
                logger.info(f"  📊 Dataset size: {len(X)} samples, {X.shape[1]} features")
                if log_cfg.edu_hints:
                    logger.info(f"  💡 Note: GPU is most efficient for large datasets (>100k samples)")
            
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
                    # Load time-aware split ratio from config
                    try:
                        from CONFIG.config_loader import get_cfg
                        time_split_ratio = float(get_cfg("preprocessing.validation.time_aware_split_ratio", default=0.8, config_name="preprocessing_config"))
                    except Exception:
                        time_split_ratio = 0.8
                    split_idx = int(len(X) * time_split_ratio)
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
                    importance_dict = importance_series.to_dict()
                    all_feature_importances['xgboost'] = importance_dict
            if hasattr(model, 'feature_importances_'):
                # Use percentage of total importance in top 10% features (0-1 scale, interpretable)
                importances = model.feature_importances_
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
            logger.warning(f"XGBoost failed: {e}")
    
    # CatBoost
    if 'catboost' in model_families:
        try:
            import catboost as cb
            from TRAINING.utils.target_utils import is_classification_target, is_binary_classification_target
            
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
                
                # CatBoost requires training dataset to compute feature importance
                importance = model.get_feature_importance(data=X, type='PredictionValuesChange')
            else:
                importance = np.array([])
            if len(importance) > 0:
                total_importance = np.sum(importance)
                if total_importance > 0:
                    top_fraction = _get_importance_top_fraction()
                    top_k = max(1, int(len(importance) * top_fraction))
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
            from TRAINING.utils.sklearn_safe import make_sklearn_dense_X
            
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
                    top_fraction = _get_importance_top_fraction()
                    top_k = max(1, int(len(importance) * top_fraction))
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
            from TRAINING.utils.sklearn_safe import make_sklearn_dense_X
            
            # Mutual information doesn't handle NaN - use sklearn-safe conversion
            X_dense, feature_names_dense = make_sklearn_dense_X(X, feature_names)
            
            # Get config values
            mi_config = get_model_config('mutual_information', multi_model_config)
            
            # Get random_state from SST (determinism system) - no hardcoded defaults
            mi_random_state = mi_config.get('random_state')
            if mi_random_state is None:
                from TRAINING.common.determinism import stable_seed_from
                mi_random_state = stable_seed_from(['mutual_information', target_column if target_column else 'default'])
            
            # Suppress warnings for zero-variance features
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                if is_binary or is_multiclass:
                    importance = mutual_info_classif(X_dense, y, 
                                                    random_state=mi_random_state,
                                                    discrete_features=mi_config.get('discrete_features', 'auto'))
                else:
                    importance = mutual_info_regression(X_dense, y, 
                                                       random_state=mi_random_state,
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
            from TRAINING.utils.sklearn_safe import make_sklearn_dense_X
            
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
                    top_fraction = _get_importance_top_fraction()
                    top_k = max(1, int(len(importance) * top_fraction))
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
            n_features_to_select = min(rfe_config['n_features_to_select'], X_imputed.shape[1])
            step = rfe_config['step']
            
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
                    top_fraction = _get_importance_top_fraction()
                    top_k = max(1, int(len(importance) * top_fraction))
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
            from TRAINING.utils.sklearn_safe import make_sklearn_dense_X
            
            # Boruta doesn't support NaN - use sklearn-safe conversion
            X_dense, feature_names_dense = make_sklearn_dense_X(X, feature_names)
            
            # Get config values
            boruta_config = get_model_config('boruta', multi_model_config)
            
            # Use random_forest config for Boruta estimator
            rf_config = get_model_config('random_forest', multi_model_config)
            
            # Get random_state from SST (determinism system) - no hardcoded defaults
            boruta_random_state = boruta_config.get('random_state')
            if boruta_random_state is None:
                from TRAINING.common.determinism import stable_seed_from
                boruta_random_state = stable_seed_from(['boruta', target_column if target_column else 'default'])
            
            # Remove random_state from rf_config to prevent double argument error
            rf_config_clean = rf_config.copy()
            rf_config_clean.pop('random_state', None)
            
            if is_binary or is_multiclass:
                rf = RandomForestClassifier(**rf_config_clean, random_state=boruta_random_state)
            else:
                rf = RandomForestRegressor(**rf_config_clean, random_state=boruta_random_state)
            
            boruta = BorutaPy(rf, n_estimators='auto', verbose=0, 
                            random_state=boruta_random_state,
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
                    top_fraction = _get_importance_top_fraction()
                    top_k = max(1, int(len(importance) * top_fraction))
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
            from TRAINING.utils.sklearn_safe import make_sklearn_dense_X
            
            # Stability selection uses Lasso/LogisticRegression which don't handle NaN
            X_dense, feature_names_dense = make_sklearn_dense_X(X, feature_names)
            
            # Get config values
            stability_config = get_model_config('stability_selection', multi_model_config)
            n_bootstrap = stability_config.get('n_bootstrap', 50)
            # Get random_state from SST (determinism system) - no hardcoded defaults
            random_state = stability_config.get('random_state')
            if random_state is None:
                from TRAINING.common.determinism import stable_seed_from
                random_state = stable_seed_from(['stability_selection', target_column if target_column else 'default'])
            stability_cv = stability_config.get('cv', 3)
            stability_n_jobs = stability_config.get('n_jobs', 1)
            stability_cs = stability_config.get('Cs', 10)
            stability_scores = np.zeros(X_dense.shape[1])
            bootstrap_r2_scores = []
            
            # Use lasso config for stability selection models
            lasso_config = get_model_config('lasso', multi_model_config)
            
            for _ in range(n_bootstrap):
                # Use deterministic seed for bootstrap sampling
                from TRAINING.common.determinism import stable_seed_from
                bootstrap_seed = stable_seed_from(['bootstrap', target_column if 'target_column' in locals() else 'default', f'iter_{i}'])
                np.random.seed(bootstrap_seed)
                indices = np.random.choice(len(X_dense), size=len(X_dense), replace=True)
                X_boot, y_boot = X_dense[indices], y[indices]
                
                try:
                    # Use TimeSeriesSplit for internal CV (even though bootstrap breaks temporal order,
                    # this maintains consistency with the rest of the codebase)
                    # Clean config to prevent double random_state argument
                    from TRAINING.utils.config_cleaner import clean_config_for_estimator
                    if is_binary or is_multiclass:
                        lr_config = {'Cs': stability_cs, 'cv': tscv, 'max_iter': lasso_config.get('max_iter', 1000), 'n_jobs': stability_n_jobs}
                        lr_config_clean = clean_config_for_estimator(LogisticRegressionCV, lr_config, extra_kwargs={'random_state': random_state}, family_name='stability_selection')
                        model = LogisticRegressionCV(**lr_config_clean, random_state=random_state)
                    else:
                        lasso_config_clean_dict = {'cv': tscv, 'max_iter': lasso_config.get('max_iter', 1000), 'n_jobs': stability_n_jobs}
                        lasso_config_clean = clean_config_for_estimator(LassoCV, lasso_config_clean_dict, extra_kwargs={'random_state': random_state}, family_name='stability_selection')
                        model = LassoCV(**lasso_config_clean, random_state=random_state)
                    
                    model.fit(X_boot, y_boot)
                    coef = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
                    stability_scores += (np.abs(coef) > 1e-6).astype(int)
                    
                    # Get R² using cross-validation (proper validation, not training score)
                    # Note: Bootstrap samples break temporal order, but we still use TimeSeriesSplit
                    # for consistency (it won't help here, but maintains the pattern)
                    # Use a quick model for CV scoring
                    if is_binary or is_multiclass:
                        lr_cv_config = {'Cs': [1.0], 'cv': tscv, 'max_iter': lasso_config.get('max_iter', 1000), 'n_jobs': 1}
                        lr_cv_config_clean = clean_config_for_estimator(LogisticRegressionCV, lr_cv_config, extra_kwargs={'random_state': random_state}, family_name='stability_selection')
                        cv_model = LogisticRegressionCV(**lr_cv_config_clean, random_state=random_state)
                    else:
                        lasso_cv_config = {'cv': tscv, 'max_iter': lasso_config.get('max_iter', 1000), 'n_jobs': 1}
                        lasso_cv_config_clean = clean_config_for_estimator(LassoCV, lasso_cv_config, extra_kwargs={'random_state': random_state}, family_name='stability_selection')
                        cv_model = LassoCV(**lasso_cv_config_clean, random_state=random_state)
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
                    top_fraction = _get_importance_top_fraction()
                    top_k = max(1, int(len(importance) * top_fraction))
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
                    # Normalize to 0-1: what % of total importance is in top 10%?
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
    view: str = "CROSS_SECTIONAL"
) -> None:
    """
    Save detailed per-model, per-feature importance scores to CSV files.
    
    Creates structure:
    {output_dir}/feature_importances/
      {target_name}/
        {symbol}/
          lightgbm_importances.csv
          xgboost_importances.csv
          random_forest_importances.csv
          ...
    
    Args:
        target_column: Name of the target being evaluated
        symbol: Symbol being evaluated
        feature_importances: Dict of {model_name: {feature: importance}}
        output_dir: Base output directory (defaults to results/)
    """
    if output_dir is None:
        output_dir = _REPO_ROOT / "results"
    
    # Create directory structure that respects view (SYMBOL_SPECIFIC vs CROSS_SECTIONAL)
    target_name_clean = target_column.replace('/', '_').replace('\\', '_')
    # Include view in path for SYMBOL_SPECIFIC to avoid collisions
    if view == "SYMBOL_SPECIFIC":
        importances_dir = output_dir / "target_rankings" / "feature_importances" / target_name_clean / view / symbol
    else:
        importances_dir = output_dir / "target_rankings" / "feature_importances" / target_name_clean / view
    importances_dir.mkdir(parents=True, exist_ok=True)
    
    # Save per-model CSV files
    # Sort model names for deterministic order (ensures reproducible file output)
    for model_name in sorted(feature_importances.keys()):
        importances = feature_importances[model_name]
        if not importances:
            continue
        
        # Create DataFrame sorted by importance
        df = pd.DataFrame([
            {'feature': feat, 'importance': imp}
            for feat, imp in sorted(importances.items())  # Sort features for deterministic order
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
        
        # Save to CSV
        csv_file = importances_dir / f"{model_name}_importances.csv"
        df.to_csv(csv_file, index=False)
        
        # Save stability snapshot (non-invasive hook)
        try:
            from TRAINING.stability.feature_importance import save_snapshot_hook
            save_snapshot_hook(
                target_name=target_column,
                method=model_name,
                importance_dict=importances,
                universe_id=view,  # Use view parameter (CROSS_SECTIONAL or SYMBOL_SPECIFIC)
                output_dir=output_dir,
                auto_analyze=None,  # Load from config
            )
        except Exception as e:
            logger.debug(f"Stability snapshot save failed (non-critical): {e}")
    
    logger.info(f"  💾 Saved feature importances to: {importances_dir}")


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


def detect_leakage(
    mean_score: float,
    composite_score: float,
    mean_importance: float,
    target_name: str = "",
    model_scores: Dict[str, float] = None,
    task_type: TaskType = TaskType.REGRESSION
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
    if _CONFIG_AVAILABLE:
        try:
            safety_cfg = get_safety_config()
            # safety_config.yaml has a top-level 'safety' key
            safety_section = safety_cfg.get('safety', {})
            leakage_cfg = safety_section.get('leakage_detection', {})
            warning_cfg = leakage_cfg.get('warning_thresholds', {})
        except Exception:
            warning_cfg = {}
    else:
        warning_cfg = {}
    
    # Determine threshold based on task type and target name
    if task_type == TaskType.REGRESSION:
        is_forward_return = target_name.startswith('fwd_ret_')
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
    
    # Check 1: Suspiciously high mean score
    if mean_score > very_high_threshold:
        flags.append("HIGH_SCORE")
        logger.warning(
            f"LEAKAGE WARNING: {metric_name}={mean_score:.3f} > {very_high_threshold:.2f} "
            f"(extremely high - likely leakage)"
        )
    elif mean_score > high_threshold:
        flags.append("HIGH_SCORE")
        logger.warning(
            f"LEAKAGE WARNING: {metric_name}={mean_score:.3f} > {high_threshold:.2f} "
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
    # Load thresholds from config
    try:
        from CONFIG.config_loader import get_cfg
        composite_high_threshold = float(get_cfg("safety.leakage_detection.model_evaluation.composite_score_high_threshold", default=0.5, config_name="safety_config"))
        regression_score_low = float(get_cfg("safety.leakage_detection.model_evaluation.regression_score_low_threshold", default=0.2, config_name="safety_config"))
        classification_score_low = float(get_cfg("safety.leakage_detection.model_evaluation.classification_score_low_threshold", default=0.6, config_name="safety_config"))
    except Exception:
        composite_high_threshold = 0.5
        regression_score_low = 0.2
        classification_score_low = 0.6
    
    score_low_threshold = regression_score_low if task_type == TaskType.REGRESSION else classification_score_low
    if composite_score > composite_high_threshold and mean_score < score_low_threshold:
        flags.append("INCONSISTENT")
        logger.warning(
            f"LEAKAGE WARNING: Composite={composite_score:.3f} but {metric_name}={mean_score:.3f} "
            f"(inconsistent - possible leakage)"
        )
    
    # Check 4: Very high importance with low score (might indicate leaked features)
    # Load thresholds from config
    try:
        from CONFIG.config_loader import get_cfg
        importance_high_threshold = float(get_cfg("safety.leakage_detection.model_evaluation.importance_high_threshold", default=0.7, config_name="safety_config"))
        regression_score_very_low = float(get_cfg("safety.leakage_detection.model_evaluation.regression_score_very_low_threshold", default=0.1, config_name="safety_config"))
        classification_score_very_low = float(get_cfg("safety.leakage_detection.model_evaluation.classification_score_very_low_threshold", default=0.5, config_name="safety_config"))
    except Exception:
        importance_high_threshold = 0.7
        regression_score_very_low = 0.1
        classification_score_very_low = 0.5
    
    score_very_low_threshold = regression_score_very_low if task_type == TaskType.REGRESSION else classification_score_very_low
    if mean_importance > importance_high_threshold and mean_score < score_very_low_threshold:
        flags.append("INCONSISTENT")
        logger.warning(
            f"LEAKAGE WARNING: Importance={mean_importance:.2f} but {metric_name}={mean_score:.3f} "
            f"(high importance with low {metric_name} - check for leaked features)"
        )
    
    if len(flags) > 1:
        return "SUSPICIOUS"
    elif len(flags) == 1:
        return flags[0]
    else:
        return "OK"


def calculate_composite_score(
    mean_score: float,
    std_score: float,
    mean_importance: float,
    n_models: int,
    task_type: TaskType = TaskType.REGRESSION
) -> float:
    """
    Calculate composite predictability score
    
    Components:
    - Mean score: Higher is better (R² for regression, ROC-AUC/Accuracy for classification)
    - Consistency: Lower std is better
    - Importance magnitude: Higher is better
    - Model agreement: More models = more confidence
    """
    
    # Normalize components based on task type
    if task_type == TaskType.REGRESSION:
        # R² can be negative, so normalize to 0-1 range
        score_component = max(0, mean_score)  # Clamp negative R² to 0
        consistency_component = 1.0 / (1.0 + std_score)
        
        # R²-weighted importance
        if mean_score > 0:
            importance_component = mean_importance * (1.0 + mean_score)
        else:
            penalty = abs(mean_score) * 0.67
            importance_component = mean_importance * max(0.5, 1.0 - penalty)
    else:
        # Classification: ROC-AUC and Accuracy are already 0-1
        score_component = mean_score  # Already 0-1
        consistency_component = 1.0 / (1.0 + std_score)
        
        # Score-weighted importance (similar logic but for 0-1 scores)
        importance_component = mean_importance * (1.0 + mean_score)
    
    # Weighted average
    composite = (
        0.50 * score_component +        # 50% weight on score
        0.25 * consistency_component + # 25% on consistency
        0.25 * importance_component    # 25% on score-weighted importance
    )
    
    # Bonus for more models (up to 10% boost)
    model_bonus = min(0.1, n_models * 0.02)
    composite = composite * (1.0 + model_bonus)
    
    return composite



def evaluate_target_predictability(
    target_name: str,
    target_config: Dict[str, Any] | TargetConfig,
    symbols: List[str],
    data_dir: Path,
    model_families: List[str],
    multi_model_config: Dict[str, Any] = None,
    output_dir: Path = None,
    min_cs: int = 10,
    max_cs_samples: Optional[int] = None,
    max_rows_per_symbol: int = None,
    explicit_interval: Optional[Union[int, str]] = None,  # Explicit interval from config (e.g., "5m")
    experiment_config: Optional[Any] = None,  # Optional ExperimentConfig (for data.bar_interval)
    view: str = "CROSS_SECTIONAL",  # "CROSS_SECTIONAL", "SYMBOL_SPECIFIC", or "LOSO"
    symbol: Optional[str] = None  # Required for SYMBOL_SPECIFIC and LOSO views
) -> TargetPredictabilityScore:
    """Evaluate predictability of a single target across symbols"""
    
    # Ensure numpy is available (imported at module level, but ensure it's accessible)
    import numpy as np  # Use global import from top of file
    
    # Get logging config for this module (at function start)
    if _LOGGING_CONFIG_AVAILABLE:
        log_cfg = get_module_logging_config('rank_target_predictability')
    else:
        log_cfg = _DummyLoggingConfig()
    
    # Load default max_rows_per_symbol from config if not provided
    if max_rows_per_symbol is None:
        if _CONFIG_AVAILABLE:
            try:
                max_rows_per_symbol = int(get_cfg("pipeline.data_limits.default_max_rows_per_symbol_ranking", default=50000, config_name="pipeline_config"))
            except Exception:
                max_rows_per_symbol = 50000
        else:
            max_rows_per_symbol = 50000
    
    # Convert dict config to TargetConfig if needed
    if isinstance(target_config, dict):
        target_column = target_config['target_column']
        display_name = target_config.get('display_name', target_name)
        # Infer task type from column name (will be refined with actual data)
        task_type = TaskType.from_target_column(target_column)
        target_config_obj = TargetConfig(
            name=target_name,
            target_column=target_column,
            task_type=task_type,
            display_name=display_name,
            **{k: v for k, v in target_config.items() 
               if k not in ['target_column', 'display_name']}
        )
    else:
        target_config_obj = target_config
        target_column = target_config_obj.target_column
        display_name = target_config_obj.display_name or target_name
    # Validate view and symbol parameters
    if view == "SYMBOL_SPECIFIC" and symbol is None:
        raise ValueError(f"symbol parameter required for SYMBOL_SPECIFIC view")
    if view == "LOSO" and symbol is None:
        raise ValueError(f"symbol parameter required for LOSO view")
    if view == "CROSS_SECTIONAL" and symbol is not None:
        logger.warning(f"symbol={symbol} provided but view=CROSS_SECTIONAL, ignoring symbol")
        symbol = None
    
    view_display = f"{view}" + (f" (symbol={symbol})" if symbol else "")
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating: {display_name} ({target_column}) - {view_display}")
    logger.info(f"{'='*60}")
    
    # Load data based on view
    from TRAINING.utils.cross_sectional_data import load_mtf_data_for_ranking, prepare_cross_sectional_data_for_ranking
    from TRAINING.utils.leakage_filtering import filter_features_for_target
    from TRAINING.utils.target_conditional_exclusions import (
        generate_target_exclusion_list,
        load_target_exclusion_list
    )
    
    # For SYMBOL_SPECIFIC and LOSO, filter symbols
    symbols_to_load = symbols
    if view == "SYMBOL_SPECIFIC":
        symbols_to_load = [symbol]
    elif view == "LOSO":
        # LOSO: train on all symbols except symbol, validate on symbol
        symbols_to_load = [s for s in symbols if s != symbol]
        validation_symbol = symbol
    else:
        validation_symbol = None
    
    logger.info(f"Loading data for {len(symbols_to_load)} symbol(s) (max {max_rows_per_symbol} rows per symbol)...")
    if view == "LOSO":
        logger.info(f"  LOSO: Training on {len(symbols_to_load)} symbols, validating on {validation_symbol}")
    mtf_data = load_mtf_data_for_ranking(data_dir, symbols_to_load, max_rows_per_symbol=max_rows_per_symbol)
    
    if not mtf_data:
        logger.error(f"No data loaded for any symbols")
        return TargetPredictabilityScore(
            target_name=target_name,
            target_column=target_column,
            task_type=TaskType.REGRESSION,
            mean_score=-999.0,
            std_score=1.0,
            mean_importance=0.0,
            consistency=0.0,
            n_models=0,
            model_scores={}
        )
    
    # Apply leakage filtering to feature list BEFORE preparing data (with registry validation)
    # Get all columns from first symbol to determine available features
    sample_df = next(iter(mtf_data.values()))
    all_columns = sample_df.columns.tolist()

    # TARGET-CONDITIONAL EXCLUSIONS: Generate per-target exclusion list
    # This implements "Target-Conditional Feature Selection" - tailoring features to target physics
    target_conditional_exclusions = []
    exclusion_metadata = {}
    target_exclusion_dir = None
    
    if output_dir:
        target_exclusion_dir = Path(output_dir) / "feature_exclusions"
        target_exclusion_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to load existing exclusion list first (from RESULTS/{cohort}/{run}/feature_exclusions/)
        # This allows reusing exclusion lists across runs for consistency
        existing_exclusions = load_target_exclusion_list(target_name, target_exclusion_dir)
        if existing_exclusions is not None:
            target_conditional_exclusions = existing_exclusions
            logger.info(
                f"📋 Loaded existing target-conditional exclusions for {target_name}: "
                f"{len(target_conditional_exclusions)} features "
                f"(from {target_exclusion_dir})"
            )
        else:
            # Generate new exclusion list
            try:
                from TRAINING.common.feature_registry import get_registry
                registry = get_registry()
            except Exception:
                registry = None
            
            # Detect interval for lookback calculation
            from TRAINING.utils.data_interval import detect_interval_from_dataframe
            temp_interval = detect_interval_from_dataframe(sample_df, explicit_interval=explicit_interval)
            
            target_conditional_exclusions, exclusion_metadata = generate_target_exclusion_list(
                target_name=target_name,
                all_features=all_columns,
                interval_minutes=temp_interval,
                output_dir=target_exclusion_dir,
                registry=registry
            )
            
            if target_conditional_exclusions:
                logger.info(
                    f"📋 Generated target-conditional exclusions for {target_name}: "
                    f"{len(target_conditional_exclusions)} features excluded "
                    f"(horizon={exclusion_metadata.get('target_horizon_minutes', 'unknown')}m, "
                    f"semantics={exclusion_metadata.get('target_semantics', {})})"
                )
    else:
        # No output_dir - skip target-conditional exclusions (backward compatibility)
        logger.debug("No output_dir provided - skipping target-conditional exclusions")

    # Detect data interval for horizon conversion (use explicit_interval if provided)
    from TRAINING.utils.data_interval import detect_interval_from_dataframe
    detected_interval = detect_interval_from_dataframe(
        sample_df,
        timestamp_column='ts', 
        default=5,
        explicit_interval=explicit_interval,
        experiment_config=experiment_config
    )
    
    # Extract target horizon for error messages
    from TRAINING.utils.leakage_filtering import _load_leakage_config, _extract_horizon
    leakage_config = _load_leakage_config()
    target_horizon_minutes = _extract_horizon(target_column, leakage_config) if target_column else None
    target_horizon_bars = None
    if target_horizon_minutes is not None and detected_interval > 0:
        target_horizon_bars = int(target_horizon_minutes // detected_interval)
    
    # Use target-aware filtering with registry validation
    # Apply target-conditional exclusions BEFORE global filtering
    # This ensures target-specific rules are applied first
    columns_after_target_exclusions = [c for c in all_columns if c not in target_conditional_exclusions]
    
    if target_conditional_exclusions:
        logger.info(
            f"  🎯 Target-conditional exclusions: Removed {len(target_conditional_exclusions)} features "
            f"({len(columns_after_target_exclusions)} remaining before global filtering)"
        )
    
    # Apply global filtering (registry, patterns, etc.)
    safe_columns = filter_features_for_target(
        columns_after_target_exclusions,  # Use pre-filtered columns
        target_column,
        verbose=True,
        use_registry=True,  # Enable registry validation
        data_interval_minutes=detected_interval,
        for_ranking=True  # Use permissive rules for ranking (allow basic OHLCV/TA)
    )
    
    excluded_count = len(all_columns) - len(safe_columns) - 1  # -1 for target itself
    features_safe = len(safe_columns)
    logger.debug(f"Filtered out {excluded_count} potentially leaking features (kept {features_safe} safe features)")
    
    # CRITICAL: Check if we have enough features to train
    # Load from config
    if _CONFIG_AVAILABLE:
        try:
            safety_cfg = get_safety_config()
            # safety_config.yaml has a top-level 'safety' key
            safety_section = safety_cfg.get('safety', {})
            leakage_cfg = safety_section.get('leakage_detection', {})
            ranking_cfg = leakage_cfg.get('ranking', {})
            MIN_FEATURES_REQUIRED = int(ranking_cfg.get('min_features_required', 2))
        except Exception:
            MIN_FEATURES_REQUIRED = 2
    else:
        MIN_FEATURES_REQUIRED = 2
    
    if len(safe_columns) < MIN_FEATURES_REQUIRED:
        horizon_info = f"horizon={target_horizon_bars} bars" if target_horizon_bars is not None else "this horizon"
        logger.error(
            f"❌ INSUFFICIENT FEATURES: Only {len(safe_columns)} features remain after filtering "
            f"(minimum required: {MIN_FEATURES_REQUIRED}). "
            f"This target may not be predictable with current feature set. "
            f"Consider:\n"
            f"  1. Adding more features to CONFIG/feature_registry.yaml with allowed_horizons including {horizon_info}\n"
            f"  2. Relaxing feature registry rules for short-horizon targets\n"
            f"  3. Checking if excluded_features.yaml is too restrictive\n"
            f"  4. Skipping this target and focusing on targets with longer horizons"
        )
        # Return -999.0 to indicate this target should be skipped (same as degenerate targets)
        return TargetPredictabilityScore(
            target_name=target_name,
            target_column=target_column,
            task_type=target_config_obj.task_type,
            mean_score=-999.0,  # Flag for filtering (same as degenerate targets)
            std_score=0.0,
            mean_importance=0.0,
            consistency=0.0,
            n_models=0,
            model_scores={},
            composite_score=0.0,
            leakage_flag="INSUFFICIENT_FEATURES"
        )
    
    # Track feature counts (will be updated after data preparation)
    features_dropped_nan = 0
    features_final = features_safe
    
    # Prepare data based on view
    if view == "SYMBOL_SPECIFIC":
        # For symbol-specific, prepare single-symbol time series data
        # Use same function but with single symbol (min_cs=1 effectively)
        X, y, feature_names, symbols_array, time_vals = prepare_cross_sectional_data_for_ranking(
            mtf_data, target_column, min_cs=1, max_cs_samples=max_cs_samples, feature_names=safe_columns
        )
        # Verify we only have one symbol
        unique_symbols = set(symbols_array) if symbols_array is not None else set()
        if len(unique_symbols) > 1:
            logger.warning(f"SYMBOL_SPECIFIC view expected 1 symbol, got {len(unique_symbols)}: {unique_symbols}")
    elif view == "LOSO":
        # LOSO: prepare training data (all symbols except validation symbol)
        X_train, y_train, feature_names_train, symbols_array_train, time_vals_train = prepare_cross_sectional_data_for_ranking(
            mtf_data, target_column, min_cs=min_cs, max_cs_samples=max_cs_samples, feature_names=safe_columns
        )
        # Load validation symbol data separately
        validation_mtf_data = load_mtf_data_for_ranking(data_dir, [validation_symbol], max_rows_per_symbol=max_rows_per_symbol)
        X_val, y_val, feature_names_val, symbols_array_val, time_vals_val = prepare_cross_sectional_data_for_ranking(
            validation_mtf_data, target_column, min_cs=1, max_cs_samples=None, feature_names=safe_columns
        )
        # For LOSO, we'll use a special CV that trains on X_train and validates on X_val
        # For now, combine them and use a custom splitter (will be implemented in train_and_evaluate_models)
        # TODO: Implement LOSO-specific CV splitter
        logger.warning("LOSO view: Using combined data for now (LOSO-specific CV splitter not yet implemented)")
        X = X_train  # Will be handled by LOSO-specific logic
        y = y_train
        feature_names = feature_names_train
        symbols_array = symbols_array_train
        time_vals = time_vals_train
    else:
        # CROSS_SECTIONAL: standard pooled data
        X, y, feature_names, symbols_array, time_vals = prepare_cross_sectional_data_for_ranking(
            mtf_data, target_column, min_cs=min_cs, max_cs_samples=max_cs_samples, feature_names=safe_columns
        )
    
    # Update feature counts after data preparation
    if feature_names is not None:
        features_final = len(feature_names)
        features_dropped_nan = features_safe - features_final
    
    # Store cohort metadata context for later use in reproducibility tracking
    # These will be used to extract cohort metadata at the end of the function
    cohort_context = {
        'X': X,
        'y': y,  # Label vector for data fingerprint
        'time_vals': time_vals,
        'symbols_array': symbols_array,
        'mtf_data': mtf_data,
        'symbols': symbols,
        'min_cs': min_cs,
        'max_cs_samples': max_cs_samples
    }
    
    if X is None or y is None:
        logger.error(f"Failed to prepare cross-sectional data for {target_column}")
        return TargetPredictabilityScore(
            target_name=target_name,
            target_column=target_column,
            task_type=TaskType.REGRESSION,
            mean_score=-999.0,
            std_score=1.0,
            mean_importance=0.0,
            consistency=0.0,
            n_models=0,
            model_scores={}
        )
    
    # CRITICAL: Initialize resolved_config early to avoid "referenced before assignment" errors
    # We'll create a baseline config first (without feature lookback), then override post-prune
    resolved_config = None
    from TRAINING.utils.resolved_config import create_resolved_config
    
    # Get n_symbols_available from mtf_data
    n_symbols_available = len(mtf_data)
    
    # Create baseline resolved_config (WITH feature lookback computation)
    # CRITICAL FIX: Compute feature lookback early to ensure purge is large enough
    # This prevents "ROLLING WINDOW LEAKAGE RISK" violations
    selected_features = feature_names.copy() if feature_names else []
    
    # Create baseline config (WITH feature lookback computation for auto-adjustment)
    # The auto-fix logic in create_resolved_config will increase purge if feature_lookback > purge
    resolved_config = create_resolved_config(
        requested_min_cs=min_cs if view != "SYMBOL_SPECIFIC" else 1,
        n_symbols_available=n_symbols_available,
        max_cs_samples=max_cs_samples,
        interval_minutes=detected_interval,
        horizon_minutes=target_horizon_minutes,
        feature_lookback_max_minutes=None,  # Will be computed from feature_names
        purge_buffer_bars=5,  # Default from config
        default_purge_minutes=None,  # Loads from safety_config.yaml (SST)
        features_safe=features_safe,
        features_dropped_nan=features_dropped_nan,
        features_final=len(selected_features),
        view=view,
        symbol=symbol,
        feature_names=selected_features,  # Pass feature names for lookback computation
        recompute_lookback=True  # CRITICAL: Compute feature lookback to auto-adjust purge
    )
    
    if log_cfg.cv_detail:
        logger.info(f"  ✅ Baseline resolved config (pre-prune): purge={resolved_config.purge_minutes:.1f}m, embargo={resolved_config.embargo_minutes:.1f}m")
    
    logger.info(f"Cross-sectional data: {len(X)} samples, {X.shape[1]} features")
    logger.info(f"Symbols: {len(set(symbols_array))} unique symbols")
    
    # Infer task type from data (needed for leak scan)
    y_sample = pd.Series(y).dropna()
    task_type = TaskType.from_target_column(target_column, y_sample.to_numpy())
    
    # PRE-TRAINING LEAK SCAN: Detect and remove near-copy features before model training
    logger.info("🔍 Pre-training leak scan: Checking for near-copy features...")
    feature_names_before_leak_scan = feature_names.copy()
    
    # Check for duplicate column names before leak scan
    if len(feature_names) != len(set(feature_names)):
        duplicates = [name for name in set(feature_names) if feature_names.count(name) > 1]
        logger.error(f"  🚨 DUPLICATE COLUMN NAMES DETECTED before leak scan: {duplicates}")
        raise ValueError(f"Duplicate feature names detected: {duplicates}")
    
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y)
    leaky_features = find_near_copy_features(X_df, y_series, task_type)
    
    if leaky_features:
        logger.error(
            f"  ❌ CRITICAL: Found {len(leaky_features)} leaky features that are near-copies of target: {leaky_features}"
        )
        logger.error(
            f"  Removing leaky features and continuing with {X.shape[1] - len(leaky_features)} features..."
        )
        
        # Remove leaky features
        leaky_indices = [i for i, name in enumerate(feature_names) if name in leaky_features]
        X = np.delete(X, leaky_indices, axis=1)
        feature_names = [name for name in feature_names if name not in leaky_features]
        
        logger.info(f"  After leak removal: {X.shape[1]} features remaining")
        from TRAINING.utils.cross_sectional_data import _log_feature_set
        _log_feature_set("AFTER_LEAK_REMOVAL", feature_names, previous_names=feature_names_before_leak_scan, logger_instance=logger)
        
        # If we removed too many features, mark as insufficient
        # Load from config
        if _CONFIG_AVAILABLE:
            try:
                safety_cfg = get_safety_config()
                # safety_config.yaml has a top-level 'safety' key
                safety_section = safety_cfg.get('safety', {})
                leakage_cfg = safety_section.get('leakage_detection', {})
                ranking_cfg = leakage_cfg.get('ranking', {})
                MIN_FEATURES_AFTER_LEAK_REMOVAL = int(ranking_cfg.get('min_features_after_leak_removal', 2))
            except Exception:
                MIN_FEATURES_AFTER_LEAK_REMOVAL = 2
        else:
            MIN_FEATURES_AFTER_LEAK_REMOVAL = 2
        
        if X.shape[1] < MIN_FEATURES_AFTER_LEAK_REMOVAL:
            logger.error(
                f"  ❌ Too few features remaining after leak removal ({X.shape[1]}). "
                f"Marking target as LEAKAGE_DETECTED."
            )
            return TargetPredictabilityScore(
                target_name=target_name,
                target_column=target_column,
                task_type=task_type,
                mean_score=-999.0,
                std_score=0.0,
                mean_importance=0.0,
                consistency=0.0,
                n_models=0,
                model_scores={},
                composite_score=0.0,
                leakage_flag="LEAKAGE_DETECTED"
            )
    else:
        logger.info("  ✅ No obvious leaky features detected in pre-training scan")
        from TRAINING.utils.cross_sectional_data import _log_feature_set
        _log_feature_set("AFTER_LEAK_REMOVAL", feature_names, previous_names=feature_names_before_leak_scan, logger_instance=logger)
    
    # CRITICAL: Early exit if too few features (before wasting time training models)
    # Load from config
    if _CONFIG_AVAILABLE:
        try:
            safety_cfg = get_safety_config()
            # safety_config.yaml has a top-level 'safety' key
            safety_section = safety_cfg.get('safety', {})
            leakage_cfg = safety_section.get('leakage_detection', {})
            ranking_cfg = leakage_cfg.get('ranking', {})
            MIN_FEATURES_FOR_MODEL = int(ranking_cfg.get('min_features_for_model', 3))
        except Exception:
            MIN_FEATURES_FOR_MODEL = 3
    else:
        MIN_FEATURES_FOR_MODEL = 3
    
    if X.shape[1] < MIN_FEATURES_FOR_MODEL:
        logger.warning(
            f"Too few features ({X.shape[1]}) after filtering (minimum: {MIN_FEATURES_FOR_MODEL}); "
            f"marking target as degenerate and skipping model training."
        )
        return TargetPredictabilityScore(
            target_name=target_name,
            target_column=target_column,
            task_type=TaskType.REGRESSION,  # Default, will be updated if we get further
            mean_score=-999.0,  # Flag for filtering
            std_score=0.0,
            mean_importance=0.0,
            consistency=0.0,
            n_models=0,
            model_scores={},
            composite_score=0.0,
            leakage_flag="INSUFFICIENT_FEATURES"
        )
    
    # Task type already inferred above for leak scan
    
    # Validate target
    is_valid, error_msg = validate_target(y, task_type=task_type)
    if not is_valid:
        logger.warning(f"Skipping: {error_msg}")
        return TargetPredictabilityScore(
            target_name=target_name,
            target_column=target_column,
            task_type=task_type,
            mean_score=-999.0,
            std_score=1.0,
            mean_importance=0.0,
            consistency=0.0,
            n_models=0,
            model_scores={}
        )
    
    # Check if target is degenerate
    unique_vals = np.unique(y[~np.isnan(y)])
    if len(unique_vals) < 2:
        logger.warning(f"Skipping: Target has only {len(unique_vals)} unique value(s)")
        return TargetPredictabilityScore(
            target_name=target_name,
            target_column=target_column,
            task_type=task_type,
            mean_score=-999.0,
            std_score=1.0,
            mean_importance=0.0,
            consistency=0.0,
            n_models=0,
            model_scores={}
        )
    
    # For classification, check if classes are too imbalanced for CV
    if len(unique_vals) <= 10:  # Likely classification
        class_counts = np.bincount(y[~np.isnan(y)].astype(int))
        min_class_count = class_counts[class_counts > 0].min()
        if min_class_count < 2:
            logger.warning(f"Skipping: Smallest class has only {min_class_count} sample(s) (too few for CV)")
            return TargetPredictabilityScore(
                target_name=target_name,
                target_column=target_column,
                task_type=task_type,
                mean_score=-999.0,
                std_score=1.0,
                mean_importance=0.0,
                consistency=0.0,
                n_models=0,
                model_scores={}
            )
    
    # CRITICAL: Recompute resolved_config AFTER pruning (if pruning happened)
    # This ensures feature_lookback_max is computed from actual pruned features
    # If pruning didn't happen or failed, we keep the baseline config (already assigned above)
    # Note: Pruning happens inside train_and_evaluate_models, so we need to handle it there
    # For now, we'll recompute here if feature_names changed (indicating pruning happened externally)
    # The actual post-prune recomputation happens in train_and_evaluate_models
    
    # Log baseline config summary
    if log_cfg.cv_detail:
        resolved_config.log_summary(logger)

    # FINAL GATEKEEPER: Enforce safety at the last possible moment
    # This runs AFTER all loading/merging/sanitization is done
    # It physically drops features that violate the purge limit from the dataframe
    # This is the "worry-free" auto-corrector that handles race conditions
    X, feature_names = _enforce_final_safety_gate(
        X=X,
        feature_names=feature_names,
        resolved_config=resolved_config,
        interval_minutes=detected_interval,
        logger=logger
    )
    
    # CRITICAL: Recompute resolved_config.feature_lookback_max AFTER Final Gatekeeper
    # The audit system uses this value, so it must reflect the ACTUAL features that will be trained
    # (not the original features before the gatekeeper dropped problematic ones)
    if feature_names and len(feature_names) > 0:
        from TRAINING.utils.resolved_config import compute_feature_lookback_max
        max_lookback_after_gatekeeper, _ = compute_feature_lookback_max(
            feature_names,
            interval_minutes=detected_interval,
            max_lookback_cap_minutes=None
        )
        # Update resolved_config with the new lookback (from features that actually remain)
        if max_lookback_after_gatekeeper is not None:
            resolved_config.feature_lookback_max_minutes = max_lookback_after_gatekeeper
            if log_cfg.cv_detail:
                logger.info(f"📊 Updated feature_lookback_max after Final Gatekeeper: {max_lookback_after_gatekeeper:.1f}m (from {len(feature_names)} remaining features)")
    
    if X.shape[1] == 0:
        logger.error("❌ FINAL GATEKEEPER: All features were dropped! Cannot train models.")
        return TargetPredictabilityScore(
            target_name=target_name,
            target_column=target_column,
            task_type=task_type,
            mean_score=-999.0,
            std_score=1.0,
            mean_importance=0.0,
            consistency=0.0,
            n_models=0,
            model_scores={}
        )

    # Train and evaluate on cross-sectional data (single evaluation, not per-symbol)
    all_model_scores = []
    all_importances = []
    all_suspicious_features = {}
    fold_timestamps = None  # Initialize fold_timestamps for later use
    
    try:
        # Use detected_interval from outer scope (already computed above)
        # No need to recompute here
        
        result = train_and_evaluate_models(
            X, y, feature_names, task_type, model_families, multi_model_config,
            target_column=target_column,
            data_interval_minutes=detected_interval,  # Auto-detected or default
            time_vals=time_vals,  # Pass timestamps for fold tracking
            explicit_interval=explicit_interval,  # Pass explicit interval for consistency
            experiment_config=experiment_config,  # Pass experiment config
            output_dir=output_dir,  # Pass output directory for stability snapshots
            resolved_config=resolved_config  # Pass resolved config with correct purge/embargo (post-pruning)
        )
        
        if result is None or len(result) != 7:
            logger.warning(f"train_and_evaluate_models returned unexpected value: {result}")
            return TargetPredictabilityScore(
                target_name=target_name,
                target_column=target_column,
                task_type=task_type,
                mean_score=-999.0,
                std_score=1.0,
                mean_importance=0.0,
                consistency=0.0,
                n_models=0,
                model_scores={}
            )
        
        model_metrics, primary_scores, importance, suspicious_features, feature_importances, fold_timestamps, _perfect_correlation_models = result
        
        # CRITICAL: Extract actual pruned feature count from feature_importances
        # feature_importances contains the features that were actually used (after pruning)
        actual_pruned_feature_count = 0
        if feature_importances:
            # Get feature count from first model's importances (all models use same features after pruning)
            first_model_importances = next(iter(feature_importances.values()))
            if isinstance(first_model_importances, dict):
                actual_pruned_feature_count = len(first_model_importances)
            elif isinstance(first_model_importances, (list, np.ndarray)):
                actual_pruned_feature_count = len(first_model_importances)
        # Fallback to len(feature_names) if we can't extract from importances
        if actual_pruned_feature_count == 0:
            actual_pruned_feature_count = len(feature_names) if feature_names else 0
        
        # NOTE: _perfect_correlation_models is now only for tracking/debugging.
        # High training accuracy alone is NOT a reliable leakage signal (especially for tree models),
        # so we no longer mark targets as LEAKAGE_DETECTED based on this.
        # Real leakage defense: schema filters + pre-training scan + time-purged CV.
        if _perfect_correlation_models:
            logger.debug(
                f"  Models with high training accuracy (may be overfitting): {_perfect_correlation_models}. "
                f"Check CV metrics to assess real predictive power."
            )
        
        # Save aggregated feature importances (respect view: CROSS_SECTIONAL vs SYMBOL_SPECIFIC)
        if feature_importances and output_dir:
            # Use view parameter if available, otherwise default to CROSS_SECTIONAL
            view_for_importances = view if 'view' in locals() else "CROSS_SECTIONAL"
            symbol_for_importances = symbol if ('symbol' in locals() and symbol) else view_for_importances
            _save_feature_importances(target_column, symbol_for_importances, feature_importances, output_dir, view=view_for_importances)
        
        # Store suspicious features
        if suspicious_features:
            all_suspicious_features = suspicious_features
            symbol_for_log = symbol if ('symbol' in locals() and symbol) else (view if 'view' in locals() else "CROSS_SECTIONAL")
            _log_suspicious_features(target_column, symbol_for_log, suspicious_features)
        
        # AUTO-FIX LEAKAGE: If leakage detected, automatically fix and re-run
        # Initialize autofix_info to None (will be set if auto-fixer runs)
        autofix_info = None
        
        # Load thresholds from config (with sensible defaults)
        if _CONFIG_AVAILABLE:
            try:
                safety_cfg = get_safety_config()
                # safety_config.yaml has a top-level 'safety' key
                safety_section = safety_cfg.get('safety', {})
                leakage_cfg = safety_section.get('leakage_detection', {})
                auto_fix_cfg = leakage_cfg.get('auto_fix_thresholds', {})
                cv_threshold = float(auto_fix_cfg.get('cv_score', 0.99))
                accuracy_threshold = float(auto_fix_cfg.get('training_accuracy', 0.999))
                r2_threshold = float(auto_fix_cfg.get('training_r2', 0.999))
                correlation_threshold = float(auto_fix_cfg.get('perfect_correlation', 0.999))
                auto_fix_enabled = leakage_cfg.get('auto_fix_enabled', True)
                auto_fix_min_confidence = float(leakage_cfg.get('auto_fix_min_confidence', 0.8))
                auto_fix_max_features = int(leakage_cfg.get('auto_fix_max_features_per_run', 20))
            except Exception as e:
                logger.debug(f"Failed to load leakage detection config: {e}, using defaults")
                cv_threshold = 0.99  # FALLBACK_DEFAULT_OK
                accuracy_threshold = 0.999  # FALLBACK_DEFAULT_OK
                r2_threshold = 0.999  # FALLBACK_DEFAULT_OK
                correlation_threshold = 0.999  # FALLBACK_DEFAULT_OK
                auto_fix_enabled = True  # FALLBACK_DEFAULT_OK
                auto_fix_min_confidence = 0.8  # FALLBACK_DEFAULT_OK
                auto_fix_max_features = 20  # FALLBACK_DEFAULT_OK
        else:
            # FALLBACK_DEFAULT_OK: Fallback defaults (config not available)
            cv_threshold = 0.99  # FALLBACK_DEFAULT_OK
            accuracy_threshold = 0.999  # FALLBACK_DEFAULT_OK
            r2_threshold = 0.999  # FALLBACK_DEFAULT_OK
            correlation_threshold = 0.999  # FALLBACK_DEFAULT_OK
            auto_fix_enabled = True  # FALLBACK_DEFAULT_OK
            auto_fix_min_confidence = 0.8
            auto_fix_max_features = 20  # FALLBACK_DEFAULT_OK
        
        # Check if auto-fixer is enabled
        if not auto_fix_enabled:
            logger.debug("Auto-fixer is disabled in config")
            should_auto_fix = False
        else:
            should_auto_fix = False
            
            # Check 1: Perfect CV scores (cross-validation)
            # CRITICAL: Use actual CV scores from model_scores (primary_scores), not model_metrics
            # model_metrics may contain training scores, but model_scores contains CV scores
            max_cv_score = None
            if primary_scores:
                # primary_scores contains CV scores from cross_val_score
                valid_cv_scores = [s for s in primary_scores.values() if s is not None and not np.isnan(s)]
                if valid_cv_scores:
                    max_cv_score = max(valid_cv_scores)
            
            # Fallback: try to extract from model_metrics if primary_scores unavailable
            # But be careful - model_metrics['accuracy'] etc. should now contain CV scores after our fix above
            if max_cv_score is None and model_metrics:
                for model_name, metrics in model_metrics.items():
                    if isinstance(metrics, dict):
                        # Get CV score (should be CV after our fix, but double-check it's not training_accuracy)
                        cv_score_val = metrics.get('roc_auc') or metrics.get('r2') or metrics.get('accuracy')
                        # Exclude training scores explicitly
                        if cv_score_val is not None and not np.isnan(cv_score_val):
                            # Skip if this looks like a training score (training_accuracy exists and matches)
                            if 'training_accuracy' in metrics and abs(cv_score_val - metrics['training_accuracy']) < 0.001:
                                continue  # This is likely a training score, skip it
                            if max_cv_score is None or cv_score_val > max_cv_score:
                                max_cv_score = cv_score_val
            
            if max_cv_score is not None and max_cv_score >= cv_threshold:
                should_auto_fix = True
                logger.warning(f"🚨 Perfect CV scores detected (max_cv={max_cv_score:.4f} >= {cv_threshold:.1%}) - enabling auto-fix mode")
            
            # Check 2: Perfect in-sample training accuracy with suspicion score gating
            # Use suspicion score to distinguish overfit noise from real leakage
            if not should_auto_fix and model_metrics:
                logger.debug(f"Checking model_metrics for perfect scores: {list(model_metrics.keys())}")
                
                # Compute suspicion score for each model with perfect train accuracy
                for model_name, metrics in model_metrics.items():
                    if isinstance(metrics, dict):
                        logger.debug(f"  {model_name} metrics: {list(metrics.keys())}")
                        
                        # Get train and CV scores
                        train_acc = metrics.get('training_accuracy')
                        cv_acc = metrics.get('accuracy')  # CV accuracy
                        train_r2 = metrics.get('training_r2')
                        cv_r2 = metrics.get('r2')  # CV R²
                        
                        # Check classification
                        if train_acc is not None and train_acc >= accuracy_threshold:
                            logger.debug(f"    {model_name} training_accuracy: {train_acc:.4f}")
                            
                            # Compute suspicion score
                            suspicion = _compute_suspicion_score(
                                train_score=train_acc,
                                cv_score=cv_acc,
                                feature_importances=feature_importances.get(model_name, {}) if feature_importances else {},
                                task_type='classification'
                            )
                            
                            # Only auto-fix if suspicion score crosses threshold
                            suspicion_threshold = 0.5  # Load from config if available
                            if suspicion >= suspicion_threshold:
                                should_auto_fix = True
                                cv_acc_str = f"{cv_acc:.3f}" if cv_acc is not None else "N/A"
                                logger.warning(f"🚨 Suspicious perfect training accuracy in {model_name} "
                                            f"(train={train_acc:.1%}, cv={cv_acc_str}, "
                                            f"suspicion={suspicion:.2f}) - enabling auto-fix mode")
                                break
                            else:
                                # Overfit noise - log once at INFO level
                                cv_acc_str = f"{cv_acc:.3f}" if cv_acc is not None else "N/A"
                                logger.info(f"⚠️  {model_name} memorized training data (train={train_acc:.1%}, "
                                         f"cv={cv_acc_str}, suspicion={suspicion:.2f}). "
                                         f"Ignoring; check CV metrics.")
                        
                        elif cv_acc is not None and cv_acc >= accuracy_threshold:
                            # CV accuracy alone is suspicious
                            should_auto_fix = True
                            logger.warning(f"🚨 Perfect CV accuracy detected in {model_name} "
                                        f"({cv_acc:.1%} >= {accuracy_threshold:.1%}) - enabling auto-fix mode")
                            break
                        
                        # Check regression
                        if train_r2 is not None and train_r2 >= r2_threshold:
                            logger.debug(f"    {model_name} training_r2 (correlation): {train_r2:.4f}")
                            
                            # Compute suspicion score
                            suspicion = _compute_suspicion_score(
                                train_score=train_r2,
                                cv_score=cv_r2,
                                feature_importances=feature_importances.get(model_name, {}) if feature_importances else {},
                                task_type='regression'
                            )
                            
                            suspicion_threshold = 0.5
                            if suspicion >= suspicion_threshold:
                                should_auto_fix = True
                                cv_r2_str = f"{cv_r2:.4f}" if cv_r2 is not None else "N/A"
                                logger.warning(f"🚨 Suspicious perfect training correlation in {model_name} "
                                            f"(train={train_r2:.4f}, cv={cv_r2_str}, "
                                            f"suspicion={suspicion:.2f}) - enabling auto-fix mode")
                                break
                            else:
                                cv_r2_str = f"{cv_r2:.4f}" if cv_r2 is not None else "N/A"
                                logger.info(f"⚠️  {model_name} memorized training data (train={train_r2:.4f}, "
                                         f"cv={cv_r2_str}, suspicion={suspicion:.2f}). "
                                         f"Ignoring; check CV metrics.")
                        
                        elif cv_r2 is not None and cv_r2 >= r2_threshold:
                            # CV R² alone is suspicious
                            should_auto_fix = True
                            logger.warning(f"🚨 Perfect CV R² detected in {model_name} "
                                        f"({cv_r2:.4f} >= {r2_threshold:.4f}) - enabling auto-fix mode")
                            break
            
            # Check 3: Models that triggered perfect correlation warnings (fallback check)
            # Note: _perfect_correlation_models is populated inside train_and_evaluate_models,
            # but we check model_metrics above which covers the same cases, so this is just a safety check
            if not should_auto_fix and _perfect_correlation_models:
                should_auto_fix = True
                logger.warning(f"🚨 Perfect correlation detected in models: {', '.join(_perfect_correlation_models)} (>= {correlation_threshold:.1%}) - enabling auto-fix mode")
        
        if should_auto_fix:
            try:
                from TRAINING.common.leakage_auto_fixer import LeakageAutoFixer
                
                logger.info("🔧 Auto-fixing detected leaks...")
                logger.info(f"   Initializing LeakageAutoFixer (backups disabled)...")
                # Backups are disabled by default - no backup directory will be created
                fixer = LeakageAutoFixer(backup_configs=False, output_dir=output_dir)
                
                # Convert X to DataFrame if needed (auto-fixer expects DataFrame)
                if not isinstance(X, pd.DataFrame):
                    X_df = pd.DataFrame(X, columns=feature_names)
                else:
                    X_df = X
                
                # Convert y to Series if needed
                if not isinstance(y, pd.Series):
                    y_series = pd.Series(y)
                else:
                    y_series = y
                
                # Aggregate feature importances across all models
                aggregated_importance = {}
                if feature_importances:
                    # Sort model names for deterministic order (ensures reproducible aggregations)
                    for model_name in sorted(feature_importances.keys()):
                        importances = feature_importances[model_name]
                        if isinstance(importances, dict):
                            for feat, imp in importances.items():
                                if feat not in aggregated_importance:
                                    aggregated_importance[feat] = []
                                aggregated_importance[feat].append(imp)
                
                # Average importance across models (sort features for deterministic order)
                avg_importance = {feat: np.mean(imps) for feat, imps in sorted(aggregated_importance.items())} if aggregated_importance else {}
                
                # Get actual training accuracy from model_metrics (not CV scores)
                # This is critical - we detected perfect training accuracy, so pass that value
                actual_train_score = None
                if model_metrics:
                    for model_name, metrics in model_metrics.items():
                        if isinstance(metrics, dict):
                            # For classification, prefer training_accuracy (in-sample), fall back to CV accuracy
                            if 'training_accuracy' in metrics and metrics['training_accuracy'] >= accuracy_threshold:
                                actual_train_score = metrics['training_accuracy']
                                logger.debug(f"Using training accuracy {actual_train_score:.4f} from {model_name} for auto-fixer")
                                break
                            elif 'accuracy' in metrics and metrics['accuracy'] >= accuracy_threshold:
                                actual_train_score = metrics['accuracy']
                                logger.debug(f"Using CV accuracy {actual_train_score:.4f} from {model_name} for auto-fixer")
                                break
                            # For regression, prefer training_r2 (in-sample correlation), fall back to CV R²
                            elif 'training_r2' in metrics and metrics['training_r2'] >= r2_threshold:
                                actual_train_score = metrics['training_r2']
                                logger.debug(f"Using training correlation {actual_train_score:.4f} from {model_name} for auto-fixer")
                                break
                            elif 'r2' in metrics and metrics['r2'] >= r2_threshold:
                                actual_train_score = metrics['r2']
                                logger.debug(f"Using CV R² {actual_train_score:.4f} from {model_name} for auto-fixer")
                                break
                
                # Fallback to CV score if no perfect training score found
                # CRITICAL: Use the same max_cv_score we computed above for consistency
                if actual_train_score is None:
                    if max_cv_score is not None:
                        actual_train_score = max_cv_score
                        logger.debug(f"Using CV score {actual_train_score:.4f} as fallback for auto-fixer (from model_metrics)")
                    else:
                        actual_train_score = max(primary_scores.values()) if primary_scores else None
                        logger.debug(f"Using CV score {actual_train_score:.4f} as fallback for auto-fixer (from primary_scores)")
                
                # Log what we're passing to auto-fixer (enhanced visibility)
                # CRITICAL: Clarify which feature set is being used for scanning vs training
                train_feature_set_size = len(feature_names)  # Features used for training (after pruning)
                scan_feature_set_size = len(safe_columns) if 'safe_columns' in locals() else len(feature_names)  # Features available for scanning
                scan_scope = "full_safe" if scan_feature_set_size > train_feature_set_size else "trained_only"
                
                train_score_str = f"{actual_train_score:.4f}" if actual_train_score is not None else "None"
                logger.info(f"🔧 Auto-fixer inputs: train_score={train_score_str}, "
                           f"train_feature_set_size={train_feature_set_size}, "
                           f"scan_feature_set_size={scan_feature_set_size}, "
                           f"scan_scope={scan_scope}, "
                           f"model_importance keys={len(avg_importance)}")
                if avg_importance:
                    top_5 = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                    logger.debug(f"   Top 5 features by importance: {', '.join([f'{f}={imp:.4f}' for f, imp in top_5])}")
                else:
                    logger.warning(f"   ⚠️  No aggregated importance available! feature_importances keys: {list(feature_importances.keys()) if feature_importances else 'None'}")
                
                # Detect leaks
                detections = fixer.detect_leaking_features(
                    X=X_df, y=y_series, feature_names=feature_names,
                    target_column=target_column,
                    symbols=pd.Series(symbols_array) if symbols_array is not None else None,
                    task_type='classification' if task_type == TaskType.BINARY_CLASSIFICATION or task_type == TaskType.MULTICLASS_CLASSIFICATION else 'regression',
                    data_interval_minutes=detected_interval,
                    model_importance=avg_importance if avg_importance else None,
                    train_score=actual_train_score,
                    test_score=None  # CV scores are already validation scores
                )
                
                if detections:
                    logger.warning(f"🔧 Auto-detected {len(detections)} leaking features")
                    # Apply fixes (with high confidence threshold to avoid false positives)
                    updates, autofix_info = fixer.apply_fixes(
                        detections, 
                        min_confidence=auto_fix_min_confidence, 
                        max_features=auto_fix_max_features,
                        dry_run=False,
                        target_name=target_name
                    )
                    if autofix_info.modified_configs:
                        logger.info(f"✅ Auto-fixed leaks. Configs updated.")
                        logger.info(f"   Updated: {len(updates.get('excluded_features_updates', {}).get('exact_patterns', []))} exact patterns, "
                                  f"{len(updates.get('excluded_features_updates', {}).get('prefix_patterns', []))} prefix patterns")
                        logger.info(f"   Rejected: {len(updates.get('feature_registry_updates', {}).get('rejected_features', []))} features in registry")
                    else:
                        logger.warning("⚠️  Auto-fix detected leaks but no configs were modified")
                        logger.warning("   This usually means all detections were below confidence threshold")
                        logger.warning(f"   Check logs above for confidence distribution details")
                    # Log backup info if available
                    if autofix_info.backup_files:
                        logger.info(f"📦 Backup created: {len(autofix_info.backup_files)} backup file(s)")
                else:
                    logger.info("🔍 Auto-fix detected no leaks (may need manual review)")
                    # Still create backup even when no leaks detected (to preserve state history)
                    # This ensures we have a backup whenever auto-fix mode is triggered
                    # But only if backup_configs is enabled
                    backup_files = []
                    if fixer.backup_configs:
                        try:
                            backup_files = fixer._backup_configs(
                                target_name=target_name,
                                max_backups_per_target=None  # Use instance config
                            )
                            if backup_files:
                                logger.info(f"📦 Backup created (no leaks detected): {len(backup_files)} backup file(s)")
                        except Exception as backup_error:
                            logger.warning(f"Failed to create backup when no leaks detected: {backup_error}")
            except Exception as e:
                logger.warning(f"Auto-fix failed: {e}", exc_info=True)
        
        # Ensure primary_scores is a dict
        if primary_scores is None:
            logger.warning(f"primary_scores is None, skipping")
            return TargetPredictabilityScore(
                target_name=target_name,
                target_column=target_column,
                task_type=task_type,
                mean_score=-999.0,
                std_score=1.0,
                mean_importance=0.0,
                consistency=0.0,
                n_models=0,
                model_scores={}
            )
        if not isinstance(primary_scores, dict):
            logger.warning(f"primary_scores is not a dict (got {type(primary_scores)}), skipping")
            return TargetPredictabilityScore(
                target_name=target_name,
                target_column=target_column,
                task_type=task_type,
                mean_score=-999.0,
                std_score=1.0,
                mean_importance=0.0,
                consistency=0.0,
                n_models=0,
                model_scores={}
            )
        
        all_model_scores.append(primary_scores)
        all_importances.append(importance)
        
        scores_str = ", ".join([f"{k}={v:.3f}" for k, v in primary_scores.items()])
        logger.info(f"Scores: {scores_str}, importance={importance:.2f}")
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        tb_str = traceback.format_exc()
        logger.warning(f"Failed: {error_msg}")
        logger.error(f"Full traceback:\n{tb_str}")
        return TargetPredictabilityScore(
            target_name=target_name,
            target_column=target_column,
            task_type=task_type,
            mean_score=-999.0,
            std_score=1.0,
            mean_importance=0.0,
            consistency=0.0,
            n_models=0,
            model_scores={}
        )
    
    if not all_model_scores:
        logger.warning(f"No successful evaluations for {target_name} (skipping)")
        return TargetPredictabilityScore(
            target_name=target_name,
            target_column=target_column,
            task_type=TaskType.REGRESSION,  # Default, will be updated if target succeeds
            mean_score=-999.0,  # Flag for degenerate/failed targets
            std_score=1.0,
            mean_importance=0.0,
            consistency=0.0,
            n_models=0,
            model_scores={}
        )
    
    # Aggregate across models (skip NaN scores)
    # Note: With cross-sectional data, we only have one evaluation, not per-symbol
    all_scores_by_model = defaultdict(list)
    for scores_dict in all_model_scores:
        # Defensive check: skip None or non-dict entries
        if scores_dict is None or not isinstance(scores_dict, dict):
            logger.warning(f"Skipping invalid scores_dict: {type(scores_dict)}")
            continue
        for model_name, score in scores_dict.items():
            if not (np.isnan(score) if isinstance(score, (float, np.floating)) else False):
                all_scores_by_model[model_name].append(score)
    
    # Calculate statistics (only from models that succeeded)
    model_means = {model: np.mean(scores) for model, scores in all_scores_by_model.items() if scores}
    if not model_means:
        logger.warning(f"No successful model evaluations for {target_name}")
        return TargetPredictabilityScore(
            target_name=target_name,
            target_column=target_column,
            task_type=TaskType.REGRESSION,  # Default
            mean_score=-999.0,
            std_score=1.0,
            mean_importance=0.0,
            consistency=0.0,
            n_models=0,
            model_scores={},
            leakage_flag="OK",
            suspicious_features=None
        )
    
    mean_score = np.mean(list(model_means.values()))
    std_score = np.std(list(model_means.values())) if len(model_means) > 1 else 0.0
    mean_importance = np.mean(all_importances)
    consistency = 1.0 - (std_score / (abs(mean_score) + 1e-6))
    
    # Determine task type (already inferred from data above)
    final_task_type = task_type
    
    # Get metric name for logging
    if final_task_type == TaskType.REGRESSION:
        metric_name = "R²"
    elif final_task_type == TaskType.BINARY_CLASSIFICATION:
        metric_name = "ROC-AUC"
    else:  # MULTICLASS_CLASSIFICATION
        metric_name = "Accuracy"
    
    # Composite score (normalize scores appropriately)
    composite = calculate_composite_score(
        mean_score, std_score, mean_importance, len(all_scores_by_model), final_task_type
    )
    
    # Detect potential leakage (use task-appropriate thresholds)
    leakage_flag = detect_leakage(mean_score, composite, mean_importance, 
                                  target_name=target_name, model_scores=model_means, task_type=final_task_type)
    
    # Build detailed leakage flags for auto-rerun logic
    leakage_flags = {
        "perfect_train_acc": len(_perfect_correlation_models) > 0,  # Any model hit 100% training accuracy
        "high_auc": mean_score > 0.95 if final_task_type == TaskType.BINARY_CLASSIFICATION else False,
        "high_r2": mean_score > 0.80 if final_task_type == TaskType.REGRESSION else False,
        "suspicious_flag": leakage_flag != "OK"
    }
    
    # Determine status: SUSPICIOUS targets should be excluded from rankings
    # High AUC/R² after auto-fix suggests structural leakage (target construction issue)
    if leakage_flag in ["SUSPICIOUS", "HIGH_SCORE"]:
        # If we have very high scores, this is likely structural leakage, not just feature leakage
        if final_task_type == TaskType.BINARY_CLASSIFICATION and mean_score > 0.95:
            final_status = "SUSPICIOUS_STRONG"
        elif final_task_type == TaskType.REGRESSION and mean_score > 0.80:
            final_status = "SUSPICIOUS_STRONG"
        else:
            final_status = "SUSPICIOUS"
    else:
        final_status = "OK"
    
    result = TargetPredictabilityScore(
        target_name=target_name,
        target_column=target_column,
        task_type=final_task_type,
        mean_score=mean_score,
        std_score=std_score,
        mean_importance=mean_importance,
        consistency=consistency,
        n_models=len(all_scores_by_model),
        model_scores=model_means,
        composite_score=composite,
        leakage_flag=leakage_flag,
        suspicious_features=all_suspicious_features if all_suspicious_features else None,
        fold_timestamps=fold_timestamps,
        leakage_flags=leakage_flags,
        autofix_info=autofix_info if 'autofix_info' in locals() else None,
        status=final_status,
        attempts=1
    )
    
    # Log canonical summary block (one block that can be screenshot for PR comments)
    # Use detected_interval from evaluate_target_predictability scope (defined at line ~2276)
    summary_interval = detected_interval if 'detected_interval' in locals() else None
    summary_horizon = target_horizon_minutes if 'target_horizon_minutes' in locals() else None
    summary_safe_features = len(safe_columns) if 'safe_columns' in locals() else 0
    summary_leaky_features = leaky_features if 'leaky_features' in locals() else []
    
    _log_canonical_summary(
        target_name=target_name,
        target_column=target_column,
        symbols=symbols,
        time_vals=time_vals,
        interval=summary_interval,
        horizon=summary_horizon,
        rows=len(X) if X is not None else 0,
        features_safe=summary_safe_features,
        features_pruned=actual_pruned_feature_count if 'actual_pruned_feature_count' in locals() else (len(feature_names) if feature_names else 0),
        leak_scan_verdict="PASS" if not summary_leaky_features else "FAIL",
        auto_fix_verdict="SKIPPED" if not should_auto_fix else ("RAN" if autofix_info and autofix_info.modified_configs else "NO_CHANGES"),
        auto_fix_reason=None if should_auto_fix else "overfit_likely; cv_not_suspicious",
        cv_metric=f"{metric_name}={mean_score:.3f}±{std_score:.3f}",
        composite=composite,
        leakage_flag=leakage_flag,
        cohort_path=None  # Will be set by reproducibility tracker
    )
    
    # Legacy summary line (backward compatibility)
    leakage_indicator = f" [{leakage_flag}]" if leakage_flag != "OK" else ""
    logger.debug(f"Legacy summary: {metric_name}={mean_score:.3f}±{std_score:.3f}, "
               f"importance={mean_importance:.2f}, composite={composite:.3f}{leakage_indicator}")
    
    # Store suspicious features in result for summary report
    result.suspicious_features = all_suspicious_features if all_suspicious_features else None
    
    # Track reproducibility: compare to previous target ranking run
    # This runs regardless of which entry point calls this function
    if output_dir and result.mean_score != -999.0:
        try:
            from TRAINING.utils.reproducibility_tracker import ReproducibilityTracker
            
            # Use module-specific directory for reproducibility log
            # output_dir might be: output_dir_YYYYMMDD_HHMMSS/target_rankings/ or just output_dir_YYYYMMDD_HHMMSS
            # We want to store in target_rankings/ subdirectory for this module
            if output_dir.name == 'target_rankings':
                # Already in target_rankings subdirectory
                module_output_dir = output_dir
            elif (output_dir.parent / 'target_rankings').exists():
                # output_dir is parent, use target_rankings subdirectory
                module_output_dir = output_dir.parent / 'target_rankings'
            else:
                # Fallback: use output_dir directly (for standalone runs)
                module_output_dir = output_dir
            
            tracker = ReproducibilityTracker(
                output_dir=module_output_dir,
                search_previous_runs=True  # Search for previous runs in parent directories
            )
            
            # Automated audit-grade reproducibility tracking using RunContext
            try:
                from TRAINING.utils.run_context import RunContext
                
                # Build RunContext from available data
                # Prefer symbols_array (from prepare_cross_sectional_data_for_ranking) over symbols list
                symbols_for_ctx = None
                if 'cohort_context' in locals() and cohort_context:
                    symbols_for_ctx = cohort_context.get('symbols_array')
                    if symbols_for_ctx is None:
                        symbols_for_ctx = cohort_context.get('symbols')
                elif 'symbols_array' in locals():
                    symbols_for_ctx = symbols_array
                elif 'symbols' in locals():
                    symbols_for_ctx = symbols
                
                # Use resolved_config values if available (single source of truth)
                if 'resolved_config' in locals() and resolved_config:
                    purge_minutes_val = resolved_config.purge_minutes
                    embargo_minutes_val = resolved_config.embargo_minutes
                    # Estimate feature lookback from resolved_config
                    if resolved_config.interval_minutes is not None:
                        max_lookback_bars = 288  # 1 day of 5m bars
                        feature_lookback_max = max_lookback_bars * resolved_config.interval_minutes
                    else:
                        feature_lookback_max = None
                elif 'purge_minutes_val' not in locals() or purge_minutes_val is None:
                    # Fallback: compute from purge_time if available
                    if 'purge_time' in locals() and purge_time is not None:
                        try:
                            if hasattr(purge_time, 'total_seconds'):
                                purge_minutes_val = purge_time.total_seconds() / 60.0
                                embargo_minutes_val = purge_minutes_val  # Assume same
                        except Exception:
                            pass
                
                # Estimate feature lookback (conservative: 1 day = 288 bars for 5m data)
                if 'feature_lookback_max' not in locals() or feature_lookback_max is None:
                    feature_lookback_max = None
                    if 'data_interval_minutes' in locals() and data_interval_minutes is not None:
                        max_lookback_bars = 288  # 1 day of 5m bars
                        feature_lookback_max = max_lookback_bars * data_interval_minutes
                
                # Build RunContext
                ctx = RunContext(
                    X=cohort_context.get('X') if 'cohort_context' in locals() and cohort_context else None,
                    y=cohort_context.get('y') if 'cohort_context' in locals() and cohort_context else None,
                    feature_names=feature_names if 'feature_names' in locals() else None,
                    symbols=symbols_for_ctx,
                    time_vals=cohort_context.get('time_vals') if 'cohort_context' in locals() and cohort_context else None,
                    target_column=target_column,
                    target_name=target_name,
                    min_cs=cohort_context.get('min_cs') if 'cohort_context' in locals() and cohort_context else (min_cs if 'min_cs' in locals() else None),
                    max_cs_samples=cohort_context.get('max_cs_samples') if 'cohort_context' in locals() and cohort_context else (max_cs_samples if 'max_cs_samples' in locals() else None),
                    mtf_data=cohort_context.get('mtf_data') if 'cohort_context' in locals() and cohort_context else None,
                    cv_method="purged_kfold",
                    cv_folds=cv_folds if 'cv_folds' in locals() else None,
                    horizon_minutes=target_horizon_minutes if 'target_horizon_minutes' in locals() else None,
                    purge_minutes=purge_minutes_val,
                    fold_timestamps=fold_timestamps if 'fold_timestamps' in locals() else None,
                    feature_lookback_max_minutes=feature_lookback_max,
                    data_interval_minutes=data_interval_minutes if 'data_interval_minutes' in locals() else None,
                    stage="target_ranking",
                    output_dir=output_dir
                )
                # Add view and symbol to RunContext if available
                if 'view' in locals():
                    ctx.view = view
                if 'symbol' in locals() and symbol:
                    ctx.symbol = symbol
                
                # Build metrics dict with regression features
                metrics_dict = {
                    "metric_name": metric_name,
                    "mean_score": result.mean_score,
                    "std_score": result.std_score,
                    "mean_importance": result.mean_importance,
                    "composite_score": result.composite_score,
                    "n_models": result.n_models,
                    "leakage_flag": result.leakage_flag,
                    "task_type": result.task_type.name if hasattr(result.task_type, 'name') else str(result.task_type),
                    # Regression features: feature counts
                    "n_features_pre": features_safe if 'features_safe' in locals() else None,
                    "n_features_post_prune": len(feature_names) if 'feature_names' in locals() and feature_names else None,
                    "features_safe": features_safe if 'features_safe' in locals() else None,
                    "features_final": len(feature_names) if 'feature_names' in locals() and feature_names else None,
                }
                
                # Add pos_rate if available (from y)
                if 'y' in locals() and y is not None:
                    try:
                        import numpy as np
                        if len(y) > 0:
                            pos_count = np.sum(y == 1) if hasattr(y, '__iter__') else 0
                            pos_rate = pos_count / len(y) if len(y) > 0 else None
                            if pos_rate is not None:
                                metrics_dict["pos_rate"] = float(pos_rate)
                    except Exception:
                        pass
                
                # Add view and symbol to RunContext if available (for dual-view target ranking)
                if 'view' in locals():
                    ctx.view = view
                if 'symbol' in locals() and symbol:
                    ctx.symbol = symbol
                
                # Use automated log_run API
                audit_result = tracker.log_run(ctx, metrics_dict)
                
                # Log audit report summary if available
                if audit_result.get("audit_report"):
                    audit_report = audit_result["audit_report"]
                    if audit_report.get("violations"):
                        logger.warning(f"🚨 Audit violations detected: {len(audit_report['violations'])}")
                        for violation in audit_report['violations']:
                            logger.warning(f"  - {violation['message']}")
                    if audit_report.get("warnings"):
                        logger.info(f"⚠️  Audit warnings: {len(audit_report['warnings'])}")
                        for warning in audit_report['warnings']:
                            logger.info(f"  - {warning['message']}")
                
                # Log trend summary if available (already logged by log_run, but include in result)
                if audit_result.get("trend_summary"):
                    trend = audit_result["trend_summary"]
                    # Trend summary is already logged by log_run, but we can add additional context here if needed
                    pass
                
            except ImportError:
                # Fallback to legacy API if RunContext not available
                logger.warning("RunContext not available, falling back to legacy reproducibility tracking")
                from TRAINING.utils.cohort_metadata_extractor import extract_cohort_metadata, format_for_reproducibility_tracker
                
                if 'cohort_context' in locals() and cohort_context:
                    symbols_for_extraction = cohort_context.get('symbols_array') or cohort_context.get('symbols')
                    cohort_metadata = extract_cohort_metadata(
                        X=cohort_context.get('X'),
                        symbols=symbols_for_extraction,
                        time_vals=cohort_context.get('time_vals'),
                        y=cohort_context.get('y'),
                        mtf_data=cohort_context.get('mtf_data'),
                        min_cs=cohort_context.get('min_cs'),
                        max_cs_samples=cohort_context.get('max_cs_samples'),
                        compute_data_fingerprint=True,
                        compute_per_symbol_stats=True
                    )
                else:
                    cohort_metadata = extract_cohort_metadata(
                        symbols=symbols if 'symbols' in locals() else None,
                        mtf_data=mtf_data if 'mtf_data' in locals() else None,
                        min_cs=min_cs if 'min_cs' in locals() else None,
                        max_cs_samples=max_cs_samples if 'max_cs_samples' in locals() else None
                    )
                
                cohort_metrics, cohort_additional_data = format_for_reproducibility_tracker(cohort_metadata)
                metrics_with_cohort = {
                    "metric_name": metric_name,
                    "mean_score": result.mean_score,
                    "std_score": result.std_score,
                    "mean_importance": result.mean_importance,
                    "composite_score": result.composite_score,
                    # Regression features: feature counts
                    "n_features_pre": features_safe if 'features_safe' in locals() else None,
                    "n_features_post_prune": len(feature_names) if 'feature_names' in locals() and feature_names else None,
                    "features_safe": features_safe if 'features_safe' in locals() else None,
                    "features_final": len(feature_names) if 'feature_names' in locals() and feature_names else None,
                    **cohort_metrics
                }
                
                # Add pos_rate if available (from y)
                if 'y' in locals() and y is not None:
                    try:
                        import numpy as np
                        if len(y) > 0:
                            pos_count = np.sum(y == 1) if hasattr(y, '__iter__') else 0
                            pos_rate = pos_count / len(y) if len(y) > 0 else None
                            if pos_rate is not None:
                                metrics_with_cohort["pos_rate"] = float(pos_rate)
                    except Exception:
                        pass
                additional_data_with_cohort = {
                    "n_models": result.n_models,
                    "leakage_flag": result.leakage_flag,
                    "task_type": result.task_type.name if hasattr(result.task_type, 'name') else str(result.task_type),
                    **cohort_additional_data
                }
                
                # Add view and symbol for dual-view target ranking
                if 'view' in locals():
                    additional_data_with_cohort['view'] = view
                if 'symbol' in locals() and symbol:
                    additional_data_with_cohort['symbol'] = symbol
                
                # Add CV details manually (legacy path)
                if 'target_horizon_minutes' in locals() and target_horizon_minutes is not None:
                    additional_data_with_cohort['horizon_minutes'] = target_horizon_minutes
                if 'purge_time' in locals() and purge_time is not None:
                    try:
                        if hasattr(purge_time, 'total_seconds'):
                            # Use purge_minutes_val if available (single source of truth)
                            if 'purge_minutes_val' in locals() and purge_minutes_val is not None:
                                additional_data_with_cohort['purge_minutes'] = purge_minutes_val
                                additional_data_with_cohort['embargo_minutes'] = purge_minutes_val
                            else:
                                purge_minutes_val = purge_time.total_seconds() / 60.0
                                additional_data_with_cohort['purge_minutes'] = purge_minutes_val
                                additional_data_with_cohort['embargo_minutes'] = purge_minutes_val
                    except Exception:
                        pass
                if 'cv_folds' in locals() and cv_folds is not None:
                    additional_data_with_cohort['cv_folds'] = cv_folds
                if 'fold_timestamps' in locals() and fold_timestamps:
                    additional_data_with_cohort['fold_timestamps'] = fold_timestamps
                if 'feature_names' in locals() and feature_names:
                    additional_data_with_cohort['feature_names'] = feature_names
                if 'data_interval_minutes' in locals() and data_interval_minutes is not None:
                    additional_data_with_cohort['data_interval_minutes'] = data_interval_minutes
                    max_lookback_bars = 288
                    additional_data_with_cohort['feature_lookback_max_minutes'] = max_lookback_bars * data_interval_minutes
                
                tracker.log_comparison(
                    stage="target_ranking",
                    item_name=target_name,
                    metrics=metrics_with_cohort,
                    additional_data=additional_data_with_cohort
                )
        except Exception as e:
            logger.warning(f"Reproducibility tracking failed for {target_name}: {e}")
            import traceback
            logger.debug(f"Reproducibility tracking traceback: {traceback.format_exc()}")
    
    return result


