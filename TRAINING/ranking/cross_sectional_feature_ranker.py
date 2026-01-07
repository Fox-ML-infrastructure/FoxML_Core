# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

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
        
        # Load LightGBM config (load_model_config returns hyperparameters directly, like Phase 3)
        try:
            lgb_hyperparams = load_model_config('lightgbm')
            default_configs['lightgbm'] = {
                'n_estimators': lgb_hyperparams.get('n_estimators', 1000),  # Match Phase 3 default
                'max_depth': lgb_hyperparams.get('max_depth', 8),  # Match Phase 3 default
                'learning_rate': lgb_hyperparams.get('learning_rate', 0.03),  # Match Phase 3 default
                'seed': cs_seed,
                'verbosity': -1,
                'n_jobs': 1
            }
        except Exception:
            default_configs['lightgbm'] = {
                'n_estimators': 1000,  # Match Phase 3 default
                'max_depth': 8,  # Match Phase 3 default
                'learning_rate': 0.03,  # Match Phase 3 default
                'seed': cs_seed,
                'verbosity': -1,
                'n_jobs': 1
            }
        
        # Load XGBoost config (load_model_config returns hyperparameters directly, like Phase 3)
        try:
            xgb_hyperparams = load_model_config('xgboost')
            default_configs['xgboost'] = {
                'n_estimators': xgb_hyperparams.get('n_estimators', 1000),  # Match Phase 3 default
                'max_depth': xgb_hyperparams.get('max_depth', 7),  # Match Phase 3 default
                'learning_rate': xgb_hyperparams.get('eta', xgb_hyperparams.get('learning_rate', 0.03)),  # Match Phase 3 default (eta is XGBoost's learning_rate)
                'seed': cs_seed,
                'n_jobs': 1
            }
        except Exception:
            default_configs['xgboost'] = {
                'n_estimators': 1000,  # Match Phase 3 default
                'max_depth': 7,  # Match Phase 3 default
                'learning_rate': 0.03,  # Match Phase 3 default
                'seed': cs_seed,
                'n_jobs': 1
            }
    except Exception:
        # Fallback to hardcoded defaults (matching Phase 3 defaults)
        default_configs = {
            'lightgbm': {
                'n_estimators': 1000,  # Match Phase 3 default
                'max_depth': 8,  # Match Phase 3 default
                'learning_rate': 0.03,  # Match Phase 3 default
                'seed': cs_seed,
                'verbosity': -1,
                'n_jobs': 1
            },
            'xgboost': {
                'n_estimators': 1000,  # Match Phase 3 default
                'max_depth': 7,  # Match Phase 3 default
                'learning_rate': 0.03,  # Match Phase 3 default
                'seed': cs_seed,
                'n_jobs': 1
            }
        }
    
    # Merge with defaults
    config = {**default_configs.get(model_family, {}), **model_config}
    
    # Import shared config cleaner utility
    from TRAINING.common.utils.config_cleaner import clean_config_for_estimator
    
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
                est_cls = LGBMClassifier
            else:
                est_cls = LGBMRegressor
            
            # Clean config to prevent duplicate/unknown param errors
            config = clean_config_for_estimator(est_cls, config, {}, model_family)
            # CRITICAL: Force deterministic mode for reproducibility
            config['deterministic'] = True
            config['force_row_wise'] = True  # Required for deterministic=True
            model = est_cls(**config)
            
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
                est_cls = xgb.XGBClassifier
            else:
                est_cls = xgb.XGBRegressor
            
            # Clean config to prevent duplicate/unknown param errors
            config = clean_config_for_estimator(est_cls, config, {}, model_family)
            model = est_cls(**config)
            
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
    model_families: List[str] = None,
    min_cs: int = None,
    max_cs_samples: int = None,
    normalization: Optional[str] = None,
    model_configs: Optional[Dict[str, Dict]] = None,
    output_dir: Optional[Path] = None,  # Optional output directory for reproducibility tracking
    universe_sig: Optional[str] = None,  # FIX: Thread universe_sig for proper scope tracking
    run_identity: Optional[Any] = None,  # SST RunIdentity for authoritative signatures
) -> pd.Series:
    """
    Compute cross-sectional feature importance using panel models.
    
    # Load defaults from config if not provided
    if model_families is None:
        model_families = ['lightgbm']
    
    if min_cs is None:
        try:
            from CONFIG.config_loader import get_cfg
            min_cs = int(get_cfg("pipeline.data_limits.min_cross_sectional_samples", default=10, config_name="pipeline_config"))
        except Exception:
            min_cs = 10
    
    if max_cs_samples is None:
        try:
            from CONFIG.config_loader import get_cfg
            max_cs_samples = int(get_cfg("pipeline.data_limits.max_cs_samples", default=1000, config_name="pipeline_config"))
        except Exception:
            max_cs_samples = 1000
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
    from TRAINING.ranking.utils.cross_sectional_data import (
        load_mtf_data_for_ranking,
        prepare_cross_sectional_data_for_ranking
    )
    
    # Load data
    mtf_data = load_mtf_data_for_ranking(data_dir, symbols)
    if not mtf_data:
        logger.warning("No data loaded, returning zero importance")
        return pd.Series(0.0, index=candidate_features)
    
    # Check if we have enough symbols BEFORE attempting cross-sectional ranking
    # This prevents the hard-stop error and allows graceful skip
    # Minimum: 3 symbols (hard requirement for cross-sectional analysis)
    # Recommended: 10+ symbols for robust results
    MIN_SYMBOLS_REQUIRED = 3
    RECOMMENDED_SYMBOLS = 10
    
    n_symbols_loaded = len(mtf_data)
    if n_symbols_loaded < MIN_SYMBOLS_REQUIRED:
        logger.warning(
            f"⚠️  Cross-sectional importance SKIPPED: insufficient symbols "
            f"(have {n_symbols_loaded}, need >= {MIN_SYMBOLS_REQUIRED}). "
            f"Returning zero importance. Use SYMBOL_SPECIFIC mode for single-symbol ranking."
        )
        return pd.Series(0.0, index=candidate_features)
    
    # Warn if using fewer than recommended symbols (but proceed)
    if n_symbols_loaded < RECOMMENDED_SYMBOLS:
        logger.warning(
            f"⚠️  Cross-sectional ranking with {n_symbols_loaded} symbols (recommended: >= {RECOMMENDED_SYMBOLS}). "
            f"Results may be less robust with fewer symbols."
        )
    
    # Build panel with candidate features only
    # Note: output_dir is optional, so mode resolution may not persist (but that's OK for this use case)
    X, y, feature_names, symbols_array, time_vals, resolved_data_config = prepare_cross_sectional_data_for_ranking(
        mtf_data, target_column,
        min_cs=min_cs,
        max_cs_samples=max_cs_samples,
        feature_names=candidate_features,  # Only candidate features
        requested_view="CROSS_SECTIONAL",  # This function is always cross-sectional
        output_dir=output_dir
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
    
    # Track reproducibility for cross-sectional feature ranking (if output_dir provided)
    # This is called from feature_selector.py, which may pass output_dir through
    # We'll add tracking here but it's optional since the main tracking happens in feature_selector.py
    # This provides separate tracking for CS ranking specifically
    if output_dir is not None:
        try:
            from TRAINING.orchestration.utils.reproducibility_tracker import ReproducibilityTracker
            from TRAINING.orchestration.utils.run_context import RunContext
            from TRAINING.orchestration.utils.cohort_metadata_extractor import extract_cohort_metadata
            from TRAINING.ranking.utils.leakage_filtering import _extract_horizon, _load_leakage_config
            
            # Use module-specific directory
            module_output_dir = output_dir.parent / 'feature_selections' if (output_dir.parent / 'feature_selections').exists() else output_dir
            
            tracker = ReproducibilityTracker(
                output_dir=module_output_dir,
                search_previous_runs=True
            )
            
            # Extract cohort metadata
            # FIX: Pass universe_sig for proper scope tracking in telemetry
            cohort_metadata = extract_cohort_metadata(
                symbols=symbols,
                mtf_data=None,  # Not available here, but symbols are enough for basic tracking
                min_cs=min_cs,
                max_cs_samples=max_cs_samples,
                universe_sig=universe_sig  # FIX: Thread universe_sig through
            )
            
            # FIX: Extract horizon_minutes from target column for COHORT_AWARE mode
            horizon_minutes_for_ctx = None
            if target_column:
                try:
                    leakage_config = _load_leakage_config()
                    horizon_minutes_for_ctx = _extract_horizon(target_column, leakage_config)
                except Exception:
                    pass
            
            # FIX: Load seed from config for reproducibility tracking (matches feature_selector.py pattern)
            ctx_seed = None
            try:
                from CONFIG.config_loader import get_cfg
                ctx_seed = get_cfg("pipeline.determinism.base_seed", default=42)
            except Exception:
                ctx_seed = 42  # Fallback to default
            
            # Build RunContext
            # FIX: Pass seed for train_seed in ComparisonGroup (required for FEATURE_SELECTION)
            # FIX: Pass min_cs and max_cs_samples for resolved_metadata validation
            ctx = RunContext(
                stage="FEATURE_SELECTION",
                target=target_column,
                target_column=target_column,
                X=X,  # Panel data
                y=y,  # Panel labels
                feature_names=feature_names,
                symbols=symbols_array if 'symbols_array' in locals() else symbols,
                time_vals=time_vals if 'time_vals' in locals() else None,
                horizon_minutes=horizon_minutes_for_ctx,  # FIX: Extract from target column
                purge_minutes=None,
                embargo_minutes=None,
                folds=None,
                fold_timestamps=None,
                data_interval_minutes=None,
                seed=ctx_seed,  # FIX: Pass seed for train_seed requirement
                min_cs=min_cs,  # FIX: Pass min_cs for resolved_metadata
                max_cs_samples=max_cs_samples,  # FIX: Pass max_cs_samples for resolved_metadata
                universe_sig=universe_sig  # FIX: Pass universe_sig for proper scope tracking
            )
            
            # Build metrics dict
            top_cs_score = cs_importance.max() if len(cs_importance) > 0 else 0.0
            mean_cs_score = cs_importance.mean() if len(cs_importance) > 0 else 0.0
            std_cs_score = cs_importance.std() if len(cs_importance) > 0 else 0.0
            
            metrics_dict = {
                "metric_name": "CS Importance Score",
                "auc": mean_cs_score,
                "std_score": std_cs_score,
                "mean_importance": top_cs_score,
                "composite_score": mean_cs_score,
                "n_features_evaluated": len(candidate_features),
                "n_selected": len(candidate_features)  # All candidates are "selected" for CS ranking
            }
            
            # Use automated log_run API (includes trend analysis)
            audit_result = tracker.log_run(
                ctx, metrics_dict,
                run_identity=run_identity,  # SST: Pass through authoritative identity
            )
            
            # Log audit report summary if available
            if audit_result.get("audit_report"):
                audit_report = audit_result["audit_report"]
                if audit_report.get("violations"):
                    logger.warning(f"⚠️  CS Ranking audit violations: {len(audit_report['violations'])}")
                if audit_report.get("warnings"):
                    logger.info(f"ℹ️  CS Ranking audit warnings: {len(audit_report['warnings'])}")
            
            # Log trend summary if available
            if audit_result.get("trend_summary"):
                trend = audit_result["trend_summary"]
                # Trend summary is already logged by log_run
                pass
                
        except ImportError:
            # RunContext not available, skip tracking
            logger.debug("RunContext not available for CS ranking reproducibility tracking")
        except Exception as e:
            logger.debug(f"CS ranking reproducibility tracking failed: {e}")
    
    return cs_importance


def compute_cross_sectional_stability(
    target_column: str,
    cs_importance: pd.Series,
    output_dir: Optional[Path] = None,
    top_k: int = 20,
    universe_sig: Optional[str] = None,
    run_identity: Optional[Any] = None,  # RunIdentity for snapshot storage
) -> Dict[str, Any]:
    """
    Compute stability metrics for cross-sectional feature importance.
    
    This tracks stability across runs using the feature importance stability system,
    providing factor robustness analysis for cross-sectional features.
    
    Args:
        target_column: Target column name
        cs_importance: Current cross-sectional importance Series
        output_dir: Optional output directory for snapshots
        top_k: Number of top features for overlap calculation
        universe_sig: Optional universe identifier (e.g., "ALL", "TOP100")
    
    Returns:
        Dict with stability metrics:
        - mean_overlap: Mean top-K overlap between runs
        - std_overlap: Std dev of overlaps
        - mean_tau: Mean Kendall tau correlation
        - std_tau: Std dev of tau
        - n_snapshots: Number of snapshots analyzed
        - n_comparisons: Number of pairwise comparisons
        - status: "stable", "drifting", or "diverged"
    """
    try:
        from TRAINING.stability.feature_importance.hooks import (
            save_snapshot_from_series_hook
        )
        from TRAINING.stability.feature_importance.analysis import (
            load_snapshots,
            compute_stability_metrics
        )
        from TRAINING.stability.feature_importance.io import get_snapshot_base_dir
    except ImportError:
        logger.debug("Stability tracking system not available, skipping CS stability analysis")
        return {
            "mean_overlap": None,
            "std_overlap": None,
            "mean_tau": None,
            "std_tau": None,
            "n_snapshots": 0,
            "n_comparisons": 0,
            "status": "system_unavailable"
        }
    
    try:
        # Save current snapshot
        method_name = "cross_sectional_panel"
        if universe_sig is None:
            universe_sig = "ALL"  # Default universe ID for cross-sectional
        
        # Use target-first structure for snapshots
        snapshot_base_dir = None
        if output_dir:
            # Find base run directory
            base_output_dir = output_dir
            for _ in range(10):
                if base_output_dir.name == "RESULTS" or (base_output_dir / "targets").exists():
                    break
                if not base_output_dir.parent.exists():
                    break
                base_output_dir = base_output_dir.parent
            
            if base_output_dir.exists():
                target_clean = target_column.replace('/', '_').replace('\\', '_')
                from TRAINING.orchestration.utils.target_first_paths import ensure_target_structure
                ensure_target_structure(base_output_dir, target_clean)
                # Cross-sectional importance is always CROSS_SECTIONAL view
                snapshot_base_dir = get_snapshot_base_dir(
                    output_dir, target=target_column,
                    view="CROSS_SECTIONAL", symbol=None
                )
        else:
            snapshot_base_dir = get_snapshot_base_dir(
                output_dir, target=target_column,
                view="CROSS_SECTIONAL", symbol=None
            )
        
        if snapshot_base_dir:
            # Compute CS identity from passed run_identity
            cs_identity = None
            partial_identity_dict = None  # Fallback: extract signatures from partial identity
            
            if run_identity is not None:
                # Always extract partial identity signatures as fallback
                # These are from FEATURE_SELECTION stage, not TARGET_RANKING
                partial_identity_dict = {
                    "dataset_signature": getattr(run_identity, 'dataset_signature', None),
                    "split_signature": getattr(run_identity, 'split_signature', None),
                    "target_signature": getattr(run_identity, 'target_signature', None),
                    "routing_signature": getattr(run_identity, 'routing_signature', None),
                    "train_seed": getattr(run_identity, 'train_seed', None),
                }
                
                try:
                    from TRAINING.common.utils.fingerprinting import (
                        RunIdentity, compute_hparams_fingerprint,
                        compute_feature_fingerprint_from_specs
                    )
                    # Hparams: cross_sectional_panel has no model-specific params
                    hparams_signature = compute_hparams_fingerprint(
                        model_family="cross_sectional_panel",
                        params={},
                    )
                    # Feature signature from CS importance features (registry-resolved)
                    from TRAINING.common.utils.fingerprinting import resolve_feature_specs_from_registry
                    feature_specs = resolve_feature_specs_from_registry(list(cs_importance.index))
                    feature_signature = compute_feature_fingerprint_from_specs(feature_specs)
                    
                    # Add computed signatures to fallback dict
                    partial_identity_dict["hparams_signature"] = hparams_signature
                    partial_identity_dict["feature_signature"] = feature_signature
                    
                    # Create updated partial and finalize
                    updated_partial = RunIdentity(
                        dataset_signature=run_identity.dataset_signature if hasattr(run_identity, 'dataset_signature') else "",
                        split_signature=run_identity.split_signature if hasattr(run_identity, 'split_signature') else "",
                        target_signature=run_identity.target_signature if hasattr(run_identity, 'target_signature') else "",
                        feature_signature=None,
                        hparams_signature=hparams_signature or "",
                        routing_signature=run_identity.routing_signature if hasattr(run_identity, 'routing_signature') else "",
                        routing_payload=run_identity.routing_payload if hasattr(run_identity, 'routing_payload') else None,
                        train_seed=run_identity.train_seed if hasattr(run_identity, 'train_seed') else None,
                        is_final=False,
                    )
                    cs_identity = updated_partial.finalize(feature_signature)
                except Exception as e:
                    # FIX: Log at WARNING level so failures are visible
                    logger.warning(
                        f"Failed to compute CS identity for cross_sectional_panel snapshot: {e}. "
                        f"Using partial identity signatures as fallback."
                    )
            
            # FIX: Check if identity is actually finalized before passing to snapshot hook
            # Partial identities (is_final=False) or None cannot be used in strict mode
            identity_is_finalized = (
                cs_identity is not None and 
                hasattr(cs_identity, 'is_final') and 
                cs_identity.is_final
            )
            
            # FIX: If identity not finalized but we have partial signatures, pass them
            effective_identity = cs_identity if identity_is_finalized else partial_identity_dict
            
            snapshot_path = save_snapshot_from_series_hook(
                target=target_column,
                method=method_name,
                importance_series=cs_importance,
                universe_sig=universe_sig,
                output_dir=snapshot_base_dir,  # Use target-first structure
                auto_analyze=False,  # We'll analyze manually to get metrics
                run_identity=effective_identity,  # Pass finalized identity or partial dict fallback
                allow_legacy=(not identity_is_finalized and partial_identity_dict is None),
                view="CROSS_SECTIONAL",  # Cross-sectional ranker is always CROSS_SECTIONAL
                symbol=None,  # No symbol for CROSS_SECTIONAL view
            )
        else:
            snapshot_path = None
        
        if snapshot_path is None:
            logger.debug("Failed to save CS importance snapshot")
            return {
                "mean_overlap": None,
                "std_overlap": None,
                "mean_tau": None,
                "std_tau": None,
                "n_snapshots": 0,
                "n_comparisons": 0,
                "status": "snapshot_failed"
            }
        
        # Load all snapshots (including the one we just saved)
        snapshots = load_snapshots(snapshot_base_dir, target_column, method_name)
        
        if len(snapshots) < 2:
            # First or second run - no comparison yet
            return {
                "mean_overlap": None,
                "std_overlap": None,
                "mean_tau": None,
                "std_tau": None,
                "n_snapshots": len(snapshots),
                "n_comparisons": 0,
                "status": "insufficient_snapshots"
            }
        
        # Compute stability metrics
        metrics = compute_stability_metrics(snapshots, top_k=top_k)
        
        # Determine status based on thresholds
        # Cross-sectional features should be more stable (global factors)
        min_overlap = 0.75  # Stricter than per-symbol (0.7)
        min_tau = 0.65  # Stricter than per-symbol (0.6)
        
        status = "stable"
        mean_overlap = metrics.get('mean_overlap')
        mean_tau = metrics.get('mean_tau')
        
        # Check overlap (primary metric)
        if mean_overlap is not None and not (isinstance(mean_overlap, float) and np.isnan(mean_overlap)):
            if mean_overlap < min_overlap * 0.8:  # Diverged threshold
                status = "diverged"
            elif mean_overlap < min_overlap:  # Drifting threshold
                status = "drifting"
        
        # Check tau (secondary metric, only if overlap is OK)
        if status == "stable" and mean_tau is not None and not (isinstance(mean_tau, float) and np.isnan(mean_tau)):
            if mean_tau < min_tau * 0.8:
                status = "diverged"
            elif mean_tau < min_tau:
                status = "drifting"
        
        return {
            "mean_overlap": metrics['mean_overlap'],
            "std_overlap": metrics['std_overlap'],
            "mean_tau": metrics['mean_tau'],
            "std_tau": metrics['std_tau'],
            "n_snapshots": metrics['n_snapshots'],
            "n_comparisons": metrics['n_comparisons'],
            "status": status
        }
        
    except Exception as e:
        logger.debug(f"CS stability analysis failed (non-critical): {e}")
        return {
            "mean_overlap": None,
            "std_overlap": None,
            "mean_tau": None,
            "std_tau": None,
            "n_snapshots": 0,
            "n_comparisons": 0,
            "status": "analysis_failed"
        }


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

