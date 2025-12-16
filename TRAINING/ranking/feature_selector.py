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
Feature Selection Module

Extracted from SCRIPTS/multi_model_feature_selection.py to enable integration
into the training pipeline. All leakage-free behavior is preserved by
reusing the original functions.
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import pandas as pd
import warnings

# Add project root to path for imports
# TRAINING/ranking/feature_selector.py -> parents[2] = repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Import original functions to preserve leakage-free behavior
from TRAINING.ranking.multi_model_feature_selection import (
    ImportanceResult as FeatureImportanceResult,
    process_single_symbol as _process_single_symbol,
    aggregate_multi_model_importance as _aggregate_multi_model_importance,
    load_multi_model_config as _load_multi_model_config,
    save_multi_model_results as _save_multi_model_results
)

# Import shared ranking harness for unified evaluation contract
from TRAINING.ranking.shared_ranking_harness import RankingHarness

# Import new config system (optional - for backward compatibility)
try:
    from CONFIG.config_builder import build_feature_selection_config
    from CONFIG.config_schemas import ExperimentConfig, FeatureSelectionConfig
    _NEW_CONFIG_AVAILABLE = True
except ImportError:
    _NEW_CONFIG_AVAILABLE = False
    # Logger not yet initialized, will be set up below
    pass

# Suppress expected warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='X does not have valid feature names')

# Import parallel execution utilities
try:
    from TRAINING.common.parallel_exec import execute_parallel, get_max_workers
    _PARALLEL_AVAILABLE = True
except ImportError:
    _PARALLEL_AVAILABLE = False

logger = logging.getLogger(__name__)


def load_multi_model_config(config_path: Path = None) -> Dict[str, Any]:
    """
    Load multi-model feature selection configuration.
    
    Args:
        config_path: Optional path to config file (default: CONFIG/multi_model_feature_selection.yaml)
    
    Returns:
        Config dictionary
    """
    return _load_multi_model_config(config_path)


def select_features_for_target(
    target_column: str,
    symbols: List[str],
    data_dir: Path,
    model_families_config: Dict[str, Dict[str, Any]] = None,
    multi_model_config: Dict[str, Any] = None,
    max_samples_per_symbol: Optional[int] = None,  # Load from config if None
    top_n: Optional[int] = None,
    output_dir: Path = None,
    feature_selection_config: Optional['FeatureSelectionConfig'] = None,  # New typed config (optional)
    explicit_interval: Optional[Union[int, str]] = None,  # Optional explicit interval (e.g., "5m" or 5)
    experiment_config: Optional[Any] = None,  # Optional ExperimentConfig (for data.bar_interval)
    view: str = "CROSS_SECTIONAL",  # "CROSS_SECTIONAL" or "SYMBOL_SPECIFIC" - must match target ranking view
    symbol: Optional[str] = None  # Required for SYMBOL_SPECIFIC view
) -> Tuple[List[str], pd.DataFrame]:
    """
    Select top features for a target using multi-model consensus.
    
    This function processes all symbols, aggregates feature importance across
    model families, and returns the top N features. All leakage-free behavior
    is preserved (PurgedTimeSeriesSplit, leakage filtering, etc.).
    
    Args:
        target_column: Target column name
        symbols: List of symbols to process
        data_dir: Directory containing symbol data
        model_families_config: Optional model families config (overrides multi_model_config) [LEGACY]
        multi_model_config: Optional multi-model config dict [LEGACY]
        max_samples_per_symbol: Maximum samples per symbol
        top_n: Number of top features to return
        output_dir: Optional output directory for results
        feature_selection_config: Optional FeatureSelectionConfig object [NEW - preferred]
    
    Returns:
        Tuple of (selected_feature_names, importance_dataframe)
    """
    # Load max_samples_per_symbol from config if not provided
    if max_samples_per_symbol is None:
        try:
            from CONFIG.config_loader import get_cfg
            max_samples_per_symbol = int(get_cfg("pipeline.data_limits.default_max_samples_feature_selection", default=50000, config_name="pipeline_config"))
        except Exception:
            max_samples_per_symbol = 50000
    
    # NEW: Use typed config if provided
    # Note: explicit_interval can be passed directly or extracted from experiment config
    if feature_selection_config is not None and _NEW_CONFIG_AVAILABLE:
        # Extract values from typed config
        model_families_config = feature_selection_config.model_families
        aggregation_config = feature_selection_config.aggregation
        if top_n is None:
            top_n = feature_selection_config.top_n
        if feature_selection_config.max_samples_per_symbol:
            max_samples_per_symbol = feature_selection_config.max_samples_per_symbol
        # Use target/symbols/data_dir from config if available
        if feature_selection_config.target:
            target_column = feature_selection_config.target
        if feature_selection_config.symbols:
            symbols = feature_selection_config.symbols
        if feature_selection_config.data_dir:
            data_dir = feature_selection_config.data_dir
        # Extract interval if available (from experiment config that built this)
        # Note: FeatureSelectionConfig doesn't have interval, but ExperimentConfig does
        # We'll check if there's an experiment_config attribute or pass it separately
    else:
        # LEGACY: Load config if not provided
        if multi_model_config is None:
            multi_model_config = load_multi_model_config()
        
        # Use model_families_config if provided, otherwise use from multi_model_config
        if model_families_config is None:
            if multi_model_config and 'model_families' in multi_model_config:
                model_families_config = multi_model_config['model_families']
            else:
                raise ValueError("Must provide either model_families_config or multi_model_config with model_families")
        
        aggregation_config = multi_model_config.get('aggregation', {}) if multi_model_config else {}
    
    # Validate view and symbol parameters
    if view == "SYMBOL_SPECIFIC" and symbol is None:
        raise ValueError(f"symbol parameter required for SYMBOL_SPECIFIC view")
    if view == "CROSS_SECTIONAL" and symbol is not None:
        logger.warning(f"symbol={symbol} provided but view=CROSS_SECTIONAL, ignoring symbol")
        symbol = None
    
    # Filter symbols based on view
    symbols_to_process = symbols
    if view == "SYMBOL_SPECIFIC" and symbol:
        symbols_to_process = [symbol]
        logger.info(f"SYMBOL_SPECIFIC view: Processing only symbol {symbol}")
    elif view == "LOSO" and symbol:
        # LOSO: train on all symbols except symbol, validate on symbol
        symbols_to_process = [s for s in symbols if s != symbol]
        logger.info(f"LOSO view: Training on {len(symbols_to_process)} symbols, validating on {symbol}")
    else:
        logger.info(f"CROSS_SECTIONAL view: Processing {len(symbols_to_process)} symbols")
    
    logger.info(f"Selecting features for target: {target_column} (view={view})")
    logger.info(f"Model families: {', '.join([f for f, cfg in model_families_config.items() if cfg.get('enabled', False)])}")
    
    # NEW: Use shared ranking harness for both CROSS_SECTIONAL and SYMBOL_SPECIFIC views
    # This reuses the same split policy, model runner, metrics, and telemetry as target ranking
    # Both views use the same evaluation contract, just with different data preparation
    use_shared_harness = (view == "CROSS_SECTIONAL" or view == "SYMBOL_SPECIFIC")
    
    # Initialize lookback cap enforcement results at function scope for telemetry tracking
    pre_cap_result = None
    post_cap_result = None
    
    # Load min_cs and max_cs_samples from config if not provided (for shared harness)
    if use_shared_harness:
        # Load defaults from config (same as target ranking)
        harness_min_cs = None
        harness_max_cs_samples = None
        try:
            from CONFIG.config_loader import get_cfg
            if experiment_config:
                try:
                    import yaml
                    exp_name = experiment_config.name
                    exp_file = Path("CONFIG/experiments") / f"{exp_name}.yaml"
                    if exp_file.exists():
                        with open(exp_file, 'r') as f:
                            exp_yaml = yaml.safe_load(f) or {}
                        exp_data = exp_yaml.get('data', {})
                        harness_min_cs = exp_data.get('min_cs')
                        harness_max_cs_samples = exp_data.get('max_cs_samples')
                except Exception:
                    pass
            
            if harness_min_cs is None:
                harness_min_cs = int(get_cfg("pipeline.data_limits.min_cross_sectional_samples", default=10, config_name="pipeline_config"))
            if harness_max_cs_samples is None:
                harness_max_cs_samples = int(get_cfg("pipeline.data_limits.max_cs_samples", default=1000, config_name="pipeline_config"))
        except Exception:
            harness_min_cs = 10
            harness_max_cs_samples = 1000
    
    if use_shared_harness:
        logger.info("🔧 Using shared ranking harness (same evaluation contract as target ranking)")
        try:
            # Extract model family names from config
            model_families_list = [f for f, cfg in model_families_config.items() if cfg.get('enabled', False)]
            
            all_results = []
            all_family_statuses = []
            
            # For SYMBOL_SPECIFIC view, process each symbol separately (same as target ranking)
            # For CROSS_SECTIONAL view, process all symbols together
            if view == "SYMBOL_SPECIFIC":
                # Process each symbol separately with shared harness (maintains view differences)
                for symbol_to_process in symbols_to_process:
                    logger.info(f"Processing {symbol_to_process} with shared harness (SYMBOL_SPECIFIC view)...")
                    
                    # Create shared harness for this symbol
                    harness = RankingHarness(
                        job_type="rank_features",
                        target_column=target_column,
                        symbols=[symbol_to_process],  # Single symbol for SYMBOL_SPECIFIC
                        data_dir=data_dir,
                        model_families=model_families_list,
                        multi_model_config=multi_model_config,
                        output_dir=output_dir,
                        view=view,
                        symbol=symbol_to_process,  # Required for SYMBOL_SPECIFIC
                        explicit_interval=explicit_interval,
                        experiment_config=experiment_config,
                        min_cs=1,  # SYMBOL_SPECIFIC uses min_cs=1
                        max_cs_samples=harness_max_cs_samples,
                        max_rows_per_symbol=max_samples_per_symbol
                    )
                    
                    # Build panel data for this symbol (includes all cleaning checks and target-conditional exclusions)
                    # Note: target-conditional exclusions are saved automatically by build_panel if output_dir is set
                    # FIX: Make unpack tolerant to signature changes
                    build_result = harness.build_panel(
                        target_column=target_column,
                        target_name=target_column,  # Use target_column as target_name for exclusions
                        feature_names=None
                    )
                    # FIX: Unpack with tolerance for signature changes, but log what we got
                    # This prevents silently masking real breakage (signature changes)
                    actual_len = len(build_result)
                    logger.debug(f"build_panel returned {actual_len} values: {[type(x).__name__ for x in build_result]}")
                    
                    if actual_len >= 8:
                        X, y, feature_names, symbols_array, time_vals, mtf_data, detected_interval, resolved_config = build_result[:8]
                        if actual_len > 8:
                            logger.warning(f"build_panel returned {actual_len} values (expected 6-8), using first 8. Extra: {build_result[8:]}")
                    elif actual_len >= 6:
                        # Fallback for older signature (6 values)
                        X, y, feature_names, symbols_array, time_vals, mtf_data = build_result[:6]
                        detected_interval = build_result[6] if actual_len > 6 else 5.0
                        resolved_config = build_result[7] if actual_len > 7 else None
                        logger.debug(f"build_panel returned {actual_len} values (legacy signature)")
                    else:
                        raise ValueError(f"build_panel returned {actual_len} values, expected at least 6. Got: {[type(x).__name__ for x in build_result]}")
                    
                    if X is None or y is None:
                        logger.warning(f"Failed to build panel data for {symbol_to_process}, skipping")
                        continue
                    
                    # Sanitize and canonicalize dtypes
                    X, feature_names = harness.sanitize_and_canonicalize_dtypes(X, feature_names)
                    
                    # Apply all cleaning and audit checks
                    X_cleaned, y_cleaned, feature_names_cleaned, resolved_config_updated, success = harness.apply_cleaning_and_audit_checks(
                        X=X, y=y, feature_names=feature_names, target_column=target_column,
                        resolved_config=resolved_config, detected_interval=detected_interval, task_type=None
                    )
                    
                    if not success:
                        logger.warning(f"Cleaning and audit checks failed for {symbol_to_process}, skipping")
                        continue
                    
                    X = X_cleaned
                    y = y_cleaned
                    feature_names = feature_names_cleaned
                    resolved_config = resolved_config_updated
                    
                    # CRITICAL: Pre-selection lookback cap enforcement (FS_PRE)
                    # Apply lookback cap BEFORE running importance producers
                    # Note: pre_cap_result is initialized at function scope for telemetry
                    # This prevents selector from even seeing unsafe features (faster + safer)
                    from TRAINING.utils.lookback_cap_enforcement import apply_lookback_cap
                    from CONFIG.config_loader import get_cfg
                    from TRAINING.common.feature_registry import get_registry
                    
                    # Load lookback cap and policy from config
                    lookback_cap = None
                    try:
                        cap_raw = get_cfg("safety.leakage_detection.lookback_budget_minutes", default="auto", config_name="safety_config")
                        if cap_raw != "auto" and isinstance(cap_raw, (int, float)):
                            lookback_cap = float(cap_raw)
                    except Exception:
                        pass
                    
                    policy = "strict"
                    try:
                        policy = get_cfg("safety.leakage_detection.policy", default="strict", config_name="safety_config")
                    except Exception:
                        pass
                    
                    log_mode = "summary"
                    try:
                        log_mode = get_cfg("safety.leakage_detection.log_mode", default="summary", config_name="safety_config")
                    except Exception:
                        pass
                    
                    if lookback_cap is not None and feature_names:
                        try:
                            registry = get_registry()
                        except Exception:
                            registry = None
                        
                        pre_cap_result = apply_lookback_cap(
                            features=feature_names,
                            interval_minutes=detected_interval,
                            cap_minutes=lookback_cap,
                            policy=policy,
                            stage=f"FS_PRE_{view}_{symbol_to_process}" if view == "SYMBOL_SPECIFIC" else f"FS_PRE_{view}",
                            registry=registry,
                            feature_time_meta_map=resolved_config.feature_time_meta_map if resolved_config and hasattr(resolved_config, 'feature_time_meta_map') else None,
                            base_interval_minutes=resolved_config.base_interval_minutes if resolved_config else None,
                            log_mode=log_mode
                        )
                        
                        # CRITICAL: Convert to EnforcedFeatureSet (SST contract)
                        enforced_fs_pre = pre_cap_result.to_enforced_set(
                            stage=f"FS_PRE_{view}_{symbol_to_process}" if view == "SYMBOL_SPECIFIC" else f"FS_PRE_{view}",
                            cap_minutes=lookback_cap
                        )
                        
                        # CRITICAL: Slice X immediately using enforced.features (no rediscovery)
                        # The enforced.features list IS the authoritative order - X columns must match it
                        feature_indices = [i for i, f in enumerate(feature_names_cleaned) if f in enforced_fs_pre.features]
                        if feature_indices and len(feature_indices) == len(enforced_fs_pre.features):
                            X = X[:, feature_indices]
                            feature_names = enforced_fs_pre.features.copy()  # Use enforced.features (the truth)
                        else:
                            logger.warning(
                                f"FS_PRE: Index mismatch for {symbol_to_process}. "
                                f"Expected {len(enforced_fs_pre.features)} features, got {len(feature_indices)} indices."
                            )
                            if not feature_indices:
                                logger.warning(f"All features quarantined for {symbol_to_process}, skipping")
                                continue
                            # Fallback: use available indices
                            X = X[:, feature_indices]
                            feature_names = [feature_names_cleaned[i] for i in feature_indices]
                        
                        # CRITICAL: Boundary assertion - validate feature_names matches FS_PRE EnforcedFeatureSet
                        from TRAINING.utils.lookback_policy import assert_featureset_fingerprint
                        try:
                            assert_featureset_fingerprint(
                                label=f"FS_PRE_{view}_{symbol_to_process}" if view == "SYMBOL_SPECIFIC" else f"FS_PRE_{view}",
                                expected=enforced_fs_pre,
                                actual_features=feature_names,
                                logger_instance=logger,
                                allow_reorder=False  # Strict order check
                            )
                        except RuntimeError as e:
                            # Log but don't fail - this is a validation check
                            logger.error(f"FS_PRE assertion failed: {e}")
                            # Fix it: use enforced.features (the truth)
                            feature_names = enforced_fs_pre.features.copy()
                            logger.info(f"Fixed: Updated feature_names to match enforced_fs_pre.features")
                        
                        # Update resolved_config with new lookback max
                        if resolved_config:
                            resolved_config.feature_lookback_max_minutes = enforced_fs_pre.actual_max_minutes
                            # Store EnforcedFeatureSet for downstream use
                            resolved_config._fs_pre_enforced = enforced_fs_pre
                    elif feature_names:
                        # No cap set, but still validate canonical map consistency
                        logger.debug(f"FS_PRE: No lookback cap set, skipping cap enforcement (view={view}, symbol={symbol_to_process})")
                    
                    # Extract horizon and create split policy
                    from TRAINING.utils.leakage_filtering import _extract_horizon, _load_leakage_config
                    leakage_config = _load_leakage_config()
                    horizon_minutes = _extract_horizon(target_column, leakage_config) if target_column else None
                    data_interval_minutes = detected_interval
                    
                    cv_splitter = harness.split_policy(
                        time_vals=time_vals, groups=None,
                        horizon_minutes=horizon_minutes, data_interval_minutes=data_interval_minutes
                    )
                    
                    # Run importance producers
                    model_metrics, model_scores, mean_importance, suspicious_features, \
                    all_feature_importances, fold_timestamps, perfect_correlation_models = harness.run_importance_producers(
                        X=X, y=y, feature_names=feature_names, time_vals=time_vals,
                        task_type=None, resolved_config=resolved_config
                    )
                    
                    # Save stability snapshots for each model family (same as target ranking)
                    # CRITICAL: Per-model-family snapshots ensure stability is computed within same family
                    # Per-symbol snapshots for SYMBOL_SPECIFIC view
                    if all_feature_importances and output_dir:
                        try:
                            from TRAINING.stability.feature_importance import save_snapshot_hook
                            # Build REPRODUCIBILITY path for snapshots (same structure as feature importances)
                            target_name_clean = target_column.replace('/', '_').replace('\\', '_')
                            # Determine base output directory (walk up from REPRODUCIBILITY/FEATURE_SELECTION structure)
                            base_output_dir = output_dir
                            while base_output_dir.name in ["CROSS_SECTIONAL", "SYMBOL_SPECIFIC", "FEATURE_SELECTION", "TARGET_RANKING", "REPRODUCIBILITY", "feature_selections", "target_rankings"]:
                                base_output_dir = base_output_dir.parent
                                if not base_output_dir.parent.exists() or base_output_dir.name == "RESULTS":
                                    break
                            
                            # Determine FEATURE_SELECTION base (avoid nested REPRODUCIBILITY)
                            # output_dir is already at: REPRODUCIBILITY/FEATURE_SELECTION/CROSS_SECTIONAL/{target}/
                            # Walk up to FEATURE_SELECTION level
                            repro_base = base_output_dir
                            while repro_base.name not in ["FEATURE_SELECTION", "TARGET_RANKING"]:
                                repro_base = repro_base.parent
                                if not repro_base.parent.exists() or repro_base.name in ["RESULTS", "REPRODUCIBILITY"]:
                                    break
                            
                            # If we hit REPRODUCIBILITY, go back down to FEATURE_SELECTION
                            if repro_base.name == "REPRODUCIBILITY":
                                repro_base = repro_base / "FEATURE_SELECTION"
                            elif repro_base.name != "FEATURE_SELECTION":
                                # We're at run level, construct path
                                repro_base = repro_base / "REPRODUCIBILITY" / "FEATURE_SELECTION"
                            
                            if view == "SYMBOL_SPECIFIC" and symbol_to_process:
                                snapshot_base_dir = repro_base / view / target_name_clean / f"symbol={symbol_to_process}"
                            else:
                                snapshot_base_dir = repro_base / view / target_name_clean
                            
                            for model_family, importance_dict in all_feature_importances.items():
                                if importance_dict:
                                    # FIX: Ensure method name is model_family (e.g., "lightgbm", "ridge")
                                    # NOT importance_method (e.g., "native") - stability must be per-family
                                    save_snapshot_hook(
                                        target_name=target_column,
                                        method=model_family,  # Use model_family as method identifier
                                        importance_dict=importance_dict,
                                        universe_id=view,  # Use view parameter
                                        output_dir=snapshot_base_dir,  # Save in REPRODUCIBILITY structure
                                        auto_analyze=None,  # Load from config
                                    )
                        except Exception as e:
                            logger.debug(f"Stability snapshot save failed for {symbol_to_process} (non-critical): {e}")
                    
                    # Convert to ImportanceResult format (per-symbol for SYMBOL_SPECIFIC)
                    for model_family in model_families_list:
                        if model_family in all_feature_importances:
                            importance_dict = all_feature_importances[model_family]
                            importance_series = pd.Series(importance_dict)
                            result = FeatureImportanceResult(
                                model_family=model_family,
                                symbol=symbol_to_process,  # Per-symbol for SYMBOL_SPECIFIC
                                importance_scores=importance_series,
                                method="native",
                                train_score=model_scores.get(model_family, 0.0)
                            )
                            all_results.append(result)
                            all_family_statuses.append({
                                "status": "success",
                                "family": model_family,
                                "symbol": symbol_to_process,
                                "score": float(model_scores.get(model_family, 0.0)),
                                "top_feature": importance_series.idxmax() if len(importance_series) > 0 else None,
                                "top_feature_score": float(importance_series.max()) if len(importance_series) > 0 else None,
                                "error": None,
                                "error_type": None
                            })
                    
                    logger.info(f"✅ {symbol_to_process}: {len([r for r in all_results if r.symbol == symbol_to_process])} model results")
            else:
                # CROSS_SECTIONAL: process all symbols together
                harness = RankingHarness(
                    job_type="rank_features",
                    target_column=target_column,
                    symbols=symbols_to_process,
                    data_dir=data_dir,
                    model_families=model_families_list,
                    multi_model_config=multi_model_config,
                    output_dir=output_dir,
                    view=view,
                    symbol=None,  # CROSS_SECTIONAL doesn't use symbol
                    explicit_interval=explicit_interval,
                    experiment_config=experiment_config,
                    min_cs=harness_min_cs,
                    max_cs_samples=harness_max_cs_samples,
                    max_rows_per_symbol=max_samples_per_symbol
                )
                
                # Build panel data using same logic as target ranking (includes target-conditional exclusions)
                # FIX: Make unpack tolerant to signature changes (use *rest to catch extra values)
                build_result = harness.build_panel(
                    target_column=target_column,
                    target_name=target_column,  # Use target_column as target_name for exclusions
                    feature_names=None  # Will filter automatically
                )
                # FIX: Unpack with tolerance for signature changes, but log what we got
                actual_len = len(build_result)
                logger.debug(f"build_panel returned {actual_len} values: {[type(x).__name__ for x in build_result]}")
                
                if actual_len >= 8:
                    X, y, feature_names, symbols_array, time_vals, mtf_data, detected_interval, resolved_config = build_result[:8]
                    if actual_len > 8:
                        logger.warning(f"build_panel returned {actual_len} values (expected 6-8), using first 8. Extra: {build_result[8:]}")
                elif actual_len >= 6:
                    # Fallback for older signature (6 values)
                    X, y, feature_names, symbols_array, time_vals, mtf_data = build_result[:6]
                    detected_interval = build_result[6] if actual_len > 6 else 5.0
                    resolved_config = build_result[7] if actual_len > 7 else None
                    logger.debug(f"build_panel returned {actual_len} values (legacy signature)")
                else:
                    raise ValueError(f"build_panel returned {actual_len} values, expected at least 6. Got: {[type(x).__name__ for x in build_result]}")
                
                if X is None or y is None:
                    logger.warning("Failed to build panel data with shared harness, falling back to per-symbol processing")
                    use_shared_harness = False
                else:
                    # Sanitize and canonicalize dtypes (prevents CatBoost object column errors)
                    X, feature_names = harness.sanitize_and_canonicalize_dtypes(X, feature_names)
                    
                    # Apply all cleaning and audit checks (same as target ranking)
                    # This includes: leak scan, duplicate checks, target validation, final gatekeeper
                    X_cleaned, y_cleaned, feature_names_cleaned, resolved_config_updated, success = harness.apply_cleaning_and_audit_checks(
                        X=X,
                        y=y,
                        feature_names=feature_names,
                        target_column=target_column,
                        resolved_config=resolved_config,
                        detected_interval=detected_interval,
                        task_type=None  # Will be inferred
                    )
                    
                    if not success:
                        logger.warning("Cleaning and audit checks failed, falling back to per-symbol processing")
                        use_shared_harness = False
                    else:
                        X = X_cleaned
                        y = y_cleaned
                        feature_names = feature_names_cleaned
                        resolved_config = resolved_config_updated
                        
                        # CRITICAL: Pre-selection lookback cap enforcement (FS_PRE)
                        # Apply lookback cap BEFORE running importance producers
                        # Note: pre_cap_result is initialized at function scope for telemetry
                        from TRAINING.utils.lookback_cap_enforcement import apply_lookback_cap
                        from CONFIG.config_loader import get_cfg
                        from TRAINING.common.feature_registry import get_registry
                        
                        # Load lookback cap and policy from config
                        lookback_cap = None
                        try:
                            cap_raw = get_cfg("safety.leakage_detection.lookback_budget_minutes", default="auto", config_name="safety_config")
                            if cap_raw != "auto" and isinstance(cap_raw, (int, float)):
                                lookback_cap = float(cap_raw)
                        except Exception:
                            pass
                        
                        policy = "strict"
                        try:
                            policy = get_cfg("safety.leakage_detection.policy", default="strict", config_name="safety_config")
                        except Exception:
                            pass
                        
                        log_mode = "summary"
                        try:
                            log_mode = get_cfg("safety.leakage_detection.log_mode", default="summary", config_name="safety_config")
                        except Exception:
                            pass
                        
                        if lookback_cap is not None and feature_names:
                            try:
                                registry = get_registry()
                            except Exception:
                                registry = None
                            
                            pre_cap_result = apply_lookback_cap(
                                features=feature_names,
                                interval_minutes=detected_interval,
                                cap_minutes=lookback_cap,
                                policy=policy,
                                stage=f"FS_PRE_{view}",
                                registry=registry,
                                feature_time_meta_map=resolved_config.feature_time_meta_map if resolved_config and hasattr(resolved_config, 'feature_time_meta_map') else None,
                                base_interval_minutes=resolved_config.base_interval_minutes if resolved_config else None,
                                log_mode=log_mode
                            )
                            
                            # CRITICAL: Convert to EnforcedFeatureSet (SST contract)
                            enforced_fs_pre = pre_cap_result.to_enforced_set(
                                stage=f"FS_PRE_{view}",
                                cap_minutes=lookback_cap
                            )
                            
                            # CRITICAL: Slice X immediately using enforced.features (no rediscovery)
                            # The enforced.features list IS the authoritative order - X columns must match it
                            feature_indices = [i for i, f in enumerate(feature_names_cleaned) if f in enforced_fs_pre.features]
                            if feature_indices and len(feature_indices) == len(enforced_fs_pre.features):
                                X = X[:, feature_indices]
                                feature_names = enforced_fs_pre.features.copy()  # Use enforced.features (the truth)
                            else:
                                logger.warning(
                                    f"FS_PRE: Index mismatch. "
                                    f"Expected {len(enforced_fs_pre.features)} features, got {len(feature_indices)} indices."
                                )
                                if not feature_indices:
                                    logger.warning("All features quarantined, skipping")
                                    use_shared_harness = False
                                    # Fall back to per-symbol processing (flag set, will skip rest of shared harness path)
                                else:
                                    # Fallback: use available indices
                                    X = X[:, feature_indices]
                                    feature_names = [feature_names_cleaned[i] for i in feature_indices]
                            
                            # CRITICAL: Boundary assertion - validate feature_names matches FS_PRE EnforcedFeatureSet
                            from TRAINING.utils.lookback_policy import assert_featureset_fingerprint
                            try:
                                assert_featureset_fingerprint(
                                    label=f"FS_PRE_{view}",
                                    expected=enforced_fs_pre,
                                    actual_features=feature_names,
                                    logger_instance=logger,
                                    allow_reorder=False  # Strict order check
                                )
                            except RuntimeError as e:
                                # Log but don't fail - this is a validation check
                                logger.error(f"FS_PRE assertion failed: {e}")
                                # Fix it: use enforced.features (the truth)
                                feature_names = enforced_fs_pre.features.copy()
                                logger.info(f"Fixed: Updated feature_names to match enforced_fs_pre.features")
                            
                            # Store EnforcedFeatureSet for downstream use
                            if resolved_config:
                                resolved_config._fs_pre_enforced = enforced_fs_pre
                            
                            # Update resolved_config with new lookback max
                            if resolved_config:
                                resolved_config.feature_lookback_max_minutes = pre_cap_result.actual_max_lookback
                        elif feature_names:
                            # No cap set, but still validate canonical map consistency
                            logger.debug(f"FS_PRE: No lookback cap set, skipping cap enforcement (view={view})")
                        
                        # Extract horizon for split policy
                        from TRAINING.utils.leakage_filtering import _extract_horizon, _load_leakage_config
                        leakage_config = _load_leakage_config()
                        horizon_minutes = _extract_horizon(target_column, leakage_config) if target_column else None
                        
                        # Use detected_interval from build_panel
                        data_interval_minutes = detected_interval
                        
                        # Create split policy (same as target ranking)
                        cv_splitter = harness.split_policy(
                            time_vals=time_vals,
                            groups=None,
                            horizon_minutes=horizon_minutes,
                            data_interval_minutes=data_interval_minutes
                        )
                        
                        # Run importance producers using same harness as target ranking
                        model_metrics, model_scores, mean_importance, suspicious_features, \
                        all_feature_importances, fold_timestamps, perfect_correlation_models = harness.run_importance_producers(
                            X=X,
                            y=y,
                            feature_names=feature_names,
                            time_vals=time_vals,
                            task_type=None,  # Will be inferred
                            resolved_config=resolved_config  # Use resolved_config from build_panel
                        )
                    
                        # Convert to ImportanceResult format for aggregation
                        for model_family in model_families_list:
                            if model_family in all_feature_importances:
                                importance_dict = all_feature_importances[model_family]
                                importance_series = pd.Series(importance_dict)
                                # For cross-sectional, we don't have per-symbol results, so use "ALL" as symbol
                                result = FeatureImportanceResult(
                                    model_family=model_family,
                                    symbol="ALL",  # Cross-sectional uses all symbols
                                    importance_scores=importance_series,
                                    method="native",  # Will be determined from config
                                    train_score=model_scores.get(model_family, 0.0)
                                )
                                all_results.append(result)
                                all_family_statuses.append({
                                    "status": "success",
                                    "family": model_family,
                                    "symbol": "ALL",
                                    "score": float(model_scores.get(model_family, 0.0)),
                                    "top_feature": importance_series.idxmax() if len(importance_series) > 0 else None,
                                    "top_feature_score": float(importance_series.max()) if len(importance_series) > 0 else None,
                                    "error": None,
                                    "error_type": None
                                })
                        
                        # Create RunContext for reproducibility tracking
                        # FIX: Extract purge/embargo from resolved_config
                        purge_minutes = resolved_config.purge_minutes if resolved_config else None
                        embargo_minutes = resolved_config.embargo_minutes if resolved_config else None
                        
                        ctx = harness.create_run_context(
                            X=X,
                            y=y,
                            feature_names=feature_names,
                            symbols_array=symbols_array,
                            time_vals=time_vals,
                            cv_splitter=cv_splitter,
                            horizon_minutes=horizon_minutes,
                            purge_minutes=purge_minutes,
                            embargo_minutes=embargo_minutes,
                            data_interval_minutes=data_interval_minutes
                        )
                        
                        logger.info(f"✅ Shared harness completed: {len(all_results)} model results")
                
        except Exception as e:
            logger.warning(f"Shared harness failed: {e}, falling back to per-symbol processing", exc_info=True)
            use_shared_harness = False
            all_results = []
            all_family_statuses = []
            # HARDENING: Re-initialize contract expected from harness
            # The fallback path must not assume harness outputs exist (feature_names, X_df, etc.)
            # This ensures per-symbol processing can run independently even if harness failed early
    
    if not use_shared_harness:
        # Fallback to original per-symbol processing (for SYMBOL_SPECIFIC view or if harness fails)
        logger.info("Using per-symbol processing (original method)")
        all_results = []
        all_family_statuses = []
        
        # Load parallel execution config
        parallel_symbols = False
        try:
            from CONFIG.config_loader import get_cfg
            feature_selection_cfg = get_cfg("multi_model_feature_selection", default={}, config_name="multi_model_feature_selection")
            parallel_symbols = feature_selection_cfg.get('parallel_symbols', False)
        except Exception:
            pass
        
        # Check if parallel execution is globally enabled
        parallel_enabled = _PARALLEL_AVAILABLE and parallel_symbols
        if parallel_enabled:
            try:
                from CONFIG.config_loader import get_cfg
                parallel_global = get_cfg("threading.parallel.enabled", default=True, config_name="threading_config")
                parallel_enabled = parallel_enabled and parallel_global
            except Exception:
                pass
        
        # Helper function for parallel symbol processing (must be picklable)
        def _process_single_symbol_wrapper(symbol):
            """Process a single symbol - wrapper for parallel execution"""
            symbol_dir = data_dir / f"symbol={symbol}"
            data_path = symbol_dir / f"{symbol}.parquet"
            
            if not data_path.exists():
                return symbol, None, None, f"Data file not found: {data_path}"
            
            try:
                # FIX: Pass selected_features to per-symbol processing (ensures consistency with pruned features)
                # This prevents features like "adjusted" from "coming back" after pruning
                # Note: In fallback path, selected_features is not available yet (computed after aggregation)
                symbol_results, symbol_statuses = _process_single_symbol(
                    symbol=symbol,
                    data_path=data_path,
                    target_column=target_column,
                    model_families_config=model_families_config,
                    max_samples=max_samples_per_symbol,
                    selected_features=None,  # Not available in fallback path (computed after aggregation)
                    explicit_interval=explicit_interval,
                    experiment_config=experiment_config,
                    output_dir=output_dir
                )
                return symbol, symbol_results, symbol_statuses, None
            except Exception as e:
                return symbol, None, None, str(e)
        
        # Process symbols (parallel or sequential)
        if parallel_enabled and len(symbols_to_process) > 1:
            logger.info(f"🚀 Parallel symbol processing enabled ({len(symbols_to_process)} symbols)")
            parallel_results = execute_parallel(
                _process_single_symbol_wrapper,
                symbols_to_process,
                max_workers=None,  # Auto-detect from config
                task_type="process",  # CPU-bound
                desc="Processing symbols",
                show_progress=True
            )
            
            # Process parallel results
            for symbol, symbol_results, symbol_statuses, error in parallel_results:
                if error:
                    logger.error(f"  ❌ {symbol} failed: {error}")
                    continue
                
                if symbol_results is None:
                    logger.warning(f"  ⚠️  {symbol}: No results")
                    continue
                
                all_results.extend(symbol_results)
                if symbol_statuses:
                    all_family_statuses.extend(symbol_statuses)
                logger.info(f"  ✅ {symbol}: {len(symbol_results)} model results")
        else:
            # Sequential processing (original code path)
            if parallel_enabled and len(symbols_to_process) == 1:
                logger.info("Running sequentially (only 1 symbol)")
            elif not parallel_enabled:
                logger.info("Parallel execution disabled (parallel_symbols=false or not available)")
            
            for idx, symbol in enumerate(symbols_to_process, 1):
                logger.info(f"[{idx}/{len(symbols_to_process)}] Processing {symbol}...")
                
                # Find symbol data file
                symbol_dir = data_dir / f"symbol={symbol}"
                data_path = symbol_dir / f"{symbol}.parquet"
                
                if not data_path.exists():
                    logger.warning(f"  Data file not found: {data_path}")
                    continue
                
                try:
                    # Process symbol (preserves all leakage-free behavior)
                    # Returns tuple: (results, family_statuses)
                    # FIX: Pass selected_features to per-symbol processing (ensures consistency with pruned features)
                    # Note: selected_features may not exist yet (computed after aggregation), so use None as fallback
                    symbol_results, symbol_statuses = _process_single_symbol(
                        symbol=symbol,
                        data_path=data_path,
                        target_column=target_column,
                        model_families_config=model_families_config,
                        max_samples=max_samples_per_symbol,
                        explicit_interval=explicit_interval,
                        experiment_config=experiment_config,
                        output_dir=output_dir,  # Pass output_dir for reproducibility tracking
                        selected_features=None  # Not available in fallback path (computed after aggregation)
                    )
                    
                    all_results.extend(symbol_results)
                    all_family_statuses.extend(symbol_statuses)
                    logger.info(f"  ✅ {symbol}: {len(symbol_results)} model results")
                
                except Exception as e:
                    logger.error(f"  ❌ {symbol} failed: {e}")
                    continue
    else:
        # Shared harness was used - all_results already populated
        # Create empty family_statuses for compatibility
        all_family_statuses = []
    
    if not all_results:
        logger.warning("No results from any symbol")
        return [], pd.DataFrame()
    
    logger.info(f"\nAggregating results from {len(all_results)} model runs...")
    
    # Aggregate across models and symbols
    # aggregation_config and model_families_config are already set above (from typed config or legacy)
    summary_df, selected_features = _aggregate_multi_model_importance(
        all_results=all_results,
        model_families_config=model_families_config,
        aggregation_config=aggregation_config,
        top_n=top_n,
        all_family_statuses=all_family_statuses  # Pass status info for logging excluded families
    )
    
    logger.info(f"✅ Selected {len(selected_features)} features")
    
    # CRITICAL: Post-selection lookback cap enforcement (FS_POST)
    # Apply lookback cap AFTER selection to catch long-lookback features that selection surfaced
    # This prevents the "pruning surfaced long-lookback" class of bugs
    if selected_features:
        from TRAINING.utils.lookback_cap_enforcement import apply_lookback_cap
        from CONFIG.config_loader import get_cfg
        from TRAINING.common.feature_registry import get_registry
        
        # Load lookback cap and policy from config
        lookback_cap = None
        try:
            cap_raw = get_cfg("safety.leakage_detection.lookback_budget_minutes", default="auto", config_name="safety_config")
            if cap_raw != "auto" and isinstance(cap_raw, (int, float)):
                lookback_cap = float(cap_raw)
        except Exception:
            pass
        
        policy = "strict"
        try:
            policy = get_cfg("safety.leakage_detection.policy", default="strict", config_name="safety_config")
        except Exception:
            pass
        
        log_mode = "summary"
        try:
            log_mode = get_cfg("safety.leakage_detection.log_mode", default="summary", config_name="safety_config")
        except Exception:
            pass
        
        # Get interval from resolved_config if available, otherwise use default
        data_interval_minutes = 5.0  # Default
        if use_shared_harness and 'resolved_config' in locals() and resolved_config:
            data_interval_minutes = resolved_config.interval_minutes if hasattr(resolved_config, 'interval_minutes') and resolved_config.interval_minutes else 5.0
        elif explicit_interval:
            if isinstance(explicit_interval, str):
                # Parse "5m" -> 5.0
                from TRAINING.utils.duration_parser import parse_duration
                duration = parse_duration(explicit_interval)
                data_interval_minutes = duration.to_minutes()
            else:
                data_interval_minutes = float(explicit_interval)
        
        if lookback_cap is not None:
            try:
                registry = get_registry()
            except Exception:
                registry = None
            
            # Get feature_time_meta_map and base_interval from resolved_config if available
            feature_time_meta_map = None
            base_interval_minutes = None
            if use_shared_harness and 'resolved_config' in locals() and resolved_config:
                feature_time_meta_map = resolved_config.feature_time_meta_map if hasattr(resolved_config, 'feature_time_meta_map') else None
                base_interval_minutes = resolved_config.base_interval_minutes if hasattr(resolved_config, 'base_interval_minutes') else None
            
            post_cap_result = apply_lookback_cap(
                features=selected_features,
                interval_minutes=data_interval_minutes,
                cap_minutes=lookback_cap,
                policy=policy,
                stage=f"FS_POST_{view}",
                registry=registry,
                feature_time_meta_map=feature_time_meta_map,
                base_interval_minutes=base_interval_minutes,
                log_mode=log_mode
            )
            
            # CRITICAL: Convert to EnforcedFeatureSet (SST contract)
            enforced_fs_post = post_cap_result.to_enforced_set(
                stage=f"FS_POST_{view}",
                cap_minutes=lookback_cap
            )
            
            # CRITICAL: Use enforced.features (the truth) - no rediscovery
            selected_features = enforced_fs_post.features.copy()
            
            # CRITICAL: Boundary assertion - validate selected_features matches FS_POST EnforcedFeatureSet
            from TRAINING.utils.lookback_policy import assert_featureset_fingerprint
            try:
                assert_featureset_fingerprint(
                    label=f"FS_POST_{view}",
                    expected=enforced_fs_post,
                    actual_features=selected_features,
                    logger_instance=logger,
                    allow_reorder=False  # Strict order check
                )
            except RuntimeError as e:
                # Log but don't fail - this is a validation check
                logger.error(f"FS_POST assertion failed: {e}")
                # Fix it: use enforced.features (the truth)
                selected_features = enforced_fs_post.features.copy()
                logger.info(f"Fixed: Updated selected_features to match enforced_fs_post.features")
            
            # Update summary_df to match (remove rows for quarantined features)
            if len(enforced_fs_post.quarantined) > 0 or len(enforced_fs_post.unknown) > 0:
                summary_df = summary_df[summary_df['feature'].isin(enforced_fs_post.features)].copy()
                quarantined_count = len(enforced_fs_post.quarantined) + len(enforced_fs_post.unknown)
                logger.info(f"✅ Post-selection cap enforcement: {len(enforced_fs_post.features)} safe features (quarantined {quarantined_count})")
            else:
                logger.debug(f"FS_POST: All {len(selected_features)} selected features passed lookback cap")
            
            # Store EnforcedFeatureSet for downstream use (if resolved_config available)
            if use_shared_harness and 'resolved_config' in locals() and resolved_config:
                resolved_config._fs_post_enforced = enforced_fs_post
        else:
            logger.debug(f"FS_POST: No lookback cap set, skipping post-selection cap enforcement (view={view})")
    
    # Save stability snapshot for aggregated feature selection (non-invasive hook)
    try:
        from TRAINING.stability.feature_importance import save_snapshot_hook
        # Convert summary_df to importance dict (consensus_score as importance)
        if summary_df is not None and len(summary_df) > 0 and output_dir:
            importance_dict = summary_df.set_index('feature')['consensus_score'].to_dict()
            
            # Build REPRODUCIBILITY path for snapshots (same structure as feature importances)
            target_name_clean = target_column.replace('/', '_').replace('\\', '_')
            # Determine base output directory (walk up from REPRODUCIBILITY/FEATURE_SELECTION structure)
            base_output_dir = output_dir
            while base_output_dir.name in ["CROSS_SECTIONAL", "SYMBOL_SPECIFIC", "FEATURE_SELECTION", "TARGET_RANKING", "REPRODUCIBILITY", "feature_selections", "target_rankings"]:
                base_output_dir = base_output_dir.parent
                if not base_output_dir.parent.exists() or base_output_dir.name == "RESULTS":
                    break
            
            repro_base = base_output_dir / "REPRODUCIBILITY" / "FEATURE_SELECTION"
            if view == "SYMBOL_SPECIFIC" and symbol:
                snapshot_base_dir = repro_base / view / target_name_clean / f"symbol={symbol}"
            else:
                snapshot_base_dir = repro_base / view / target_name_clean
            
            save_snapshot_hook(
                target_name=target_column,
                method="multi_model_aggregated",
                importance_dict=importance_dict,
                universe_id=view,  # Use view parameter
                output_dir=snapshot_base_dir,  # Save in REPRODUCIBILITY structure
                auto_analyze=None,  # Load from config
            )
    except Exception as e:
        logger.debug(f"Stability snapshot save failed for aggregated selection (non-critical): {e}")
    
    # Optional: Cross-sectional ranking (if enabled and enough symbols)
    cs_importance = None
    cs_stability_results = None  # Will store CS stability metrics
    # Load cross-sectional ranking config from preprocessing_config.yaml
    try:
        from CONFIG.config_loader import get_cfg
        cs_config_base = get_cfg("preprocessing.multi_model_feature_selection.cross_sectional_ranking", default={}, config_name="preprocessing_config")
        # Merge with aggregation_config (aggregation_config takes precedence if both exist)
        cs_config = {**cs_config_base, **aggregation_config.get('cross_sectional_ranking', {})}
    except Exception:
        cs_config = aggregation_config.get('cross_sectional_ranking', {})
    
    # Store cohort metadata context for later use in reproducibility tracking
    # Extract from available data: symbols, cs_config, and optionally load mtf_data for date ranges
    cohort_context = {
        'symbols': symbols,
        'data_dir': data_dir,
        'min_cs': cs_config.get('min_cs', 10) if cs_config.get('enabled', False) else None,
        'max_cs_samples': cs_config.get('max_cs_samples', 1000) if cs_config.get('enabled', False) else None,
        'mtf_data': None  # Will try to load if needed
    }
    
    # Try to load mtf_data for date range extraction (optional, for better metadata)
    # NOTE: pd is imported at module scope - do not import locally to avoid UnboundLocalError
    try:
        mtf_data_for_cohort = {}
        for symbol in symbols[:5]:  # Limit to first 5 symbols to avoid loading too much
            symbol_dir = data_dir / f"symbol={symbol}"
            data_path = symbol_dir / f"{symbol}.parquet"
            if data_path.exists():
                df = pd.read_parquet(data_path, columns=['timestamp'] if 'timestamp' in pd.read_parquet(data_path, nrows=0).columns else [])
                if not df.empty:
                    mtf_data_for_cohort[symbol] = df
        if mtf_data_for_cohort:
            cohort_context['mtf_data'] = mtf_data_for_cohort
    except Exception as e:
        logger.debug(f"Could not load mtf_data for cohort metadata: {e}")
    
    if (cs_config.get('enabled', False) and 
        len(symbols) >= cs_config.get('min_symbols', 5)):
        
        try:
            from TRAINING.ranking.cross_sectional_feature_ranker import (
                compute_cross_sectional_importance,
                tag_features_by_importance
            )
            
            top_k_candidates = cs_config.get('top_k_candidates', 50)
            candidates = selected_features[:top_k_candidates]
            
            logger.info(f"🔍 Computing cross-sectional importance for {len(candidates)} candidate features...")
            cs_importance = compute_cross_sectional_importance(
                candidate_features=candidates,
                target_column=target_column,
                symbols=symbols,
                data_dir=data_dir,
                model_families=cs_config.get('model_families', ['lightgbm']),
                min_cs=cs_config.get('min_cs', 10),
                max_cs_samples=cs_config.get('max_cs_samples', 1000),
                normalization=cs_config.get('normalization'),
                model_configs=cs_config.get('model_configs'),
                output_dir=output_dir  # Pass output_dir for reproducibility tracking
            )
            
            # Merge CS scores into summary_df
            summary_df['cs_importance_score'] = summary_df['feature'].map(cs_importance).fillna(0.0)
            
            # Tag features
            symbol_importance = summary_df.set_index('feature')['consensus_score']
            cs_importance_aligned = cs_importance.reindex(symbol_importance.index, fill_value=0.0)
            feature_categories = tag_features_by_importance(
                symbol_importance=symbol_importance,
                cs_importance=cs_importance_aligned,
                symbol_threshold=cs_config.get('symbol_threshold', 0.1),
                cs_threshold=cs_config.get('cs_threshold', 0.1)
            )
            # Map categories back to summary_df (preserve original index)
            summary_df['feature_category'] = summary_df['feature'].map(feature_categories).fillna('UNKNOWN')
            
            logger.info(f"   ✅ Cross-sectional ranking complete")
            category_counts = summary_df['feature_category'].value_counts()
            for cat, count in category_counts.items():
                logger.info(f"      {cat}: {count} features")
            
            # Cross-sectional stability tracking
            try:
                from TRAINING.ranking.cross_sectional_feature_ranker import (
                    compute_cross_sectional_stability
                )
                
                cs_stability = compute_cross_sectional_stability(
                    target_column=target_column,
                    cs_importance=cs_importance,
                    output_dir=output_dir,
                    top_k=20,
                    universe_id="ALL"  # Cross-sectional uses all symbols
                )
                
                # Compact logging (similar to per-model reproducibility)
                if cs_stability['status'] == 'stable':
                    logger.info(
                        f"   [CS-STABILITY] ✅ STABLE: "
                        f"overlap={cs_stability['mean_overlap']:.3f}±{cs_stability['std_overlap']:.3f}, "
                        f"tau={cs_stability['mean_tau']:.3f if cs_stability['mean_tau'] is not None else 'N/A'}, "
                        f"snapshots={cs_stability['n_snapshots']}"
                    )
                elif cs_stability['status'] == 'drifting':
                    logger.warning(
                        f"   [CS-STABILITY] ⚠️  DRIFTING: "
                        f"overlap={cs_stability['mean_overlap']:.3f}±{cs_stability['std_overlap']:.3f}, "
                        f"tau={cs_stability['mean_tau']:.3f if cs_stability['mean_tau'] is not None else 'N/A'}, "
                        f"snapshots={cs_stability['n_snapshots']}"
                    )
                elif cs_stability['status'] == 'diverged':
                    logger.warning(
                        f"   [CS-STABILITY] ⚠️  DIVERGED: "
                        f"overlap={cs_stability['mean_overlap']:.3f}±{cs_stability['std_overlap']:.3f}, "
                        f"tau={cs_stability['mean_tau']:.3f if cs_stability['mean_tau'] is not None else 'N/A'}, "
                        f"snapshots={cs_stability['n_snapshots']}"
                    )
                elif cs_stability['n_snapshots'] < 2:
                    logger.debug(
                        f"   [CS-STABILITY] First run (snapshots={cs_stability['n_snapshots']})"
                    )
                
                # Store stability results for metadata
                cs_stability_results = cs_stability
                
            except Exception as e:
                logger.debug(f"CS stability tracking failed (non-critical): {e}")
                cs_stability_results = None
                
        except Exception as e:
            logger.warning(f"Cross-sectional ranking failed: {e}", exc_info=True)
            summary_df['cs_importance_score'] = 0.0
            summary_df['feature_category'] = 'UNKNOWN'
            cs_stability_results = None
    else:
        summary_df['cs_importance_score'] = 0.0
        summary_df['feature_category'] = 'SYMBOL_ONLY'  # CS ranking not run
        cs_stability_results = None
        if len(symbols) < cs_config.get('min_symbols', 5):
            logger.debug(f"Skipping cross-sectional ranking: only {len(symbols)} symbols (min: {cs_config.get('min_symbols', 5)})")
        elif not cs_config.get('enabled', False):
            logger.debug("Cross-sectional ranking disabled in config")
    
    # Run importance diff detector if enabled (optional diagnostic)
    # This compares models trained with all features vs. safe features only
    try:
        from TRAINING.common.importance_diff_detector import ImportanceDiffDetector
        from TRAINING.common.feature_registry import get_registry
        from TRAINING.utils.data_interval import detect_interval_from_dataframe
        from TRAINING.utils.leakage_filtering import filter_features_for_target, _extract_horizon, _load_leakage_config
        
        # Check if we should run importance diff detection
        # (This would require training two sets of models - full vs safe)
        # For now, we'll add it as an optional post-processing step
        # that can be enabled via config
        
        # Placeholder for future implementation:
        # 1. Train models with all features (already done)
        # 2. Train models with only safe features (registry-validated)
        # 3. Compare importances to detect suspicious features
        
        logger.debug("Importance diff detector available (not yet integrated into selection pipeline)")
    except ImportError:
        logger.debug("Importance diff detector not available")
    
    # Save results if output_dir provided
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            'target_column': target_column,
            'symbols': symbols,
            'n_symbols_processed': len(symbols),
            'n_model_results': len(all_results),
            'top_n': top_n or len(selected_features),
            'model_families_config': model_families_config,  # Include for confidence computation
            'family_statuses': all_family_statuses,  # Include family status tracking for debugging
            'view': view,
            'symbol': symbol if view == "SYMBOL_SPECIFIC" else None
        }
        
        # Add cross-sectional stability to metadata if available
        if cs_stability_results is not None:
            metadata['cross_sectional_stability'] = cs_stability_results
        
        # Save using existing multi-model results function (detailed CSVs, etc.)
        _save_multi_model_results(
            summary_df=summary_df,
            selected_features=selected_features,
            all_results=all_results,
            output_dir=output_dir,
            metadata=metadata
        )
        
        # NEW: Also save in same format as target ranking (CSV, YAML, REPRODUCIBILITY structure)
        try:
            from TRAINING.ranking.feature_selection_reporting import (
                save_feature_selection_rankings,
                save_dual_view_feature_selections,
                save_feature_importances_for_reproducibility
            )
            
            # Save rankings in target ranking format
            save_feature_selection_rankings(
                summary_df=summary_df,
                selected_features=selected_features,
                target_column=target_column,
                output_dir=output_dir,
                view=view,
                symbol=symbol,
                metadata=metadata
            )
            
            # Save feature importances (if available from shared harness)
            # For CROSS_SECTIONAL: all_feature_importances is available from run_importance_producers
            # For SYMBOL_SPECIFIC: we need to collect from each symbol's results
            if use_shared_harness:
                if view == "CROSS_SECTIONAL":
                    # Try to get from shared harness results (if available in scope)
                    if 'all_feature_importances' in locals() and all_feature_importances:
                        save_feature_importances_for_reproducibility(
                            all_feature_importances=all_feature_importances,
                            target_column=target_column,
                            output_dir=output_dir,
                            view=view,
                            symbol=None
                        )
                elif view == "SYMBOL_SPECIFIC":
                    # Collect importances from all_results (per-symbol results from shared harness)
                    symbol_importances = {}
                    for result in all_results:
                        if result.symbol not in symbol_importances:
                            symbol_importances[result.symbol] = {}
                        # Convert Series to dict for JSON serialization
                        if hasattr(result.importance_scores, 'to_dict'):
                            symbol_importances[result.symbol][result.model_family] = result.importance_scores.to_dict()
                        else:
                            symbol_importances[result.symbol][result.model_family] = dict(result.importance_scores)
                    
                    # Save per-symbol importances (same structure as target ranking)
                    for sym, importances_dict in symbol_importances.items():
                        save_feature_importances_for_reproducibility(
                            all_feature_importances=importances_dict,
                            target_column=target_column,
                            output_dir=output_dir,
                            view=view,
                            symbol=sym
                        )
            else:
                # Fallback: collect from all_results (per-symbol processing)
                symbol_importances = {}
                for result in all_results:
                    if result.symbol not in symbol_importances:
                        symbol_importances[result.symbol] = {}
                    if hasattr(result.importance_scores, 'to_dict'):
                        symbol_importances[result.symbol][result.model_family] = result.importance_scores.to_dict()
                    else:
                        symbol_importances[result.symbol][result.model_family] = dict(result.importance_scores)
                
                # Save per-symbol importances
                for sym, importances_dict in symbol_importances.items():
                    save_feature_importances_for_reproducibility(
                        all_feature_importances=importances_dict,
                        target_column=target_column,
                        output_dir=output_dir,
                        view="SYMBOL_SPECIFIC",  # Per-symbol processing is SYMBOL_SPECIFIC
                        symbol=sym
                    )
            
            # Prepare dual-view results for saving (if we have both views)
            # Note: This function is called once per view, so we save what we have
            results_cs = None
            results_sym = None
            if view == "CROSS_SECTIONAL":
                results_cs = {
                    'target_column': target_column,
                    'selected_features': selected_features,
                    'n_features': len(selected_features),
                    'top_n': top_n or len(selected_features)
                }
            elif view == "SYMBOL_SPECIFIC" and symbol:
                results_sym = {
                    symbol: {
                        'target_column': target_column,
                        'selected_features': selected_features,
                        'n_features': len(selected_features),
                        'top_n': top_n or len(selected_features)
                    }
                }
            
            # Save dual-view structure (same as target ranking)
            save_dual_view_feature_selections(
                results_cs=results_cs,
                results_sym=results_sym,
                target_column=target_column,
                output_dir=output_dir
            )
            
        except ImportError as e:
            logger.debug(f"Feature selection reporting module not available: {e}, using basic save only")
        except Exception as e:
            logger.warning(f"Failed to save feature selection results in target ranking format: {e}", exc_info=True)
        
        # Save leak detection summary (same as target ranking)
        # Collect suspicious features from shared harness results if available
        try:
            from TRAINING.ranking.predictability.reporting import save_leak_report_summary
            all_suspicious_features = {}
            
            # Collect from shared harness results (if used)
            if use_shared_harness:
                # Check if we have suspicious features from the harness
                if view == "CROSS_SECTIONAL" and 'all_suspicious_features' in locals():
                    # all_suspicious_features is a dict from run_importance_producers
                    if all_suspicious_features:
                        all_suspicious_features[target_column] = all_suspicious_features
                elif view == "SYMBOL_SPECIFIC":
                    # Collect per-symbol suspicious features
                    symbol_suspicious = {}
                    for result in all_results:
                        if hasattr(result, 'suspicious_features') and result.suspicious_features:
                            symbol = getattr(result, 'symbol', 'ALL')
                            model_family = getattr(result, 'model_family', 'unknown')
                            if symbol not in symbol_suspicious:
                                symbol_suspicious[symbol] = {}
                            symbol_suspicious[symbol][model_family] = result.suspicious_features
                    if symbol_suspicious:
                        all_suspicious_features[target_column] = symbol_suspicious
            
            # Also collect from all_results (fallback for non-harness path)
            if not all_suspicious_features:
                for result in all_results:
                    if hasattr(result, 'suspicious_features') and result.suspicious_features:
                        model_key = f"{getattr(result, 'model_family', 'unknown')}_{getattr(result, 'symbol', 'ALL')}"
                        if target_column not in all_suspicious_features:
                            all_suspicious_features[target_column] = {}
                        # Convert to list of tuples if needed
                        if isinstance(result.suspicious_features, dict):
                            suspicious_list = list(result.suspicious_features.items())
                        else:
                            suspicious_list = result.suspicious_features
                        all_suspicious_features[target_column][model_key] = suspicious_list
            
            if all_suspicious_features:
                save_leak_report_summary(output_dir, all_suspicious_features)
                logger.info("✅ Saved leak detection summary (same format as target ranking)")
        except ImportError:
            logger.debug("Leak detection summary not available (non-critical)")
        except Exception as e:
            logger.debug(f"Failed to save leak detection summary: {e}")
        
        # Analyze stability for all feature selection methods (same as target ranking)
        try:
            from TRAINING.stability.feature_importance import analyze_all_stability_hook
            logger.info("\n" + "="*60)
            logger.info("Feature Importance Stability Analysis")
            logger.info("="*60)
            analyze_all_stability_hook(output_dir=output_dir)
        except ImportError:
            logger.debug("Stability analysis hook not available (non-critical)")
        except Exception as e:
            logger.debug(f"Stability analysis failed (non-critical): {e}")
        
        # Save CS stability metadata separately → metadata/ (matching target ranking structure)
        if cs_stability_results is not None and output_dir:
            try:
                import json
                metadata_dir = output_dir / "metadata"
                metadata_dir.mkdir(parents=True, exist_ok=True)
                cs_metadata_file = metadata_dir / "cross_sectional_stability_metadata.json"
                cs_metadata = {
                    "target_column": target_column,
                    "universe_id": "ALL",
                    "method": "cross_sectional_panel",
                    "stability": cs_stability_results,
                    "timestamp": pd.Timestamp.now().isoformat()
                }
                with open(cs_metadata_file, 'w') as f:
                    json.dump(cs_metadata, f, indent=2)
                logger.debug(f"Saved CS stability metadata to {cs_metadata_file}")
            except Exception as e:
                logger.debug(f"Failed to save CS stability metadata: {e}")
    
    # Track reproducibility: compare to previous feature selection run with trend analysis
    # This runs regardless of which entry point calls this function
    if output_dir and summary_df is not None and len(summary_df) > 0:
        try:
            from TRAINING.utils.reproducibility_tracker import ReproducibilityTracker
            
            # Use run-level directory for reproducibility tracking
            # output_dir is now: REPRODUCIBILITY/FEATURE_SELECTION/CROSS_SECTIONAL/{target}/
            # Walk up to find the run-level directory
            module_output_dir = output_dir
            while module_output_dir.name in ["CROSS_SECTIONAL", "SYMBOL_SPECIFIC", "FEATURE_SELECTION", "TARGET_RANKING", "REPRODUCIBILITY", "feature_selections", "target_rankings"]:
                module_output_dir = module_output_dir.parent
                if not module_output_dir.parent.exists() or module_output_dir.name == "RESULTS":
                    break
            
            tracker = ReproducibilityTracker(
                output_dir=module_output_dir,
                search_previous_runs=True  # Search for previous runs in parent directories
            )
            
            # Calculate summary metrics for reproducibility tracking
            top_feature_score = summary_df.iloc[0]['consensus_score'] if not summary_df.empty else 0.0
            mean_consensus = summary_df['consensus_score'].mean()
            std_consensus = summary_df['consensus_score'].std()
            n_features_selected = len(selected_features)
            n_successful_families = len([s for s in all_family_statuses if s.get('status') == 'success'])
            
            # Extract cohort metadata using unified extractor
            from TRAINING.utils.cohort_metadata_extractor import extract_cohort_metadata, format_for_reproducibility_tracker
            
            # Extract cohort metadata from stored context (symbols, mtf_data, cs_config)
            # cohort_context is defined earlier in the function
            if 'cohort_context' in locals() and cohort_context:
                cohort_metadata = extract_cohort_metadata(
                    symbols=cohort_context.get('symbols'),
                    mtf_data=cohort_context.get('mtf_data'),
                    min_cs=cohort_context.get('min_cs'),
                    max_cs_samples=cohort_context.get('max_cs_samples')
                )
            else:
                # Fallback: try to extract from function variables (shouldn't happen if cohort_context is set)
                cohort_metadata = extract_cohort_metadata(
                    symbols=symbols,
                    mtf_data=mtf_data if 'mtf_data' in locals() else None,
                    min_cs=None,
                    max_cs_samples=None
                )
            
            # Format for reproducibility tracker
            cohort_metrics, cohort_additional_data = format_for_reproducibility_tracker(cohort_metadata)
            
            # Try to use new log_run API with RunContext (includes trend analysis)
            try:
                from TRAINING.utils.run_context import RunContext
                from TRAINING.utils.cohort_metadata_extractor import extract_cohort_metadata
                
                # Use RunContext from shared harness if available, otherwise build from available data
                if use_shared_harness and 'ctx' in locals():
                    # Use the RunContext created by the shared harness (has all required fields)
                    ctx_to_use = ctx
                else:
                    # Build RunContext from available data (fallback for per-symbol processing)
                    X_for_ctx = None
                    y_for_ctx = None
                    feature_names_for_ctx = selected_features if selected_features else []
                    
                    # Try to get X, y from cohort_context if available (for data fingerprint)
                    if 'cohort_context' in locals() and cohort_context:
                        mtf_data_for_ctx = cohort_context.get('mtf_data')
                        if mtf_data_for_ctx:
                            # We can't easily reconstruct X, y here, so pass None
                            # The data fingerprint will be computed from symbols/time_vals if available
                            pass
                    
                    # Build RunContext
                    # FIX: Try to get time_vals and horizon_minutes from available data
                    time_vals_for_ctx = None
                    horizon_minutes_for_ctx = None
                    if use_shared_harness and 'time_vals' in locals():
                        time_vals_for_ctx = time_vals
                    if use_shared_harness and 'horizon_minutes' in locals():
                        horizon_minutes_for_ctx = horizon_minutes
                    elif target_column:
                        # Try to extract horizon from target column name
                        try:
                            from TRAINING.utils.leakage_filtering import _extract_horizon, _load_leakage_config
                            leakage_config = _load_leakage_config()
                            horizon_minutes_for_ctx = _extract_horizon(target_column, leakage_config)
                        except Exception:
                            pass
                    
                    # FIX: Ensure view and symbol are set for proper telemetry scoping
                    # Telemetry must be scoped by: target, view (CROSS_SECTIONAL vs SYMBOL_SPECIFIC), and symbol
                    # CRITICAL: For CROSS_SECTIONAL, symbol must be None to prevent history forking
                    symbol_for_ctx = symbol if view == "SYMBOL_SPECIFIC" else None
                    # Get seed from config for reproducibility
                    try:
                        from CONFIG.config_loader import get_cfg
                        seed_value = get_cfg("pipeline.determinism.base_seed", default=42)
                    except Exception:
                        seed_value = 42
                    
                    ctx_to_use = RunContext(
                        stage="FEATURE_SELECTION",
                        target_name=target_column,
                        target_column=target_column,
                        X=X_for_ctx,  # May be None - fingerprint will use symbols/time_vals
                        y=y_for_ctx,  # May be None
                        feature_names=feature_names_for_ctx,
                        symbols=symbols,
                        time_vals=time_vals_for_ctx,  # Use from shared harness if available
                        horizon_minutes=horizon_minutes_for_ctx,  # Extract from target if available
                        purge_minutes=None,
                        embargo_minutes=None,
                        cv_folds=None,
                        fold_timestamps=None,
                        data_interval_minutes=None,
                        seed=seed_value,
                        view=view,  # FIX: Set view for proper telemetry scoping (CROSS_SECTIONAL vs SYMBOL_SPECIFIC)
                        symbol=symbol_for_ctx  # FIX: Set symbol for SYMBOL_SPECIFIC view only (None for CROSS_SECTIONAL to prevent history forking)
                    )
                
                # Build metrics dict
                metrics_dict = {
                    "metric_name": "Consensus Score",
                    "mean_score": mean_consensus,
                    "std_score": std_consensus,
                    "mean_importance": top_feature_score,
                    "composite_score": mean_consensus,
                    "n_features_selected": n_features_selected,
                    "n_successful_families": n_successful_families,
                    "n_selected": n_features_selected  # For trend analyzer
                }
                
                # Use automated log_run API (includes trend analysis)
                # FIX: Pass RunContext to log_run (required for COHORT_AWARE mode)
                try:
                    audit_result = tracker.log_run(ctx_to_use, metrics_dict)
                except Exception as e:
                    # If COHORT_AWARE fails due to missing fields, fall back to legacy mode
                    if "Missing required fields" in str(e) or "COHORT_AWARE" in str(e):
                        logger.debug(f"COHORT_AWARE mode failed (missing fields), using legacy tracking: {e}")
                        # Disable COHORT_AWARE and retry with minimal context
                        # FIX: Ensure view and symbol are set for proper telemetry scoping
                        # CRITICAL: For CROSS_SECTIONAL, symbol must be None to prevent history forking
                        symbol_for_ctx = symbol if view == "SYMBOL_SPECIFIC" else None
                        # Get seed from config for reproducibility (same as above)
                        try:
                            from CONFIG.config_loader import get_cfg
                            seed_value = get_cfg("pipeline.determinism.base_seed", default=42)
                        except Exception:
                            seed_value = 42
                        
                        ctx_minimal = RunContext(
                            stage="FEATURE_SELECTION",
                            target_name=target_column,
                            target_column=target_column,
                            X=None,  # Not available in fallback
                            y=None,
                            feature_names=selected_features if selected_features else [],
                            symbols=symbols,
                            time_vals=None,
                            horizon_minutes=None,
                            purge_minutes=None,
                            embargo_minutes=None,
                            data_interval_minutes=None,
                            seed=seed_value,
                            cv_splitter=None,
                            view=view,  # FIX: Set view for proper telemetry scoping
                            symbol=symbol_for_ctx  # FIX: Set symbol for SYMBOL_SPECIFIC view only (None for CROSS_SECTIONAL)
                        )
                        audit_result = tracker.log_run(ctx_minimal, metrics_dict)
                    else:
                        raise
                
                # Log audit report summary if available
                if audit_result.get("audit_report"):
                    audit_report = audit_result["audit_report"]
                    if audit_report.get("violations"):
                        logger.warning(f"⚠️  Audit violations: {len(audit_report['violations'])}")
                    if audit_report.get("warnings"):
                        logger.info(f"ℹ️  Audit warnings: {len(audit_report['warnings'])}")
                
                # Log trend summary if available
                if audit_result.get("trend_summary"):
                    trend = audit_result["trend_summary"]
                    # Trend summary is already logged by log_run, but we can add additional context here if needed
                    pass
                    
            except ImportError:
                # Fallback to legacy API if RunContext not available
                logger.debug("RunContext not available, falling back to legacy reproducibility tracking")
                
                # Merge with existing metrics and additional_data
                metrics_with_cohort = {
                    "metric_name": "Consensus Score",
                    "mean_score": mean_consensus,
                    "std_score": std_consensus,
                    "mean_importance": top_feature_score,  # Use top feature score as importance proxy
                    "composite_score": mean_consensus,  # Use mean consensus as composite
                    "n_features_selected": n_features_selected,
                    "n_successful_families": n_successful_families,
                    "n_selected": n_features_selected,  # For trend analyzer
                    **cohort_metrics  # Adds N_effective_cs if available
                }
                
                # FIX: Map view to route_type for FEATURE_SELECTION (ensures proper telemetry scoping)
                route_type_for_legacy = None
                if view:
                    if view.upper() == "CROSS_SECTIONAL":
                        route_type_for_legacy = "CROSS_SECTIONAL"
                    elif view.upper() in ["SYMBOL_SPECIFIC", "INDIVIDUAL"]:
                        route_type_for_legacy = "INDIVIDUAL"  # SYMBOL_SPECIFIC maps to INDIVIDUAL
                
                # Track lookback cap enforcement results (pre and post selection) in telemetry
                lookback_cap_metadata = {}
                # pre_cap_result and post_cap_result are initialized at function scope
                # They may be None if cap wasn't set or if we're in a different code path
                if pre_cap_result is not None:
                    lookback_cap_metadata['pre_selection'] = {
                        'quarantine_count': pre_cap_result.quarantine_count,
                        'actual_max_lookback': pre_cap_result.actual_max_lookback,
                        'safe_features_count': len(pre_cap_result.safe_features),
                        'quarantined_features_sample': pre_cap_result.quarantined_features[:10]  # Top 10
                    }
                if post_cap_result is not None:
                    lookback_cap_metadata['post_selection'] = {
                        'quarantine_count': post_cap_result.quarantine_count,
                        'actual_max_lookback': post_cap_result.actual_max_lookback,
                        'safe_features_count': len(post_cap_result.safe_features),
                        'quarantined_features_sample': post_cap_result.quarantined_features[:10]  # Top 10
                    }
                
                additional_data_with_cohort = {
                    "top_feature": summary_df.iloc[0]['feature'] if not summary_df.empty else None,
                    "top_n": top_n or len(selected_features),
                    "view": view,  # FIX: Include view for proper telemetry scoping
                    "symbol": symbol,  # FIX: Include symbol for SYMBOL_SPECIFIC view
                    "route_type": route_type_for_legacy,  # FIX: Map view to route_type
                    **cohort_additional_data,  # Adds n_symbols, date_range, cs_config if available
                    'lookback_cap_enforcement': lookback_cap_metadata if lookback_cap_metadata else None
                }
                
                # Add seed for reproducibility tracking
                try:
                    from CONFIG.config_loader import get_cfg
                    seed = get_cfg("pipeline.determinism.base_seed", default=42)
                    additional_data_with_cohort['seed'] = seed
                except Exception:
                    # Fallback to default if config not available
                    additional_data_with_cohort['seed'] = 42
                
                tracker.log_comparison(
                    stage="feature_selection",
                    item_name=target_column,  # FIX: item_name is just target (view/symbol handled by route_type/symbol params)
                    metrics=metrics_with_cohort,
                    additional_data=additional_data_with_cohort,
                    route_type=route_type_for_legacy,  # FIX: Properly scoped by view
                    symbol=symbol  # FIX: Properly scoped by symbol (for SYMBOL_SPECIFIC view)
                )
        except Exception as e:
            logger.warning(f"Reproducibility tracking failed for {target_column}: {e}")
            import traceback
            logger.debug(f"Reproducibility tracking traceback: {traceback.format_exc()}")
    
    # Cross-sectional stability summary (if CS ranking was run)
    if cs_stability_results is not None:
        try:
            status_emoji = {
                'stable': '✅',
                'drifting': '⚠️',
                'diverged': '⚠️',
                'insufficient_snapshots': '📊',
                'snapshot_failed': '❌',
                'analysis_failed': '❌',
                'system_unavailable': '❌'
            }.get(cs_stability_results.get('status', 'unknown'), '❓')
            
            logger.info(
                f"📊 Cross-sectional stability summary: {status_emoji} {cs_stability_results.get('status', 'unknown').upper()} "
                f"(overlap={cs_stability_results.get('mean_overlap', 'N/A')}, "
                f"tau={cs_stability_results.get('mean_tau', 'N/A')}, "
                f"snapshots={cs_stability_results.get('n_snapshots', 0)})"
            )
        except Exception:
            pass  # Non-critical summary logging
    
    return selected_features, summary_df


def rank_features_multi_model(
    target_column: str,
    symbols: List[str],
    data_dir: Path,
    model_families_config: Dict[str, Dict[str, Any]] = None,
    multi_model_config: Dict[str, Any] = None,
    max_samples_per_symbol: Optional[int] = None,  # Load from config if None
    output_dir: Path = None
) -> pd.DataFrame:
    """
    Rank all features for a target using multi-model consensus.
    
    Similar to select_features_for_target but returns full ranking
    instead of just top N features.
    
    Args:
        target_column: Target column name
        symbols: List of symbols to process
        data_dir: Directory containing symbol data
        model_families_config: Optional model families config
        multi_model_config: Optional multi-model config dict
        max_samples_per_symbol: Maximum samples per symbol
        output_dir: Optional output directory for results
    
    Returns:
        DataFrame with features ranked by consensus score
    """
    # Load max_samples_per_symbol from config if not provided
    if max_samples_per_symbol is None:
        try:
            from CONFIG.config_loader import get_cfg
            max_samples_per_symbol = int(get_cfg("pipeline.data_limits.default_max_samples_feature_selection", default=50000, config_name="pipeline_config"))
        except Exception:
            max_samples_per_symbol = 50000
    
    selected_features, summary_df = select_features_for_target(
        target_column=target_column,
        symbols=symbols,
        data_dir=data_dir,
        model_families_config=model_families_config,
        multi_model_config=multi_model_config,
        max_samples_per_symbol=max_samples_per_symbol,
        top_n=None,  # Return all features
        output_dir=output_dir
    )
    
    return summary_df

