# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

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

# Import config loader for centralized path resolution
try:
    from CONFIG.config_loader import get_experiment_config_path, load_experiment_config
    _CONFIG_LOADER_AVAILABLE = True
except ImportError:
    _CONFIG_LOADER_AVAILABLE = False
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


def _compute_feature_selection_config_hash(
    target_column: str,
    symbols: List[str],
    model_families_config: Dict[str, Dict[str, Any]],
    view: str,
    symbol: Optional[str],
    max_samples_per_symbol: Optional[int],
    min_cs: Optional[int],
    max_cs_samples: Optional[int],
    explicit_interval: Optional[Union[int, str]],
    aggregation_config: Dict[str, Any],
    top_n: Optional[int]
) -> str:
    """
    Compute a deterministic hash of feature selection configuration.
    
    This hash is used to determine if cached results can be reused,
    bypassing Phase 2 (symbol processing) when configs are identical.
    
    Uses centralized config hashing utility for consistency.
    
    Args:
        target_column: Target column name
        symbols: List of symbols to process (sorted for hash stability)
        model_families_config: Model families configuration
        view: View type (CROSS_SECTIONAL/SYMBOL_SPECIFIC/LOSO)
        symbol: Symbol for SYMBOL_SPECIFIC view
        max_samples_per_symbol: Maximum samples per symbol
        min_cs: Minimum cross-sectional samples
        max_cs_samples: Maximum cross-sectional samples
        explicit_interval: Explicit interval
        aggregation_config: Aggregation configuration
        top_n: Number of top features to return
    
    Returns:
        Hex digest of config hash (truncated to 16 characters for backward compatibility)
    """
    from TRAINING.common.utils.config_hashing import compute_config_hash
    
    # Build config dict for hashing (only include parameters that affect results)
    config_dict = {
        'target': target_column,
        'symbols': sorted(symbols),  # Sort for stability
        'view': view,
        'symbol': symbol,  # For SYMBOL_SPECIFIC view
        'max_samples_per_symbol': max_samples_per_symbol,
        'min_cs': min_cs,
        'max_cs_samples': max_cs_samples,
        'explicit_interval': explicit_interval,
        'top_n': top_n,
        'aggregation': aggregation_config,
        # Include enabled model families and their configs (only enabled ones)
        'model_families': {
            name: cfg for name, cfg in model_families_config.items()
            if isinstance(cfg, dict) and cfg.get('enabled', False)
        }
    }
    
    # Compute hash using centralized utility (SHA256, truncated to 16 chars for backward compatibility)
    config_hash = compute_config_hash(config_dict)[:16]
    
    return config_hash


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
    symbol: Optional[str] = None,  # Required for SYMBOL_SPECIFIC view
    force_refresh: bool = False,  # If True, bypass cache and re-run Phase 2
    universe_sig: Optional[str] = None,  # Universe signature from SST (resolved_data_config)
    run_identity: Optional[Any] = None,  # Finalized RunIdentity for hash-based storage
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
    # Load max_samples_per_symbol from config if not provided (same logic as target ranking)
    if max_samples_per_symbol is None:
        # First check experiment config if available (same as target ranking)
        if experiment_config:
            # Check experiment_config attribute first
            if hasattr(experiment_config, 'max_samples_per_symbol'):
                max_samples_per_symbol = experiment_config.max_samples_per_symbol
                logger.debug(f"Using max_samples_per_symbol={max_samples_per_symbol} from experiment config attribute")
            else:
                # Fallback: check experiment YAML file (same as target ranking)
                try:
                    exp_name = experiment_config.name
                    if _CONFIG_LOADER_AVAILABLE:
                        exp_yaml = load_experiment_config(exp_name)
                    else:
                        import yaml
                        exp_file = Path("CONFIG/experiments") / f"{exp_name}.yaml"
                        if exp_file.exists():
                            with open(exp_file, 'r') as f:
                                exp_yaml = yaml.safe_load(f) or {}
                        else:
                            exp_yaml = {}
                        exp_data = exp_yaml.get('data', {})
                        if 'max_samples_per_symbol' in exp_data:
                            max_samples_per_symbol = exp_data['max_samples_per_symbol']
                            logger.debug(f"Using max_samples_per_symbol={max_samples_per_symbol} from experiment config YAML")
                except Exception:
                    pass
        
        # Fallback to pipeline config (use same key as target ranking for consistency)
        if max_samples_per_symbol is None:
            try:
                from CONFIG.config_loader import get_cfg
                # Use same config key as target ranking for consistency
                max_samples_per_symbol = int(get_cfg("pipeline.data_limits.default_max_rows_per_symbol_ranking", default=50000, config_name="pipeline_config"))
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
    harness_min_cs = None
    harness_max_cs_samples = None
    if use_shared_harness:
        # Load defaults from config (same as target ranking)
        try:
            from CONFIG.config_loader import get_cfg
            if experiment_config:
                try:
                    exp_name = experiment_config.name
                    if _CONFIG_LOADER_AVAILABLE:
                        exp_yaml = load_experiment_config(exp_name)
                    else:
                        import yaml
                        exp_file = Path("CONFIG/experiments") / f"{exp_name}.yaml"
                        if exp_file.exists():
                            with open(exp_file, 'r') as f:
                                exp_yaml = yaml.safe_load(f) or {}
                        else:
                            exp_yaml = {}
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
    
    # Compute config hash for cache key
    config_hash = _compute_feature_selection_config_hash(
        target_column=target_column,
        symbols=symbols_to_process,
        model_families_config=model_families_config,
        view=view,
        symbol=symbol,
        max_samples_per_symbol=max_samples_per_symbol,
        min_cs=harness_min_cs,
        max_cs_samples=harness_max_cs_samples,
        explicit_interval=explicit_interval,
        aggregation_config=aggregation_config,
        top_n=top_n
    )
    
    # Check cache before Phase 2 (symbol processing)
    cache_hit = False
    cached_features = None
    cached_summary_df = None
    
    if not force_refresh and output_dir:
        from TRAINING.common.utils.cache_manager import (
            build_cache_key_with_symbol,
            get_cache_path,
            load_cache
        )
        
        # Create cache key using centralized utility
        cache_key = build_cache_key_with_symbol(target_column, config_hash, view, symbol)
        cache_path = get_cache_path(output_dir, "feature_selection", cache_key)
        
        # Load cache using centralized utility
        cache_data = load_cache(cache_path, config_hash, verify_hash=True)
        
        if cache_data:
            cached_features = cache_data.get('selected_features', [])
            # Load summary DataFrame if available
            if 'summary_df' in cache_data and cache_data['summary_df']:
                cached_summary_df = pd.DataFrame(cache_data['summary_df'])
            
            if cached_features:
                cache_hit = True
                logger.info(f"✅ Using cached feature selection results (config hash: {config_hash[:8]}, {len(cached_features)} features)")
            else:
                logger.debug(f"Cache file exists but has no features, proceeding with Phase 2")
    
    # If cache hit, skip Phase 2 and return cached results
    if cache_hit and cached_features is not None:
        # Apply top_n filter if specified
        if top_n is not None and len(cached_features) > top_n:
            cached_features = cached_features[:top_n]
        
        # Return cached results (bypass Phase 2)
        if cached_summary_df is not None:
            # Filter summary_df to match selected features
            if top_n is not None:
                cached_summary_df = cached_summary_df.head(top_n)
            return cached_features, cached_summary_df
        else:
            # Create minimal summary_df from cached features
            summary_df = pd.DataFrame({
                'feature': cached_features,
                'consensus_score': [1.0] * len(cached_features)  # Placeholder
            })
            return cached_features, summary_df
    
    # Phase 2: Process symbols (only if cache miss or force_refresh)
    # Initialize SST scope variables for downstream use (may be overwritten by shared harness)
    # These are needed by reproducibility tracking even if shared harness fails
    universe_sig_for_writes = None
    view_for_writes = view  # Default to caller's view
    symbol_for_writes = symbol  # Default to caller's symbol
    
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
                        target=target_column,  # Use target_column as target for exclusions
                        feature_names=None,
                        use_strict_registry=True  # Use strict registry mode for feature selection (same as training)
                    )
                    # FIX: Unpack with tolerance for signature changes, but log what we got
                    # This prevents silently masking real breakage (signature changes)
                    actual_len = len(build_result)
                    logger.debug(f"build_panel returned {actual_len} values: {[type(x).__name__ for x in build_result]}")
                    
                    if actual_len >= 9:
                        # Current signature: 9 values (includes resolved_data_config)
                        X, y, feature_names, symbols_array, time_vals, mtf_data, detected_interval, resolved_config, resolved_data_config = build_result[:9]
                    elif actual_len >= 8:
                        # Legacy signature: 8 values (missing resolved_data_config)
                        X, y, feature_names, symbols_array, time_vals, mtf_data, detected_interval, resolved_config = build_result[:8]
                        resolved_data_config = None
                        logger.debug(f"build_panel returned {actual_len} values (legacy signature without resolved_data_config)")
                    elif actual_len >= 6:
                        # Current signature (6 values): X, y, feature_names, symbols, time_vals, resolved_config
                        # Note: resolved_config (position 5) contains universe_sig, view, etc.
                        X, y, feature_names, symbols_array, time_vals, resolved_data_config = build_result[:6]
                        detected_interval = 5.0  # Default if not provided
                        resolved_config = resolved_data_config  # Alias for compatibility
                        mtf_data = None  # Not returned in 6-value signature
                        logger.debug(f"build_panel returned {actual_len} values (current signature with resolved_data_config)")
                    
                    # Extract universe_sig from resolved_data_config (SST) if not passed by caller
                    if not universe_sig and resolved_data_config:
                        universe_sig = resolved_data_config.get('universe_sig')
                        if universe_sig:
                            logger.debug(f"Extracted universe_sig={universe_sig[:8]}... from resolved_data_config")
                    
                    # Also extract effective_view from resolved_data_config if available
                    if resolved_data_config and resolved_data_config.get('view'):
                        effective_view = resolved_data_config.get('view')
                        if effective_view != view:
                            logger.debug(f"Using effective_view={effective_view} from resolved_data_config (requested: {view})")
                    
                    # FIX: Call resolve_write_scope for SYMBOL_SPECIFIC to set universe_sig_for_writes
                    # This ensures proper scope tracking for reproducibility
                    try:
                        from TRAINING.orchestration.utils.scope_resolution import resolve_write_scope
                        strict_scope = False
                        try:
                            from CONFIG.config_loader import load_config
                            cfg = load_config()
                            strict_scope = getattr(getattr(getattr(cfg, 'safety', None), 'output_layout', None), 'strict_scope_partitioning', False)
                        except Exception:
                            pass
                        view_for_writes, symbol_for_writes, universe_sig_for_writes = resolve_write_scope(
                            resolved_data_config=resolved_data_config,
                            caller_view=view,
                            caller_symbol=symbol_to_process,
                            strict=strict_scope
                        )
                        if universe_sig_for_writes and not universe_sig:
                            universe_sig = universe_sig_for_writes
                    except Exception as e:
                        logger.debug(f"resolve_write_scope failed for {symbol_to_process}: {e}")
                    
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
                    from TRAINING.ranking.utils.lookback_cap_enforcement import apply_lookback_cap
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
                        from TRAINING.ranking.utils.lookback_policy import assert_featureset_hash
                        try:
                            assert_featureset_hash(
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
                    from TRAINING.ranking.utils.leakage_filtering import _extract_horizon, _load_leakage_config
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
                    
                    # CRITICAL: Create RunContext for each symbol (SYMBOL_SPECIFIC view)
                    # This ensures COHORT_AWARE mode works correctly for per-symbol processing
                    purge_minutes = resolved_config.purge_minutes if resolved_config else None
                    embargo_minutes = resolved_config.embargo_minutes if resolved_config else None
                    
                    # Store ctx for this symbol (will be collected later for aggregation)
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
                    # Store ctx in a dict keyed by symbol for later use
                    if 'symbol_ctx_map' not in locals():
                        symbol_ctx_map = {}
                    symbol_ctx_map[symbol_to_process] = ctx
                    
                    logger.debug(f"Created RunContext for {symbol_to_process} with X.shape={X.shape if X is not None else None}, "
                               f"y.shape={y.shape if y is not None else None}, "
                               f"time_vals.shape={time_vals.shape if time_vals is not None else None}, "
                               f"horizon_minutes={horizon_minutes}")
                    
                    # Save stability snapshots for each model family (same as target ranking)
                    # CRITICAL: Per-model-family snapshots ensure stability is computed within same family
                    # Per-symbol snapshots for SYMBOL_SPECIFIC view
                    if all_feature_importances and output_dir:
                        try:
                            from TRAINING.stability.feature_importance import save_snapshot_hook
                            # Build snapshot path (matching TARGET_RANKING structure)
                            # output_dir is already at: REPRODUCIBILITY/FEATURE_SELECTION/CROSS_SECTIONAL/{target}/
                            # Use it directly to avoid nested structures
                            target_clean = target_column.replace('/', '_').replace('\\', '_')
                            
                            # Find base run directory for target-first structure
                            # REMOVED: Legacy REPRODUCIBILITY/FEATURE_SELECTION path construction
                            base_output_dir = output_dir
                            for _ in range(10):
                                # Only stop if we find a run directory (has targets/, globals/, or cache/)
                                # Don't stop at RESULTS/ - continue to find actual run directory
                                if (base_output_dir / "targets").exists() or (base_output_dir / "globals").exists() or (base_output_dir / "cache").exists():
                                    break
                                if not base_output_dir.parent.exists():
                                    break
                                base_output_dir = base_output_dir.parent
                            
                            # Compute split_signature from fold info (folds are now finalized)
                            split_signature = None
                            if run_identity is not None and fold_timestamps:
                                try:
                                    from TRAINING.common.utils.fingerprinting import compute_split_fingerprint
                                    from datetime import datetime
                                    # Convert fold_timestamps to boundaries
                                    fold_boundaries = []
                                    for fold_info in fold_timestamps:
                                        if isinstance(fold_info, dict):
                                            start = fold_info.get('start') or fold_info.get('train_start')
                                            end = fold_info.get('end') or fold_info.get('val_end')
                                            if start and end:
                                                fold_boundaries.append((start, end))
                                    if fold_boundaries:
                                        split_signature = compute_split_fingerprint(
                                            cv_method=harness.cv_method if hasattr(harness, 'cv_method') else "purged_kfold",
                                            n_folds=len(fold_boundaries),
                                            purge_minutes=purge_minutes or 0.0,
                                            embargo_minutes=embargo_minutes or 0.0,
                                            fold_boundaries=fold_boundaries,
                                            split_seed=None,
                                        )
                                except Exception as e:
                                    logger.debug(f"Failed to compute split_signature: {e}")
                            
                            # Use base run directory - save_snapshot_hook will use target-first structure
                            for model_family, importance_dict in all_feature_importances.items():
                                if importance_dict:
                                    # Finalize identity for this model family
                                    final_identity = None
                                    partial_identity_dict = None  # Fallback: extract signatures from partial identity
                                    
                                    if run_identity is not None:
                                        # Always extract partial identity signatures as fallback
                                        # These are from FEATURE_SELECTION stage, not TARGET_RANKING
                                        partial_identity_dict = {
                                            "dataset_signature": getattr(run_identity, 'dataset_signature', None),
                                            "split_signature": split_signature or getattr(run_identity, 'split_signature', None),
                                            "target_signature": getattr(run_identity, 'target_signature', None),
                                            "routing_signature": getattr(run_identity, 'routing_signature', None),
                                            "train_seed": getattr(run_identity, 'train_seed', None),
                                        }
                                        
                                        try:
                                            from TRAINING.common.utils.fingerprinting import (
                                                RunIdentity, compute_hparams_fingerprint,
                                                compute_feature_fingerprint_from_specs
                                            )
                                            # Compute hparams_signature for this model family
                                            family_config = model_families_config.get(model_family, {}) if model_families_config else {}
                                            hparams_signature = compute_hparams_fingerprint(
                                                model_family=model_family,
                                                params=family_config.get('params', family_config),
                                            )
                                            
                                            # Compute feature_signature from final features (registry-resolved)
                                            from TRAINING.common.utils.fingerprinting import resolve_feature_specs_from_registry
                                            feature_specs = resolve_feature_specs_from_registry(list(importance_dict.keys()))
                                            feature_signature = compute_feature_fingerprint_from_specs(feature_specs)
                                            
                                            # Add computed signatures to fallback dict
                                            partial_identity_dict["hparams_signature"] = hparams_signature
                                            partial_identity_dict["feature_signature"] = feature_signature
                                            
                                            # Create updated partial with split + hparams
                                            updated_partial = RunIdentity(
                                                dataset_signature=run_identity.dataset_signature,
                                                split_signature=split_signature or "",
                                                target_signature=run_identity.target_signature,
                                                feature_signature=None,
                                                hparams_signature=hparams_signature or "",
                                                routing_signature=run_identity.routing_signature,
                                                routing_payload=run_identity.routing_payload,
                                                train_seed=run_identity.train_seed,
                                                is_final=False,
                                            )
                                            
                                            # Finalize - will raise if required signatures missing
                                            final_identity = updated_partial.finalize(feature_signature)
                                        except ValueError as ve:
                                            # Required signatures missing - RE-RAISE (contract violation)
                                            logger.error(f"Identity finalization failed for {model_family}: {ve}")
                                            raise
                                        except Exception as e:
                                            # FIX: Log at WARNING level so failures are visible
                                            logger.warning(
                                                f"Failed to finalize identity for {model_family}: {e}. "
                                                f"Using partial identity signatures as fallback."
                                            )
                                    
                                    # FIX: Ensure method name is model_family (e.g., "lightgbm", "ridge")
                                    # NOT importance_method (e.g., "native") - stability must be per-family
                                    # FIX: Pass universe_sig (from SST), not view - prevents scope bugs
                                    # NEVER fall back to view - that's the exact bug we're fixing
                                    snapshot_universe_sig = None
                                    if universe_sig:
                                        from TRAINING.orchestration.utils.cohort_metadata import validate_universe_sig
                                        try:
                                            validate_universe_sig(universe_sig)
                                            snapshot_universe_sig = universe_sig
                                        except ValueError as ve:
                                            logger.warning(f"Invalid universe_sig for snapshot: {ve}")
                                            snapshot_universe_sig = None
                                    else:
                                        logger.warning(
                                            "universe_sig missing; snapshot will be unscoped (legacy). "
                                            "NEVER falling back to view-as-universe. "
                                            f"target={target_column} view={view} symbol={symbol_to_process}"
                                        )
                                    
                                    # FIX: If identity not finalized but we have partial signatures, pass them
                                    effective_identity = final_identity if final_identity else partial_identity_dict
                                    
                                    # Compute feature_fingerprint_input for per-model snapshots
                                    per_model_candidate_features = list(importance_dict.keys()) if importance_dict else []
                                    per_model_feature_input_hash = None
                                    if per_model_candidate_features:
                                        import hashlib
                                        import json as json_mod
                                        sorted_features = sorted(per_model_candidate_features)
                                        per_model_feature_input_hash = hashlib.sha256(json_mod.dumps(sorted_features).encode()).hexdigest()
                                    
                                    per_model_inputs = {
                                        "candidate_features": per_model_candidate_features,
                                        "feature_fingerprint_input": per_model_feature_input_hash,
                                    }
                                    
                                    save_snapshot_hook(
                                        target=target_column,
                                        method=model_family,  # Use model_family as method identifier
                                        importance_dict=importance_dict,
                                        universe_sig=snapshot_universe_sig,  # SST or None, NEVER view
                                        output_dir=base_output_dir,  # Pass run directory - will use target-first structure
                                        auto_analyze=None,  # Load from config
                                        run_identity=effective_identity,  # Pass finalized identity or partial dict fallback
                                        allow_legacy=(final_identity is None and partial_identity_dict is None),
                                        view=view,  # Pass view for proper scoping
                                        symbol=symbol_to_process,  # Pass symbol for SYMBOL_SPECIFIC view
                                        inputs=per_model_inputs,  # Pass inputs with feature_fingerprint_input
                                        stage="FEATURE_SELECTION",  # Explicit stage for proper path scoping
                                    )
                        except ValueError as ve:
                            # Identity validation failure - respect config mode
                            try:
                                from TRAINING.common.utils.fingerprinting import get_identity_mode
                                mode = get_identity_mode()
                            except Exception:
                                mode = "strict"
                            if mode == "strict":
                                logger.error(f"Snapshot save failed (strict mode): {ve}")
                                raise  # Re-raise in strict mode
                            else:
                                logger.error(f"Snapshot save failed ({mode} mode, continuing): {ve}")
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
                    
                    # Check for missing model families (failed or skipped) - same as CROSS_SECTIONAL
                    enabled_families_set = set(model_families_list)
                    families_with_results = set(all_feature_importances.keys())
                    missing_families = enabled_families_set - families_with_results
                    
                    if missing_families:
                        logger.warning(f"⚠️  {symbol_to_process}: {len(missing_families)} model families missing from harness results: {', '.join(missing_families)}")
                        # Record failure statuses for missing families
                        for family_name in missing_families:
                            all_family_statuses.append({
                                "status": "failed",
                                "family": family_name,
                                "symbol": symbol_to_process,
                                "score": None,
                                "top_feature": None,
                                "top_feature_score": None,
                                "error": "Model family not in harness results (likely failed silently)",
                                "error_type": "MissingFromHarness"
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
                    target=target_column,  # Use target_column as target for exclusions
                    feature_names=None,  # Will filter automatically
                    use_strict_registry=True  # Use strict registry mode for feature selection (same as training)
                )
                # FIX: Unpack with tolerance for signature changes, but log what we got
                actual_len = len(build_result)
                logger.debug(f"build_panel returned {actual_len} values: {[type(x).__name__ for x in build_result]}")
                
                if actual_len >= 9:
                    # Current signature: 9 values (includes resolved_data_config)
                    X, y, feature_names, symbols_array, time_vals, mtf_data, detected_interval, resolved_config, resolved_data_config = build_result[:9]
                elif actual_len >= 8:
                    # Legacy signature: 8 values (missing resolved_data_config)
                    X, y, feature_names, symbols_array, time_vals, mtf_data, detected_interval, resolved_config = build_result[:8]
                    resolved_data_config = None
                    logger.debug(f"build_panel returned {actual_len} values (legacy signature without resolved_data_config)")
                elif actual_len >= 6:
                    # Current signature (6 values): X, y, feature_names, symbols, time_vals, resolved_config
                    # Note: resolved_config (position 5) contains universe_sig, view, etc.
                    X, y, feature_names, symbols_array, time_vals, resolved_data_config = build_result[:6]
                    detected_interval = 5.0  # Default if not provided
                    resolved_config = resolved_data_config  # Alias for compatibility
                    mtf_data = None  # Not returned in 6-value signature
                    logger.debug(f"build_panel returned {actual_len} values (current signature with resolved_data_config)")
                else:
                    raise ValueError(f"build_panel returned {actual_len} values, expected at least 6. Got: {[type(x).__name__ for x in build_result]}")
                
                # ========================================================================
                # PATCH 0 (SST VIEW OVERRIDE): Use resolve_write_scope for ALL downstream writes
                # ========================================================================
                # Canonical SST-derived scope resolution with:
                # - Asymmetric rule: blocks SS→CS promotion (min_cs=1 bug)
                # - Symbol derivation: auto-derive from SST symbols[0] when unambiguous
                # - Strict mode: raises on any scope ambiguity
                from TRAINING.orchestration.utils.scope_resolution import resolve_write_scope
                
                # Check if strict mode is enabled
                strict_scope = False
                try:
                    from CONFIG.config_loader import load_config
                    cfg = load_config()
                    strict_scope = getattr(getattr(getattr(cfg, 'safety', None), 'output_layout', None), 'strict_scope_partitioning', False)
                except Exception:
                    pass
                
                view_for_writes, symbol_for_writes, universe_sig_for_writes = resolve_write_scope(
                    resolved_data_config=resolved_data_config,
                    caller_view=view,
                    caller_symbol=symbol,
                    strict=strict_scope
                )
                
                # Use SST-derived universe_sig if not already provided
                if not universe_sig and universe_sig_for_writes:
                    universe_sig = universe_sig_for_writes
                
                if view_for_writes != view:
                    logger.warning(
                        f"SST OVERRIDE: Using view={view_for_writes} instead of "
                        f"caller view={view} for downstream writes"
                    )
                if universe_sig:
                    logger.debug(f"SST universe_sig={universe_sig[:8]}... for writes")
                # ========================================================================
                # END PATCH 0
                # ========================================================================
                
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
                        from TRAINING.ranking.utils.lookback_cap_enforcement import apply_lookback_cap
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
                            from TRAINING.ranking.utils.lookback_policy import assert_featureset_hash
                            try:
                                assert_featureset_hash(
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
                        from TRAINING.ranking.utils.leakage_filtering import _extract_horizon, _load_leakage_config
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
                        
                        # Check for missing model families (failed or skipped)
                        enabled_families_set = set(model_families_list)
                        families_with_results = set(all_feature_importances.keys())
                        missing_families = enabled_families_set - families_with_results
                        
                        if missing_families:
                            logger.warning(f"⚠️  {len(missing_families)} model families missing from harness results: {', '.join(missing_families)}")
                            # Record failure statuses for missing families
                            for family_name in missing_families:
                                all_family_statuses.append({
                                    "status": "failed",
                                    "family": family_name,
                                    "symbol": "ALL",
                                    "score": None,
                                    "top_feature": None,
                                    "top_feature_score": None,
                                    "error": "Model family not in harness results (likely failed silently)",
                                    "error_type": "MissingFromHarness"
                                })
                        
                        # Create RunContext for reproducibility tracking
                        # FIX: Extract purge/embargo from resolved_config
                        purge_minutes = resolved_config.purge_minutes if resolved_config else None
                        embargo_minutes = resolved_config.embargo_minutes if resolved_config else None
                        
                        # Create RunContext for reproducibility tracking (CRITICAL: Must have all required fields)
                        # This ensures COHORT_AWARE mode works correctly
                        # FIX: Get min_cs and max_cs_samples from resolved_config for diff telemetry
                        min_cs_for_ctx = resolved_config.effective_min_cs if resolved_config else None
                        max_cs_samples_for_ctx = resolved_config.max_cs_samples if resolved_config else None
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
                            data_interval_minutes=data_interval_minutes,
                            min_cs=min_cs_for_ctx,  # FIX: Populate for diff telemetry
                            max_cs_samples=max_cs_samples_for_ctx  # FIX: Populate for diff telemetry
                        )
                        
                        logger.info(f"✅ Shared harness completed: {len(all_results)} model results")
                        logger.debug(f"Created RunContext with X.shape={X.shape if X is not None else None}, "
                                   f"y.shape={y.shape if y is not None else None}, "
                                   f"time_vals.shape={time_vals.shape if time_vals is not None else None}, "
                                   f"horizon_minutes={horizon_minutes}")
                
        except Exception as e:
            # FIX: Check if this is an expected error (insufficient symbols for CROSS_SECTIONAL or insufficient data span)
            # If so, log at INFO level instead of WARNING to reduce noise
            error_msg = str(e)
            if "CROSS_SECTIONAL mode requires" in error_msg and "symbols" in error_msg:
                logger.info(f"Shared harness: {error_msg}. Falling back to per-symbol processing.")
            elif "Insufficient data span for long-horizon target" in error_msg:
                logger.info(f"Shared harness: {error_msg}")
            else:
                logger.warning(f"Shared harness failed: {e}, falling back to per-symbol processing", exc_info=True)
            
            # Preserve partial status information if any was collected before the exception
            partial_results_count = len(all_results) if 'all_results' in locals() else 0
            partial_statuses_count = len(all_family_statuses) if 'all_family_statuses' in locals() else 0
            if partial_results_count > 0 or partial_statuses_count > 0:
                logger.warning(
                    f"⚠️  Shared harness exception occurred after partial success: "
                    f"{partial_results_count} results and {partial_statuses_count} statuses were collected but will be discarded in fallback path"
                )
            
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
                    output_dir=output_dir,
                    run_identity=run_identity,  # Pass partial identity for snapshot storage
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
                        selected_features=None,  # Not available in fallback path (computed after aggregation)
                        run_identity=run_identity,  # Pass partial identity for snapshot storage
                    )
                    
                    all_results.extend(symbol_results)
                    all_family_statuses.extend(symbol_statuses)
                    logger.info(f"  ✅ {symbol}: {len(symbol_results)} model results")
                
                except Exception as e:
                    logger.error(f"  ❌ {symbol} failed: {e}")
                    continue
        
        # FIX: Compute universe_sig in fallback path for proper scope tracking
        # This ensures reproducibility tracking has universe_sig even when shared harness fails
        if (universe_sig_for_writes is None or universe_sig is None) and symbols_to_process:
            try:
                from TRAINING.orchestration.utils.run_context import compute_universe_signature
                computed_sig = compute_universe_signature(symbols_to_process)
                if universe_sig is None:
                    universe_sig = computed_sig
                if universe_sig_for_writes is None:
                    universe_sig_for_writes = computed_sig
                    view_for_writes = view
                    symbol_for_writes = symbol
                logger.debug(f"Computed universe_sig={computed_sig[:8]}... in fallback path")
            except Exception as e:
                logger.debug(f"Failed to compute universe_sig in fallback path: {e}")
    else:
        # Shared harness was used - all_results and all_family_statuses already populated
        # Statuses were collected in the shared harness block (lines 684-693 for SYMBOL_SPECIFIC or 913-942 for CROSS_SECTIONAL)
        # Do not overwrite all_family_statuses here - it would destroy the failure statuses for missing families like Boruta
        pass
    
    # Validation: Ensure statuses are preserved through control flow
    if use_shared_harness and model_families_config:
        enabled_families = set(f for f, cfg in model_families_config.items() if cfg.get('enabled', False))
        if enabled_families:
            # Check that we have status information for all enabled families (success or failure)
            families_with_statuses = set(s.get('family') for s in all_family_statuses if s.get('family'))
            missing_statuses = enabled_families - families_with_statuses
            if missing_statuses:
                logger.warning(
                    f"⚠️  Validation: {len(missing_statuses)} enabled families have no status information: {', '.join(sorted(missing_statuses))}. "
                    f"This may indicate families were silently skipped."
                )
    
    if not all_results:
        logger.warning("No results from any symbol")
        return [], pd.DataFrame()
    
    logger.info(f"\nAggregating results from {len(all_results)} model runs...")
    
    # Debug logging: show expected vs actual families
    if model_families_config:
        expected_families = sorted([f for f, cfg in model_families_config.items() if cfg.get('enabled', False)])
        actual_families = sorted(set(r.model_family for r in all_results))
        logger.debug(f"Expected model families: {expected_families}")
        logger.debug(f"Families with results: {actual_families}")
        missing = set(expected_families) - set(actual_families)
        if missing:
            logger.warning(f"⚠️  Missing families in results: {sorted(missing)}")
    
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
        from TRAINING.ranking.utils.lookback_cap_enforcement import apply_lookback_cap
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
                from TRAINING.common.utils.duration_parser import parse_duration
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
            from TRAINING.ranking.utils.lookback_policy import assert_featureset_hash
            try:
                assert_featureset_hash(
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
            
            # Use target-first structure for snapshots
            target_clean = target_column.replace('/', '_').replace('\\', '_')
            # Find base run directory
            base_output_dir = output_dir
            for _ in range(10):
                if base_output_dir.name == "RESULTS" or (base_output_dir / "targets").exists():
                    break
                if not base_output_dir.parent.exists():
                    break
                base_output_dir = base_output_dir.parent
            
            if base_output_dir.exists():
                from TRAINING.orchestration.utils.target_first_paths import (
                    get_target_reproducibility_dir, ensure_target_structure
                )
                ensure_target_structure(base_output_dir, target_clean)
                target_repro_dir = get_target_reproducibility_dir(base_output_dir, target_clean)
                
                # FIX: Pass universe_sig (from SST), not view - prevents scope bugs
                # NEVER fall back to view - that's the exact bug we're fixing
                snapshot_universe_sig = None
                if universe_sig:
                    from TRAINING.orchestration.utils.cohort_metadata import validate_universe_sig
                    try:
                        validate_universe_sig(universe_sig)
                        snapshot_universe_sig = universe_sig
                    except ValueError as ve:
                        logger.warning(f"Invalid universe_sig for aggregated snapshot: {ve}")
                        snapshot_universe_sig = None
                else:
                    logger.warning(
                        "universe_sig missing; aggregated snapshot will be unscoped (legacy). "
                        "NEVER falling back to view-as-universe. "
                        f"target={target_column} view={view}"
                    )
                
                # Compute identity for aggregated snapshot
                # Use "multi_model_aggregated" as hparams (hash of all enabled families)
                aggregated_identity = None
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
                        # Hparams: hash all enabled model families together
                        enabled_families = sorted([f for f, cfg in model_families_config.items() if cfg.get('enabled', False)]) if model_families_config else []
                        hparams_signature = compute_hparams_fingerprint(
                            model_family="multi_model_aggregated",
                            params={"enabled_families": enabled_families},
                        )
                        # Feature signature from aggregated features (registry-resolved)
                        from TRAINING.common.utils.fingerprinting import resolve_feature_specs_from_registry
                        feature_specs = resolve_feature_specs_from_registry(list(importance_dict.keys()))
                        feature_signature = compute_feature_fingerprint_from_specs(feature_specs)
                        
                        # Add computed signatures to fallback dict
                        partial_identity_dict["hparams_signature"] = hparams_signature
                        partial_identity_dict["feature_signature"] = feature_signature
                        
                        # Create updated partial with hparams and finalize
                        updated_partial = RunIdentity(
                            dataset_signature=run_identity.dataset_signature,
                            split_signature=run_identity.split_signature if hasattr(run_identity, 'split_signature') else "",
                            target_signature=run_identity.target_signature,
                            feature_signature=None,
                            hparams_signature=hparams_signature or "",
                            routing_signature=run_identity.routing_signature,
                            routing_payload=run_identity.routing_payload if hasattr(run_identity, 'routing_payload') else None,
                            train_seed=run_identity.train_seed,
                            is_final=False,
                        )
                        aggregated_identity = updated_partial.finalize(feature_signature)
                    except Exception as e:
                        # FIX: Log at WARNING level so failures are visible
                        logger.warning(
                            f"Failed to compute aggregated identity for FEATURE_SELECTION snapshot: {e}. "
                            f"Using partial identity signatures as fallback."
                        )
                
                # CRITICAL: Use passed-in run_identity as fallback if aggregated computation failed
                # This ensures stability hooks get a valid identity for auditability
                identity_for_snapshot = aggregated_identity or run_identity
                
                # FIX: Check if identity is actually finalized before passing to snapshot hook
                # Partial identities (is_final=False) cannot be used in strict mode
                identity_is_finalized = (
                    identity_for_snapshot is not None and 
                    hasattr(identity_for_snapshot, 'is_final') and 
                    identity_for_snapshot.is_final
                )
                
                # FIX: Build inputs dict with selected_targets (from FEATURE_SELECTION stage)
                candidate_features = list(importance_dict.keys()) if importance_dict else []
                
                # Compute feature_fingerprint_input (hash of candidate features before selection)
                feature_fingerprint_input = None
                if candidate_features:
                    import hashlib
                    import json as json_mod
                    sorted_features = sorted(candidate_features)
                    feature_fingerprint_input = hashlib.sha256(json_mod.dumps(sorted_features).encode()).hexdigest()
                
                fs_inputs = {
                    "selected_targets": [target_column],  # This target passed TR
                    "candidate_features": candidate_features,
                    "feature_fingerprint_input": feature_fingerprint_input,  # Hash of candidate features
                }
                
                # FIX: If identity not finalized but we have partial signatures, pass them
                # This ensures fingerprints are populated even when finalization fails
                effective_identity = identity_for_snapshot if identity_is_finalized else partial_identity_dict
                
                save_snapshot_hook(
                    target=target_column,
                    method="multi_model_aggregated",
                    importance_dict=importance_dict,
                    universe_sig=snapshot_universe_sig,  # SST or None, NEVER view
                    output_dir=target_repro_dir,  # Use target-first structure
                    auto_analyze=None,  # Load from config
                    run_identity=effective_identity,  # Pass finalized identity or partial dict fallback
                    allow_legacy=(not identity_is_finalized and partial_identity_dict is None),
                    view=view,  # Pass view for proper scoping
                    symbol=symbol,  # Pass symbol for SYMBOL_SPECIFIC view
                    inputs=fs_inputs,  # Pass inputs with selected_targets
                    stage="FEATURE_SELECTION",  # Explicit stage for proper path scoping
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
    # CRITICAL: Always populate min_cs and max_cs_samples (required for diff_telemetry validation)
    # Use cs_config values if enabled, otherwise use harness values from earlier config loading
    cohort_min_cs = cs_config.get('min_cs', 10) if cs_config.get('enabled', False) else None
    cohort_max_cs = cs_config.get('max_cs_samples', 1000) if cs_config.get('enabled', False) else None
    # Fallback to harness values if not set (loaded from pipeline config earlier)
    if cohort_min_cs is None and 'harness_min_cs' in dir() and harness_min_cs is not None:
        cohort_min_cs = harness_min_cs
    if cohort_max_cs is None and 'harness_max_cs_samples' in dir() and harness_max_cs_samples is not None:
        cohort_max_cs = harness_max_cs_samples
    # Ultimate fallback to defaults
    if cohort_min_cs is None:
        cohort_min_cs = 10
    if cohort_max_cs is None:
        cohort_max_cs = 1000
    
    cohort_context = {
        'symbols': symbols,
        'data_dir': data_dir,
        'min_cs': cohort_min_cs,
        'max_cs_samples': cohort_max_cs,
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
    
    # Check if cross-sectional ranking is enabled and we have enough symbols
    min_cs_required = cs_config.get('min_cs', 10) if cs_config.get('enabled', False) else None
    if (cs_config.get('enabled', False) and 
        min_cs_required is not None and
        len(symbols) >= min_cs_required):
        
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
                min_cs=min_cs_required,
                max_cs_samples=cs_config.get('max_cs_samples', 1000),
                max_rows_per_symbol=max_samples_per_symbol,  # FIX: Consistent sample limit across stages
                normalization=cs_config.get('normalization'),
                model_configs=cs_config.get('model_configs'),
                output_dir=output_dir,  # Pass output_dir for reproducibility tracking
                universe_sig=universe_sig,  # FIX: Thread universe_sig for proper scope tracking
                run_identity=run_identity,  # SST: Pass through authoritative identity
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
                    universe_sig="ALL",  # Cross-sectional uses all symbols
                    run_identity=run_identity,  # Pass partial identity for snapshot storage
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
                
        except ValueError as e:
            # Handle insufficient symbols gracefully
            error_msg = str(e)
            if "requires >=" in error_msg and "symbols" in error_msg:
                logger.warning(
                    f"Cross-sectional ranking skipped: insufficient symbols "
                    f"(have {len(symbols)}, need {min_cs_required}). "
                    f"Falling back to symbol-specific ranking only."
                )
            else:
                logger.warning(f"Cross-sectional ranking failed: {e}", exc_info=True)
            summary_df['cs_importance_score'] = 0.0
            summary_df['feature_category'] = 'SYMBOL_ONLY'  # Mark as symbol-only when CS fails
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
        from TRAINING.ranking.utils.data_interval import detect_interval_from_dataframe
        from TRAINING.ranking.utils.leakage_filtering import filter_features_for_target, _extract_horizon, _load_leakage_config
        
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
        # Compute registry filtering stats for metadata
        # Track how many features were available before/after registry filtering
        registry_filtering_stats = {
            'selected_features_total': len(selected_features),
            'selected_features_registry_allowed': len(selected_features),  # Will be updated if we have tracking
            'registry_filtering_applied': True,
            'registry_mode': 'strict'  # Feature selection uses strict mode (same as training)
        }
        
        # If we used shared harness, feature_names already passed strict registry filtering
        # Count how many features were in the original dataset vs after filtering
        if use_shared_harness and 'feature_names' in locals() and feature_names:
            # feature_names from shared harness are already registry-filtered (strict mode)
            # We don't have the "before" count easily, but we can note that filtering was applied
            registry_filtering_stats['features_after_registry'] = len(feature_names)
            registry_filtering_stats['registry_filtering_note'] = 'Applied via shared harness (strict mode)'
        
        metadata = {
            'target_column': target_column,
            'symbols': symbols,
            'n_symbols_processed': len(symbols),
            'n_model_results': len(all_results),
            'top_n': top_n or len(selected_features),
            'model_families_config': model_families_config,  # Include for confidence computation
            'family_statuses': all_family_statuses,  # Include family status tracking for debugging
            'view': view,
            'symbol': symbol if view == "SYMBOL_SPECIFIC" else None,
            'registry_filtering': registry_filtering_stats  # Add registry filtering metadata
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
            metadata=metadata,
            universe_sig=universe_sig,  # Pass universe_sig for canonical paths
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
                metadata=metadata,
                universe_sig=universe_sig,  # Pass universe_sig for canonical paths
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
                            symbol=None,
                            universe_sig=universe_sig,  # Pass universe_sig for canonical paths
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
                            symbol=sym,
                            universe_sig=universe_sig,  # Pass universe_sig for canonical paths
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
                
                # CRITICAL: If view is CROSS_SECTIONAL, aggregate importances across symbols and save once
                # If view is SYMBOL_SPECIFIC, save per-symbol
                if view == "CROSS_SECTIONAL":
                    from TRAINING.ranking.feature_selection_reporting import aggregate_importances_cross_sectional
                    aggregated_importances = aggregate_importances_cross_sectional(symbol_importances)
                    # Save once to CROSS_SECTIONAL/feature_importances/
                    save_feature_importances_for_reproducibility(
                        all_feature_importances=aggregated_importances,
                        target_column=target_column,
                        output_dir=output_dir,
                        view="CROSS_SECTIONAL",
                        symbol=None,
                        universe_sig=universe_sig,  # Pass universe_sig for canonical paths
                    )
                else:
                    # Save per-symbol importances for SYMBOL_SPECIFIC view
                    for sym, importances_dict in symbol_importances.items():
                        save_feature_importances_for_reproducibility(
                            all_feature_importances=importances_dict,
                            target_column=target_column,
                            output_dir=output_dir,
                            view="SYMBOL_SPECIFIC",
                            symbol=sym,
                            universe_sig=universe_sig,  # Pass universe_sig for canonical paths
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
                    "universe_sig": "ALL",
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
        # FIX: Initialize cohort metadata variables at the very beginning (before any try-except)
        # This ensures they exist even if an exception occurs early in the try block
        cohort_metadata = None
        cohort_metrics = {}
        cohort_additional_data = {}
        
        try:
            from TRAINING.orchestration.utils.reproducibility_tracker import ReproducibilityTracker
            
            # Use run-level directory for reproducibility tracking
            # output_dir is now: REPRODUCIBILITY/FEATURE_SELECTION/{view}/{target}/[symbol={symbol}/]
            # Walk up to find the run-level directory
            module_output_dir = output_dir
            # For SYMBOL_SPECIFIC, also skip symbol= directories
            while module_output_dir.name in ["CROSS_SECTIONAL", "SYMBOL_SPECIFIC", "FEATURE_SELECTION", "TARGET_RANKING", "REPRODUCIBILITY", "feature_selections", "target_rankings"] or module_output_dir.name.startswith("symbol="):
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
            
            # CRITICAL: Initialize cohort variables before any try blocks to prevent "referenced before assignment" errors
            # This ensures they're always defined, even if extraction fails or exception occurs
            cohort_metadata = None
            cohort_metrics = {}
            cohort_additional_data = {}
            
            # Extract cohort metadata using unified extractor
            try:
                from TRAINING.orchestration.utils.cohort_metadata_extractor import extract_cohort_metadata, format_for_reproducibility_tracker
                
                # Extract cohort metadata from stored context (symbols, mtf_data, cs_config)
                # cohort_context is defined earlier in the function
                # FIX: Pass universe_sig from SST to cohort metadata for proper scope tracking
                sst_universe_sig = universe_sig_for_writes if 'universe_sig_for_writes' in locals() else (universe_sig if 'universe_sig' in locals() else None)
                if 'cohort_context' in locals() and cohort_context:
                    cohort_metadata = extract_cohort_metadata(
                        symbols=cohort_context.get('symbols'),
                        mtf_data=cohort_context.get('mtf_data'),
                        min_cs=cohort_context.get('min_cs'),
                        max_cs_samples=cohort_context.get('max_cs_samples'),
                        universe_sig=sst_universe_sig  # FIX: Pass universe_sig from SST
                    )
                else:
                    # Fallback: try to extract from function variables (shouldn't happen if cohort_context is set)
                    # Use harness values or defaults for min_cs/max_cs_samples
                    fallback_min_cs = harness_min_cs if 'harness_min_cs' in dir() and harness_min_cs else 10
                    fallback_max_cs = harness_max_cs_samples if 'harness_max_cs_samples' in dir() and harness_max_cs_samples else 1000
                    cohort_metadata = extract_cohort_metadata(
                        symbols=symbols,
                        mtf_data=mtf_data if 'mtf_data' in locals() else None,
                        min_cs=fallback_min_cs,
                        max_cs_samples=fallback_max_cs,
                        universe_sig=sst_universe_sig  # FIX: Pass universe_sig from SST
                    )
                
                # Format for reproducibility tracker
                if cohort_metadata is not None:
                    cohort_metrics, cohort_additional_data = format_for_reproducibility_tracker(cohort_metadata)
            except Exception as e:
                logger.debug(f"Failed to extract cohort metadata for reproducibility tracking: {e}")
                # Use empty dicts as fallbacks (already initialized above)
            
            # Try to use new log_run API with RunContext (includes trend analysis)
            try:
                from TRAINING.orchestration.utils.run_context import RunContext
                from TRAINING.orchestration.utils.cohort_metadata_extractor import extract_cohort_metadata
                
                # Extract hyperparameters from model_families_config for FEATURE_SELECTION
                # CRITICAL: Different hyperparameters = different features selected
                # This needs to be available for both shared harness and fallback paths
                additional_data_override = {}
                if model_families_config:
                    # Find primary model family (usually lightgbm or first enabled)
                    enabled_families = [f for f, cfg in model_families_config.items() 
                                      if isinstance(cfg, dict) and cfg.get('enabled', False)]
                    if enabled_families:
                        primary_family = 'lightgbm' if 'lightgbm' in enabled_families else enabled_families[0]
                        if primary_family in model_families_config:
                            family_config = model_families_config[primary_family]
                            if isinstance(family_config, dict) and 'config' in family_config:
                                hp_config = family_config['config']
                                # Copy all hyperparameters (exclude non-hyperparameter keys)
                                excluded_keys = {'verbose', 'verbosity', 'objective', 'metric', 'device', 'gpu_device_id'}
                                training_config = {k: v for k, v in hp_config.items() 
                                                  if k not in excluded_keys and v is not None}
                                if training_config:
                                    additional_data_override['training'] = training_config
                
                # Use RunContext from shared harness if available, otherwise build from available data
                # CRITICAL: ctx is created inside the use_shared_harness block, so check both conditions
                ctx_to_use = None
                if use_shared_harness:
                    # Try to get ctx from the shared harness block
                    # For CROSS_SECTIONAL: ctx is created at function scope
                    # For SYMBOL_SPECIFIC: ctx is stored in symbol_ctx_map (use first symbol's ctx for aggregation-level tracking)
                    try:
                        if view == "CROSS_SECTIONAL" and 'ctx' in locals() and ctx is not None:
                            ctx_to_use = ctx
                            logger.debug("Using RunContext from shared harness (CROSS_SECTIONAL, has all required fields for COHORT_AWARE)")
                        elif view == "SYMBOL_SPECIFIC" and 'symbol_ctx_map' in locals() and symbol_ctx_map:
                            # For SYMBOL_SPECIFIC, use the first symbol's ctx (aggregation-level tracking)
                            # Individual symbol ctxs are already stored in symbol_ctx_map
                            first_symbol = next(iter(symbol_ctx_map.keys()))
                            ctx_to_use = symbol_ctx_map[first_symbol]
                            logger.debug(f"Using RunContext from shared harness (SYMBOL_SPECIFIC, using {first_symbol}'s ctx for aggregation-level tracking)")
                        else:
                            logger.warning("Shared harness used but ctx not found - will build fallback RunContext")
                    except (NameError, StopIteration):
                        logger.warning("ctx not in scope - will build fallback RunContext")
                
                if ctx_to_use is None:
                    # Build RunContext from available data (fallback for per-symbol processing)
                    # FIX: Try to get X, y, time_vals from available sources (per-symbol contexts, cohort_context, etc.)
                    X_for_ctx = None
                    y_for_ctx = None
                    time_vals_for_ctx = None
                    feature_names_for_ctx = selected_features if selected_features else []
                    horizon_minutes_for_ctx = None
                    purge_minutes_for_ctx = None
                    embargo_minutes_for_ctx = None
                    folds_for_ctx = None
                    
                    # Try to get data from symbol_ctx_map (for SYMBOL_SPECIFIC view with per-symbol processing)
                    min_cs_for_ctx = None
                    max_cs_samples_for_ctx = None
                    if view == "SYMBOL_SPECIFIC" and 'symbol_ctx_map' in locals() and symbol_ctx_map:
                        # Use first symbol's ctx data (aggregation-level tracking)
                        first_symbol = next(iter(symbol_ctx_map.keys()))
                        first_ctx = symbol_ctx_map[first_symbol]
                        X_for_ctx = first_ctx.X
                        y_for_ctx = first_ctx.y
                        time_vals_for_ctx = first_ctx.time_vals
                        horizon_minutes_for_ctx = first_ctx.horizon_minutes
                        purge_minutes_for_ctx = first_ctx.purge_minutes
                        embargo_minutes_for_ctx = first_ctx.embargo_minutes
                        folds_for_ctx = first_ctx.folds
                        min_cs_for_ctx = first_ctx.min_cs  # FIX: Get from ctx for diff telemetry
                        max_cs_samples_for_ctx = first_ctx.max_cs_samples  # FIX: Get from ctx for diff telemetry
                    
                    # Fallback: Try to get from shared harness variables if available
                    if X_for_ctx is None and use_shared_harness:
                        if 'X' in locals() and X is not None:
                            X_for_ctx = X
                        if 'y' in locals() and y is not None:
                            y_for_ctx = y
                        if 'time_vals' in locals() and time_vals is not None:
                            time_vals_for_ctx = time_vals
                        if 'horizon_minutes' in locals() and horizon_minutes is not None:
                            horizon_minutes_for_ctx = horizon_minutes
                    
                    # Fallback: Try to get from cohort_context
                    if X_for_ctx is None and 'cohort_context' in locals() and cohort_context:
                        X_for_ctx = cohort_context.get('X')
                        y_for_ctx = cohort_context.get('y')
                        time_vals_for_ctx = cohort_context.get('time_vals')
                        if min_cs_for_ctx is None:
                            min_cs_for_ctx = cohort_context.get('min_cs')  # FIX: Get from cohort_context for diff telemetry
                        if max_cs_samples_for_ctx is None:
                            max_cs_samples_for_ctx = cohort_context.get('max_cs_samples')  # FIX: Get from cohort_context for diff telemetry
                        if 'horizon_minutes' not in locals() or horizon_minutes_for_ctx is None:
                            # Try to extract from target if not already set
                            if target_column:
                                try:
                                    from TRAINING.ranking.utils.leakage_filtering import _extract_horizon, _load_leakage_config
                                    leakage_config = _load_leakage_config()
                                    horizon_minutes_for_ctx = _extract_horizon(target_column, leakage_config)
                                except Exception:
                                    pass
                    
                    # Fallback: Try to get from resolved_config if available
                    if min_cs_for_ctx is None and 'resolved_config' in locals() and resolved_config:
                        min_cs_for_ctx = getattr(resolved_config, 'effective_min_cs', None) or getattr(resolved_config, 'requested_min_cs', None)
                    if max_cs_samples_for_ctx is None and 'resolved_config' in locals() and resolved_config:
                        max_cs_samples_for_ctx = getattr(resolved_config, 'max_cs_samples', None)
                    
                    # Final fallback: Extract horizon from target column name if still not set
                    if horizon_minutes_for_ctx is None and target_column:
                        try:
                            from TRAINING.ranking.utils.leakage_filtering import _extract_horizon, _load_leakage_config
                            leakage_config = _load_leakage_config()
                            horizon_minutes_for_ctx = _extract_horizon(target_column, leakage_config)
                        except Exception:
                            pass
                    
                    # FIX: Ensure view and symbol are set for proper telemetry scoping
                    # Telemetry must be scoped by: target, view (CROSS_SECTIONAL vs SYMBOL_SPECIFIC), and symbol
                    # CRITICAL: For CROSS_SECTIONAL, symbol must be None to prevent history forking
                    # Use SST-resolved view_for_writes if available (handles auto-flip case)
                    effective_view_for_ctx = view_for_writes if 'view_for_writes' in locals() else view
                    symbol_for_ctx = symbol if effective_view_for_ctx == "SYMBOL_SPECIFIC" else None
                    # FIX: Get universe_sig from SST-resolved value
                    universe_sig_for_ctx = universe_sig_for_writes if 'universe_sig_for_writes' in locals() else (universe_sig if 'universe_sig' in locals() else None)
                    # Get seed from config for reproducibility
                    try:
                        from CONFIG.config_loader import get_cfg
                        seed_value = get_cfg("pipeline.determinism.base_seed", default=42)
                    except Exception:
                        seed_value = 42
                    
                    ctx_to_use = RunContext(
                        stage="FEATURE_SELECTION",
                        target=target_column,
                        target_column=target_column,
                        X=X_for_ctx,  # FIX: Populated from available sources (symbol_ctx_map, shared harness, or cohort_context)
                        y=y_for_ctx,  # FIX: Populated from available sources
                        feature_names=feature_names_for_ctx,
                        symbols=symbols,
                        time_vals=time_vals_for_ctx,  # FIX: Populated from available sources
                        horizon_minutes=horizon_minutes_for_ctx,  # Extract from target if available
                        purge_minutes=purge_minutes_for_ctx,  # FIX: Use from symbol_ctx_map if available
                        embargo_minutes=embargo_minutes_for_ctx,  # FIX: Use from symbol_ctx_map if available
                        folds=folds_for_ctx,  # FIX: Use from symbol_ctx_map if available
                        fold_timestamps=None,
                        data_interval_minutes=None,
                        seed=seed_value,
                        view=effective_view_for_ctx,  # FIX: Use SST-resolved view (handles auto-flip from CS to SS)
                        symbol=symbol_for_ctx,  # FIX: Set symbol for SYMBOL_SPECIFIC view only (None for CROSS_SECTIONAL to prevent history forking)
                        min_cs=min_cs_for_ctx,  # FIX: Populate for diff telemetry
                        max_cs_samples=max_cs_samples_for_ctx,  # FIX: Populate for diff telemetry
                        universe_sig=universe_sig_for_ctx  # FIX: Pass universe_sig for proper scope tracking
                    )
                
                # Build metrics dict
                metrics_dict = {
                    "metric_name": "Consensus Score",
                    "auc": mean_consensus,
                    "std_score": std_consensus,
                    "mean_importance": top_feature_score,
                    "composite_score": mean_consensus,
                    "n_features_selected": n_features_selected,
                    "n_successful_families": n_successful_families,
                    "n_selected": n_features_selected  # For trend analyzer
                }
                
                # Use automated log_run API (includes trend analysis)
                # FIX: Pass RunContext to log_run (required for COHORT_AWARE mode)
                # Pass hyperparameters via additional_data_override
                try:
                    audit_result = tracker.log_run(
                        ctx_to_use, metrics_dict,
                        additional_data_override=additional_data_override,
                        run_identity=run_identity,  # SST: Pass through authoritative identity
                    )
                except Exception as e:
                    # If COHORT_AWARE fails due to missing fields, fall back to legacy mode
                    if "Missing required fields" in str(e) or "COHORT_AWARE" in str(e):
                        logger.debug(f"COHORT_AWARE mode failed (missing fields), using legacy tracking: {e}")
                        # Disable COHORT_AWARE and retry with minimal context
                        # FIX: Ensure view and symbol are set for proper telemetry scoping
                        # CRITICAL: For CROSS_SECTIONAL, symbol must be None to prevent history forking
                        # Use SST-resolved view_for_writes if available (handles auto-flip case)
                        effective_view_fallback = view_for_writes if 'view_for_writes' in locals() else view
                        symbol_for_ctx = symbol if effective_view_fallback == "SYMBOL_SPECIFIC" else None
                        # FIX: Get universe_sig from SST-resolved value (same as main path)
                        universe_sig_fallback = universe_sig_for_writes if 'universe_sig_for_writes' in locals() else (universe_sig if 'universe_sig' in locals() else None)
                        # Get seed from config for reproducibility (same as above)
                        try:
                            from CONFIG.config_loader import get_cfg
                            seed_value = get_cfg("pipeline.determinism.base_seed", default=42)
                        except Exception:
                            seed_value = 42
                        
                        ctx_minimal = RunContext(
                            stage="FEATURE_SELECTION",
                            target=target_column,
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
                            view=effective_view_fallback,  # FIX: Use SST-resolved view (handles auto-flip from CS to SS)
                            symbol=symbol_for_ctx,  # FIX: Set symbol for SYMBOL_SPECIFIC view only (None for CROSS_SECTIONAL)
                            universe_sig=universe_sig_fallback  # FIX: Pass universe_sig for proper scope tracking
                        )
                        audit_result = tracker.log_run(
                            ctx_minimal, metrics_dict,
                            run_identity=run_identity,  # SST: Pass through authoritative identity
                        )
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
                
                # FIX: Ensure cohort variables are initialized (may not exist if exception occurred before initialization)
                if 'cohort_metadata' not in locals():
                    cohort_metadata = None
                if 'cohort_metrics' not in locals():
                    cohort_metrics = {}
                if 'cohort_additional_data' not in locals():
                    cohort_additional_data = {}
                
                # Merge with existing metrics and additional_data
                # Use cohort_metrics if available (may be empty dict if extraction failed)
                metrics_with_cohort = {
                    "metric_name": "Consensus Score",
                    "auc": mean_consensus,
                    "std_score": std_consensus,
                    "mean_importance": top_feature_score,  # Use top feature score as importance proxy
                    "composite_score": mean_consensus,  # Use mean consensus as composite
                    "n_features_selected": n_features_selected,
                    "n_successful_families": n_successful_families,
                    "n_selected": n_features_selected,  # For trend analyzer
                }
                # Add cohort_metrics if available (safe to unpack empty dict)
                if cohort_metrics:
                    metrics_with_cohort.update(cohort_metrics)
                
                # PATCH 0: Map SST view_for_writes to view (not caller view)
                # Use SYMBOL_SPECIFIC directly to match directory structure (not INDIVIDUAL)
                view_for_legacy = None
                effective_view = view_for_writes if 'view_for_writes' in locals() else view
                if effective_view:
                    if effective_view.upper() == "CROSS_SECTIONAL":
                        view_for_legacy = "CROSS_SECTIONAL"
                    else:
                        view_for_legacy = "SYMBOL_SPECIFIC"
                
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
                    "view": view_for_legacy,  # FIX: Map view to view
                    'lookback_cap_enforcement': lookback_cap_metadata if lookback_cap_metadata else None
                }
                # Add cohort_additional_data if available (safe to unpack empty dict)
                if cohort_additional_data:
                    additional_data_with_cohort.update(cohort_additional_data)
                
                # Add seed for reproducibility tracking
                try:
                    from CONFIG.config_loader import get_cfg
                    seed = get_cfg("pipeline.determinism.base_seed", default=42)
                    additional_data_with_cohort['seed'] = seed
                    additional_data_with_cohort['train_seed'] = seed  # Also pass as train_seed for FEATURE_SELECTION
                except Exception:
                    # Fallback to default if config not available
                    additional_data_with_cohort['seed'] = 42
                    additional_data_with_cohort['train_seed'] = 42
                
                # Extract hyperparameters from model_families_config for reproducibility tracking
                # CRITICAL: Different hyperparameters = different features selected
                if model_families_config:
                    # Collect hyperparameters from all enabled model families
                    # Since we use multiple models, we'll collect all unique hyperparameters
                    # and create a combined dict (or use the primary model's config)
                    training_config = {}
                    primary_family = None
                    
                    # Find primary model family (usually lightgbm or first enabled)
                    enabled_families = [f for f, cfg in model_families_config.items() 
                                      if isinstance(cfg, dict) and cfg.get('enabled', False)]
                    if enabled_families:
                        # Prefer lightgbm as primary, otherwise use first enabled
                        if 'lightgbm' in enabled_families:
                            primary_family = 'lightgbm'
                        else:
                            primary_family = enabled_families[0]
                        
                        # Extract hyperparameters from primary family's config
                        if primary_family in model_families_config:
                            family_config = model_families_config[primary_family]
                            if isinstance(family_config, dict) and 'config' in family_config:
                                hp_config = family_config['config']
                                # Copy all hyperparameters (exclude non-hyperparameter keys)
                                excluded_keys = {'verbose', 'verbosity', 'objective', 'metric', 'device', 'gpu_device_id'}
                                for key, value in hp_config.items():
                                    if key not in excluded_keys and value is not None:
                                        training_config[key] = value
                    
                    # If we have hyperparameters, add them to additional_data
                    if training_config:
                        additional_data_with_cohort['training'] = training_config
                
                # Add library_versions for diff telemetry comparison group
                # CRITICAL: Different library versions = different feature selection outcomes
                try:
                    from TRAINING.common.utils.config_hashing import get_library_versions
                    library_versions = get_library_versions()
                    if library_versions:
                        additional_data_with_cohort['library_versions'] = library_versions
                except ImportError:
                    # Fallback: collect versions manually
                    import sys
                    library_versions = {'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"}
                    try:
                        import lightgbm
                        library_versions['lightgbm'] = lightgbm.__version__
                    except ImportError:
                        pass
                    try:
                        import sklearn
                        library_versions['sklearn'] = sklearn.__version__
                    except ImportError:
                        pass
                    try:
                        import numpy
                        library_versions['numpy'] = numpy.__version__
                    except ImportError:
                        pass
                    try:
                        import pandas
                        library_versions['pandas'] = pandas.__version__
                    except ImportError:
                        pass
                    additional_data_with_cohort['library_versions'] = library_versions
                except Exception:
                    pass  # Non-critical, diff telemetry will warn but continue
                
                if not model_families_config and multi_model_config and isinstance(multi_model_config, dict):
                    # Fallback: try to extract from multi_model_config
                    # This is a legacy format, but try to extract hyperparameters if available
                    if 'model_families' in multi_model_config:
                        families = multi_model_config['model_families']
                        if isinstance(families, dict):
                            # Try to get lightgbm config
                            if 'lightgbm' in families and isinstance(families['lightgbm'], dict):
                                lgb_config = families['lightgbm'].get('config', {})
                                if isinstance(lgb_config, dict):
                                    training_config = {}
                                    excluded_keys = {'verbose', 'verbosity', 'objective', 'metric'}
                                    for key, value in lgb_config.items():
                                        if key not in excluded_keys and value is not None:
                                            training_config[key] = value
                                    if training_config:
                                        additional_data_with_cohort['training'] = training_config
                
                # Add resolved_data_config (mode, loader contract) to additional_data for telemetry
                if 'resolved_data_config' in locals() and resolved_data_config:
                    additional_data_with_cohort['view'] = resolved_data_config.get('view') or resolved_data_config.get('resolved_data_mode')
                    additional_data_with_cohort['view_reason'] = resolved_data_config.get('view_reason')
                    additional_data_with_cohort['loader_contract'] = resolved_data_config.get('loader_contract')
                
                # PATCH 0: Create WriteScope from available context for type-safe scope handling
                from TRAINING.orchestration.utils.scope_resolution import (
                    WriteScope, ScopePurpose, Stage, View
                )
                
                # Determine scope from SST-derived values
                scope_view = view_for_writes if 'view_for_writes' in locals() else view
                scope_symbol = symbol_for_writes if 'symbol_for_writes' in locals() else None
                scope_universe_sig = universe_sig if 'universe_sig' in locals() else None
                
                # Create WriteScope if we have required data
                scope = None
                if scope_universe_sig:
                    try:
                        if scope_view == "CROSS_SECTIONAL" or scope_symbol is None:
                            scope = WriteScope.for_cross_sectional(
                                universe_sig=scope_universe_sig,
                                stage=Stage.FEATURE_SELECTION,
                                purpose=ScopePurpose.FINAL
                            )
                        else:
                            scope = WriteScope.for_symbol_specific(
                                universe_sig=scope_universe_sig,
                                symbol=scope_symbol,
                                stage=Stage.FEATURE_SELECTION,
                                purpose=ScopePurpose.FINAL
                            )
                    except ValueError as e:
                        logger.warning(f"Failed to create WriteScope: {e}")
                
                # Use WriteScope to populate additional_data correctly
                if scope:
                    scope.to_additional_data(additional_data_with_cohort)
                else:
                    # Legacy fallback (deprecated)
                    from TRAINING.orchestration.utils.scope_resolution import populate_additional_data
                    populate_additional_data(
                        additional_data_with_cohort,
                        view_for_writes=scope_view,
                        symbol_for_writes=scope_symbol,
                        universe_sig_for_writes=scope_universe_sig
                    )
                
                # Add fields needed for enhanced metadata (matching target ranking)
                if selected_features:
                    additional_data_with_cohort['feature_names'] = selected_features
                    additional_data_with_cohort['n_features'] = len(selected_features)  # Required for diff_telemetry validation
                
                # Add data interval for data_source metadata
                if 'detected_interval' in locals() and detected_interval is not None:
                    additional_data_with_cohort['data_interval_minutes'] = detected_interval
                elif explicit_interval:
                    # Convert explicit_interval to minutes if it's a string like "5m"
                    try:
                        if isinstance(explicit_interval, str) and explicit_interval.endswith('m'):
                            additional_data_with_cohort['data_interval_minutes'] = float(explicit_interval[:-1])
                        elif isinstance(explicit_interval, (int, float)):
                            additional_data_with_cohort['data_interval_minutes'] = float(explicit_interval)
                    except Exception:
                        pass
                
                # Add feature registry hash if available (for comparable_key)
                try:
                    from TRAINING.common.feature_registry import get_registry
                    registry = get_registry()
                    if registry and selected_features:
                        # Compute hash from feature names (sorted for stability)
                        feature_names_sorted = sorted([str(f) for f in selected_features])
                        feature_registry_str = "|".join(feature_names_sorted)
                        import hashlib
                        additional_data_with_cohort['feature_registry_hash'] = hashlib.sha256(feature_registry_str.encode()).hexdigest()[:16]
                except Exception:
                    pass
                
                # Use WriteScope-derived values for tracker call
                # NEW: Pass run_identity for authoritative signatures in snapshot
                tracker.log_comparison(
                    stage=scope.stage.value if scope else "FEATURE_SELECTION",  # FIX: Use uppercase
                    target=target_column,
                    metrics=metrics_with_cohort,
                    additional_data=additional_data_with_cohort,
                    view=scope.view.value if scope else scope_view,
                    symbol=scope.symbol if scope else scope_symbol,
                    run_identity=run_identity,  # NEW: Pass RunIdentity SST object
                )
        except Exception as e:
            # FIX: Ensure cohort variables exist before logging (may not be initialized if exception occurred early)
            if 'cohort_metadata' not in locals():
                cohort_metadata = None
            if 'cohort_metrics' not in locals():
                cohort_metrics = {}
            if 'cohort_additional_data' not in locals():
                cohort_additional_data = {}
            
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
    
    # NOTE: Metrics rollups should be generated ONCE at the run level after ALL targets are processed,
    # not per-target. This is handled by the orchestrator (intelligent_trainer.py) after all feature
    # selections complete. Per-target rollups would create duplicate/conflicting trend data.
    # If you need per-target rollups, they should be generated separately at the target level.
    
    # Save results to cache after Phase 2 completion
    if output_dir and selected_features:
        from TRAINING.common.utils.cache_manager import (
            build_cache_key_with_symbol,
            get_cache_path,
            save_cache
        )
        
        # Create cache key (same as check above)
        cache_key = build_cache_key_with_symbol(target_column, config_hash, view, symbol)
        cache_path = get_cache_path(output_dir, "feature_selection", cache_key)
        
        # Prepare cache data
        cache_data = {
            'target': target_column,
            'view': view,
            'symbol': symbol,
            'selected_features': selected_features,
            'summary_df': summary_df.to_dict('records') if summary_df is not None else None
        }
        
        # Save to cache using centralized utility
        save_cache(cache_path, cache_data, config_hash=config_hash, include_timestamp=True)
    
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
    # Load max_samples_per_symbol from config if not provided (same logic as target ranking)
    # Note: This function doesn't have experiment_config, so it will use pipeline config
    if max_samples_per_symbol is None:
        try:
            from CONFIG.config_loader import get_cfg
            # Use same config key as target ranking for consistency
            max_samples_per_symbol = int(get_cfg("pipeline.data_limits.default_max_rows_per_symbol_ranking", default=50000, config_name="pipeline_config"))
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

