# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Cross-sectional data loading utilities for ranking scripts.

This module provides functions to load and prepare cross-sectional data
that matches the training pipeline's data structure.
"""


import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
import logging
import warnings

from TRAINING.common.utils.fingerprinting import _compute_feature_fingerprint

logger = logging.getLogger(__name__)


def _log_feature_set(
    stage: str,
    feature_names: List[str],
    previous_names: Optional[List[str]] = None,
    logger_instance: Optional[logging.Logger] = None
) -> None:
    """
    Log feature set with fingerprint and delta tracking.
    
    Args:
        stage: Stage name (e.g., "SAFE_CANDIDATES", "AFTER_DROP_ALL_NAN")
        feature_names: Current feature names
        previous_names: Previous feature names (for delta computation)
        logger_instance: Logger to use (defaults to module logger)
    """
    if logger_instance is None:
        logger_instance = logger
    
    n_features = len(feature_names)
    set_fingerprint, order_fingerprint = _compute_feature_fingerprint(feature_names, set_invariant=True)
    fingerprint = set_fingerprint  # Use set-invariant for logging (backward compatibility)
    
    # Check for duplicates
    unique_names = set(feature_names)
    has_duplicates = len(unique_names) != n_features
    if has_duplicates:
        duplicates = [name for name in unique_names if feature_names.count(name) > 1]
        logger_instance.error(
            f"  🚨 FEATURESET [{stage}]: {n_features} features, fingerprint={fingerprint}, "
            f"DUPLICATES DETECTED: {duplicates}"
        )
        return
    
    # Compute delta if previous set provided
    if previous_names is not None:
        prev_set = set(previous_names)
        curr_set = set(feature_names)
        added = sorted(curr_set - prev_set)
        removed = sorted(prev_set - curr_set)
        
        # Check for order changes (if sets are equal but order differs)
        order_changed = False
        if not added and not removed and len(previous_names) == len(feature_names):
            prev_order_fp, _ = _compute_feature_fingerprint(previous_names, set_invariant=False)
            _, curr_order_fp = _compute_feature_fingerprint(feature_names, set_invariant=False)
            if prev_order_fp != curr_order_fp:
                order_changed = True
        
        if added or removed:
            delta_str = f", added={len(added)}, removed={len(removed)}"
            if added and len(added) <= 5:
                delta_str += f" (added: {added})"
            elif added:
                delta_str += f" (added: {added[:3]}... +{len(added)-3} more)"
            if removed and len(removed) <= 5:
                delta_str += f" (removed: {removed})"
            elif removed:
                delta_str += f" (removed: {removed[:3]}... +{len(removed)-3} more)"
        elif order_changed:
            delta_str = " (order changed)"
        else:
            delta_str = " (no changes)"
    else:
        delta_str = ""
    
    logger_instance.info(
        f"  📊 FEATURESET [{stage}]: n={n_features}, fingerprint={fingerprint}{delta_str}"
    )


def load_mtf_data_for_ranking(
    data_dir: Path,
    symbols: List[str],
    max_rows_per_symbol: Optional[int] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load MTF data for multiple symbols (matches training pipeline structure).
    
    Args:
        data_dir: Directory containing symbol data
        symbols: List of symbols to load
        max_rows_per_symbol: Optional limit on rows per symbol (most recent rows)
                            Default: None (load all). For ranking, use 10000-50000 for speed.
    
    Returns:
        Dictionary mapping symbol -> DataFrame
    """
    """
    Load MTF data for multiple symbols (matches training pipeline structure).
    
    Args:
        data_dir: Directory containing symbol data
        symbols: List of symbols to load
        max_rows_per_symbol: Optional limit on rows per symbol (most recent)
    
    Returns:
        Dictionary mapping symbol -> DataFrame
    """
    mtf_data = {}
    dropped_symbols = []  # Track dropped symbols with reasons
    
    for symbol in symbols:
        # Try different possible file locations (matching training pipeline)
        possible_paths = [
            data_dir / f"symbol={symbol}" / f"{symbol}.parquet",  # New structure
            data_dir / f"{symbol}.parquet",  # Direct file
            data_dir / f"{symbol}_mtf.parquet",  # Legacy format
        ]
        
        symbol_file = None
        for path in possible_paths:
            if path.exists():
                symbol_file = path
                break
        
        if symbol_file and symbol_file.exists():
            try:
                df = pd.read_parquet(symbol_file)
                
                # Check for empty DataFrame
                if df.empty:
                    dropped_symbols.append({
                        'symbol': symbol,
                        'reason': 'empty_dataframe',
                        'details': 'File exists but contains no rows'
                    })
                    logger.warning(f"Dropping {symbol}: empty DataFrame")
                    continue
                
                # Apply row limit if specified (most recent rows)
                if max_rows_per_symbol and len(df) > max_rows_per_symbol:
                    df = df.tail(max_rows_per_symbol)
                    logger.debug(f"Limited {symbol} to {max_rows_per_symbol} most recent rows")
                
                mtf_data[symbol] = df
                logger.debug(f"Loaded {symbol}: {df.shape}")
            except Exception as e:
                dropped_symbols.append({
                    'symbol': symbol,
                    'reason': 'load_error',
                    'details': str(e)
                })
                logger.error(f"Error loading {symbol}: {e}")
        else:
            dropped_symbols.append({
                'symbol': symbol,
                'reason': 'file_not_found',
                'details': f'Tried: {possible_paths}'
            })
            logger.warning(f"File not found for {symbol}. Tried: {possible_paths}")
    
    # Log loader contract (requested vs loaded)
    n_requested = len(symbols)
    n_loaded = len(mtf_data)
    loaded_symbols = list(mtf_data.keys())
    
    logger.info(f"📦 Loader contract: requested={n_requested} symbols → loaded={n_loaded} symbols")
    logger.info(f"   Loaded symbols: {loaded_symbols}")
    
    if dropped_symbols:
        logger.warning(f"   Dropped {len(dropped_symbols)} symbols:")
        for drop_info in dropped_symbols:
            logger.warning(f"     - {drop_info['symbol']}: {drop_info['reason']} ({drop_info['details']})")
    
    # Store loader contract in mtf_data metadata (for later use)
    if mtf_data:
        # Attach metadata as a special key (will be filtered out during processing)
        mtf_data['__loader_contract__'] = {
            'requested_symbols': symbols,
            'loaded_symbols': loaded_symbols,
            'n_requested': n_requested,
            'n_loaded': n_loaded,
            'dropped_symbols': dropped_symbols
        }
    
    return mtf_data


def prepare_cross_sectional_data_for_ranking(
    mtf_data: Dict[str, pd.DataFrame],
    target_column: str,
    min_cs: int = 10,
    max_cs_samples: Optional[int] = None,
    feature_names: Optional[List[str]] = None,
    feature_time_meta_map: Optional[Dict[str, Any]] = None,  # NEW: Optional map of feature_name -> FeatureTimeMeta
    base_interval_minutes: Optional[float] = None,  # NEW: Base training grid interval (for alignment)
    allow_single_symbol: bool = False,  # NEW: Allow single symbol for SYMBOL_SPECIFIC view
    requested_view: Optional[str] = None,  # SST: View requested by caller/config
    output_dir: Optional[Path] = None  # NEW: Output directory for persisting view (SST)
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[str]], Optional[np.ndarray], Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """
    Prepare cross-sectional data for ranking (simplified version of training pipeline).
    
    Args:
        mtf_data: Dictionary of symbol -> DataFrame
        target_column: Target column name
        min_cs: Minimum cross-sectional size per timestamp
        max_cs_samples: Maximum samples per timestamp (default: 1000)
        feature_names: Optional list of feature names (auto-discovered if None)
    
    Returns:
        Tuple of (X, y, feature_names, symbols, time_vals) or (None,)*5 on error
    """
    if not mtf_data:
        logger.error("No data provided")
        return (None,) * 6
    
    # Default max_cs_samples to match training pipeline
    if max_cs_samples is None:
        max_cs_samples = 1000
        logger.info(f"Using default max_cs_samples={max_cs_samples}")
    
    logger.info(f"🎯 Building cross-sectional data for target: {target_column}")
    
    # Extract loader contract if available (and remove from mtf_data)
    loader_contract = mtf_data.pop('__loader_contract__', None)
    
    # CRITICAL: Always log symbol load report (requested vs loaded vs dropped)
    # This prevents confusion and makes regressions obvious
    n_symbols_available = len(mtf_data)
    loaded_symbols_list = list(mtf_data.keys())
    
    if loader_contract:
        n_requested = loader_contract['n_requested']
        requested_symbols_list = loader_contract.get('requested_symbols', [])
        dropped_symbols = loader_contract.get('dropped_symbols', [])
        
        # Build structured symbol load report
        logger.info(f"📦 Symbol load report:")
        logger.info(f"   Requested: {n_requested} symbols {requested_symbols_list}")
        logger.info(f"   Loaded: {n_symbols_available} symbols {loaded_symbols_list}")
        
        if dropped_symbols:
            dropped_dict = {d['symbol']: d.get('reason', 'unknown') for d in dropped_symbols}
            logger.warning(f"   Dropped: {len(dropped_symbols)} symbols {dropped_dict}")
        else:
            logger.info(f"   Dropped: 0 symbols")
    else:
        # Fallback: if loader contract not available, still log what we have
        logger.info(f"📦 Symbol load report:")
        logger.info(f"   Requested: unknown (loader contract not available)")
        logger.info(f"   Loaded: {n_symbols_available} symbols {loaded_symbols_list}")
        logger.warning(f"   Dropped: unknown (loader contract not available)")
        # Create minimal loader contract for error messages
        loader_contract = {
            'requested_symbols': [],
            'n_requested': None,
            'loaded_symbols': loaded_symbols_list,
            'n_loaded': n_symbols_available,
            'dropped_symbols': []
        }
    
    # CRITICAL: Enforce minimum symbols BEFORE building cross-sectional data
    # Cross-sectional ranking with too few symbols should hard-stop, not degrade silently
    # 
    # IMPORTANT DISTINCTION:
    #   - This check enforces: "N symbols loaded overall" (global availability)
    #   - Later, per-timestamp filtering enforces: "effective cross-sectional width per timestamp"
    #   - The hard-stop prevents "only 1-2 symbols total", while per-timestamp sampling enforces cross-sectional width
    # 
    # NOTE: For LOSO view:
    #   - This function is called with mtf_data containing (N-1) training symbols (validation symbol excluded)
    #   - The check is applied to the training set size (N-1), which is correct
    #   - We need at least MIN_SYMBOLS symbols loaded in the training set (global availability)
    #   - Per-timestamp filtering (below) ensures each timestamp has >= effective_min_cs symbols present
    #   - The validation symbol is loaded separately with min_cs=1 (see evaluate_target_predictability)
    # 
    # For CROSS_SECTIONAL view:
    #   - This function is called with mtf_data containing all N symbols
    #   - The check ensures we have at least MIN_SYMBOLS symbols loaded overall (global availability)
    #   - Per-timestamp filtering (below) ensures each timestamp has >= effective_min_cs symbols present
    
    # Minimum symbols required for meaningful cross-sectional analysis
    # Hard minimum: 3 symbols (below this, it's not truly cross-sectional)
    # Recommended: 10+ symbols for robust cross-sectional ranking
    # Exception: allow_single_symbol=True for SYMBOL_SPECIFIC view (intentional single-symbol time series)
    MIN_SYMBOLS_REQUIRED = 3
    RECOMMENDED_SYMBOLS = 10
    
    # Skip symbol count check for SYMBOL_SPECIFIC view (intentional single-symbol)
    if not allow_single_symbol:
        if n_symbols_available < MIN_SYMBOLS_REQUIRED:
            error_msg = (
                f"CROSS_SECTIONAL mode requires >= {MIN_SYMBOLS_REQUIRED} symbols, but only {n_symbols_available} loaded. "
                f"Loaded symbols: {loaded_symbols_list}. "
                f"This would silently degrade into single-symbol time series masquerading as cross-sectional ranking. "
                f"Use SYMBOL_SPECIFIC mode for single-symbol ranking, or ensure sufficient symbols are available."
            )
            # Always include dropped symbols info if available
            if loader_contract and loader_contract.get('dropped_symbols'):
                dropped_dict = {d['symbol']: d.get('reason', 'unknown') for d in loader_contract['dropped_symbols']}
                error_msg += f" Dropped symbols: {dropped_dict}"
            elif loader_contract and loader_contract.get('requested_symbols'):
                # If we have requested list, show what was requested but not loaded
                requested_set = set(loader_contract['requested_symbols'])
                loaded_set = set(loaded_symbols_list)
                missing = requested_set - loaded_set
                if missing:
                    error_msg += f" Missing from requested list: {sorted(missing)}"
            
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Warn if using fewer than recommended symbols (but allow it)
        if n_symbols_available < RECOMMENDED_SYMBOLS:
            logger.warning(
                f"⚠️  CROSS_SECTIONAL mode with {n_symbols_available} symbols (recommended: >= {RECOMMENDED_SYMBOLS}). "
                f"Cross-sectional ranking may be less robust with fewer symbols. "
                f"Consider using SYMBOL_SPECIFIC mode for more reliable per-symbol ranking."
            )
    else:
        # SYMBOL_SPECIFIC view: single symbol is expected and valid
        if n_symbols_available == 1:
            logger.debug(f"SYMBOL_SPECIFIC view: processing single symbol {loaded_symbols_list[0]}")
        elif n_symbols_available > 1:
            logger.warning(
                f"SYMBOL_SPECIFIC view expected 1 symbol, but {n_symbols_available} loaded: {loaded_symbols_list}. "
                f"Proceeding with all loaded symbols."
            )
    
    # Compute effective_min_cs (should equal min_cs now, but keep for consistency)
    effective_min_cs = min(min_cs, n_symbols_available)
    min_cs_reason = "requested"  # Always requested now (we hard-stop if insufficient)
    
    # Compute universe signature for this symbol set
    from TRAINING.orchestration.utils.run_context import (
        compute_universe_signature, get_view_for_universe, save_run_context, validate_view_contract
    )
    universe_sig = compute_universe_signature(loaded_symbols_list)
    symbols_sample = loaded_symbols_list[:3] if len(loaded_symbols_list) > 3 else loaded_symbols_list
    logger.info(f"🔑 Universe: sig={universe_sig} n_symbols={n_symbols_available} sample={symbols_sample}")
    
    # SST: Use requested_view if provided, fall back to deprecated requested_view
    effective_requested_view = requested_view or requested_view
    
    # Load existing view for THIS universe only (not global)
    existing_entry = None
    if output_dir is not None:
        try:
            existing_entry = get_view_for_universe(output_dir, universe_sig)
            if existing_entry:
                # SST: Read view from entry
                cached_view = existing_entry.get('view')
                logger.debug(f"Found cached view for universe={universe_sig}: {cached_view}")
        except Exception as e:
            logger.debug(f"Could not load existing run context: {e}")
    
    # Load view policy from config (with backward compat for view_policy)
    view_policy = "auto"  # Default
    auto_flip_min_symbols = RECOMMENDED_SYMBOLS  # Default
    try:
        from CONFIG.config_loader import get_cfg
        # New config keys (view_policy, requested_view) with fallback to old keys (view_policy, requested_view)
        view_policy = get_cfg("training_config.routing.view_policy", 
                              default=get_cfg("training_config.routing.view_policy", default="auto"))
        auto_flip_min_symbols = get_cfg("training_config.routing.auto_flip_min_symbols", default=RECOMMENDED_SYMBOLS)
        if effective_requested_view is None:
            effective_requested_view = get_cfg("training_config.routing.requested_view",
                                     default=get_cfg("training_config.routing.requested_view", default=None))
    except Exception as e:
        logger.debug(f"Could not load view policy from config: {e}, using defaults")
    
    # Determine resolved view based on policy
    # Only reuse if we have a cached entry for THIS universe
    if existing_entry:
        # Reuse cached view for this universe - reference stored original_reason directly (no nesting)
        view = existing_entry.get('view')
        original_reason = existing_entry.get("original_reason", "N/A")
        view_reason = f"reusing cached view for universe={universe_sig} (originally: {original_reason})"
    elif view_policy == "force":
        # Force view: use requested_view exactly (no auto-flip)
        if effective_requested_view is None:
            logger.warning("view_policy=force but requested_view not set, defaulting to CROSS_SECTIONAL")
            effective_requested_view = "CROSS_SECTIONAL"
        view = effective_requested_view
        view_reason = f"view_policy=force, requested_view={effective_requested_view}"
    else:
        # Auto mode: resolve fresh based on panel size
        if n_symbols_available == 1:
            if effective_requested_view and effective_requested_view != "SINGLE_SYMBOL_TS":
                view = effective_requested_view
                view_reason = f"n_symbols=1, using requested_view={effective_requested_view} (per-symbol loop)"
            else:
                view = "SINGLE_SYMBOL_TS"
                view_reason = "n_symbols=1"
        elif n_symbols_available < auto_flip_min_symbols:
            view = "SYMBOL_SPECIFIC"
            view_reason = f"n_symbols={n_symbols_available} (small panel, < {auto_flip_min_symbols} recommended)"
        else:
            view = "CROSS_SECTIONAL"
            view_reason = f"n_symbols={n_symbols_available} (full panel, >= {auto_flip_min_symbols})"
    
    # Validate view contract (only if we resolved a new view, not cached)
    if not existing_entry:
        try:
            validate_view_contract(view, effective_requested_view, view_policy)
        except ValueError as e:
            logger.error(f"View contract validation failed: {e}")
            raise
    
    # Set data_scope based on current n_symbols_available (can vary per-symbol, non-immutable)
    if n_symbols_available == 1:
        data_scope = "SINGLE_SYMBOL"
    else:
        data_scope = "PANEL"
    
    # Persist view and data_scope to run context (SST)
    # view is immutable PER UNIVERSE, data_scope can be updated
    if output_dir is not None:
        try:
            # For cached entries, pass the original_reason (not the "reusing..." message)
            save_view_reason = existing_entry["original_reason"] if existing_entry else view_reason
            save_run_context(
                output_dir=output_dir,
                view=view,
                requested_view=effective_requested_view,
                view_reason=save_view_reason,
                n_symbols=n_symbols_available,
                data_scope=data_scope,
                universe_signature=universe_sig,
                symbols=loaded_symbols_list
            )
        except Exception as e:
            logger.warning(f"Could not save run context: {e}")
    
    # Log requested vs effective (single authoritative line)
    data_type_label = "Cross-sectional sampling" if view == "CROSS_SECTIONAL" else "Panel data sampling"
    logger.info(
        f"📊 {data_type_label}: "
        f"requested_min_cs={min_cs} → effective_min_cs={effective_min_cs} "
        f"(reason={min_cs_reason}, n_symbols={n_symbols_available}), "
        f"max_cs_samples={max_cs_samples}"
    )
    logger.info(f"📋 View resolution: requested_view={effective_requested_view or 'N/A'}, view={view} (reason: {view_reason})")
    
    # NEW: Multi-interval alignment support
    # If feature_time_meta_map and base_interval_minutes are provided, apply alignment
    use_alignment = (
        feature_time_meta_map is not None 
        and base_interval_minutes is not None 
        and len(feature_time_meta_map) > 0
    )
    
    # Combine all symbol data (this becomes our base dataframe)
    all_data = []
    for symbol, df in mtf_data.items():
        if target_column not in df.columns:
            logger.debug(f"Skipping {symbol}: target '{target_column}' not found")
            continue
        
        df_copy = df.copy()
        df_copy['symbol'] = symbol
        all_data.append(df_copy)
    
    if not all_data:
        logger.error(f"Target '{target_column}' not found in any symbol")
        return (None,) * 6
    
    # CRITICAL: Ensure we have a time column for panel data validation
    # Without timestamps, we cannot use time-based purging and will cause data leakage
    
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Combined data shape: {combined_df.shape}")
    
    # Normalize time column name
    time_col = "timestamp" if "timestamp" in combined_df.columns else ("ts" if "ts" in combined_df.columns else None)
    
    if use_alignment and time_col is not None:
        # Import alignment function
        from TRAINING.ranking.utils.feature_alignment import align_features_asof
        
        # Separate features by whether they need alignment
        features_need_alignment = []
        features_no_alignment = []
        
        # Auto-discover features if not provided
        if feature_names is None:
            sample_cols = next(iter(mtf_data.values())).columns.tolist()
            feature_names = [col for col in sample_cols 
                            if not any(col.startswith(prefix) for prefix in 
                                     ['fwd_ret_', 'will_peak', 'will_valley', 'mdd_', 'mfe_', 'y_will_',
                                      'tth_', 'p_', 'barrier_', 'hit_'])
                            and col not in [time_col, target_column, 'symbol']]
        
        for feat_name in feature_names:
            if feat_name in feature_time_meta_map:
                meta = feature_time_meta_map[feat_name]
                native_interval = meta.native_interval_minutes or base_interval_minutes
                # Need alignment if different interval OR has embargo OR has publish_offset
                if (native_interval != base_interval_minutes or 
                    meta.embargo_minutes != 0.0 or 
                    meta.publish_offset_minutes != 0.0):
                    features_need_alignment.append(feat_name)
                else:
                    features_no_alignment.append(feat_name)
            else:
                features_no_alignment.append(feat_name)
                logger.debug(f"Feature {feat_name} not in feature_time_meta_map - using standard merge")
        
        if features_need_alignment:
            logger.info(
                f"🔧 Multi-interval alignment: {len(features_need_alignment)} features need alignment "
                f"(native intervals differ or have embargo/publish_offset), {len(features_no_alignment)} use standard merge"
            )
            
            # Extract feature DataFrames for alignment (from original mtf_data)
            feature_dfs = {}
            for feat_name in features_need_alignment:
                # Combine feature across all symbols
                feat_data = []
                for symbol, df in mtf_data.items():
                    if feat_name in df.columns and time_col in df.columns:
                        feat_df = df[[time_col, feat_name]].copy()
                        feat_df['symbol'] = symbol
                        feat_data.append(feat_df)
                if feat_data:
                    feature_dfs[feat_name] = pd.concat(feat_data, ignore_index=True)
                else:
                    logger.warning(f"Feature {feat_name} marked for alignment but not found in any symbol - skipping")
            
            if feature_dfs:
                # Use existing combined_df as base (not full cross product)
                # This is the base grid: all (symbol, timestamp) pairs that exist in the data
                base_df = combined_df[['symbol', time_col]].drop_duplicates().copy()
                
                # Align features onto existing base dataframe
                aligned_features_df = align_features_asof(
                    base_df,
                    feature_dfs,
                    {k: v for k, v in feature_time_meta_map.items() if k in features_need_alignment},
                    base_interval_minutes,
                    timestamp_column=time_col
                )
                
                # Merge aligned features back into combined_df
                # Drop aligned features from combined_df first (they'll be replaced by aligned versions)
                cols_to_drop = [f for f in features_need_alignment if f in combined_df.columns]
                if cols_to_drop:
                    combined_df = combined_df.drop(columns=cols_to_drop)
                
                # Merge aligned features
                aligned_cols = [f for f in features_need_alignment if f in aligned_features_df.columns]
                if aligned_cols:
                    combined_df = combined_df.merge(
                        aligned_features_df[['symbol', time_col] + aligned_cols],
                        on=['symbol', time_col],
                        how='left'
                    )
                
                # Log alignment stats
                for feat_name in aligned_cols:
                    null_rate = combined_df[feat_name].isna().mean()
                    if null_rate > 0:
                        logger.debug(f"  Aligned {feat_name}: null_rate={null_rate:.1%} (from embargo/staleness)")
            else:
                logger.warning("No features found for alignment - falling back to standard merge")
                use_alignment = False
        else:
            use_alignment = False
    
    # CRITICAL: Sort by timestamp IMMEDIATELY after combining
    # This ensures data is always sorted and prevents warnings later
    if time_col is not None:
        combined_df = combined_df.sort_values(time_col).reset_index(drop=True)
        logger.debug(f"Sorted combined data by {time_col}")
    
    # Enforce min_cs: filter timestamps that don't meet cross-sectional size
    # But be lenient: if we have fewer symbols than min_cs, use what we have
    if time_col is not None:
        cs = combined_df.groupby(time_col)["symbol"].transform("size")
        combined_df = combined_df[cs >= effective_min_cs]
        logger.debug(f"After effective_min_cs={effective_min_cs} filter: {combined_df.shape}")
        
        if len(combined_df) == 0:
            logger.warning(f"No data after min_cs filter - all timestamps have < {effective_min_cs} symbols")
            return (None,) * 6
        
        # Apply cross-sectional sampling per timestamp
        # CRITICAL: Shuffle symbols within each timestamp to avoid bias
        # If data is sorted alphabetically, we'd always sample AAPL, AMZN, etc. and miss ZZZ
        if max_cs_samples:
            # Add random shuffle column per timestamp group
            # CRITICAL FIX: Use deterministic timestamp-based seeding instead of hash()
            # hash() output changes every Python restart (salted for security), breaking reproducibility
            # Using timestamp integer ensures same shuffle for same timestamp across all runs
            def _get_deterministic_shuffle(group):
                """Generate deterministic shuffle key based on timestamp"""
                timestamp = group.name  # The timestamp value for this group
                # Convert timestamp to integer seed (works for pd.Timestamp, datetime, or numeric)
                if isinstance(timestamp, pd.Timestamp):
                    seed = int(timestamp.timestamp()) % (2**31)  # Use timestamp as seed
                elif isinstance(timestamp, (int, float)):
                    seed = int(timestamp) % (2**31)
                else:
                    # Fallback: use string hash but with fixed seed
                    seed = hash(str(timestamp)) % (2**31)
                return np.random.RandomState(seed).permutation(len(group))
            
            combined_df["_shuffle_key"] = combined_df.groupby(time_col)["symbol"].transform(_get_deterministic_shuffle)
            # Count timestamps that hit the cap before filtering
            timestamp_counts = combined_df.groupby(time_col).size()
            cap_hit_count = (timestamp_counts > max_cs_samples).sum()
            total_timestamps = len(timestamp_counts)
            
            combined_df = (combined_df
                           .sort_values([time_col, "_shuffle_key"])
                           .groupby(time_col, group_keys=False)
                           .head(max_cs_samples)
                           .drop(columns=["_shuffle_key"]))
            
            # INFO: Show shape + cap hit info (readable)
            if cap_hit_count > 0:
                logger.info(f"After max_cs_samples={max_cs_samples} filter: {combined_df.shape} "
                          f"(cap_hit: {cap_hit_count}/{total_timestamps} timestamps)")
            else:
                logger.info(f"After max_cs_samples={max_cs_samples} filter: {combined_df.shape} "
                          f"(cap_hit: 0/{total_timestamps} timestamps)")
            # DEBUG: Full timestamp-level detail
            if cap_hit_count > 0:
                logger.debug(f"max_cs_samples cap details: {cap_hit_count} timestamps exceeded limit "
                           f"(sample: {list(timestamp_counts[timestamp_counts > max_cs_samples].head(5).index)})")
            # Data is already sorted by [time_col, _shuffle_key], so it's sorted by time_col
    else:
        # CRITICAL: Panel data REQUIRES timestamps for time-based purging
        # Without timestamps, row-count purging causes catastrophic leakage (1 bar = N rows, not 1 row)
        logger.error("CRITICAL: No time column found in panel data. Time-based purging is REQUIRED.")
        logger.error("  Panel data structure: multiple symbols per timestamp means row-count purging is invalid.")
        logger.error("  Example: With 50 symbols, 1 bar = 50 rows. Purging 17 rows = ~20 seconds, not 60 minutes!")
        return (None,) * 6
    
    # Auto-discover features if not provided
    if feature_names is None:
        feature_names = [col for col in combined_df.columns 
                        if not any(col.startswith(prefix) for prefix in 
                                 ['fwd_ret_', 'will_peak', 'will_valley', 'mdd_', 'mfe_', 'y_will_',
                                  'tth_', 'p_', 'barrier_', 'hit_'])
                        and col not in (['symbol', time_col, target_column] if time_col else ['symbol', target_column])]
        logger.info(f"Auto-discovered {len(feature_names)} features")
    
    # Extract target
    if target_column not in combined_df.columns:
        logger.error(f"Target '{target_column}' not in combined data")
        return (None,) * 6
    
    y = combined_df[target_column].values
    y = pd.Series(y).replace([np.inf, -np.inf], np.nan).values
    
    # Extract features
    feature_df = combined_df[feature_names].copy()
    
    # Convert to numeric and handle infinities
    for col in feature_df.columns:
        feature_df.loc[:, col] = pd.to_numeric(feature_df[col], errors='coerce')
    feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Track feature counts for resolved_config
    features_safe = len(feature_names)  # Features before dropping NaN
    _log_feature_set("SAFE_CANDIDATES", feature_names)
    
    # Drop columns that are entirely NaN
    before_cols = feature_df.shape[1]
    feature_names_before_nan = feature_names.copy()
    feature_df = feature_df.dropna(axis=1, how='all')
    features_dropped_nan = before_cols - feature_df.shape[1]
    if features_dropped_nan:
        logger.debug(f"Dropped {features_dropped_nan} all-NaN feature columns")
        # Update feature_names to match
        dropped_nan_names = [f for f in feature_names if f not in feature_df.columns]
        feature_names = [f for f in feature_names if f in feature_df.columns]
        if dropped_nan_names:
            logger.debug(f"  Dropped all-NaN columns: {dropped_nan_names[:10]}{'...' if len(dropped_nan_names) > 10 else ''}")
    _log_feature_set("AFTER_DROP_ALL_NAN", feature_names, previous_names=feature_names_before_nan)
    
    features_final = len(feature_names)  # Features after all filtering
    
    # Ensure only numeric dtypes
    feature_names_before_numeric = feature_names.copy()
    numeric_cols = [c for c in feature_df.columns if pd.api.types.is_numeric_dtype(feature_df[c])]
    if len(numeric_cols) != feature_df.shape[1]:
        non_numeric_dropped = feature_df.shape[1] - len(numeric_cols)
        dropped_non_numeric = [c for c in feature_df.columns if c not in numeric_cols]
        feature_df = feature_df[numeric_cols]
        feature_names = [f for f in feature_names if f in numeric_cols]
        logger.info(f"🔧 Dropped {non_numeric_dropped} non-numeric feature columns: {dropped_non_numeric[:10]}{'...' if len(dropped_non_numeric) > 10 else ''}")
    # CRITICAL: Reindex DataFrame columns to match feature_names order (prevent order drift)
    # After cleaning, DataFrame columns might be reordered - enforce authoritative order
    # The feature_names list IS the authoritative order - DataFrame columns must match it
    if isinstance(feature_df, pd.DataFrame):
        # Reindex columns to match feature_names order exactly
        # This prevents "(order changed)" warnings and ensures deterministic column alignment
        feature_df = feature_df.loc[:, [f for f in feature_names if f in feature_df.columns]]
    
    _log_feature_set("AFTER_CLEANING", feature_names, previous_names=feature_names_before_numeric)
    
    # Check if we have any features left
    if len(feature_names) == 0:
        logger.error("❌ No features remaining after filtering - cannot train models")
        return (None,) * 6
    
    # Warn if very few features (may cause training issues)
    if len(feature_names) < 5:
        logger.warning(
            f"⚠️  Very few features ({len(feature_names)}) remaining after filtering. "
            f"Model training may fail or produce poor results. "
            f"Consider relaxing feature filtering rules or adding more features to the registry."
        )
    
    # Build feature matrix
    X = feature_df.to_numpy(dtype=np.float32, copy=False)
    
    # Check if we have any data
    if X.shape[0] == 0:
        logger.error("Feature matrix is empty - no data to process")
        return (None,) * 6
    
    # Clean data: remove rows with invalid target or too many NaN features
    target_valid = ~np.isnan(y) & np.isfinite(y)
    
    # Compute feature NaN ratio safely (suppress warning for empty slices)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if X.shape[0] > 0 and X.shape[1] > 0:
            feature_nan_ratio = np.isnan(X).mean(axis=1)
        else:
            feature_nan_ratio = np.ones(X.shape[0])  # All invalid if empty
            logger.warning("Feature matrix has zero columns - all features invalid")
    
    feature_valid = feature_nan_ratio <= 0.5  # Allow up to 50% NaN in features
    
    valid_mask = target_valid & feature_valid
    
    if not valid_mask.any():
        logger.error(f"No valid data after cleaning:")
        logger.error(f"  Target: {len(y)} total, {target_valid.sum()} valid ({np.isnan(y).sum()} NaN, {np.sum(~np.isfinite(y))} inf)")
        logger.error(f"  Features: {X.shape[0]} rows, {X.shape[1]} cols, {feature_valid.sum()} valid rows")
        if X.shape[0] > 0:
            logger.error(f"  Feature NaN: {np.isnan(X).sum()} total, mean per row: {feature_nan_ratio.mean():.2%}")
        return (None,) * 6
    
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]
    symbols_clean = combined_df['symbol'].values[valid_mask]
    time_vals = combined_df[time_col].values[valid_mask] if time_col else None
    
    # CRITICAL FIX: Do NOT impute here - this causes data leakage!
    # Imputation must happen INSIDE the CV loop (fit on train, transform test)
    # The imputation will be handled by sklearn Pipeline in train_and_evaluate_models
    # We only remove rows with >50% NaN features, but keep NaN values for proper CV imputation
    
    # CRITICAL: Ensure time_vals is sorted (required for PurgedTimeSeriesSplit)
    # Data should already be sorted from earlier steps, but verify and fix if needed
    if time_vals is not None and len(time_vals) > 1:
        time_series = pd.Series(time_vals) if not isinstance(time_vals, pd.Series) else time_vals
        if not time_series.is_monotonic_increasing:
            # Sort silently - data should be pre-sorted, but handle edge cases
            sort_idx = np.argsort(time_vals)
            X_clean = X_clean[sort_idx]
            y_clean = y_clean[sort_idx]
            symbols_clean = symbols_clean[sort_idx]
            time_vals = time_series.iloc[sort_idx].values if isinstance(time_series, pd.Series) else time_series[sort_idx]
            logger.debug(f"  Re-sorted data by timestamp (should be rare)")
    
    logger.info(f"✅ Cleaned data: {len(X_clean)} samples, {X_clean.shape[1]} features")
    logger.info(f"   Removed {len(X) - len(X_clean)} rows due to cleaning")
    logger.info(f"   ⚠️  Note: NaN values preserved for CV-safe imputation (no leakage)")
    
    # Store resolved view and loader contract in a metadata dict (for telemetry)
    # This will be extracted by callers and passed to reproducibility tracker
    resolved_config = {
        # SST canonical fields
        'view': view,
        'requested_view': effective_requested_view,
        'view_reason': view_reason,
        'view_policy': view_policy,
        # DEPRECATED aliases (for backward compat)
        'resolved_data_mode': view,  # DEPRECATED: Use view
        'view': view,  # DEPRECATED: Use view
        'requested_view': effective_requested_view,  # DEPRECATED: Use requested_view
        'view_reason': view_reason,  # DEPRECATED: Use view_reason
        'view_policy': view_policy,  # DEPRECATED: Use view_policy
        # Other fields
        'data_scope': data_scope,  # Data scope (PANEL or SINGLE_SYMBOL) - can vary per-symbol
        'n_symbols_loaded': n_symbols_available,
        'min_cs_required': min_cs,
        'effective_min_cs': effective_min_cs,
        'loader_contract': loader_contract,
        # SST for scope partitioning - universe_sig is born here from loaded (not requested) symbols
        'universe_sig': universe_sig,
        'loaded_symbols': loaded_symbols_list,
        'requested_symbols': loader_contract.get('requested_symbols') if loader_contract else None,
    }
    
    # Attach to mtf_data for callers to extract (non-intrusive)
    # Callers can access via: resolved_config = mtf_data.get('__resolved_config__')
    # But we don't have mtf_data in return, so we'll need to pass this separately
    # For now, log it and callers can extract from logs or we'll add it to return later
    
    return X_clean, y_clean, feature_names, symbols_clean, time_vals, resolved_config

