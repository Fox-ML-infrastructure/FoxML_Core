"""
Shared Ranking Harness

Unified harness for both target ranking and feature selection that ensures:
- Same split policy (PurgedTimeSeriesSplit with time-based purging)
- Same model evaluation (train_and_evaluate_models)
- Same telemetry (RunContext, reproducibility tracking)
- Same data sanitization and dtype canonicalization

This prevents "in-sample-ish" mistakes from scaling and ensures both paths
use identical evaluation contracts.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RankingHarness:
    """
    Shared harness for ranking operations (target ranking and feature selection).
    
    Ensures both use:
    - Same split generator (walk-forward / purged, same embargo, same grouping)
    - Same scoring function + metric normalization
    - Same leakage-safe imputation policy
    - Same RunContext + reproducibility tracker payload
    - Same logging/artifact writer
    """
    
    def __init__(
        self,
        job_type: str,  # "rank_targets" or "rank_features"
        target_column: str,
        symbols: List[str],
        data_dir: Path,
        model_families: List[str],
        multi_model_config: Dict[str, Any] = None,
        output_dir: Optional[Path] = None,
        view: str = "CROSS_SECTIONAL",  # "CROSS_SECTIONAL", "SYMBOL_SPECIFIC", "LOSO"
        symbol: Optional[str] = None,  # Required for SYMBOL_SPECIFIC and LOSO views
        explicit_interval: Optional[Union[int, str]] = None,
        experiment_config: Optional[Any] = None,
        min_cs: Optional[int] = None,
        max_cs_samples: Optional[int] = None,
        max_rows_per_symbol: Optional[int] = None
    ):
        """
        Initialize the ranking harness.
        
        Args:
            job_type: "rank_targets" or "rank_features"
            target_column: Target column name
            symbols: List of symbols to process
            data_dir: Directory containing symbol data
            model_families: List of model family names
            multi_model_config: Multi-model configuration dict
            output_dir: Optional output directory for results
            view: Evaluation view ("CROSS_SECTIONAL", "SYMBOL_SPECIFIC", "LOSO")
            symbol: Symbol name (required for SYMBOL_SPECIFIC and LOSO views)
            explicit_interval: Explicit data interval (e.g., "5m" or 5)
            experiment_config: Optional ExperimentConfig
            min_cs: Minimum cross-sectional samples
            max_cs_samples: Maximum cross-sectional samples
            max_rows_per_symbol: Maximum rows per symbol
        """
        self.job_type = job_type
        self.target_column = target_column
        self.symbols = symbols
        self.data_dir = data_dir
        self.model_families = model_families
        self.multi_model_config = multi_model_config or {}
        self.output_dir = output_dir
        self.view = view
        self.symbol = symbol
        self.explicit_interval = explicit_interval
        self.experiment_config = experiment_config
        self.min_cs = min_cs
        self.max_cs_samples = max_cs_samples
        self.max_rows_per_symbol = max_rows_per_symbol
        
        # Validate view and symbol parameters
        if view == "SYMBOL_SPECIFIC" and symbol is None:
            raise ValueError(f"symbol parameter required for SYMBOL_SPECIFIC view")
        if view == "LOSO" and symbol is None:
            raise ValueError(f"symbol parameter required for LOSO view")
        if view == "CROSS_SECTIONAL" and symbol is not None:
            logger.warning(f"symbol={symbol} provided but view=CROSS_SECTIONAL, ignoring symbol")
            self.symbol = None
    
    def build_panel(
        self,
        target_column: str,
        target_name: Optional[str] = None,
        feature_names: Optional[List[str]] = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[str]], 
               Optional[np.ndarray], Optional[np.ndarray], Optional[Dict[str, pd.DataFrame]], 
               Optional[float], Optional[Any], Optional[Dict[str, Any]]]:
        """
        Build panel data (X, y, feature_names, symbols_array, time_vals, mtf_data, detected_interval, resolved_config, resolved_data_config).
        
        This is the same data build logic used by target ranking, including:
        - Target-conditional exclusions
        - Leakage filtering with registry validation
        - Resolved config creation
        
        Args:
            target_column: Target column name
            target_name: Optional target name (for exclusion list generation)
            feature_names: Optional list of feature names to use (None = all safe features)
        
        Returns:
            Tuple of 9 values: (X, y, feature_names, symbols_array, time_vals, mtf_data, detected_interval, resolved_config, resolved_data_config)
            - X: Feature matrix (np.ndarray)
            - y: Target array (np.ndarray)
            - feature_names: List of feature names
            - symbols_array: Symbol array (np.ndarray)
            - time_vals: Timestamp array (np.ndarray)
            - mtf_data: Multi-timeframe data dict (Dict[str, pd.DataFrame])
            - detected_interval: Detected data interval in minutes (float)
        
        PERFORMANCE AUDIT: This function is tracked for call counts and timing.
            - resolved_config: ResolvedConfig object with purge/embargo settings
            - resolved_data_config: Dict with resolved data mode and loader contract
            Any can be None if data preparation fails
        """
        from TRAINING.ranking.utils.cross_sectional_data import (
            load_mtf_data_for_ranking,
            prepare_cross_sectional_data_for_ranking
        )
        from TRAINING.ranking.utils.leakage_filtering import filter_features_for_target, _extract_horizon, _load_leakage_config
        from TRAINING.ranking.utils.data_interval import detect_interval_from_dataframe
        from TRAINING.ranking.utils.target_conditional_exclusions import (
            generate_target_exclusion_list,
            load_target_exclusion_list
        )
        from TRAINING.ranking.utils.resolved_config import create_resolved_config
        from TRAINING.ranking.predictability.scoring import TaskType
        
        # PERFORMANCE AUDIT: Track build_panel calls
        import time
        build_start_time = time.time()
        try:
            from TRAINING.common.utils.performance_audit import get_auditor
            auditor = get_auditor()
            if auditor.enabled:
                fingerprint_kwargs = {
                    'target': target_column,
                    'n_symbols': len(self.symbols),
                    'view': self.view,
                    'symbol': self.symbol,
                    'n_features_requested': len(feature_names) if feature_names else None
                }
                fingerprint = auditor._compute_fingerprint('RankingHarness.build_panel', **fingerprint_kwargs)
        except Exception:
            auditor = None
            fingerprint = None
        
        # Filter symbols based on view
        symbols_to_load = self.symbols
        if self.view == "SYMBOL_SPECIFIC":
            symbols_to_load = [self.symbol]
        elif self.view == "LOSO":
            # LOSO: train on all symbols except symbol, validate on symbol
            symbols_to_load = [s for s in self.symbols if s != self.symbol]
            validation_symbol = self.symbol
        else:
            validation_symbol = None
        
        logger.info(f"Loading data for {len(symbols_to_load)} symbol(s) (max {self.max_rows_per_symbol} rows per symbol)...")
        if self.view == "LOSO":
            logger.info(f"  LOSO: Training on {len(symbols_to_load)} symbols, validating on {validation_symbol}")
        
        mtf_data = load_mtf_data_for_ranking(
            self.data_dir, 
            symbols_to_load, 
            max_rows_per_symbol=self.max_rows_per_symbol
        )
        
        if not mtf_data:
            logger.error(f"No data loaded for any symbols")
            return None, None, None, None, None, None, None, None
        
        # FIX: Check if CROSS_SECTIONAL mode has insufficient symbols and handle gracefully
        n_symbols_loaded = len(mtf_data)
        MIN_SYMBOLS_FOR_CROSS_SECTIONAL = 3
        
        if self.view == "CROSS_SECTIONAL" and n_symbols_loaded < MIN_SYMBOLS_FOR_CROSS_SECTIONAL:
            loaded_symbols_list = list(mtf_data.keys())
            if n_symbols_loaded == 1:
                # Single symbol: automatically switch to SYMBOL_SPECIFIC mode
                logger.warning(
                    f"⚠️  CROSS_SECTIONAL mode requires >= {MIN_SYMBOLS_FOR_CROSS_SECTIONAL} symbols, "
                    f"but only {n_symbols_loaded} loaded ({loaded_symbols_list}). "
                    f"Automatically switching to SYMBOL_SPECIFIC mode for symbol {loaded_symbols_list[0]}."
                )
                self.view = "SYMBOL_SPECIFIC"
                self.symbol = loaded_symbols_list[0]
            else:
                # 2 symbols: still insufficient, but log warning and proceed with allow_single_symbol=True
                logger.warning(
                    f"⚠️  CROSS_SECTIONAL mode recommended >= {MIN_SYMBOLS_FOR_CROSS_SECTIONAL} symbols, "
                    f"but only {n_symbols_loaded} loaded ({loaded_symbols_list}). "
                    f"Proceeding with reduced cross-sectional width (may be less robust)."
                )
        
        # Get sample dataframe for interval detection and feature filtering
        sample_df = next(iter(mtf_data.values()))
        all_columns = sample_df.columns.tolist()
        
        # TARGET-CONDITIONAL EXCLUSIONS: Generate per-target exclusion list
        # This implements "Target-Conditional Feature Selection" - tailoring features to target physics
        target_conditional_exclusions = []
        exclusion_metadata = {}
        target_exclusion_dir = None
        
        if self.output_dir and target_name:
            # Determine base output directory (handle both old and new call patterns)
            base_output_dir = Path(self.output_dir)
            if base_output_dir.name in ["target_rankings", "feature_selections"]:
                base_output_dir = base_output_dir.parent
            
            # FIX: Check if we're already inside REPRODUCIBILITY structure at target level
            # If output_dir is already REPRODUCIBILITY/FEATURE_SELECTION/CROSS_SECTIONAL/{target}/,
            # use it directly instead of reconstructing the path
            target_name_clean = target_name.replace('/', '_').replace('\\', '_')
            
            # Check if we're already at the target level inside REPRODUCIBILITY
            is_already_in_repro = "REPRODUCIBILITY" in str(base_output_dir)
            is_at_target_level = (
                base_output_dir.name == target_name_clean or
                (base_output_dir.parent.name in ["CROSS_SECTIONAL", "SYMBOL_SPECIFIC"] and
                 base_output_dir.parent.parent.name in ["FEATURE_SELECTION", "TARGET_RANKING"])
            )
            
            if is_already_in_repro and is_at_target_level:
                # Already at target level - use it directly
                target_exclusion_dir = base_output_dir / "feature_exclusions"
            else:
                # Need to construct path - walk up to find run level if we're inside REPRODUCIBILITY
                if is_already_in_repro:
                    # Walk up until we find the run directory (parent of REPRODUCIBILITY)
                    current = base_output_dir
                    while current.name != "REPRODUCIBILITY" and current.parent != current:
                        current = current.parent
                    if current.name == "REPRODUCIBILITY":
                        base_output_dir = current.parent
                    # If we couldn't find it, log warning and use original
                    if "REPRODUCIBILITY" in str(base_output_dir):
                        logger.warning(f"Could not resolve REPRODUCIBILITY base from {self.output_dir}, using as-is")
                
                # Save to target-first structure (targets/<target>/reproducibility/feature_exclusions/)
                from TRAINING.orchestration.utils.target_first_paths import (
                    get_target_reproducibility_dir, ensure_target_structure
                )
                ensure_target_structure(base_output_dir, target_name_clean)
                target_repro_dir = get_target_reproducibility_dir(base_output_dir, target_name_clean)
                target_exclusion_dir = target_repro_dir / "feature_exclusions"
            
            target_exclusion_dir.mkdir(parents=True, exist_ok=True)
            
            # Try to load existing exclusion list first (check target-first structure)
            existing_exclusions = load_target_exclusion_list(target_name, target_exclusion_dir)
            if existing_exclusions is None:
                # Fallback to legacy location (for reading existing runs only)
                if self.job_type == "rank_targets":
                    legacy_exclusion_dir = base_output_dir / "REPRODUCIBILITY" / "TARGET_RANKING" / self.view / target_name_clean / "feature_exclusions"
                elif self.job_type == "rank_features":
                    view_subdir = self.view if self.view else "CROSS_SECTIONAL"
                    legacy_exclusion_dir = base_output_dir / "REPRODUCIBILITY" / "FEATURE_SELECTION" / view_subdir / target_name_clean / "feature_exclusions"
                    if view_subdir == "SYMBOL_SPECIFIC" and self.symbol:
                        legacy_exclusion_dir = legacy_exclusion_dir.parent.parent / f"symbol={self.symbol}" / "feature_exclusions"
                else:
                    legacy_exclusion_dir = base_output_dir / "feature_exclusions"
                existing_exclusions = load_target_exclusion_list(target_name, legacy_exclusion_dir)
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
                temp_interval = detect_interval_from_dataframe(
                    sample_df, 
                    explicit_interval=self.explicit_interval,
                    experiment_config=self.experiment_config
                )
                
                target_conditional_exclusions, exclusion_metadata = generate_target_exclusion_list(
                    target_name=target_name,
                    all_features=all_columns,
                    interval_minutes=temp_interval,
                    output_dir=target_exclusion_dir,
                    registry=registry
                )
                
                # Also save to legacy location for backward compatibility
                if target_conditional_exclusions and 'legacy_exclusion_dir' in locals():
                    try:
                        import shutil
                        safe_target_name = target_name.replace('/', '_').replace('\\', '_')
                        exclusion_file = target_exclusion_dir / f"{safe_target_name}_exclusions.yaml"
                        legacy_exclusion_file = legacy_exclusion_dir / f"{safe_target_name}_exclusions.yaml"
                        if exclusion_file.exists():
                            shutil.copy2(exclusion_file, legacy_exclusion_file)
                            logger.debug(f"Saved exclusion file to legacy location: {legacy_exclusion_file}")
                    except Exception as e:
                        logger.debug(f"Failed to copy exclusion file to legacy location: {e}")
                
                if target_conditional_exclusions:
                    logger.info(
                        f"📋 Generated target-conditional exclusions for {target_name}: "
                        f"{len(target_conditional_exclusions)} features excluded "
                        f"(horizon={exclusion_metadata.get('target_horizon_minutes', 'unknown')}m, "
                        f"semantics={exclusion_metadata.get('target_semantics', {})})"
                    )
        else:
            logger.debug("No output_dir or target_name provided - skipping target-conditional exclusions")
        
        # Detect data interval
        detected_interval = detect_interval_from_dataframe(
            sample_df,
            timestamp_column='ts',
            default=5,
            explicit_interval=self.explicit_interval,
            experiment_config=self.experiment_config
        )
        
        # Extract target horizon for error messages and resolved config
        leakage_config = _load_leakage_config()
        target_horizon_minutes = _extract_horizon(target_column, leakage_config) if target_column else None
        
        # Apply target-conditional exclusions BEFORE global filtering
        columns_after_target_exclusions = [c for c in all_columns if c not in target_conditional_exclusions]
        
        if target_conditional_exclusions:
            logger.info(
                f"  🎯 Target-conditional exclusions: Removed {len(target_conditional_exclusions)} features "
                f"({len(columns_after_target_exclusions)} remaining before global filtering)"
            )
        
        # Apply leakage filtering if feature_names not provided
        if feature_names is None:
            safe_columns = filter_features_for_target(
                columns_after_target_exclusions,  # Use pre-filtered columns
                target_column,
                verbose=True,
                use_registry=True,
                data_interval_minutes=detected_interval,
                for_ranking=True  # Use permissive rules for ranking
            )
            feature_names = safe_columns
        
        excluded_count = len(all_columns) - len(feature_names) - 1  # -1 for target itself
        features_safe = len(feature_names)
        logger.debug(f"Filtered out {excluded_count} potentially leaking features (kept {features_safe} safe features)")
        
        # Check if we have enough features to train
        try:
            from CONFIG.config_loader import get_cfg
            from CONFIG.config_loader import get_safety_config
            safety_cfg = get_safety_config()
            safety_section = safety_cfg.get('safety', {})
            leakage_cfg = safety_section.get('leakage_detection', {})
            ranking_cfg = leakage_cfg.get('ranking', {})
            MIN_FEATURES_REQUIRED = int(ranking_cfg.get('min_features_required', 2))
        except Exception:
            MIN_FEATURES_REQUIRED = 2
        
        if len(feature_names) < MIN_FEATURES_REQUIRED:
            target_horizon_bars = None
            if target_horizon_minutes is not None and detected_interval > 0:
                target_horizon_bars = int(target_horizon_minutes // detected_interval)
            # Always log both minutes and bars for clarity
            if target_horizon_minutes is not None and target_horizon_bars is not None:
                horizon_info = f"horizon_minutes={target_horizon_minutes:.1f}m, horizon_bars={target_horizon_bars} bars @ interval={detected_interval:.1f}m"
            elif target_horizon_bars is not None:
                horizon_info = f"horizon_bars={target_horizon_bars} bars @ interval={detected_interval:.1f}m"
            else:
                horizon_info = "this horizon"
            logger.error(
                f"❌ INSUFFICIENT FEATURES: Only {len(feature_names)} features remain after filtering "
                f"(minimum required: {MIN_FEATURES_REQUIRED}). "
                f"This target may not be predictable with current feature set."
            )
            return None, None, None, None, None, None, detected_interval, None
        
        # Create resolved_config EARLY (before data prep) to get feature_time_meta_map for alignment
        # We'll update feature counts after data prep
        n_symbols_available = len(mtf_data)
        resolved_config = create_resolved_config(
            requested_min_cs=self.min_cs if self.view != "SYMBOL_SPECIFIC" else 1,
            n_symbols_available=n_symbols_available,
            max_cs_samples=self.max_cs_samples,
            interval_minutes=detected_interval,
            horizon_minutes=target_horizon_minutes,
            feature_lookback_max_minutes=None,  # Will be computed from feature_names
            purge_buffer_bars=5,
            default_purge_minutes=None,  # Loads from safety_config.yaml
            features_safe=features_safe,
            features_dropped_nan=0,  # Will be updated after data prep
            features_final=features_safe,  # Will be updated after data prep
            view=self.view,
            symbol=self.symbol,
            feature_names=feature_names,  # Pass feature names for feature_time_meta_map building
            recompute_lookback=True,  # CRITICAL: Compute feature lookback to auto-adjust purge
            experiment_config=getattr(self, 'experiment_config', None)  # NEW: Pass experiment_config if available
        )
        
        # Prepare data based on view (with alignment args from resolved_config)
        # NOTE: view may have been changed to SYMBOL_SPECIFIC above if only 1 symbol loaded
        if self.view == "SYMBOL_SPECIFIC":
            # For symbol-specific, prepare single-symbol time series data
            X, y, feature_names_out, symbols_array, time_vals, resolved_data_config = prepare_cross_sectional_data_for_ranking(
                mtf_data, target_column, min_cs=1, max_cs_samples=self.max_cs_samples, feature_names=feature_names,
                feature_time_meta_map=resolved_config.feature_time_meta_map,  # NEW: Pass from resolved_config
                base_interval_minutes=resolved_config.base_interval_minutes,  # NEW: Pass from resolved_config
                allow_single_symbol=True  # SYMBOL_SPECIFIC always allows single symbol
            )
        elif self.view == "LOSO":
            # LOSO: prepare training data (all symbols except validation symbol)
            X, y, feature_names_out, symbols_array, time_vals, resolved_data_config = prepare_cross_sectional_data_for_ranking(
                mtf_data, target_column, min_cs=self.min_cs, max_cs_samples=self.max_cs_samples, feature_names=feature_names,
                feature_time_meta_map=resolved_config.feature_time_meta_map,  # NEW: Pass from resolved_config
                base_interval_minutes=resolved_config.base_interval_minutes  # NEW: Pass from resolved_config
            )
            # TODO: Handle validation symbol separately for LOSO
            logger.warning("LOSO view: Using combined data for now (LOSO-specific CV splitter not yet implemented)")
        else:
            # CROSS_SECTIONAL: standard pooled data
            # FIX: Pass allow_single_symbol=True if we have < 3 symbols (graceful degradation)
            # This prevents ValueError when only 1-2 symbols are loaded
            # Note: If view was changed to SYMBOL_SPECIFIC above, we won't reach this block
            allow_single_symbol = (n_symbols_loaded < 3)  # Use literal to avoid scope issues
            X, y, feature_names_out, symbols_array, time_vals, resolved_data_config = prepare_cross_sectional_data_for_ranking(
                mtf_data, target_column, min_cs=self.min_cs, max_cs_samples=self.max_cs_samples, feature_names=feature_names,
                feature_time_meta_map=resolved_config.feature_time_meta_map,  # NEW: Pass from resolved_config
                base_interval_minutes=resolved_config.base_interval_minutes,  # NEW: Pass from resolved_config
                allow_single_symbol=allow_single_symbol  # FIX: Allow graceful degradation for < 3 symbols
            )
        
        # Update feature counts after data preparation
        features_dropped_nan = 0
        features_final = features_safe
        if feature_names_out is not None:
            features_final = len(feature_names_out)
            features_dropped_nan = features_safe - features_final
        
        # Update resolved_config with final feature counts (using dataclasses.replace for frozen dataclass)
        if hasattr(resolved_config, '__dict__'):
            # ResolvedConfig is not frozen, can update directly
            resolved_config.features_dropped_nan = features_dropped_nan
            resolved_config.features_final = features_final
        else:
            # If frozen, would need dataclasses.replace, but ResolvedConfig is not frozen
            resolved_config.features_dropped_nan = features_dropped_nan
            resolved_config.features_final = features_final
        
        # PERFORMANCE AUDIT: Track build_panel completion
        build_duration = time.time() - build_start_time
        if auditor and auditor.enabled:
            rows = X.shape[0] if X is not None else None
            cols = len(feature_names_out) if feature_names_out else None
            auditor.track_call(
                func_name='RankingHarness.build_panel',
                duration=build_duration,
                rows=rows,
                cols=cols,
                stage=self.job_type,  # 'rank_targets' or 'rank_features'
                cache_hit=False,  # build_panel doesn't use cache currently
                input_fingerprint=fingerprint
            )
        
        return X, y, feature_names_out, symbols_array, time_vals, mtf_data, detected_interval, resolved_config, resolved_data_config
    
    def split_policy(
        self,
        time_vals: np.ndarray,
        groups: Optional[np.ndarray] = None,
        horizon_minutes: Optional[float] = None,
        data_interval_minutes: float = 5.0
    ) -> Any:
        """
        Create split policy (PurgedTimeSeriesSplit) with time-based purging.
        
        This is the SAME split policy used by target ranking to ensure
        identical evaluation contracts.
        
        Args:
            time_vals: Timestamps for each sample
            groups: Optional grouping array (for panel data)
            horizon_minutes: Target horizon in minutes (for purge calculation)
            data_interval_minutes: Data bar interval in minutes
        
        Returns:
            PurgedTimeSeriesSplit instance
        
        Raises:
            ValueError: If data span is insufficient for the required purge/embargo
        """
        from TRAINING.ranking.utils.purged_time_series_split import PurgedTimeSeriesSplit
        from TRAINING.ranking.utils.resolved_config import derive_purge_embargo
        
        # Get CV config
        cv_config = self.multi_model_config.get('cross_validation', {}) if self.multi_model_config else {}
        try:
            from CONFIG.config_loader import get_cfg
            cv_folds = int(get_cfg("training.cv_folds", default=cv_config.get('cv_folds', 3), config_name="intelligent_training_config"))
        except Exception:
            cv_folds = cv_config.get('cv_folds', 3)
        
        # Derive purge and embargo from horizon
        if horizon_minutes is not None:
            purge_minutes, embargo_minutes = derive_purge_embargo(
                horizon_minutes=horizon_minutes,
                interval_minutes=data_interval_minutes,
                feature_lookback_max_minutes=None,  # Will be computed from features if needed
                purge_buffer_bars=5,
                default_purge_minutes=None  # Loads from safety_config.yaml
            )
            purge_time = pd.Timedelta(minutes=purge_minutes)
            
            # CRITICAL: Validate that data span is sufficient for purge/embargo
            # Rule of thumb: need at least (purge + embargo) * 2 * cv_folds worth of data
            # This ensures each fold has enough training data after purging
            if time_vals is not None and len(time_vals) > 1:
                time_vals_sorted = np.sort(time_vals)
                data_span_minutes = (time_vals_sorted[-1] - time_vals_sorted[0]) / (60 * 1e9)  # Convert ns to minutes
                required_span_minutes = (purge_minutes + embargo_minutes) * 2 * cv_folds
                
                if data_span_minutes < required_span_minutes:
                    raise ValueError(
                        f"Insufficient data span for long-horizon target (horizon={horizon_minutes:.1f}m). "
                        f"Data span: {data_span_minutes:.1f}m, Required: {required_span_minutes:.1f}m "
                        f"(purge={purge_minutes:.1f}m + embargo={embargo_minutes:.1f}m) * 2 * {cv_folds} folds. "
                        f"Falling back to per-symbol processing (expected for long-horizon targets with limited data). "
                        f"To use shared harness: 1) Increase max_rows_per_symbol, 2) Reduce horizon, or 3) Skip this target."
                    )
        else:
            # Fallback: use default purge (60m = 12 bars + buffer)
            purge_time = pd.Timedelta(minutes=60)
        
        # Create time-based purged splitter (REQUIRED for panel data)
        cv_splitter = PurgedTimeSeriesSplit(
            n_splits=cv_folds,
            purge_overlap_time=purge_time,
            time_column_values=time_vals
        )
        
        return cv_splitter
    
    def run_importance_producers(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        time_vals: Optional[np.ndarray] = None,
        task_type: Any = None,
        resolved_config: Optional[Any] = None
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float], float,
               Dict[str, List[Tuple[str, float]]], Dict[str, Dict[str, float]],
               List[Dict[str, Any]], set]:
        """
        Run importance producers (models) using the SAME evaluation harness.
        
        This calls train_and_evaluate_models which uses:
        - PurgedTimeSeriesSplit with time-based purging
        - Same scoring functions
        - Same leakage-safe imputation
        - Same fold timestamp tracking
        
        Args:
            X: Feature matrix
            y: Target array
            feature_names: List of feature names
            time_vals: Timestamps for each sample
            task_type: TaskType enum
            resolved_config: Optional ResolvedConfig with purge/embargo
        
        Returns:
            Tuple of (model_metrics, model_scores, mean_importance,
                     all_suspicious_features, all_feature_importances, fold_timestamps, perfect_correlation_models)
        """
        from TRAINING.ranking.predictability.model_evaluation import train_and_evaluate_models
        from TRAINING.ranking.predictability.scoring import TaskType
        
        # Infer task type if not provided
        if task_type is None:
            y_sample = pd.Series(y).dropna()
            task_type = TaskType.from_target_column(self.target_column, y_sample.to_numpy())
        
        # Detect data interval from time_vals if available
        data_interval_minutes = 5.0  # Default
        if time_vals is not None and len(time_vals) > 1:
            try:
                time_series = pd.Series(time_vals) if not isinstance(time_vals, pd.Series) else time_vals
                if pd.api.types.is_datetime64_any_dtype(time_series):
                    unique_times = time_series.unique()
                    unique_times_sorted = pd.Series(unique_times).sort_values()
                    time_diffs = unique_times_sorted.diff().dropna()
                    if isinstance(time_diffs, pd.TimedeltaIndex) and len(time_diffs) > 0:
                        median_diff_minutes = abs(time_diffs.median().total_seconds()) / 60.0
                        # Round to common intervals
                        common_intervals = [1, 5, 15, 30, 60]
                        detected_interval = min(common_intervals, key=lambda x: abs(x - median_diff_minutes))
                        data_interval_minutes = detected_interval
            except Exception:
                pass
        
        # Call the SAME train_and_evaluate_models function used by target ranking
        results = train_and_evaluate_models(
            X=X,
            y=y,
            feature_names=feature_names,
            task_type=task_type,
            model_families=self.model_families,
            multi_model_config=self.multi_model_config,
            target_column=self.target_column,
            data_interval_minutes=data_interval_minutes,
            time_vals=time_vals,
            explicit_interval=self.explicit_interval,
            experiment_config=self.experiment_config,
            output_dir=self.output_dir,
            resolved_config=resolved_config
        )
        
        return results
    
    def create_run_context(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        symbols_array: np.ndarray,
        time_vals: np.ndarray,
        cv_splitter: Any,
        horizon_minutes: Optional[float] = None,
        purge_minutes: Optional[float] = None,
        embargo_minutes: Optional[float] = None,
        data_interval_minutes: Optional[float] = None,
        min_cs: Optional[int] = None,  # FIX: Add min_cs for diff telemetry
        max_cs_samples: Optional[int] = None  # FIX: Add max_cs_samples for diff telemetry
    ) -> Any:
        """
        Create RunContext for reproducibility tracking.
        
        This ensures both target ranking and feature selection use the SAME
        RunContext fields for consistent telemetry.
        
        Args:
            X: Feature matrix
            y: Target array
            feature_names: List of feature names
            symbols_array: Symbol array
            time_vals: Timestamps
            cv_splitter: CV splitter instance
            horizon_minutes: Target horizon
            purge_minutes: Purge minutes
            embargo_minutes: Embargo minutes
            data_interval_minutes: Data interval
        
        Returns:
            RunContext instance
        """
        from TRAINING.orchestration.utils.run_context import RunContext
        
        # Extract cv_folds from splitter if available (required for COHORT_AWARE mode)
        cv_folds = None
        if cv_splitter is not None:
            # Try to get n_splits from splitter (PurgedTimeSeriesSplit has n_splits attribute)
            if hasattr(cv_splitter, 'n_splits'):
                cv_folds = cv_splitter.n_splits
            elif hasattr(cv_splitter, 'get_n_splits'):
                try:
                    cv_folds = cv_splitter.get_n_splits()
                except Exception:
                    pass
        
        ctx = RunContext(
            stage="FEATURE_SELECTION" if self.job_type == "rank_features" else "TARGET_RANKING",
            target_name=self.target_column,
            target_column=self.target_column,
            X=X,
            y=y,
            feature_names=feature_names,
            symbols=symbols_array.tolist() if isinstance(symbols_array, np.ndarray) else symbols_array,
            time_vals=time_vals,
            horizon_minutes=horizon_minutes,
            purge_minutes=purge_minutes,
            embargo_minutes=embargo_minutes,
            data_interval_minutes=data_interval_minutes,
            cv_splitter=cv_splitter,
            cv_folds=cv_folds,  # FIX: Extract and pass cv_folds for COHORT_AWARE mode
            view=self.view,
            symbol=self.symbol,
            min_cs=min_cs if min_cs is not None else self.min_cs,  # FIX: Populate min_cs for diff telemetry
            max_cs_samples=max_cs_samples if max_cs_samples is not None else self.max_cs_samples  # FIX: Populate max_cs_samples for diff telemetry
        )
        
        return ctx
    
    def sanitize_and_canonicalize_dtypes(
        self,
        X: np.ndarray,
        feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Sanitize data and enforce numeric dtypes.
        
        This is done ONCE upstream and reused everywhere to prevent
        dtype issues (e.g., CatBoost object column errors).
        
        Args:
            X: Feature matrix (can be DataFrame or array)
            feature_names: List of feature names
        
        Returns:
            Tuple of (X_sanitized, feature_names_sanitized)
        """
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=feature_names)
        else:
            X_df = X.copy()
        
        # FIX: Hard guardrail - enforce numeric dtypes BEFORE any model training
        # This prevents CatBoost from treating numeric columns as text/categorical
        # CatBoost "categoricalizing" numeric features causes fake performance (perfect scores)
        # NOTE: np and pd are imported at module scope - do not import locally to avoid UnboundLocalError
        
        # Step 1: Try to convert object columns to numeric (don't drop immediately)
        object_cols = X_df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
        if object_cols:
            logger.warning(f"Found {len(object_cols)} object/string/category columns: {object_cols[:10]}")
            logger.warning("Attempting to convert to numeric (coercing errors to NaN)")
            for col in object_cols:
                try:
                    X_df[col] = pd.to_numeric(X_df[col], errors='coerce').astype('float32')
                    logger.debug(f"Converted object column {col} to float32")
                except Exception as e:
                    logger.warning(f"Failed to convert {col} to numeric: {e}, will drop")
        
        # Step 2: Ensure all remaining columns are numeric (hard-fail if not)
        still_bad = [c for c in X_df.columns if not pd.api.types.is_numeric_dtype(X_df[c])]
        if still_bad:
            logger.error(f"Non-numeric columns remain after conversion: {still_bad[:10]}")
            logger.error("Dropping non-numeric columns to prevent dtype errors")
            X_df = X_df.drop(columns=still_bad)
            feature_names = [f for f in feature_names if f not in still_bad]
        
        # Step 3: Hard-cast all numeric columns to float32 (prevents object dtype from NaN/mixed types)
        # This is critical - CatBoost can interpret float64 with NaN as object dtype
        for col in X_df.columns:
            if pd.api.types.is_numeric_dtype(X_df[col]):
                X_df[col] = X_df[col].astype('float32')
            else:
                # Should not happen after Step 2, but defensive check
                logger.error(f"Column {col} is still not numeric (dtype={X_df[col].dtype}), dropping")
                X_df = X_df.drop(columns=[col])
                feature_names = [f for f in feature_names if f != col]
        
        # Step 4: Final verification - fail fast if any non-numeric remain
        final_bad = [c for c in X_df.columns if not np.issubdtype(X_df[c].dtype, np.number)]
        if final_bad:
            raise TypeError(f"CRITICAL: Non-numeric columns remain after all conversions: {final_bad[:10]}. "
                          f"This will cause CatBoost to treat them as text/categorical and fake performance.")
        
        # FIX: Replace inf/-inf with nan before fail-fast (prevents phantom issues)
        # Some models (e.g., Ridge) may fail on inf values
        X_df = X_df.replace([np.inf, -np.inf], np.nan)
        # Drop columns that are all nan/inf
        X_df = X_df.dropna(axis=1, how='all')
        feature_names = [f for f in feature_names if f in X_df.columns]
        
        # Convert back to numpy array
        X_sanitized = X_df.values
        
        return X_sanitized, feature_names
    
    def apply_cleaning_and_audit_checks(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        target_column: str,
        resolved_config: Any,
        detected_interval: float,
        task_type: Any = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[str]], Optional[Any], bool]:
        """
        Apply all cleaning and audit checks from target ranking.
        
        This includes:
        - Duplicate column name checks
        - Pre-training leak scan (find_near_copy_features)
        - Target validation
        - Degenerate target checks
        - Class imbalance checks
        - Final gatekeeper (ghost buster)
        
        Args:
            X: Feature matrix
            y: Target array
            feature_names: List of feature names
            target_column: Target column name
            resolved_config: ResolvedConfig object
            detected_interval: Data interval in minutes
            task_type: Optional TaskType (will be inferred if None)
        
        Returns:
            Tuple of (X_cleaned, y_cleaned, feature_names_cleaned, resolved_config_updated, success)
            If success is False, X_cleaned will be None
        """
        from TRAINING.ranking.predictability.leakage_detection import find_near_copy_features
        from TRAINING.ranking.predictability.model_evaluation import _enforce_final_safety_gate, validate_target
        from TRAINING.ranking.predictability.scoring import TaskType
        from TRAINING.ranking.utils.cross_sectional_data import _log_feature_set
        from TRAINING.ranking.utils.resolved_config import compute_feature_lookback_max
        
        # Infer task type if not provided
        if task_type is None:
            y_sample = pd.Series(y).dropna()
            task_type = TaskType.from_target_column(target_column, y_sample.to_numpy())
        
        # Check for duplicate column names
        if len(feature_names) != len(set(feature_names)):
            duplicates = [name for name in set(feature_names) if feature_names.count(name) > 1]
            logger.error(f"  🚨 DUPLICATE COLUMN NAMES DETECTED: {duplicates}")
            raise ValueError(f"Duplicate feature names detected: {duplicates}")
        
        # PRE-TRAINING LEAK SCAN: Detect and remove near-copy features
        logger.info("🔍 Pre-training leak scan: Checking for near-copy features...")
        feature_names_before_leak_scan = feature_names.copy()
        
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
            _log_feature_set("AFTER_LEAK_REMOVAL", feature_names, previous_names=feature_names_before_leak_scan, logger_instance=logger)
            
            # Check if we removed too many features
            try:
                from CONFIG.config_loader import get_safety_config
                safety_cfg = get_safety_config()
                safety_section = safety_cfg.get('safety', {})
                leakage_cfg = safety_section.get('leakage_detection', {})
                ranking_cfg = leakage_cfg.get('ranking', {})
                MIN_FEATURES_AFTER_LEAK_REMOVAL = int(ranking_cfg.get('min_features_after_leak_removal', 2))
            except Exception:
                MIN_FEATURES_AFTER_LEAK_REMOVAL = 2
            
            if X.shape[1] < MIN_FEATURES_AFTER_LEAK_REMOVAL:
                logger.error(
                    f"  ❌ Too few features remaining after leak removal ({X.shape[1]}). "
                    f"Marking as LEAKAGE_DETECTED."
                )
                return None, None, None, None, False
        else:
            logger.info("  ✅ No obvious leaky features detected in pre-training scan")
            _log_feature_set("AFTER_LEAK_REMOVAL", feature_names, previous_names=feature_names_before_leak_scan, logger_instance=logger)
        
        # Early exit if too few features
        try:
            from CONFIG.config_loader import get_safety_config
            safety_cfg = get_safety_config()
            safety_section = safety_cfg.get('safety', {})
            leakage_cfg = safety_section.get('leakage_detection', {})
            ranking_cfg = leakage_cfg.get('ranking', {})
            MIN_FEATURES_FOR_MODEL = int(ranking_cfg.get('min_features_for_model', 3))
        except Exception:
            MIN_FEATURES_FOR_MODEL = 3
        
        if X.shape[1] < MIN_FEATURES_FOR_MODEL:
            logger.warning(
                f"Too few features ({X.shape[1]}) after filtering (minimum: {MIN_FEATURES_FOR_MODEL}); "
                f"marking as degenerate and skipping."
            )
            return None, None, None, None, False
        
        # Validate target
        is_valid, error_msg = validate_target(y, task_type=task_type)
        if not is_valid:
            logger.warning(f"Skipping: {error_msg}")
            return None, None, None, None, False
        
        # Check if target is degenerate
        unique_vals = np.unique(y[~np.isnan(y)])
        if len(unique_vals) < 2:
            logger.warning(f"Skipping: Target has only {len(unique_vals)} unique value(s)")
            return None, None, None, None, False
        
        # For classification, check if classes are too imbalanced for CV
        if len(unique_vals) <= 10:  # Likely classification
            class_counts = np.bincount(y[~np.isnan(y)].astype(int))
            min_class_count = class_counts[class_counts > 0].min()
            if min_class_count < 2:
                logger.warning(f"Skipping: Smallest class has only {min_class_count} sample(s) (too few for CV)")
                return None, None, None, None, False
        
        # FINAL GATEKEEPER: Enforce safety at the last possible moment (ghost buster)
        # This runs AFTER all loading/merging/sanitization is done
        X, feature_names = _enforce_final_safety_gate(
            X=X,
            feature_names=feature_names,
            resolved_config=resolved_config,
            interval_minutes=detected_interval,
            logger=logger
        )
        
        # Recompute resolved_config.feature_lookback_max AFTER Final Gatekeeper
        if feature_names and len(feature_names) > 0:
            result = compute_feature_lookback_max(
                feature_names,
                interval_minutes=detected_interval,
                max_lookback_cap_minutes=None,
                stage="shared_harness_post_gatekeeper"
            )
            max_lookback_after_gatekeeper = result.max_minutes
            fingerprint = result.fingerprint
            if max_lookback_after_gatekeeper is not None:
                resolved_config.feature_lookback_max_minutes = max_lookback_after_gatekeeper
                logger.debug(f"📊 Updated feature_lookback_max after Final Gatekeeper: {max_lookback_after_gatekeeper:.1f}m (from {len(feature_names)} remaining features)")
        
        if X.shape[1] == 0:
            logger.error("❌ FINAL GATEKEEPER: All features were dropped! Cannot proceed.")
            return None, None, None, None, False
        
        return X, y, feature_names, resolved_config, True
