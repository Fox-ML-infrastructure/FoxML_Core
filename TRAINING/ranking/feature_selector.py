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
    experiment_config: Optional[Any] = None  # Optional ExperimentConfig (for data.bar_interval)
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
    
    logger.info(f"Selecting features for target: {target_column}")
    logger.info(f"Processing {len(symbols)} symbols")
    logger.info(f"Model families: {', '.join([f for f, cfg in model_families_config.items() if cfg.get('enabled', False)])}")
    
    # Process each symbol
    all_results = []
    all_family_statuses = []  # Collect status info for all families across all symbols
    for idx, symbol in enumerate(symbols, 1):
        logger.info(f"[{idx}/{len(symbols)}] Processing {symbol}...")
        
        # Find symbol data file
        symbol_dir = data_dir / f"symbol={symbol}"
        data_path = symbol_dir / f"{symbol}.parquet"
        
        if not data_path.exists():
            logger.warning(f"  Data file not found: {data_path}")
            continue
        
        try:
            # Process symbol (preserves all leakage-free behavior)
            # Returns tuple: (results, family_statuses)
            symbol_results, symbol_statuses = _process_single_symbol(
                symbol=symbol,
                data_path=data_path,
                target_column=target_column,
                model_families_config=model_families_config,
                max_samples=max_samples_per_symbol,
                explicit_interval=explicit_interval,
                experiment_config=experiment_config,
                output_dir=output_dir  # Pass output_dir for reproducibility tracking
            )
            
            all_results.extend(symbol_results)
            all_family_statuses.extend(symbol_statuses)
            logger.info(f"  ✅ {symbol}: {len(symbol_results)} model results")
        
        except Exception as e:
            logger.error(f"  ❌ {symbol} failed: {e}")
            continue
    
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
    
    # Save stability snapshot for aggregated feature selection (non-invasive hook)
    try:
        from TRAINING.stability.feature_importance import save_snapshot_hook
        # Convert summary_df to importance dict (consensus_score as importance)
        if summary_df is not None and len(summary_df) > 0:
            importance_dict = summary_df.set_index('feature')['consensus_score'].to_dict()
            universe_id = ",".join(sorted(symbols)) if len(symbols) <= 10 else "ALL"
            save_snapshot_hook(
                target_name=target_column,
                method="multi_model_aggregated",
                importance_dict=importance_dict,
                universe_id=universe_id,
                output_dir=output_dir,
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
    try:
        import pandas as pd
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
            'family_statuses': all_family_statuses  # Include family status tracking for debugging
        }
        
        # Add cross-sectional stability to metadata if available
        if cs_stability_results is not None:
            metadata['cross_sectional_stability'] = cs_stability_results
        
        _save_multi_model_results(
            summary_df=summary_df,
            selected_features=selected_features,
            all_results=all_results,
            output_dir=output_dir,
            metadata=metadata
        )
        
        # Save CS stability metadata separately (similar to model_metadata.json)
        if cs_stability_results is not None and output_dir:
            try:
                import json
                cs_metadata_file = output_dir / "cross_sectional_stability_metadata.json"
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
            
            # Use module-specific directory for reproducibility log
            # output_dir is typically: output_dir_YYYYMMDD_HHMMSS/feature_selections/{target}/
            # We want to store in feature_selections/ subdirectory for this module
            if output_dir.name == 'feature_selections' or (output_dir.parent / 'feature_selections').exists():
                # Already in or can find feature_selections subdirectory
                if output_dir.name != 'feature_selections':
                    module_output_dir = output_dir.parent / 'feature_selections'
                else:
                    module_output_dir = output_dir
            else:
                # Fallback: use output_dir directly (for standalone runs)
                module_output_dir = output_dir
            
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
                
                # Build RunContext from available data
                # Note: For feature selection, we don't have X, y directly, but we can extract from cohort_context
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
                ctx = RunContext(
                    stage="FEATURE_SELECTION",
                    target_name=target_column,
                    target_column=target_column,
                    X=X_for_ctx,  # May be None - fingerprint will use symbols/time_vals
                    y=y_for_ctx,  # May be None
                    feature_names=feature_names_for_ctx,
                    symbols=symbols,
                    time_vals=None,  # Not directly available here
                    horizon_minutes=None,  # Not applicable for feature selection
                    purge_minutes=None,
                    embargo_minutes=None,
                    cv_folds=None,
                    fold_timestamps=None,
                    data_interval_minutes=None,
                    seed=None
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
                audit_result = tracker.log_run(ctx, metrics_dict)
                
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
                
                additional_data_with_cohort = {
                    "top_feature": summary_df.iloc[0]['feature'] if not summary_df.empty else None,
                    "top_n": top_n or len(selected_features),
                    **cohort_additional_data  # Adds n_symbols, date_range, cs_config if available
                }
                
                tracker.log_comparison(
                    stage="feature_selection",
                    item_name=target_column,
                    metrics=metrics_with_cohort,
                    additional_data=additional_data_with_cohort
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

