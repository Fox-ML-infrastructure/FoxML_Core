# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Feature Importance Stability Hooks

Non-invasive hooks that can be called from pipeline endpoints.
These functions handle snapshot creation and automatic stability analysis.
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, Optional, List, Union, Any
from datetime import datetime
import uuid

from .schema import FeatureImportanceSnapshot
from .io import save_importance_snapshot, get_snapshot_base_dir
from .analysis import analyze_stability_auto

# SST: Import View enum for consistent view handling
from TRAINING.orchestration.utils.scope_resolution import View

logger = logging.getLogger(__name__)


def save_snapshot_hook(
    target: str,
    method: str,
    importance_dict: Dict[str, float],
    universe_sig: Optional[str] = None,
    output_dir: Optional[Path] = None,
    run_id: Optional[str] = None,
    auto_analyze: Optional[bool] = None,  # None = load from config
    run_identity: Optional[Any] = None,   # RunIdentity object or dict
    allow_legacy: bool = False,  # If True, allow saving without identity (legacy path)
    prediction_fingerprint: Optional[Dict] = None,  # PredictionFingerprint.to_dict() for hash tracking
    view: Optional[str] = None,  # "CROSS_SECTIONAL" or "SYMBOL_SPECIFIC"
    symbol: Optional[str] = None,  # Symbol name for SYMBOL_SPECIFIC views
    inputs: Optional[Dict] = None,  # Optional inputs dict for fs_snapshot
    process: Optional[Dict] = None,  # Optional process dict for fs_snapshot
    write_fs_snapshot: bool = True,  # If True, also write fs_snapshot.json and update global index
    stage: str = "FEATURE_SELECTION",  # Pipeline stage for fs_snapshot labeling
    # Full parity with TARGET_RANKING fields (sourced from FS stage)
    snapshot_seq: int = 0,  # Sequence number for this run
    n_effective: Optional[int] = None,  # Effective sample count from FS
    feature_registry_hash: Optional[str] = None,  # Hash of feature registry
    comparable_key: Optional[str] = None,  # Pre-computed comparison key
) -> Optional[Path]:
    """
    Hook function to save feature importance snapshot.
    
    This is the main entry point for saving snapshots from pipeline code.
    
    Args:
        target: Target name (e.g., "peak_60m_0.8")
        method: Method name (e.g., "lightgbm", "quick_pruner", "rfe")
        importance_dict: Dictionary mapping feature names to importance values
        universe_sig: Optional universe identifier (symbol name, "ALL", etc.)
        output_dir: Optional output directory (defaults to artifacts/feature_importance)
        run_id: Optional run ID (generates UUID if not provided)
        auto_analyze: If True, automatically run stability analysis after saving.
                     If None, loads from config (safety.feature_importance.auto_analyze_stability)
        run_identity: Optional RunIdentity object or dict with identity signatures.
                     PREFERRED: Pass RunIdentity SST object for full reproducibility.
        allow_legacy: If True, allow saving without identity (legacy path).
                     If False (default), raise if no identity provided.
        prediction_fingerprint: Optional PredictionFingerprint.to_dict() containing:
                     - prediction_hash: Strict bitwise hash for determinism verification
                     - prediction_hash_live: Quantized hash for drift detection
                     - prediction_row_ids_hash: Hash of row identifiers
                     - prediction_classes_hash: Hash of class order (classification)
                     - prediction_kind: "regression", "binary_proba", etc.
        stage: Pipeline stage for fs_snapshot - "FEATURE_SELECTION" (default) or "TARGET_RANKING"
    
    Returns:
        Path to saved snapshot, or None if saving failed
    """
    try:
        # Load auto_analyze setting from config if not explicitly provided
        if auto_analyze is None:
            try:
                from CONFIG.config_loader import get_cfg
                auto_analyze = get_cfg(
                    "safety.feature_importance.auto_analyze_stability",
                    default=True,
                    config_name="safety_config"
                )
            except Exception:
                auto_analyze = True  # Default to enabled
        
        # Load identity enforcement mode from config
        identity_mode = "strict"  # Default to strict
        try:
            from TRAINING.common.utils.fingerprinting import get_identity_mode
            identity_mode = get_identity_mode()
        except Exception:
            pass  # Use default strict mode
        
        # Validate and convert RunIdentity
        identity_dict = None
        use_hash_path = False

        if run_identity is not None:
            # Check if it's a RunIdentity object with is_final
            if hasattr(run_identity, 'is_final'):
                if not run_identity.is_final:
                    # Partial identity - check allow_legacy FIRST before strict mode check
                    error_msg = (
                        "Cannot save snapshot with partial RunIdentity (is_final=False). "
                        "Call run_identity.finalize(feature_signature) before saving. "
                        f"Current identity: {run_identity.debug_key if hasattr(run_identity, 'debug_key') else 'unknown'}"
                    )
                    if allow_legacy:
                        # FIX: Explicit escape hatch - use legacy path instead of failing
                        logger.warning(
                            f"Partial identity with allow_legacy=True - using legacy path. "
                            f"target={target} method={method}"
                        )
                        # Fall through to legacy path (identity_dict stays None, use_hash_path stays False)
                    elif identity_mode == "strict":
                        raise ValueError(error_msg + " (strict mode, allow_legacy=False)")
                    elif identity_mode == "relaxed":
                        logger.error(f"Identity validation failed (relaxed mode): {error_msg}")
                        # Continue with degraded identity (no hash path)
                    else:  # legacy mode
                        logger.warning(f"Partial identity ignored (legacy mode): {error_msg}")
                else:
                    # Valid finalized identity - use hash-based path
                    identity_dict = run_identity.to_dict()
                    use_hash_path = True
            elif hasattr(run_identity, 'to_dict'):
                # Has to_dict but no is_final - treat as legacy RunIdentity
                identity_dict = run_identity.to_dict()
                # Check if it has the keys (finalized)
                if identity_dict.get('replicate_key') and identity_dict.get('strict_key'):
                    use_hash_path = True
            elif isinstance(run_identity, dict):
                identity_dict = run_identity
                # Check if dict has required keys for hash path
                if run_identity.get('replicate_key') and run_identity.get('strict_key'):
                    use_hash_path = True
            else:
                raise TypeError(
                    f"run_identity must be RunIdentity object or dict, got {type(run_identity).__name__}. "
                    "Use RunIdentity.finalize(feature_signature) for full reproducibility."
                )
        else:
            # No run_identity provided - behavior depends on mode
            error_msg = (
                "Cannot save snapshot without run_identity. "
                "Provide a finalized RunIdentity for proper reproducibility tracking."
            )
            if identity_mode == "strict" and not allow_legacy:
                raise ValueError(error_msg + " (strict mode, allow_legacy=False)")
            elif identity_mode == "strict" and allow_legacy:
                # Explicit escape hatch in strict mode - warn loudly
                logger.warning(
                    f"Saving snapshot WITHOUT identity in STRICT mode (allow_legacy=True override). "
                    f"target={target} method={method}"
                )
            elif identity_mode == "relaxed":
                logger.error(f"No identity provided (relaxed mode): {error_msg}")
            else:  # legacy mode
                logger.debug("No identity provided (legacy mode)")
        
        # Merge prediction fingerprint into identity_dict if provided
        if prediction_fingerprint:
            if identity_dict is None:
                identity_dict = {}
            # Map PredictionFingerprint fields to snapshot fields
            if "prediction_hash" in prediction_fingerprint:
                identity_dict["prediction_hash"] = prediction_fingerprint["prediction_hash"]
            if "prediction_hash_live" in prediction_fingerprint:
                identity_dict["prediction_hash_live"] = prediction_fingerprint["prediction_hash_live"]
            if "row_ids_hash" in prediction_fingerprint:
                identity_dict["prediction_row_ids_hash"] = prediction_fingerprint["row_ids_hash"]
            if "classes_hash" in prediction_fingerprint:
                identity_dict["prediction_classes_hash"] = prediction_fingerprint["classes_hash"]
            if "kind" in prediction_fingerprint:
                identity_dict["prediction_kind"] = prediction_fingerprint["kind"]
        
        # Create snapshot
        snapshot = FeatureImportanceSnapshot.from_dict_series(
            target=target,
            method=method,
            importance_dict=importance_dict,
            universe_sig=universe_sig,
            run_id=run_id,
            run_identity=identity_dict,
        )
        
        # Save snapshot
        # Use target for target-first structure with view/symbol/universe scoping
        # Snapshots should only be saved to target-specific directories, never at root level
        base_dir = get_snapshot_base_dir(
            output_dir, target=target,
            view=view or "CROSS_SECTIONAL", symbol=symbol, universe_sig=universe_sig,
            stage=stage  # Pass stage for proper stage-scoped paths
        )
        snapshot_path = save_importance_snapshot(snapshot, base_dir, use_hash_path=use_hash_path)
        
        logger.debug(f"Saved importance snapshot: {snapshot_path}")
        
        # Create and save FeatureSelectionSnapshot if requested (human-readable format)
        if write_fs_snapshot:
            try:
                from .io import create_fs_snapshot_from_importance
                # Determine view from universe_sig if not explicitly provided
                effective_view = view or ("SYMBOL_SPECIFIC" if (symbol or (universe_sig and ':' in universe_sig)) else "CROSS_SECTIONAL")
                # Determine cohort directory from snapshot_path
                cohort_dir = snapshot_path.parent if snapshot_path else None
                
                create_fs_snapshot_from_importance(
                    importance_snapshot=snapshot,
                    view=effective_view,
                    symbol=symbol,
                    cohort_dir=cohort_dir,
                    output_dir=output_dir,
                    inputs=inputs,
                    process=process,
                    stage=stage,  # Pass stage for correct labeling
                    # Full parity fields (sourced from FS stage)
                    snapshot_seq=snapshot_seq,
                    n_effective=n_effective,
                    feature_registry_hash=feature_registry_hash,
                    comparable_key=comparable_key,
                )
            except Exception as e:
                # FIX: Log at warning level so per-model fs_snapshot failures are visible
                logger.warning(
                    f"Failed to create fs_snapshot for {method} (target={target}, view={effective_view}): {e}. "
                    f"This may indicate per-model snapshots are not being saved correctly."
                )
                import traceback
                logger.debug(f"fs_snapshot creation traceback: {traceback.format_exc()}")
        
        # Auto-analyze if enabled
        if auto_analyze:
            try:
                # Load config for auto-analysis settings
                min_overlap_threshold = 0.7
                min_tau_threshold = 0.6
                top_k = 20
                
                try:
                    from CONFIG.config_loader import get_cfg
                    stability_thresholds = get_cfg(
                        "safety.feature_importance.stability_thresholds",
                        default={},
                        config_name="safety_config"
                    )
                    min_overlap_threshold = stability_thresholds.get('min_top_k_overlap', 0.7)
                    min_tau_threshold = stability_thresholds.get('min_kendall_tau', 0.6)
                    top_k = stability_thresholds.get('top_k', 20)
                except Exception:
                    pass  # Use defaults
                
                stability_metrics = analyze_stability_auto(
                    base_dir=base_dir,
                    target=target,
                    method=method,
                    log_to_console=True,
                    save_report=True,
                    min_overlap_threshold=min_overlap_threshold,
                    min_tau_threshold=min_tau_threshold,
                    top_k=top_k,
                )
                # Log when analysis is skipped due to insufficient snapshots
                if stability_metrics is None:
                    # Get snapshot count for informative message
                    from .io import load_snapshots
                    try:
                        # Use allow_legacy=True since we may have just saved a legacy snapshot
                        snapshots = load_snapshots(base_dir, target, method, allow_legacy=True)
                        snapshot_count = len(snapshots)
                        if snapshot_count < 2:
                            # First snapshot(s) - quiet log, analysis not possible yet
                            logger.debug(
                                f"Stability snapshot saved for {target}/{method} "
                                f"({snapshot_count} available, need 2+ for analysis)"
                            )
                        # When >= 2 snapshots, stability_metrics will be logged by analyze_stability_auto
                    except Exception:
                        # Fallback if loading snapshots fails
                        logger.debug(
                            f"Stability snapshot saved for {target}/{method} (count check failed)"
                        )
            except Exception as e:
                logger.debug(f"Auto-analysis failed (non-critical): {e}")
        
        return snapshot_path
    
    except (ValueError, TypeError) as e:
        # ValueError/TypeError indicates programming error (e.g., partial identity) - re-raise
        logger.error(f"Failed to save importance snapshot: {e}")
        raise
    except Exception as e:
        logger.warning(f"Failed to save importance snapshot: {e}")
        return None


def save_snapshot_from_series_hook(
    target: str,
    method: str,
    importance_series,  # pd.Series
    universe_sig: Optional[str] = None,
    output_dir: Optional[Path] = None,
    run_id: Optional[str] = None,
    auto_analyze: Optional[bool] = None,  # None = load from config
    run_identity: Optional[Any] = None,   # RunIdentity object or dict
    allow_legacy: bool = False,  # If True, allow saving without identity (legacy path)
    prediction_fingerprint: Optional[Dict] = None,  # PredictionFingerprint.to_dict()
    view: Optional[str] = None,  # "CROSS_SECTIONAL" or "SYMBOL_SPECIFIC"
    symbol: Optional[str] = None,  # Symbol name for SYMBOL_SPECIFIC views
    inputs: Optional[Dict] = None,  # Optional inputs dict for fs_snapshot
    process: Optional[Dict] = None,  # Optional process dict for fs_snapshot
    write_fs_snapshot: bool = True,  # If True, also write fs_snapshot.json
    stage: str = "FEATURE_SELECTION",  # Pipeline stage for fs_snapshot labeling
    # Full parity with TARGET_RANKING fields (sourced from FS stage)
    snapshot_seq: int = 0,  # Sequence number for this run
    n_effective: Optional[int] = None,  # Effective sample count from FS
    feature_registry_hash: Optional[str] = None,  # Hash of feature registry
    comparable_key: Optional[str] = None,  # Pre-computed comparison key
) -> Optional[Path]:
    """
    Hook function to save snapshot from pandas Series.

    Convenience wrapper for Series-based importance data.

    Args:
        target: Target name
        method: Method name
        importance_series: pandas Series with feature names as index
        universe_sig: Optional universe identifier
        output_dir: Optional output directory
        run_id: Optional run ID
        auto_analyze: If True, automatically run stability analysis.
                     If None, loads from config (safety.feature_importance.auto_analyze_stability)
        run_identity: Optional RunIdentity object or dict with identity signatures.
        allow_legacy: If True, allow saving without identity (legacy path).
        prediction_fingerprint: Optional PredictionFingerprint.to_dict() for hash tracking.
        view: "CROSS_SECTIONAL" or "SYMBOL_SPECIFIC" for fs_snapshot
        symbol: Symbol name for SYMBOL_SPECIFIC views
        inputs: Optional inputs dict for fs_snapshot
        process: Optional process dict for fs_snapshot
        write_fs_snapshot: If True, also write fs_snapshot.json and update global index
        stage: Pipeline stage for fs_snapshot - "FEATURE_SELECTION" (default) or "TARGET_RANKING"
    
    Returns:
        Path to saved snapshot, or None if saving failed
    """
    # Convert Series to dict
    importance_dict = importance_series.to_dict()
    return save_snapshot_hook(
        target=target,
        method=method,
        importance_dict=importance_dict,
        universe_sig=universe_sig,
        output_dir=output_dir,
        run_id=run_id,
        auto_analyze=auto_analyze,
        run_identity=run_identity,
        allow_legacy=allow_legacy,
        prediction_fingerprint=prediction_fingerprint,
        view=view,
        symbol=symbol,
        inputs=inputs,
        process=process,
        write_fs_snapshot=write_fs_snapshot,
        stage=stage,  # Pass stage for correct labeling
        # Full parity fields
        snapshot_seq=snapshot_seq,
        n_effective=n_effective,
        feature_registry_hash=feature_registry_hash,
        comparable_key=comparable_key,
    )


def analyze_all_stability_hook(
    output_dir: Optional[Path] = None,
    target: Optional[str] = None,
    method: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Hook function to analyze stability for all available snapshots.
    
    **IMPORTANT**: Stability is computed PER-METHOD (not across methods).
    Low overlap between different methods (e.g., RFE vs Boruta vs Lasso) is EXPECTED
    because they use different importance definitions. Only compare snapshots from the
    SAME method across different runs/time periods.
    
    Can be called at end of pipeline run to generate comprehensive stability report.
    
    Args:
        output_dir: Optional output directory (defaults to artifacts/feature_importance)
                    Can be RESULTS/{run}/target_rankings/ or RESULTS/{run}/feature_selections/
                    Function will search REPRODUCIBILITY structure automatically
        target: Optional target name filter (None = all targets)
        method: Optional method filter (None = all methods)
    
    Returns:
        Dictionary mapping "{target}/{method}" to metrics dict
    """
    all_metrics = {}
    
    # Determine base output directory (RESULTS/{run}/)
    # Walk up to find run directory (has targets/, globals/, or cache/)
    # REMOVED: Legacy REPRODUCIBILITY path construction - only use target-first structure
    if output_dir:
        base_output_dir = output_dir
        from TRAINING.orchestration.utils.target_first_paths import run_root as get_run_root
        base_output_dir = get_run_root(base_output_dir)
    else:
        # Default: use get_snapshot_base_dir with None (artifacts/feature_importance)
        base_dir = get_snapshot_base_dir(None)
        if base_dir.exists():
            # Legacy structure: artifacts/feature_importance/{target}/{method}/
            for target_path in base_dir.iterdir():
                if not target_path.is_dir():
                    continue
                
                target = target_path.name
                if target and target != target:
                    continue
                
                for method_path in target_path.iterdir():
                    if not method_path.is_dir():
                        continue
                    
                    method_name = method_path.name
                    if method and method_name != method:
                        continue
                    
                    metrics = analyze_stability_auto(
                        base_dir=base_dir,
                        target=target,
                        method=method_name,
                        log_to_console=True,
                        save_report=True,
                    )
                    
                    if metrics:
                        all_metrics[f"{target}/{method_name}"] = metrics
        return all_metrics
    
    # Search target-first structure using iter_stage_dirs for dual-structure support
    # This handles both new (stage=TARGET_RANKING/) and legacy (no stage prefix) paths
    try:
        from TRAINING.orchestration.utils.target_first_paths import iter_stage_dirs
    except ImportError:
        logger.debug("iter_stage_dirs not available, skipping target-first scanning")
        return all_metrics
    
    targets_dir = base_output_dir / "targets"
    if not targets_dir.exists():
        logger.debug(f"No targets directory found: {targets_dir}")
        return all_metrics
    
    # Helper function to process snapshots and log stability
    def _process_snapshots(snapshot_base_dir: Path, target_name: str, stage_name: str, method_filter: str = None):
        """Process snapshots in a directory and compute stability metrics."""
        from .io import load_snapshots
        from .analysis import compute_stability_metrics
        
        # Look for replicate directories (hash-based snapshots)
        replicate_dir = snapshot_base_dir / "replicate"
        if replicate_dir.exists():
            for replicate_key_dir in replicate_dir.iterdir():
                if not replicate_key_dir.is_dir():
                    continue
                
                replicate_key = replicate_key_dir.name
                snapshots = load_snapshots(snapshot_base_dir, replicate_key=replicate_key)
                if len(snapshots) < 2:
                    continue
                
                # Determine method from first snapshot
                method_name = snapshots[0].method if snapshots else "unknown"
                if method_filter and method_name != method_filter:
                    continue
                
                metrics = compute_stability_metrics(snapshots, top_k=20, filter_by_universe_sig=True)
                if metrics:
                    # Include stage in key for clear separation
                    key = f"{stage_name or 'LEGACY'}/{target_name}/{method_name}"
                    all_metrics[key] = metrics
                    _log_stability(method_name, metrics, stage_name)
        
        # Also check legacy target/method structure
        for method_path in snapshot_base_dir.iterdir():
            if not method_path.is_dir() or method_path.name == "replicate":
                continue
            
            method_name = method_path.name
            if method_filter and method_name != method_filter:
                continue
            
            snapshots = load_snapshots(snapshot_base_dir, target=target_name, method=method_name, allow_legacy=True)
            if len(snapshots) < 2:
                continue
            
            metrics = compute_stability_metrics(snapshots, top_k=20, filter_by_universe_sig=True)
            if metrics:
                key = f"{stage_name or 'LEGACY'}/{target_name}/{method_name}"
                if key not in all_metrics:  # Don't overwrite replicate-based metrics
                    all_metrics[key] = metrics
                    _log_stability(method_name, metrics, stage_name)
    
    def _log_stability(method_name: str, metrics: dict, stage_name: str = None):
        """Log stability metrics with appropriate thresholds."""
        status = metrics.get('status', 'unknown')
        mean_overlap = metrics.get('mean_overlap', 0.0)
        mean_tau = metrics.get('mean_tau', None)
        n_snapshots = metrics.get('n_snapshots', 0)
        
        # Adjust thresholds based on method type
        high_variance_methods = {'stability_selection', 'boruta', 'rfe', 'neural_network'}
        if method_name in high_variance_methods:
            overlap_threshold = 0.5
            tau_threshold = 0.4
        else:
            overlap_threshold = 0.7
            tau_threshold = 0.6
        
        stage_prefix = f"[{stage_name}] " if stage_name else ""
        
        if status == 'stable':
            logger.info(f"  {stage_prefix}[{method_name}] ✅ STABLE: overlap={mean_overlap:.3f}, tau={mean_tau:.3f if mean_tau else 'N/A'}, snapshots={n_snapshots}")
        elif mean_overlap < overlap_threshold or (mean_tau is not None and mean_tau < tau_threshold):
            logger.warning(
                f"  {stage_prefix}[{method_name}] ⚠️  LOW STABILITY: overlap={mean_overlap:.3f} (threshold={overlap_threshold:.1f}), "
                f"tau={mean_tau:.3f if mean_tau else 'N/A'} (threshold={tau_threshold:.1f}), snapshots={n_snapshots}. "
                f"This is comparing {method_name} snapshots across runs - low overlap may indicate method variability or data changes."
            )
        else:
            logger.info(f"  {stage_prefix}[{method_name}] ℹ️  DRIFTING: overlap={mean_overlap:.3f}, tau={mean_tau:.3f if mean_tau else 'N/A'}, snapshots={n_snapshots}")
    
    # Iterate through all targets
    for target_path in targets_dir.iterdir():
        if not target_path.is_dir():
            continue
        
        target_name = target_path.name
        if target and target_name != target:
            continue
        
        # Use iter_stage_dirs to scan both new (stage=*/) and legacy structures
        for stage_name, stage_path in iter_stage_dirs(base_output_dir, target_name):
            # Search both views
            for view in ["CROSS_SECTIONAL", "SYMBOL_SPECIFIC"]:
                view_path = stage_path / view
                if not view_path.exists():
                    continue
                
                # Scan universe directories within view
                for universe_or_symbol_path in view_path.iterdir():
                    if not universe_or_symbol_path.is_dir():
                        continue
                    
                    # Handle both universe= and symbol= prefixes
                    # SST: Use View enum for comparison
                    view_enum = View.from_string(view) if isinstance(view, str) else view
                    if view_enum == View.SYMBOL_SPECIFIC and not universe_or_symbol_path.name.startswith("symbol="):
                        # For SYMBOL_SPECIFIC, we might have universe= then symbol= nested
                        if universe_or_symbol_path.name.startswith("universe="):
                            for symbol_path in universe_or_symbol_path.iterdir():
                                if symbol_path.is_dir() and symbol_path.name.startswith("symbol="):
                                    snapshot_dir = symbol_path / "feature_importance_snapshots"
                                    if snapshot_dir.exists():
                                        _process_snapshots(snapshot_dir, target_name, stage_name, method)
                        continue
                    
                    # Direct universe= or symbol= directory
                    snapshot_dir = universe_or_symbol_path / "feature_importance_snapshots"
                    if snapshot_dir.exists():
                        _process_snapshots(snapshot_dir, target_name, stage_name, method)
    
    return all_metrics
