# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

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
                    # Partial identity - behavior depends on mode
                    error_msg = (
                        "Cannot save snapshot with partial RunIdentity (is_final=False). "
                        "Call run_identity.finalize(feature_signature) before saving. "
                        f"Current identity: {run_identity.debug_key if hasattr(run_identity, 'debug_key') else 'unknown'}"
                    )
                    if identity_mode == "strict":
                        raise ValueError(error_msg)
                    elif identity_mode == "relaxed":
                        logger.error(f"Identity validation failed (relaxed mode): {error_msg}")
                        # Continue with degraded identity
                    else:  # legacy mode
                        logger.warning(f"Partial identity ignored (legacy mode): {error_msg}")
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
        # Use target for target-first structure
        # Snapshots should only be saved to target-specific directories, never at root level
        base_dir = get_snapshot_base_dir(output_dir, target=target)
        snapshot_path = save_importance_snapshot(snapshot, base_dir, use_hash_path=use_hash_path)
        
        logger.debug(f"Saved importance snapshot: {snapshot_path}")
        
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
                        if snapshot_count == 1:
                            logger.info(
                                f"📊 Stability analysis for {target}/{method}: "
                                f"Snapshot saved (1 snapshot available, need 2+ for analysis). "
                                f"Stats will be available after the next symbol/run."
                            )
                        else:
                            logger.info(
                                f"📊 Stability analysis for {target}/{method}: "
                                f"Snapshot saved ({snapshot_count} snapshots available, need 2+ for analysis). "
                                f"Analysis will run automatically once more snapshots are available."
                            )
                    except Exception:
                        # Fallback if loading snapshots fails
                        logger.info(
                            f"📊 Stability analysis for {target}/{method}: "
                            f"Snapshot saved (analysis will run once more snapshots are available)"
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
        for _ in range(10):
            # Only stop if we find a run directory (has targets/, globals/, or cache/)
            # Don't stop at RESULTS/ - continue to find actual run directory
            if (base_output_dir / "targets").exists() or (base_output_dir / "globals").exists() or (base_output_dir / "cache").exists():
                break
            if not base_output_dir.parent.exists():
                break
            base_output_dir = base_output_dir.parent
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
    
    # Search REPRODUCIBILITY structure
    repro_dir = base_output_dir / "REPRODUCIBILITY"
    if not repro_dir.exists():
        logger.debug(f"No REPRODUCIBILITY directory found: {repro_dir}")
        return {}
    
    # Search both TARGET_RANKING and FEATURE_SELECTION
    for stage in ["TARGET_RANKING", "FEATURE_SELECTION"]:
        stage_dir = repro_dir / stage
        if not stage_dir.exists():
            continue
        
        # Search both CROSS_SECTIONAL and SYMBOL_SPECIFIC views
        for view in ["CROSS_SECTIONAL", "SYMBOL_SPECIFIC"]:
            view_dir = stage_dir / view
            if not view_dir.exists():
                continue
            
            # Iterate through targets
            for target_path in view_dir.iterdir():
                if not target_path.is_dir():
                    continue
                
                target = target_path.name
                if target and target != target:
                    continue
                
                # For SYMBOL_SPECIFIC, iterate through symbol directories
                if view == "SYMBOL_SPECIFIC":
                    for symbol_path in target_path.iterdir():
                        if not symbol_path.is_dir() or not symbol_path.name.startswith("symbol="):
                            continue
                        
                        # Get snapshot base directory for this target/symbol
                        snapshot_base_dir = get_snapshot_base_dir(symbol_path)
                        if not snapshot_base_dir.exists():
                            continue
                        
                        # Find all methods
                        for method_path in snapshot_base_dir.iterdir():
                            if not method_path.is_dir():
                                continue
                            
                            method_name = method_path.name
                            if method and method_name != method:
                                continue
                            
                            # Load snapshots for this target/method
                            from .io import load_snapshots
                            snapshots = load_snapshots(snapshot_base_dir, target=target, method=method_name, allow_legacy=True)
                            if len(snapshots) < 2:
                                continue
                            
                            # Analyze stability (filter by universe_sig to avoid cross-symbol comparisons)
                            # filter_mode defaults to "replicate" - silently ignores legacy snapshots without signatures
                            from .analysis import compute_stability_metrics
                            metrics = compute_stability_metrics(snapshots, top_k=20, filter_by_universe_sig=True)
                            if metrics:
                                all_metrics[f"{target}/{method_name}"] = metrics
                                # Log per-method stability with context
                                status = metrics.get('status', 'unknown')
                                mean_overlap = metrics.get('mean_overlap', 0.0)
                                mean_tau = metrics.get('mean_tau', None)
                                n_snapshots = metrics.get('n_snapshots', 0)
                                
                                # Adjust warning thresholds based on method type
                                high_variance_methods = {'stability_selection', 'boruta', 'rfe', 'neural_network'}
                                if method_name in high_variance_methods:
                                    overlap_threshold = 0.5
                                    tau_threshold = 0.4
                                else:
                                    overlap_threshold = 0.7
                                    tau_threshold = 0.6
                                
                                if status == 'stable':
                                    logger.info(f"  [{method_name}] ✅ STABLE: overlap={mean_overlap:.3f}, tau={mean_tau:.3f if mean_tau else 'N/A'}, snapshots={n_snapshots}")
                                elif mean_overlap < overlap_threshold or (mean_tau is not None and mean_tau < tau_threshold):
                                    logger.warning(
                                        f"  [{method_name}] ⚠️  LOW STABILITY: overlap={mean_overlap:.3f} (threshold={overlap_threshold:.1f}), "
                                        f"tau={mean_tau:.3f if mean_tau else 'N/A'} (threshold={tau_threshold:.1f}), snapshots={n_snapshots}. "
                                        f"This is comparing {method_name} snapshots across runs - low overlap may indicate method variability or data changes."
                                    )
                                else:
                                    logger.info(f"  [{method_name}] ℹ️  DRIFTING: overlap={mean_overlap:.3f}, tau={mean_tau:.3f if mean_tau else 'N/A'}, snapshots={n_snapshots}")
                else:
                    # CROSS_SECTIONAL: target directory directly contains feature_importance_snapshots
                    snapshot_base_dir = get_snapshot_base_dir(target_path)
                    if not snapshot_base_dir.exists():
                        continue
                    
                    # Find all methods
                    for method_path in snapshot_base_dir.iterdir():
                        if not method_path.is_dir():
                            continue
                        
                        method_name = method_path.name
                        if method and method_name != method:
                            continue
                        
                        # Load snapshots for this target/method
                        from .io import load_snapshots
                        snapshots = load_snapshots(snapshot_base_dir, target=target, method=method_name, allow_legacy=True)
                        if len(snapshots) < 2:
                            continue
                        
                        # Analyze stability (filter by universe_sig to avoid cross-symbol comparisons)
                        # filter_mode defaults to "replicate" - silently ignores legacy snapshots without signatures
                        from .analysis import compute_stability_metrics
                        metrics = compute_stability_metrics(snapshots, top_k=20, filter_by_universe_sig=True)
                        if metrics:
                            all_metrics[f"{target}/{method_name}"] = metrics
                            # Log per-method stability with context
                            status = metrics.get('status', 'unknown')
                            mean_overlap = metrics.get('mean_overlap', 0.0)
                            mean_tau = metrics.get('mean_tau', None)
                            n_snapshots = metrics.get('n_snapshots', 0)
                            
                            # Adjust warning thresholds based on method type
                            # Some methods (stability_selection, boruta) are inherently more variable
                            high_variance_methods = {'stability_selection', 'boruta', 'rfe', 'neural_network'}
                            if method_name in high_variance_methods:
                                # Lower thresholds for high-variance methods
                                overlap_threshold = 0.5  # vs 0.7 default
                                tau_threshold = 0.4  # vs 0.6 default
                            else:
                                overlap_threshold = 0.7
                                tau_threshold = 0.6
                            
                            if status == 'stable':
                                logger.info(f"  [{method_name}] ✅ STABLE: overlap={mean_overlap:.3f}, tau={mean_tau:.3f if mean_tau else 'N/A'}, snapshots={n_snapshots}")
                            elif mean_overlap < overlap_threshold or (mean_tau is not None and mean_tau < tau_threshold):
                                # Only warn if below threshold (not just "drifting" status)
                                logger.warning(
                                    f"  [{method_name}] ⚠️  LOW STABILITY: overlap={mean_overlap:.3f} (threshold={overlap_threshold:.1f}), "
                                    f"tau={mean_tau:.3f if mean_tau else 'N/A'} (threshold={tau_threshold:.1f}), snapshots={n_snapshots}. "
                                    f"This is comparing {method_name} snapshots across runs - low overlap may indicate method variability or data changes."
                                )
                            else:
                                logger.info(f"  [{method_name}] ℹ️  DRIFTING: overlap={mean_overlap:.3f}, tau={mean_tau:.3f if mean_tau else 'N/A'}, snapshots={n_snapshots}")
                        if metrics:
                            all_metrics[f"{target}/{method_name}"] = metrics
                            # Log per-method stability with context
                            status = metrics.get('status', 'unknown')
                            mean_overlap = metrics.get('mean_overlap', 0.0)
                            mean_tau = metrics.get('mean_tau', None)
                            n_snapshots = metrics.get('n_snapshots', 0)
                            
                            # Adjust warning thresholds based on method type
                            high_variance_methods = {'stability_selection', 'boruta', 'rfe', 'neural_network'}
                            if method_name in high_variance_methods:
                                overlap_threshold = 0.5
                                tau_threshold = 0.4
                            else:
                                overlap_threshold = 0.7
                                tau_threshold = 0.6
                            
                            if status == 'stable':
                                logger.info(f"  [{method_name}] ✅ STABLE: overlap={mean_overlap:.3f}, tau={mean_tau:.3f if mean_tau else 'N/A'}, snapshots={n_snapshots}")
                            elif mean_overlap < overlap_threshold or (mean_tau is not None and mean_tau < tau_threshold):
                                logger.warning(
                                    f"  [{method_name}] ⚠️  LOW STABILITY: overlap={mean_overlap:.3f} (threshold={overlap_threshold:.1f}), "
                                    f"tau={mean_tau:.3f if mean_tau else 'N/A'} (threshold={tau_threshold:.1f}), snapshots={n_snapshots}. "
                                    f"This is comparing {method_name} snapshots across runs - low overlap may indicate method variability or data changes."
                                )
                            else:
                                logger.info(f"  [{method_name}] ℹ️  DRIFTING: overlap={mean_overlap:.3f}, tau={mean_tau:.3f if mean_tau else 'N/A'}, snapshots={n_snapshots}")
    
    return all_metrics
