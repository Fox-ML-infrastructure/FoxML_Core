# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Reporting Functions

Functions for logging summaries and saving feature importances.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

# Add project root for _REPO_ROOT
import sys
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def log_canonical_summary(
    target: str,
    target_column: str,
    symbols: List[str],
    time_vals: Optional[np.ndarray],
    interval: Optional[Any],
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
    cohort_path: Optional[str],
    splitter_name: Optional[str] = None,
    purge_minutes: Optional[float] = None,
    embargo_minutes: Optional[float] = None,
    max_feature_lookback_minutes: Optional[float] = None,
    n_splits: Optional[int] = None,
    lookback_budget_minutes: Optional[Any] = None,
    purge_include_feature_lookback: Optional[bool] = None,
    gatekeeper_threshold_source: Optional[str] = None
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
    
    # CV splitter and leakage budget details (CRITICAL for audit)
    if splitter_name:
        logger.info(f"splitter: {splitter_name}")
    if n_splits is not None:
        logger.info(f"n_splits: {n_splits}")
    if purge_minutes is not None:
        logger.info(f"purge_minutes: {purge_minutes:.1f}m")
    if embargo_minutes is not None:
        logger.info(f"embargo_minutes: {embargo_minutes:.1f}m")
    if max_feature_lookback_minutes is not None:
        logger.info(f"max_feature_lookback_minutes: {max_feature_lookback_minutes:.1f}m")
    
    # Config trace for leakage detection settings (CRITICAL for auditability)
    logger.info("")
    logger.info("📋 CONFIG TRACE: Leakage Detection Settings")
    logger.info("-" * 60)
    if lookback_budget_minutes is not None:
        if isinstance(lookback_budget_minutes, str):
            logger.info(f"  lookback_budget_minutes: {lookback_budget_minutes} (source: config)")
        else:
            logger.info(f"  lookback_budget_minutes: {lookback_budget_minutes:.1f}m (source: config)")
    else:
        logger.info(f"  lookback_budget_minutes: auto (not set, using actual max)")
    if purge_include_feature_lookback is not None:
        logger.info(f"  purge_include_feature_lookback: {purge_include_feature_lookback} (source: config)")
    else:
        logger.info(f"  purge_include_feature_lookback: N/A (not available)")
    if gatekeeper_threshold_source is not None:
        logger.info(f"  gatekeeper_threshold_source: {gatekeeper_threshold_source}")
    else:
        logger.info(f"  gatekeeper_threshold_source: N/A (not available)")
    logger.info("-" * 60)
    logger.info("")
    
    if cohort_path:
        logger.info(f"repro: {cohort_path}")
    logger.info("=" * 60)


def save_feature_importances(
    target_column: str,
    symbol: str,
    feature_importances: Dict[str, Dict[str, float]],
    output_dir: Path = None,
    view: str = "CROSS_SECTIONAL",
    universe_sig: Optional[str] = None,  # PATCH 4: Required for proper scoping
    run_identity: Optional[Any] = None,  # Finalized RunIdentity OR Dict[str, RunIdentity] per model
    model_metrics: Optional[Dict[str, Dict]] = None,  # Model metrics with prediction fingerprints
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
        run_identity: Finalized RunIdentity for hash-based storage, OR Dict[str, RunIdentity]
                      mapping model_name -> identity for per-model snapshots
        model_metrics: Model metrics dict containing prediction fingerprints
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
    
    # Find base run directory for target-first structure using SST helper
    from TRAINING.orchestration.utils.target_first_paths import run_root as get_run_root
    base_output_dir = get_run_root(output_dir)
    
    from TRAINING.orchestration.utils.target_first_paths import normalize_target_name
    target_clean = normalize_target_name(target_column)
    
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
            stage="TARGET_RANKING",  # Explicit stage for proper path scoping
        )
        target_importances_dir = layout.feature_importance_dir()
        target_importances_dir.mkdir(parents=True, exist_ok=True)
        
        # For stability snapshots, use the importances dir as base
        target_repro_dir = target_importances_dir.parent
        
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
            
            # Save to properly scoped location
            target_csv_file = target_importances_dir / f"{model_name}_importances.csv"
            df.to_csv(target_csv_file, index=False)
            
            # Save stability snapshot (non-invasive hook)
            try:
                from TRAINING.stability.feature_importance import save_snapshot_hook
                
                # Extract prediction fingerprint if available
                prediction_fp = None
                if model_metrics and model_name in model_metrics:
                    prediction_fp = model_metrics[model_name].get('prediction_fingerprint')
                
                # FIX: Look up per-model identity if run_identity is a dict
                # This ensures each model gets its own strict_key/replicate_key
                model_identity = None
                if isinstance(run_identity, dict):
                    # Per-model identity dict: {model_name: RunIdentity}
                    model_identity = run_identity.get(model_name)
                    if model_identity is None:
                        logger.debug(f"No per-model identity for {model_name}, using legacy path")
                else:
                    # Shared identity (fallback) - all models use same keys (may overwrite)
                    model_identity = run_identity
                
                # Use properly scoped structure for snapshots
                # NOTE: This is called from TARGET_RANKING stage (model_evaluation.py)
                # We disable fs_snapshot here - TARGET_RANKING uses snapshot.json, not fs_snapshot.json
                save_snapshot_hook(
                    target=target_column,
                    method=model_name,
                    importance_dict=importances,
                    universe_sig=universe_sig,  # Use universe_sig, not view
                    output_dir=target_repro_dir,  # Save snapshots in scoped structure
                    auto_analyze=None,  # Load from config
                    run_identity=model_identity,  # FIX: Use per-model identity
                    allow_legacy=True,  # FIX: Ensure ALL model families get snapshots
                    prediction_fingerprint=prediction_fp,  # Pass prediction hash for determinism tracking
                    stage="TARGET_RANKING",  # Correctly label as TARGET_RANKING stage
                    write_fs_snapshot=False,  # TARGET_RANKING uses snapshot.json, not fs_snapshot
                    view=view,  # Pass view for proper scoping
                    symbol=symbol,  # Pass symbol for SYMBOL_SPECIFIC view
                )
            except Exception as e:
                # FIX: Log at WARNING level to diagnose why non-xgboost models fail
                import traceback
                logger.warning(f"Stability snapshot save failed for {model_name}: {e}")
                logger.debug(f"Traceback for {model_name}: {traceback.format_exc()}")
        
        logger.info(f"  💾 Saved feature importances to: {target_importances_dir}")
    except Exception as e:
        logger.warning(f"Failed to save feature importances to target-first structure: {e}")


def log_suspicious_features(
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

