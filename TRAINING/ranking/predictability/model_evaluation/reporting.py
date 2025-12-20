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
    target_name: str,
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
    
    # Find base run directory for target-first structure
    base_output_dir = output_dir
    for _ in range(10):
        if base_output_dir.name == "RESULTS" or (base_output_dir / "targets").exists():
            break
        if not base_output_dir.parent.exists():
            break
        base_output_dir = base_output_dir.parent
    
    target_name_clean = target_column.replace('/', '_').replace('\\', '_')
    
    # Save to target-first structure only
    try:
        from TRAINING.orchestration.utils.target_first_paths import (
            get_target_reproducibility_dir, ensure_target_structure
        )
        ensure_target_structure(base_output_dir, target_name_clean)
        target_repro_dir = get_target_reproducibility_dir(base_output_dir, target_name_clean)
        target_importances_dir = target_repro_dir / "feature_importances"
        target_importances_dir.mkdir(parents=True, exist_ok=True)
        
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
            
            # Save to target-first location
            target_csv_file = target_importances_dir / f"{model_name}_importances.csv"
            df.to_csv(target_csv_file, index=False)
            
            # Save stability snapshot (non-invasive hook)
            try:
                from TRAINING.stability.feature_importance import save_snapshot_hook
                # Use target-first structure for snapshots
                save_snapshot_hook(
                    target_name=target_column,
                    method=model_name,
                    importance_dict=importances,
                    universe_id=view,  # Use view parameter (CROSS_SECTIONAL or SYMBOL_SPECIFIC)
                    output_dir=target_repro_dir,  # Save snapshots in target-first structure
                    auto_analyze=None,  # Load from config
                )
            except Exception as e:
                logger.debug(f"Stability snapshot save failed (non-critical): {e}")
        
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

