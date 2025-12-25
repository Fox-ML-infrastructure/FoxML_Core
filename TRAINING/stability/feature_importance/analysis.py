# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

"""
Feature Importance Stability Analysis

Compute stability metrics and generate reports for feature importance snapshots.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

try:
    from scipy.stats import kendalltau
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("scipy not available, Kendall tau will be unavailable")

from .schema import FeatureImportanceSnapshot
from .io import load_snapshots

logger = logging.getLogger(__name__)


def top_k_overlap(s1: FeatureImportanceSnapshot, s2: FeatureImportanceSnapshot, k: int = 20) -> float:
    """
    Compute Jaccard similarity of top-K features between two snapshots.
    
    **IMPORTANT**: This compares features by NAME, not by importance magnitude.
    Features are already sorted by importance (descending) in the snapshot.
    
    For stability analysis, this should only be called on snapshots from the SAME
    model family and importance method (e.g., LightGBM gain across runs).
    Comparing across different methods (RFE vs Boruta) will naturally have low overlap.
    
    Args:
        s1: First snapshot
        s2: Second snapshot
        k: Number of top features to compare
    
    Returns:
        Jaccard similarity (intersection / union) of top-K features
    """
    # Features are already sorted by importance (descending)
    # Take top k (or all if fewer than k)
    k1 = min(k, len(s1.features))
    k2 = min(k, len(s2.features))
    top1 = set(s1.features[:k1])
    top2 = set(s2.features[:k2])
    
    if not top1 and not top2:
        return 1.0  # Both empty = perfect match
    
    intersection = len(top1 & top2)
    union = len(top1 | top2)
    
    return intersection / union if union > 0 else 0.0


def rank_correlation(s1: FeatureImportanceSnapshot, s2: FeatureImportanceSnapshot) -> float:
    """
    Compute Kendall tau rank correlation between two snapshots.
    
    Args:
        s1: First snapshot
        s2: Second snapshot
    
    Returns:
        Kendall tau correlation coefficient, or NaN if insufficient common features
    """
    if not SCIPY_AVAILABLE:
        logger.warning("scipy not available, cannot compute Kendall tau")
        return np.nan
    
    # Find common features
    common = set(s1.features) & set(s2.features)
    if len(common) < 3:
        return np.nan  # Need at least 3 features for meaningful correlation
    
    # Get ranks for common features (lower rank = higher importance)
    # Features are sorted by importance (descending), so rank = position
    rank1 = {feat: i for i, feat in enumerate(s1.features)}
    rank2 = {feat: i for i, feat in enumerate(s2.features)}
    
    # Extract ranks for common features
    ranks1 = [rank1[feat] for feat in common]
    ranks2 = [rank2[feat] for feat in common]
    
    # Compute Kendall tau
    tau, _ = kendalltau(ranks1, ranks2)
    return float(tau) if not np.isnan(tau) else np.nan


def selection_frequency(
    snapshots: List[FeatureImportanceSnapshot],
    top_k: int = 20
) -> Dict[str, float]:
    """
    Compute how often each feature appears in top-K across snapshots.
    
    Args:
        snapshots: List of snapshots
        top_k: Number of top features to consider
    
    Returns:
        Dictionary mapping feature names to selection frequency (0.0 to 1.0)
    """
    counts: Dict[str, int] = {}
    total = len(snapshots)
    
    if total == 0:
        return {}
    
    for snapshot in snapshots:
        # Features are already sorted by importance (descending)
        top_features = set(snapshot.features[:top_k])
        for feat in top_features:
            counts[feat] = counts.get(feat, 0) + 1
    
    # Convert to frequencies
    return {feat: count / total for feat, count in counts.items()}


def compute_stability_metrics(
    snapshots: List[FeatureImportanceSnapshot],
    top_k: int = 20,
    filter_by_universe_id: bool = True  # NEW: Filter by universe_id to avoid cross-symbol comparisons
) -> Dict[str, float]:
    """
    Compute stability metrics for a list of snapshots.
    
    **CRITICAL**: This function assumes all snapshots are from the SAME model family
    and importance method (e.g., all LightGBM with "native" importance).
    Comparing snapshots from different methods (RFE vs Boruta vs Lasso) will
    naturally have low overlap because they use different importance definitions.
    
    **CRITICAL**: For SYMBOL_SPECIFIC mode, snapshots should be from the SAME symbol.
    Comparing snapshots across different symbols (AAPL vs MSFT) will show low overlap
    due to symbol heterogeneity, not instability. Use filter_by_universe_id=True to
    filter snapshots by the symbol part of universe_id.
    
    The snapshots should be sorted by importance (descending) already, so we
    compare top-K by feature name (not magnitude, since magnitudes are not comparable
    across different importance definitions).
    
    Args:
        snapshots: List of snapshots to analyze (must be same method/family)
        top_k: Number of top features to consider for overlap
        filter_by_universe_id: If True, filter snapshots to only include those with
            the same universe_id (or same symbol prefix if universe_id format is "SYMBOL:...")
    
    Returns:
        Dictionary with stability metrics:
        - mean_overlap: Mean Jaccard similarity of top-K features
        - std_overlap: Std dev of overlap
        - mean_tau: Mean Kendall tau rank correlation
        - std_tau: Std dev of tau
        - n_snapshots: Number of snapshots
        - n_comparisons: Number of pairwise comparisons
        - status: "stable", "drifting", "diverged", or "insufficient"
    """
    # Filter snapshots by universe_id if requested (for SYMBOL_SPECIFIC mode)
    if filter_by_universe_id and len(snapshots) > 0:
        # Extract symbol from universe_id if format is "SYMBOL:..."
        # Group snapshots by symbol (first part before ":")
        universe_id_groups = {}
        for snapshot in snapshots:
            if snapshot.universe_id:
                # Extract symbol part (before ":")
                symbol_part = snapshot.universe_id.split(":")[0] if ":" in snapshot.universe_id else snapshot.universe_id
                if symbol_part not in universe_id_groups:
                    universe_id_groups[symbol_part] = []
                universe_id_groups[symbol_part].append(snapshot)
            else:
                # No universe_id - treat as separate group
                if "NO_UNIVERSE" not in universe_id_groups:
                    universe_id_groups["NO_UNIVERSE"] = []
                universe_id_groups["NO_UNIVERSE"].append(snapshot)
        
        # If we have multiple groups (different symbols), use the largest group
        # and log a warning that we're filtering to avoid cross-symbol comparisons
        if len(universe_id_groups) > 1:
            largest_group = max(universe_id_groups.values(), key=len)
            symbol_for_group = [k for k, v in universe_id_groups.items() if v == largest_group][0]
            logger.warning(
                f"⚠️  Stability computation: Found snapshots from {len(universe_id_groups)} different symbols/universes. "
                f"Filtering to largest group (symbol={symbol_for_group}, n={len(largest_group)} snapshots) to avoid "
                f"cross-symbol comparisons. Low overlap across symbols is expected (symbol heterogeneity), not instability."
            )
            snapshots = largest_group
    
    if len(snapshots) < 2:
        return {
            "mean_overlap": np.nan,
            "std_overlap": np.nan,
            "mean_tau": np.nan,
            "std_tau": np.nan,
            "n_comparisons": 0,
        }
    
    # Compute pairwise metrics (adjacent runs)
    overlaps = []
    taus = []
    
    for i in range(len(snapshots) - 1):
        s1 = snapshots[i]
        s2 = snapshots[i + 1]
        
        overlap = top_k_overlap(s1, s2, k=top_k)
        overlaps.append(overlap)
        
        tau = rank_correlation(s1, s2)
        if not np.isnan(tau):
            taus.append(tau)
    
    # Compute statistics
    overlaps_array = np.array(overlaps)
    taus_array = np.array(taus) if taus else np.array([np.nan])
    
    return {
        "mean_overlap": float(np.nanmean(overlaps_array)),
        "std_overlap": float(np.nanstd(overlaps_array)),
        "mean_tau": float(np.nanmean(taus_array)) if len(taus) > 0 else np.nan,
        "std_tau": float(np.nanstd(taus_array)) if len(taus) > 0 else np.nan,
        "n_comparisons": len(overlaps),
        "n_snapshots": len(snapshots),
    }


def analyze_stability_auto(
    base_dir: Path,
    target_name: str,
    method: str,
    min_snapshots: int = 2,
    top_k: int = 20,
    log_to_console: bool = True,
    save_report: bool = True,
    report_path: Optional[Path] = None,
    min_overlap_threshold: float = 0.7,
    min_tau_threshold: float = 0.6,
) -> Optional[Dict[str, float]]:
    """
    Automatically analyze stability if enough snapshots exist.
    
    This is the main hook function that can be called from pipeline endpoints.
    
    Args:
        base_dir: Base directory for snapshots
        target_name: Target name
        method: Method name
        min_snapshots: Minimum snapshots required for analysis
        top_k: Number of top features to consider
        log_to_console: If True, log metrics to console
        save_report: If True, save text report to disk
        report_path: Optional path for report (defaults to base_dir/stability_reports/)
        min_overlap_threshold: Warning threshold for overlap (default: 0.7)
        min_tau_threshold: Warning threshold for tau (default: 0.6)
    
    Returns:
        Dictionary with stability metrics, or None if insufficient snapshots
    """
    snapshots = load_snapshots(base_dir, target_name, method)
    
    if len(snapshots) < min_snapshots:
        logger.debug(
            f"Insufficient snapshots for {target_name}/{method}: "
            f"{len(snapshots)} < {min_snapshots}"
        )
        return None
    
    metrics = compute_stability_metrics(snapshots, top_k=top_k)
    
    if log_to_console:
        logger.info(f"📊 Stability for {target_name}/{method}:")
        logger.info(f"   Snapshots: {metrics['n_snapshots']}")
        logger.info(f"   Top-{top_k} overlap: {metrics['mean_overlap']:.3f} ± {metrics['std_overlap']:.3f}")
        if not np.isnan(metrics['mean_tau']):
            logger.info(f"   Kendall tau: {metrics['mean_tau']:.3f} ± {metrics['std_tau']:.3f}")
        
        # Warn if stability is low
        if metrics['mean_overlap'] < min_overlap_threshold:
            logger.warning(
                f"   ⚠️  Low stability detected (overlap {metrics['mean_overlap']:.3f} < {min_overlap_threshold})"
            )
        if not np.isnan(metrics['mean_tau']) and metrics['mean_tau'] < min_tau_threshold:
            logger.warning(
                f"   ⚠️  Low rank correlation (tau {metrics['mean_tau']:.3f} < {min_tau_threshold})"
            )
    
    if save_report:
        if report_path is None:
            report_dir = base_dir / "stability_reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_dir / f"{target_name}_{method}.txt"
        
        save_stability_report(metrics, snapshots, report_path, top_k=top_k)
    
    return metrics


def save_stability_report(
    metrics: Dict[str, float],
    snapshots: List[FeatureImportanceSnapshot],
    report_path: Path,
    top_k: int = 20
) -> None:
    """
    Save stability report to text file.
    
    Args:
        metrics: Stability metrics dictionary
        snapshots: List of snapshots analyzed
        report_path: Path to save report
        top_k: Number of top features to include in report
    """
    try:
        with report_path.open("w") as f:
            f.write(f"Feature Importance Stability Report\n")
            f.write(f"{'='*60}\n\n")
            
            if len(snapshots) > 0:
                f.write(f"Target: {snapshots[0].target_name}\n")
                f.write(f"Method: {snapshots[0].method}\n")
                f.write(f"Universe: {snapshots[0].universe_id or 'N/A'}\n")
            
            f.write(f"\nMetrics:\n")
            f.write(f"  Snapshots analyzed: {metrics['n_snapshots']}\n")
            f.write(f"  Comparisons: {metrics['n_comparisons']}\n")
            f.write(f"  Top-{top_k} overlap: {metrics['mean_overlap']:.3f} ± {metrics['std_overlap']:.3f}\n")
            if not np.isnan(metrics['mean_tau']):
                f.write(f"  Kendall tau: {metrics['mean_tau']:.3f} ± {metrics['std_tau']:.3f}\n")
            
            # Selection frequency
            freq = selection_frequency(snapshots, top_k=top_k)
            if freq:
                f.write(f"\nTop-{top_k} Selection Frequency:\n")
                sorted_freq = sorted(freq.items(), key=lambda x: -x[1])
                for feat, p in sorted_freq[:30]:  # Top 30
                    f.write(f"  {feat:40s} {p:5.2%}\n")
            
            f.write(f"\nSnapshot History:\n")
            for i, snapshot in enumerate(snapshots, 1):
                f.write(f"  {i}. {snapshot.run_id} ({snapshot.created_at.isoformat()})\n")
        
        logger.debug(f"Saved stability report: {report_path}")
    except Exception as e:
        logger.warning(f"Failed to save stability report to {report_path}: {e}")
