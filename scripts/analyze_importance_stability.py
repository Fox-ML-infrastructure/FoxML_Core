#!/usr/bin/env python3
# MIT License - see LICENSE file

"""
Feature Importance Stability Analysis Script

CLI tool for analyzing feature importance stability across runs.
No UI, just prints metrics to stdout.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from TRAINING.stability.feature_importance import (
    load_snapshots,
    compute_stability_metrics,
    selection_frequency,
    analyze_stability_auto,
    get_snapshot_base_dir,
)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze feature importance stability across runs"
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target name (e.g., 'peak_60m_0.8')"
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        help="Method name (e.g., 'lightgbm', 'quick_pruner', 'rfe')"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Base directory for snapshots (default: artifacts/feature_importance)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (snapshots in {output_dir}/feature_importance_snapshots)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top features to analyze (default: 20)"
    )
    parser.add_argument(
        "--min-snapshots",
        type=int,
        default=2,
        help="Minimum snapshots required (default: 2)"
    )
    
    args = parser.parse_args()
    
    # Determine base directory
    if args.base_dir:
        base_dir = args.base_dir
    elif args.output_dir:
        base_dir = args.output_dir / "feature_importance_snapshots"
    else:
        base_dir = get_snapshot_base_dir()
    
    # Load snapshots
    snapshots = load_snapshots(base_dir, args.target, args.method)
    
    if len(snapshots) < args.min_snapshots:
        print(f"❌ Insufficient snapshots: {len(snapshots)} < {args.min_snapshots}")
        print(f"   Need at least {args.min_snapshots} snapshots for stability analysis")
        return 1
    
    print(f"\n{'='*60}")
    print(f"Feature Importance Stability Analysis")
    print(f"{'='*60}")
    print(f"Target: {args.target}")
    print(f"Method: {args.method}")
    print(f"Snapshots: {len(snapshots)}")
    print(f"Top-K: {args.top_k}")
    print(f"\n")
    
    # Compute metrics
    metrics = compute_stability_metrics(snapshots, top_k=args.top_k)
    
    print(f"Stability Metrics:")
    print(f"  Top-{args.top_k} overlap: {metrics['mean_overlap']:.3f} ± {metrics['std_overlap']:.3f}")
    if not metrics['mean_tau'] is None and not str(metrics['mean_tau']) == 'nan':
        print(f"  Kendall tau:        {metrics['mean_tau']:.3f} ± {metrics['std_tau']:.3f}")
    print(f"  Comparisons:        {metrics['n_comparisons']}")
    print(f"\n")
    
    # Selection frequency
    freq = selection_frequency(snapshots, top_k=args.top_k)
    if freq:
        print(f"Top-{args.top_k} Selection Frequency:")
        sorted_freq = sorted(freq.items(), key=lambda x: -x[1])
        for feat, p in sorted_freq[:30]:  # Top 30
            print(f"  {feat:40s} {p:5.2%}")
        print(f"\n")
    
    # Snapshot history
    print(f"Snapshot History:")
    for i, snapshot in enumerate(snapshots, 1):
        print(f"  {i}. {snapshot.run_id} ({snapshot.created_at.isoformat()})")
    
    print(f"\n{'='*60}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
