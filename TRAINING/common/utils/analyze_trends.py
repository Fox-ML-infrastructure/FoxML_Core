#!/usr/bin/env python3
"""
Trend Analysis CLI

Analyzes trends across runs in the REPRODUCIBILITY directory.

Usage:
    python -m TRAINING.utils.analyze_trends [--view STRICT|PROGRESS] [--half-life-days 7.0] [--output TREND_REPORT.json]
"""

import argparse
import logging
from pathlib import Path
from TRAINING.common.utils.trend_analyzer import TrendAnalyzer, SeriesView

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Analyze trends across reproducibility runs")
    parser.add_argument(
        '--reproducibility-dir',
        type=Path,
        default=None,
        help='Path to REPRODUCIBILITY directory (default: auto-detect from RESULTS)'
    )
    parser.add_argument(
        '--view',
        type=str,
        choices=['STRICT', 'PROGRESS'],
        default='STRICT',
        help='Series view: STRICT (all keys match) or PROGRESS (allow feature changes)'
    )
    parser.add_argument(
        '--half-life-days',
        type=float,
        default=7.0,
        help='Exponential decay half-life in days (default: 7.0)'
    )
    parser.add_argument(
        '--min-runs',
        type=int,
        default=5,
        help='Minimum runs required for trend fitting (default: 5)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output path for trend report (default: REPRODUCIBILITY/TREND_REPORT.json)'
    )
    
    args = parser.parse_args()
    
    # Auto-detect reproducibility directory
    if args.reproducibility_dir is None:
        # Try to find RESULTS directory
        repo_root = Path(__file__).parent.parent.parent
        results_dir = repo_root / "RESULTS"
        
        # Find most recent run's REPRODUCIBILITY directory
        reproducibility_dirs = list(results_dir.glob("*/REPRODUCIBILITY"))
        if not reproducibility_dirs:
            logger.error("Could not find REPRODUCIBILITY directory. Specify --reproducibility-dir")
            return 1
        
        # Use most recent (by modification time)
        args.reproducibility_dir = max(reproducibility_dirs, key=lambda p: p.stat().st_mtime)
        logger.info(f"Auto-detected REPRODUCIBILITY directory: {args.reproducibility_dir}")
    
    # Initialize analyzer
    analyzer = TrendAnalyzer(
        reproducibility_dir=args.reproducibility_dir,
        half_life_days=args.half_life_days,
        min_runs_for_trend=args.min_runs
    )
    
    # Analyze trends
    view = SeriesView(args.view)
    logger.info(f"Analyzing trends with {view.value} view...")
    trends = analyzer.analyze_all_series(view=view)
    
    # Write report
    if args.output is None:
        args.output = args.reproducibility_dir / "TREND_REPORT.json"
    
    analyzer.write_trend_report(trends, args.output)
    
    # Print summary
    n_series = len(trends)
    n_trends = sum(len(t) for t in trends.values())
    logger.info(f"✅ Analyzed {n_series} series, {n_trends} trends")
    
    # Count alerts
    total_alerts = 0
    for trend_list in trends.values():
        for trend in trend_list:
            total_alerts += len(trend.alerts)
    
    if total_alerts > 0:
        logger.warning(f"⚠️  Found {total_alerts} alerts across all trends")
    
    return 0


if __name__ == '__main__':
    exit(main())
