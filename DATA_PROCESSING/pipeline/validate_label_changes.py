#!/usr/bin/env python3

# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

"""
Validate Label Changes

Compares old vs new labeled datasets to verify the horizon unit fix.
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compare_targets(
    old_path: Path,
    new_path: Path,
    target_col: str,
    sample_size: int = 1000
) -> dict:
    """Compare old vs new target values for a specific target column."""
    try:
        # Load old data
        old_df = pd.read_parquet(old_path)
        if target_col not in old_df.columns:
            return {"error": f"Target column {target_col} not found in old data"}
        
        # Load new data
        new_df = pd.read_parquet(new_path)
        if target_col not in new_df.columns:
            return {"error": f"Target column {target_col} not found in new data"}
        
        # Align by timestamp (assuming 'ts' column exists)
        if 'ts' in old_df.columns and 'ts' in new_df.columns:
            merged = pd.merge(
                old_df[['ts', target_col]].rename(columns={target_col: 'old_value'}),
                new_df[['ts', target_col]].rename(columns={target_col: 'new_value'}),
                on='ts',
                how='inner'
            )
        else:
            # Fallback: align by index
            min_len = min(len(old_df), len(new_df))
            merged = pd.DataFrame({
                'old_value': old_df[target_col].iloc[:min_len].values,
                'new_value': new_df[target_col].iloc[:min_len].values
            })
        
        # Sample if needed
        if len(merged) > sample_size:
            merged = merged.sample(n=sample_size, random_state=42)
        
        # Compute statistics
        old_vals = merged['old_value'].dropna()
        new_vals = merged['new_value'].dropna()
        
        # Count differences
        both_present = merged.dropna()
        if len(both_present) > 0:
            exact_match = (both_present['old_value'] == both_present['new_value']).sum()
            different = len(both_present) - exact_match
            match_rate = exact_match / len(both_present) if len(both_present) > 0 else 0
        else:
            exact_match = 0
            different = 0
            match_rate = 0
        
        # Compute distribution stats
        stats = {
            "total_comparable": len(both_present),
            "exact_matches": exact_match,
            "differences": different,
            "match_rate": match_rate,
            "old_stats": {
                "count": len(old_vals),
                "mean": float(old_vals.mean()) if len(old_vals) > 0 else None,
                "std": float(old_vals.std()) if len(old_vals) > 0 else None,
                "min": float(old_vals.min()) if len(old_vals) > 0 else None,
                "max": float(old_vals.max()) if len(old_vals) > 0 else None,
                "unique_values": int(old_vals.nunique()) if len(old_vals) > 0 else 0
            },
            "new_stats": {
                "count": len(new_vals),
                "mean": float(new_vals.mean()) if len(new_vals) > 0 else None,
                "std": float(new_vals.std()) if len(new_vals) > 0 else None,
                "min": float(new_vals.min()) if len(new_vals) > 0 else None,
                "max": float(new_vals.max()) if len(new_vals) > 0 else None,
                "unique_values": int(new_vals.nunique()) if len(new_vals) > 0 else 0
            }
        }
        
        # Show examples of differences
        if different > 0:
            diff_rows = both_present[both_present['old_value'] != both_present['new_value']]
            stats["difference_examples"] = diff_rows.head(10).to_dict('records')
        
        return stats
        
    except Exception as e:
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Validate label changes between old and new datasets")
    parser.add_argument("--old-dir", required=True, help="Old labeled data directory")
    parser.add_argument("--new-dir", required=True, help="New labeled data directory")
    parser.add_argument("--symbol", required=True, help="Symbol to compare")
    parser.add_argument("--target", required=True, help="Target column to compare (e.g., y_will_peak_60m_0.8)")
    parser.add_argument("--sample-size", type=int, default=1000, help="Sample size for comparison (default: 1000)")
    
    args = parser.parse_args()
    
    # Find files
    old_path = Path(args.old_dir) / "interval=5m" / f"symbol={args.symbol}" / "*.parquet"
    new_path = Path(args.new_dir) / "interval=5m" / f"symbol={args.symbol}" / "*.parquet"
    
    old_files = list(Path(args.old_dir).glob(f"interval=5m/symbol={args.symbol}/*.parquet"))
    new_files = list(Path(args.new_dir).glob(f"interval=5m/symbol={args.symbol}/*.parquet"))
    
    if not old_files:
        logger.error(f"No old files found for {args.symbol}")
        return
    
    if not new_files:
        logger.error(f"No new files found for {args.symbol}")
        return
    
    # Use first file from each
    old_file = old_files[0]
    new_file = new_files[0]
    
    logger.info(f"📊 Comparing {args.target} for {args.symbol}")
    logger.info(f"   Old: {old_file}")
    logger.info(f"   New: {new_file}")
    
    # Compare
    result = compare_targets(old_file, new_file, args.target, args.sample_size)
    
    if "error" in result:
        logger.error(f"❌ Error: {result['error']}")
        return
    
    # Print results
    logger.info(f"\n{'='*60}")
    logger.info(f"Comparison Results: {args.target}")
    logger.info(f"{'='*60}")
    logger.info(f"Total comparable rows: {result['total_comparable']}")
    logger.info(f"Exact matches: {result['exact_matches']} ({result['match_rate']*100:.1f}%)")
    logger.info(f"Differences: {result['differences']} ({(1-result['match_rate'])*100:.1f}%)")
    
    logger.info(f"\nOld Statistics:")
    logger.info(f"  Count: {result['old_stats']['count']}")
    logger.info(f"  Mean: {result['old_stats']['mean']:.6f}" if result['old_stats']['mean'] is not None else "  Mean: N/A")
    logger.info(f"  Std: {result['old_stats']['std']:.6f}" if result['old_stats']['std'] is not None else "  Std: N/A")
    logger.info(f"  Unique values: {result['old_stats']['unique_values']}")
    
    logger.info(f"\nNew Statistics:")
    logger.info(f"  Count: {result['new_stats']['count']}")
    logger.info(f"  Mean: {result['new_stats']['mean']:.6f}" if result['new_stats']['mean'] is not None else "  Mean: N/A")
    logger.info(f"  Std: {result['new_stats']['std']:.6f}" if result['new_stats']['std'] is not None else "  Std: N/A")
    logger.info(f"  Unique values: {result['new_stats']['unique_values']}")
    
    # Expected: For 60m horizon on 5m data, old code used 60 bars, new code uses 12 bars
    # This should result in significant differences
    if result['differences'] > 0:
        logger.info(f"\n✅ Differences detected (expected due to horizon unit fix)")
        logger.info(f"   Old code: horizon_minutes used as bars (e.g., 60 bars for 60m)")
        logger.info(f"   New code: horizon_minutes converted to bars (e.g., 12 bars for 60m on 5m data)")
        if 'difference_examples' in result:
            logger.info(f"\n   Example differences (first 5):")
            for i, ex in enumerate(result['difference_examples'][:5], 1):
                logger.info(f"     {i}. Old: {ex['old_value']}, New: {ex['new_value']}")
    else:
        logger.warning(f"\n⚠️  No differences detected - this may indicate:")
        logger.warning(f"   1. Both datasets use the same (fixed) code")
        logger.warning(f"   2. Target column doesn't exist in one dataset")
        logger.warning(f"   3. Data alignment issue")


if __name__ == "__main__":
    main()
