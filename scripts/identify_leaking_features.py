#!/usr/bin/env python
"""
Identify and Remove Data-Leaking Features

Data leakage occurs when features use future information that wouldn't
be available at prediction time.
"""

import sys
from pathlib import Path
import pandas as pd
import logging

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def identify_leaking_features(df: pd.DataFrame) -> dict:
    """
    Identify features that likely leak future information.
    
    Returns:
        dict with 'definite_leaks', 'probable_leaks', 'safe_features'
    """
    
    all_features = [c for c in df.columns 
                   if not c.startswith(('y_', 'fwd_ret_', 'barrier_', 'zigzag_', 'p_'))
                   and c not in ['ts', 'datetime', 'symbol', 'interval', 'source']]
    
    definite_leaks = []
    probable_leaks = []
    safe_features = []
    
    # Keywords that DEFINITELY indicate future-looking
    definite_leak_keywords = [
        'time_in_profit',  # Looks ahead to see if profitable
        'time_in_drawdown',
        'mfe_share',       # Max favorable excursion (future path)
        'mdd_share',       # Max drawdown (future path)
        'tth_',            # Time to hit (when future barrier is hit)
        'flipcount',       # Number of future crossings
        'excursion_up',    # Future path metrics
        'excursion_down',
    ]
    
    # Keywords that PROBABLY indicate future-looking
    probable_leak_keywords = [
        'mfe',    # Max favorable excursion
        'mdd',    # Max drawdown
    ]
    
    # Keywords for SAFE features (calculated from past only)
    safe_keywords = [
        'ret_',      # Returns (as long as not fwd_ret_)
        'rsi_',      # RSI (calculated from past)
        'macd_',     # MACD (past)
        'bb_',       # Bollinger Bands (past)
        'atr_',      # ATR (past)
        'vol_',      # Volatility (past)
        'regime',    # Regime detection (past)
    ]
    
    for feature in all_features:
        feature_lower = feature.lower()
        
        # Check definite leaks
        if any(keyword in feature_lower for keyword in definite_leak_keywords):
            definite_leaks.append(feature)
        # Check probable leaks
        elif any(keyword in feature_lower for keyword in probable_leak_keywords):
            probable_leaks.append(feature)
        # Likely safe
        elif any(keyword in feature_lower for keyword in safe_keywords):
            safe_features.append(feature)
        else:
            # Unknown - mark as probable leak to be safe
            probable_leaks.append(feature)
    
    return {
        'definite_leaks': definite_leaks,
        'probable_leaks': probable_leaks,
        'safe_features': safe_features,
        'total_features': len(all_features)
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, default='AAPL',
                       help='Symbol to check (default: AAPL)')
    parser.add_argument('--data-dir', type=Path,
                       default=_REPO_ROOT / 'data/data_labeled/interval=5m')
    parser.add_argument('--save-safe-list', action='store_true',
                       help='Save list of safe features to file')
    args = parser.parse_args()
    
    # Load data
    symbol_dir = args.data_dir / f"symbol={args.symbol}"
    parquet_file = symbol_dir / f"{args.symbol}.parquet"
    
    if not parquet_file.exists():
        logger.error(f"❌ Data not found: {parquet_file}")
        return 1
    
    logger.info(f"Loading {args.symbol} data...")
    df = pd.read_parquet(parquet_file)
    
    # Identify leaking features
    results = identify_leaking_features(df)
    
    # Report
    logger.info("")
    logger.info("="*80)
    logger.info("DATA LEAKAGE ANALYSIS")
    logger.info("="*80)
    logger.info(f"Total features: {results['total_features']}")
    logger.info("")
    
    logger.info(f"🚨 DEFINITE LEAKS: {len(results['definite_leaks'])} features")
    logger.info("   These MUST be removed (they use future information):")
    for feature in sorted(results['definite_leaks'])[:20]:
        logger.info(f"     ❌ {feature}")
    if len(results['definite_leaks']) > 20:
        logger.info(f"     ... and {len(results['definite_leaks']) - 20} more")
    logger.info("")
    
    logger.info(f"⚠️  PROBABLE LEAKS: {len(results['probable_leaks'])} features")
    logger.info("   These should be reviewed manually:")
    for feature in sorted(results['probable_leaks'])[:10]:
        logger.info(f"     ⚠️  {feature}")
    if len(results['probable_leaks']) > 10:
        logger.info(f"     ... and {len(results['probable_leaks']) - 10} more")
    logger.info("")
    
    logger.info(f"✅ SAFE FEATURES: {len(results['safe_features'])} features")
    logger.info("   These are calculated from past data only:")
    for feature in sorted(results['safe_features'])[:20]:
        logger.info(f"     ✅ {feature}")
    if len(results['safe_features']) > 20:
        logger.info(f"     ... and {len(results['safe_features']) - 20} more")
    logger.info("")
    
    # Summary
    leak_pct = (len(results['definite_leaks']) / results['total_features']) * 100
    logger.info("="*80)
    logger.info(f"SUMMARY: {leak_pct:.1f}% of features are definite leaks!")
    logger.info("="*80)
    logger.info("")
    
    logger.info("🔧 RECOMMENDED ACTION:")
    logger.info("")
    logger.info("1. Remove leaking features:")
    logger.info("   - Create filtered dataset with only safe features")
    logger.info("   - Re-run baseline validation with clean features")
    logger.info("")
    logger.info("2. Compare results:")
    logger.info("   - R² will drop (this is GOOD - it's now honest)")
    logger.info("   - Lower R² with clean data > High R² with leakage")
    logger.info("")
    logger.info("3. Use multi-model system to find best SAFE features:")
    logger.info("   - Run: python scripts/multi_model_feature_selection.py")
    logger.info("   - Only include safe features")
    logger.info("")
    
    # Save safe features list
    if args.save_safe_list:
        output_file = _REPO_ROOT / "config" / "safe_features_list.txt"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            for feature in sorted(results['safe_features']):
                f.write(f"{feature}\n")
        logger.info(f"✅ Saved safe features to: {output_file}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

