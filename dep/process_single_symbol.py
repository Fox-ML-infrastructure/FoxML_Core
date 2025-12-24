#!/usr/bin/env python3

# MIT License - see LICENSE file

"""
Process Single Symbol
Run complete data processing pipeline for one symbol.
"""


import sys
import argparse
from pathlib import Path
import polars as pl
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from DATA_PROCESSING.pipeline import normalize_interval
from DATA_PROCESSING.targets import compute_barrier_targets
from DATA_PROCESSING.utils import MemoryManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_symbol(symbol: str, verbose: bool = False):
    """Process single symbol through complete pipeline"""
    
    if verbose:
        logger.setLevel(logging.DEBUG)
    
    mem = MemoryManager()
    mem.check_memory("Start")
    
    # Paths
    input_file = PROJECT_ROOT / f"data/data_labeled/interval=5m/{symbol}.parquet"
    output_file = PROJECT_ROOT / f"DATA_PROCESSING/data/labeled/{symbol}_labeled.parquet"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"{'='*80}")
    logger.info(f"Processing {symbol}")
    logger.info(f"{'='*80}")
    
    # 1. Load data
    logger.info(f"[1/4] Loading data from {input_file}")
    if not input_file.exists():
        logger.error(f"❌ File not found: {input_file}")
        return False
    
    try:
        df = pl.read_parquet(input_file)
        logger.info(f"  ✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
    except Exception as e:
        logger.error(f"❌ Failed to load data: {e}")
        return False
    
    mem.check_memory("After load")
    
    # 2. Normalize session
    logger.info(f"[2/4] Normalizing session (RTH, grid-aligned)")
    try:
        df_clean = normalize_interval(df, interval="5m")
        logger.info(f"  ✅ Normalized to {len(df_clean):,} rows")
    except Exception as e:
        logger.error(f"❌ Failed to normalize: {e}")
        return False
    
    # 3. Build features (simplified - just add basic returns and volatility)
    logger.info(f"[3/4] Building features")
    try:
        # Add basic features
        df_features = df_clean.with_columns([
            # Returns
            pl.col("close").pct_change(1).alias("ret_1m"),
            pl.col("close").pct_change(5).alias("ret_5m"),
            pl.col("close").pct_change(15).alias("ret_15m"),
            
            # Volatility
            pl.col("ret_1m").rolling_std(window_size=5).alias("vol_5m"),
            pl.col("ret_1m").rolling_std(window_size=15).alias("vol_15m"),
            
            # Volume
            (pl.col("close") * pl.col("volume")).alias("dollar_volume"),
        ])
        logger.info(f"  ✅ Added {len(df_features.columns) - len(df_clean.columns)} features")
    except Exception as e:
        logger.error(f"❌ Failed to build features: {e}")
        return False
    
    mem.check_memory("After features")
    
    # 4. Generate targets
    logger.info(f"[4/4] Generating barrier targets (15m horizon)")
    try:
        # Convert to pandas for target generation
        df_pd = df_features.to_pandas()
        
        # Compute barrier targets
        targets = compute_barrier_targets(
            prices=df_pd['close'],
            horizon_minutes=15,
            barrier_size=0.5,
            vol_window=20
        )
        
        # Add back to dataframe
        for col in targets.columns:
            df_features = df_features.with_columns(
                pl.Series(name=col, values=targets[col].values)
            )
        
        logger.info(f"  ✅ Added {len(targets.columns)} target columns")
    except Exception as e:
        logger.error(f"❌ Failed to generate targets: {e}")
        return False
    
    mem.check_memory("After targets")
    
    # 5. Clean and save
    logger.info(f"Cleaning and saving to {output_file}")
    try:
        # Drop warmup period with NaN
        df_final = df_features.drop_nulls()
        df_final.write_parquet(output_file)
        logger.info(f"  ✅ Saved {len(df_final):,} rows to {output_file}")
        
        # Summary
        feature_cols = [c for c in df_final.columns if not c.startswith('y_') and c not in ['ts', 'open', 'high', 'low', 'close', 'volume']]
        target_cols = [c for c in df_final.columns if c.startswith('y_')]
        
        logger.info(f"\n📊 Summary:")
        logger.info(f"  Features: {len(feature_cols)}")
        logger.info(f"  Targets: {len(target_cols)}")
        logger.info(f"  Total columns: {len(df_final.columns)}")
        logger.info(f"  Output size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
        
    except Exception as e:
        logger.error(f"❌ Failed to save: {e}")
        return False
    
    mem.check_memory("Complete")
    
    logger.info(f"\n✅ Successfully processed {symbol}!")
    return True

def main():
    parser = argparse.ArgumentParser(description="Process single symbol through pipeline")
    parser.add_argument("symbol", help="Symbol to process (e.g., AAPL)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    success = process_symbol(args.symbol, args.verbose)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

