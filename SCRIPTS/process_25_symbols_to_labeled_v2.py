#!/usr/bin/env python3
"""
Process 25 symbols with reworked barrier targets and output to data/data_labeled_v2/interval=5m/
Uses the actual barrier processing code from dep/sort_py/barrier_targets.py
"""

import sys
from pathlib import Path
import pandas as pd
import polars as pl
import json
from datetime import datetime
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import barrier processing from dep/sort_py
sys.path.insert(0, str(PROJECT_ROOT / "dep" / "sort_py"))
from barrier_targets import add_barrier_targets_to_dataframe

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 25 symbols for testing
SYMBOLS_25 = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "JNJ",
    "WMT", "PG", "MA", "UNH", "HD", "DIS", "BAC", "ADBE", "PYPL", "NFLX",
    "CMCSA", "PFE", "KO", "AVGO", "COST"
]

def process_symbol(symbol: str, input_dir: Path, output_dir: Path, interval: str = "5m") -> bool:
    """Process a single symbol: load, add barrier targets, save."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {symbol}")
    logger.info(f"{'='*60}")
    
    # Input file path (handle symbol=SYMBOL structure)
    input_symbol_dir = input_dir / f"symbol={symbol}"
    if not input_symbol_dir.exists():
        logger.warning(f"⚠️  Skipping {symbol}: Directory not found at {input_symbol_dir}")
        return False
    
    # Find parquet file
    parquet_files = list(input_symbol_dir.glob("*.parquet"))
    if not parquet_files:
        logger.warning(f"⚠️  Skipping {symbol}: No parquet files found in {input_symbol_dir}")
        return False
    
    input_file = parquet_files[0]  # Use first parquet file found
    
    # Load raw data
    logger.info(f"📂 Loading: {input_file}")
    try:
        # Try polars first (faster), fallback to pandas
        try:
            df = pl.read_parquet(input_file)
            df = df.to_pandas()
        except:
            df = pd.read_parquet(input_file)
        
        logger.info(f"   Loaded {len(df):,} rows, {len(df.columns)} columns")
    except Exception as e:
        logger.error(f"❌ Error loading {symbol}: {e}")
        return False
    
    # Check for required columns
    required_cols = ['close', 'ts'] if 'ts' in df.columns else ['close']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error(f"❌ Missing required columns for {symbol}: {missing}")
        return False
    
    # Generate barrier targets (with reworked barrier logic)
    logger.info(f"🎯 Generating barrier targets...")
    try:
        # Use the actual barrier processing function
        # Horizons: 5, 10, 15, 30, 60 minutes
        # Barrier sizes: 0.3, 0.5, 0.8 (k * sigma)
        df_with_targets = add_barrier_targets_to_dataframe(
            df,
            price_col='close',
            horizons=[5, 10, 15, 30, 60],  # Multiple horizons
            barrier_sizes=[0.3, 0.5, 0.8],  # Multiple barrier sizes
            vol_window=20
        )
        
        target_cols = [c for c in df_with_targets.columns if c.startswith('y_') or 'will_' in c.lower() or 'barrier' in c.lower()]
        logger.info(f"   ✅ Added {len(target_cols)} target columns")
        logger.info(f"   Total columns: {len(df_with_targets.columns)}")
    except Exception as e:
        logger.error(f"❌ Error generating targets for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Save labeled data to correct structure
    output_symbol_dir = output_dir / f"symbol={symbol}"
    output_symbol_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_symbol_dir / f"{symbol}.parquet"
    
    logger.info(f"💾 Saving to: {output_file}")
    try:
        df_with_targets.to_parquet(output_file, index=False)
        file_size_mb = output_file.stat().st_size / 1024 / 1024
        logger.info(f"   ✅ Successfully processed {symbol}: {len(df_with_targets):,} rows, {len(df_with_targets.columns)} columns, {file_size_mb:.1f} MB")
        return True
    except Exception as e:
        logger.error(f"❌ Error saving {symbol}: {e}")
        return False

def update_metadata(output_dir: Path, symbols: list):
    """Update metadata.json with processed symbols."""
    metadata_file = output_dir.parent / "metadata.json"
    
    metadata = {
        "barrier_version": 2,
        "horizon_units": "minutes",
        "interval_minutes": 5.0,
        "generation_date": datetime.now().isoformat(),
        "symbols": sorted(symbols),
        "horizons": [5, 10, 15, 30, 60],
        "barrier_sizes": [0.3, 0.5, 0.8],
        "fixes": {
            "horizon_unit_bug": "Fixed horizon_minutes being used as horizon_bars in target computation",
            "description": "Horizon is now correctly converted from minutes to bars using interval_minutes"
        }
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"\n✅ Updated metadata: {metadata_file}")

def main():
    """Main processing function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process 25 symbols with barrier targets to data/data_labeled_v2/interval=5m/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all 25 default symbols from data/data_labeled/interval=5m/
  python SCRIPTS/process_25_symbols_to_labeled_v2.py
  
  # Process specific symbols
  python SCRIPTS/process_25_symbols_to_labeled_v2.py --symbols AAPL MSFT GOOGL
  
  # Custom input directory
  python SCRIPTS/process_25_symbols_to_labeled_v2.py \\
      --input-dir data/data_labeled/interval=5m \\
      --output-dir data/data_labeled_v2/interval=5m
        """
    )
    parser.add_argument("--input-dir", type=Path, 
                       default=PROJECT_ROOT / "data" / "data_labeled" / "interval=5m",
                       help="Input directory with raw data (default: data/data_labeled/interval=5m)")
    parser.add_argument("--output-dir", type=Path, 
                       default=PROJECT_ROOT / "data" / "data_labeled_v2" / "interval=5m",
                       help="Output directory for labeled data (default: data/data_labeled_v2/interval=5m)")
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS_25,
                       help=f"List of symbols to process (default: {len(SYMBOLS_25)} common symbols)")
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("Processing Symbols with Barrier Targets")
    logger.info("="*60)
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Symbols to process: {len(args.symbols)}")
    logger.info(f"Symbols: {', '.join(args.symbols[:10])}{'...' if len(args.symbols) > 10 else ''}")
    logger.info("="*60)
    logger.info("\nThis script:")
    logger.info("  - Reads from: {input_dir}/symbol={{SYMBOL}}/{{SYMBOL}}.parquet")
    logger.info("  - Adds barrier targets: horizons=[5,10,15,30,60]m, sizes=[0.3,0.5,0.8]")
    logger.info("  - Outputs to: {output_dir}/symbol={{SYMBOL}}/{{SYMBOL}}.parquet")
    logger.info("="*60)
    
    # Check if input directory exists
    if not args.input_dir.exists():
        logger.error(f"\n❌ Input directory not found: {args.input_dir}")
        logger.error(f"   Please ensure raw data files exist in: {args.input_dir}/symbol={{SYMBOL}}/")
        sys.exit(1)
    
    # Process each symbol
    success_count = 0
    successful_symbols = []
    
    for symbol in args.symbols:
        if process_symbol(symbol, args.input_dir, args.output_dir):
            success_count += 1
            successful_symbols.append(symbol)
    
    # Update metadata
    if successful_symbols:
        update_metadata(args.output_dir, successful_symbols)
    
    logger.info("\n" + "="*60)
    logger.info(f"Processing complete: {success_count}/{len(args.symbols)} symbols successful")
    logger.info("="*60)
    
    if success_count < len(args.symbols):
        failed = set(args.symbols) - set(successful_symbols)
        logger.warning(f"\n⚠️  Failed symbols ({len(failed)}): {', '.join(sorted(failed))}")
        sys.exit(1)
    else:
        logger.info("\n✅ All symbols processed successfully!")
        logger.info(f"\n📁 Output location: {args.output_dir}")
        logger.info(f"📊 Processed {len(successful_symbols)} symbols")
        sys.exit(0)

if __name__ == "__main__":
    main()
