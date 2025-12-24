#!/usr/bin/env python3
"""
Process 25 symbols with reworked barrier targets and output to data/data_labeled_v3/interval=5m/
Uses DATA_PROCESSING pipeline with correct data layout structure
"""

import sys
from pathlib import Path
import json
from datetime import datetime
import logging
import subprocess

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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

def main():
    """Main processing function using generate_versioned_labels.py."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process 25 symbols with barrier targets to data/data_labeled_v3/interval=5m/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all 25 default symbols from data/data_labeled/interval=5m/
  python SCRIPTS/process_25_symbols_to_v3.py
  
  # Process specific symbols
  python SCRIPTS/process_25_symbols_to_v3.py --symbols AAPL MSFT GOOGL
  
  # Custom input directory
  python SCRIPTS/process_25_symbols_to_v3.py \\
      --input-dir data/data_labeled \\
      --output-dir data/data_labeled_v3
        """
    )
    parser.add_argument("--input-dir", type=Path, 
                       default=PROJECT_ROOT / "data" / "data_labeled",
                       help="Input data directory (default: data/data_labeled - will look for interval=5m/ inside)")
    parser.add_argument("--output-dir", type=Path, 
                       default=PROJECT_ROOT / "data" / "data_labeled_v3",
                       help="Output directory for labeled data (default: data/data_labeled_v3 - creates interval=5m/ inside)")
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS_25,
                       help=f"List of symbols to process (default: {len(SYMBOLS_25)} common symbols)")
    parser.add_argument("--horizons", nargs="+", type=int, default=[5, 10, 15, 30, 60],
                       help="Horizons in minutes (default: 5 10 15 30 60)")
    parser.add_argument("--barrier-sizes", nargs="+", type=float, default=[0.3, 0.5, 0.8],
                       help="Barrier sizes (default: 0.3 0.5 0.8)")
    parser.add_argument("--interval-minutes", type=float, default=5.0,
                       help="Bar interval in minutes (default: 5.0)")
    parser.add_argument("--n-workers", type=int, default=4,
                       help="Number of parallel workers (default: 4 to limit CPU/temp)")
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("Processing 25 Symbols to data/data_labeled_v3/interval=5m/")
    logger.info("="*60)
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Symbols to process: {len(args.symbols)}")
    logger.info(f"Symbols: {', '.join(args.symbols[:10])}{'...' if len(args.symbols) > 10 else ''}")
    logger.info(f"Horizons: {args.horizons}")
    logger.info(f"Barrier sizes: {args.barrier_sizes}")
    logger.info(f"Interval: {args.interval_minutes} minutes")
    logger.info(f"Workers: {args.n_workers} (limited to prevent CPU temp spikes)")
    logger.info("="*60)
    
    # Use the existing generate_versioned_labels.py script
    script_path = PROJECT_ROOT / "DATA_PROCESSING" / "pipeline" / "generate_versioned_labels.py"
    
    if not script_path.exists():
        logger.error(f"❌ Script not found: {script_path}")
        logger.error("   Make sure DATA_PROCESSING files are copied correctly")
        sys.exit(1)
    
    # Build command
    cmd = [
        sys.executable,
        str(script_path),
        "--data-dir", str(args.input_dir),
        "--output-dir", str(args.output_dir),
        "--symbols"] + args.symbols + [
        "--horizons"] + [str(h) for h in args.horizons] + [
        "--barrier-sizes"] + [str(b) for b in args.barrier_sizes] + [
        "--interval-minutes", str(args.interval_minutes)
    ]
    
    # Always pass n-workers (defaults to 4 to limit CPU/temp)
    cmd.extend(["--n-workers", str(args.n_workers)])
    
    logger.info(f"\n🚀 Running: {' '.join(cmd)}\n")
    
    # Run the script
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            check=True
        )
        
        logger.info("\n" + "="*60)
        logger.info("✅ Processing complete!")
        logger.info(f"📁 Output location: {args.output_dir}/interval=5m/")
        logger.info(f"📊 Processed {len(args.symbols)} symbols")
        logger.info("="*60)
        
        sys.exit(0)
        
    except subprocess.CalledProcessError as e:
        logger.error(f"\n❌ Processing failed with exit code {e.returncode}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

