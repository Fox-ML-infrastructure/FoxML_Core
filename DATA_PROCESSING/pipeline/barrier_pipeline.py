#!/usr/bin/env python3

"""
Copyright (c) 2025 Fox ML Infrastructure

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
Smart Barrier Processing Script

This script intelligently processes barrier targets by:
1. Checking which symbols already have barrier targets
2. Only processing symbols that need barrier targets
3. Resuming from where it left off
4. Providing progress tracking and status updates
"""


import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Set, List, Dict, Optional
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Add project root to path

from ml.barrier_targets import (
    add_barrier_targets_to_dataframe,
    add_zigzag_targets_to_dataframe,
    add_mfe_mdd_targets_to_dataframe,
    add_enhanced_targets_to_dataframe
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/smart_barrier_processing.log')
    ]
)
logger = logging.getLogger(__name__)

# Target column prefixes for idempotent re-runs
TARGET_PREFIXES = (
    # barrier-like
    'will_peak_', 'will_valley_', 'y_will_', 'y_first_touch', 'p_up_', 'p_down_', 
    'barrier_up_', 'barrier_down_', 'vol_at_t_',
    # zigzag/swing
    'zigzag_peak_', 'zigzag_valley_', 'y_will_swing_',
    # mfe/mdd families
    'mfe_', 'mdd_', 'max_return_', 'min_return_',
    # enhanced targets (TTH, ordinal, path quality, asymmetric)
    'tth_', 'tth_abs_', 'hit_direction_', 'hit_asym_', 'tth_asym_',
    'ret_ord_', 'ret_zscore_', 'mfe_share_', 'time_in_profit_', 'flipcount_',
)

def ensure_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate-named columns (keep first) and log what we removed."""
    if not df.columns.is_unique:
        dupes = df.columns[df.columns.duplicated(keep='first')]
        logger.warning(f"De-duplicating {dupes.size} duplicate column name(s); "
                       f"examples: {list(dict.fromkeys(dupes))[:10]}")
        df = df.loc[:, ~df.columns.duplicated(keep='first')].copy()
    return df

def drop_existing_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove any pre-existing target columns so downstream add_* functions are idempotent."""
    to_drop = [c for c in df.columns if any(c.startswith(p) for p in TARGET_PREFIXES)]
    if to_drop:
        logger.info(f"Removing {len(to_drop)} pre-existing target cols before recompute "
                    f"(e.g., {to_drop[:8]}...)")
        df = df.drop(columns=to_drop, errors='ignore')
    return df

def read_parquet_with_fallback(path: Path) -> pd.DataFrame:
    """Try fastparquet first (your current default), then pyarrow if available."""
    try:
        return pd.read_parquet(path, engine='fastparquet')
    except Exception as e_fast:
        logger.warning(f"fastparquet failed on {path.name}: {e_fast}. Trying pyarrow...")
        try:
            return pd.read_parquet(path, engine='pyarrow')
        except Exception as e_arrow:
            # Re-raise the fast error; the arrow error is usually similar/noisier.
            raise RuntimeError(f"Failed to read {path} with both engines. "
                               f"fastparquet: {e_fast}; pyarrow: {e_arrow}")

class SmartBarrierProcessor:
    """Smart barrier processing with resume capability."""
    
    def __init__(self, data_dir: str, output_dir: str, horizons: List[int], 
                 barrier_sizes: List[float], n_workers: int = 8, throttle_delay: float = 0.1, 
                 force: bool = False):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.horizons = horizons
        self.barrier_sizes = barrier_sizes
        self.n_workers = n_workers
        self.throttle_delay = throttle_delay  # Delay between operations to reduce CPU heat
        self.force = force  # Force reprocessing of all symbols
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Progress tracking
        self.progress_file = self.output_dir / "barrier_processing_progress.json"
        self.completed_symbols: Set[str] = set()
        self.failed_symbols: Set[str] = set()
        self.load_progress()
        
        # Statistics
        self.stats = {
            "total_symbols": 0,
            "completed_symbols": 0,
            "failed_symbols": 0,
            "skipped_symbols": 0,
            "start_time": time.time(),
            "last_update": time.time()
        }
    
    def load_progress(self):
        """Load progress from previous runs."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                    self.completed_symbols = set(data.get('completed_symbols', []))
                    self.failed_symbols = set(data.get('failed_symbols', []))
                logger.info(f"Loaded progress: {len(self.completed_symbols)} completed, {len(self.failed_symbols)} failed")
            except Exception as e:
                logger.warning(f"Could not load progress file: {e}")
    
    def save_progress(self):
        """Save current progress."""
        try:
            data = {
                'completed_symbols': list(self.completed_symbols),
                'failed_symbols': list(self.failed_symbols),
                'timestamp': time.time(),
                'stats': self.stats
            }
            with open(self.progress_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save progress: {e}")
    
    def check_symbol_has_targets(self, symbol: str) -> bool:
        """Check if symbol already has barrier targets."""
        # If force flag is set, always reprocess
        if self.force:
            return False
        
        # Check output directory structure (interval=5m/symbol=SYMBOL)
        symbol_output_dir = None
        for interval_dir in self.output_dir.glob("interval=*"):
            symbol_dir = interval_dir / f"symbol={symbol}"
            if symbol_dir.exists():
                symbol_output_dir = symbol_dir
                break
        
        if not symbol_output_dir:
            return False
        
        # Check for parquet files with barrier targets
        parquet_files = list(symbol_output_dir.glob("*.parquet"))
        if not parquet_files:
            return False
        
        # Check if any file has ALL expected target families (old + enhanced)
        for parquet_file in parquet_files:
            try:
                df = pd.read_parquet(parquet_file)
                
                # Check for basic barrier targets (old)
                basic_barrier_cols = [col for col in df.columns if any(
                    target in col for target in ['will_peak', 'will_valley', 'y_will_']
                )]
                
                # Check for enhanced targets (new)
                enhanced_cols = [col for col in df.columns if any(
                    target in col for target in ['tth_', 'ret_ord_', 'mfe_share_', 'hit_asym_']
                )]
                
                # Symbol is only "complete" if it has BOTH old and new targets
                if basic_barrier_cols and enhanced_cols:
                    logger.debug(f"Symbol {symbol} has {len(basic_barrier_cols)} basic + {len(enhanced_cols)} enhanced targets")
                    return True
                elif basic_barrier_cols and not enhanced_cols:
                    logger.info(f"Symbol {symbol} has OLD targets only - will reprocess to add enhanced targets")
                    return False
                    
            except Exception as e:
                logger.warning(f"Could not check {parquet_file}: {e}")
                continue
        
        return False
    
    def get_symbols_to_process(self) -> List[str]:
        """Get list of symbols that need barrier target processing."""
        all_symbols = []
        
        # Find all symbols in data directory (handle interval=5m/symbol=SYMBOL structure)
        for interval_dir in self.data_dir.glob("interval=*"):
            for symbol_dir in interval_dir.glob("symbol=*"):
                symbol = symbol_dir.name.split("=")[1]
                all_symbols.append(symbol)
        
        self.stats["total_symbols"] = len(all_symbols)
        
        # Filter out symbols that already have targets
        symbols_to_process = []
        for symbol in all_symbols:
            if symbol in self.completed_symbols:
                self.stats["skipped_symbols"] += 1
                continue
            
            if symbol in self.failed_symbols:
                # Retry failed symbols
                symbols_to_process.append(symbol)
                continue
            
            if self.check_symbol_has_targets(symbol):
                self.completed_symbols.add(symbol)
                self.stats["skipped_symbols"] += 1
                logger.info(f"Symbol {symbol} already has barrier targets, skipping")
            else:
                symbols_to_process.append(symbol)
        
        logger.info(f"Found {len(symbols_to_process)} symbols to process")
        logger.info(f"Skipping {self.stats['skipped_symbols']} symbols that already have targets")
        
        return symbols_to_process
    
    def process_symbol_file(self, symbol: str) -> Dict[str, any]:
        """Process a single symbol file with optimized operations."""
        start_time = time.time()
        
        try:
            # Find input file (handle interval=5m/symbol=SYMBOL structure)
            input_dir = None
            for interval_dir in self.data_dir.glob("interval=*"):
                symbol_dir = interval_dir / f"symbol={symbol}"
                if symbol_dir.exists():
                    input_dir = symbol_dir
                    break
            
            if not input_dir:
                return {"symbol": symbol, "status": "error", "message": f"Symbol directory not found for {symbol}"}
            
            parquet_files = list(input_dir.glob("*.parquet"))
            
            if not parquet_files:
                return {"symbol": symbol, "status": "error", "message": "No parquet files found"}
            
            # Process each parquet file for this symbol
            processed_files = 0
            for parquet_file in parquet_files:
                try:
                    # Load data with engine fallback
                    df = read_parquet_with_fallback(parquet_file)
                    
                    # 1) Immediately ensure column-name uniqueness from source
                    df = ensure_unique_columns(df)
                    
                    # 2) If the file already carries any target columns, drop them before recompute
                    df = drop_existing_target_columns(df)
                    
                    # Use 'close' as price column (matching optimized script)
                    price_col = 'close'
                    if price_col not in df.columns:
                        # Try alternative price columns
                        for alt_col in ['vwap', 'mid', 'last']:
                            if alt_col in df.columns:
                                price_col = alt_col
                                break
                        else:
                            logger.warning(f"No suitable price column found in {parquet_file.name}")
                            continue
                    
                    # Add barrier targets with optimized parameters
                    df = add_barrier_targets_to_dataframe(
                        df, 
                        price_col=price_col,
                        horizons=self.horizons,
                        barrier_sizes=self.barrier_sizes
                    )
                    
                    # Add ZigZag targets
                    df = add_zigzag_targets_to_dataframe(
                        df,
                        price_col=price_col,
                        horizons=self.horizons,
                        reversal_pcts=[0.05, 0.1, 0.2]  # Default reversal percentages
                    )
                    
                    # Add MFE/MDD targets
                    df = add_mfe_mdd_targets_to_dataframe(
                        df,
                        price_col=price_col,
                        horizons=self.horizons,
                        thresholds=[0.001, 0.002, 0.005]  # Default thresholds
                    )
                    
                    # Add enhanced targets (TTH, ordinal, path quality, asymmetric)
                    df = add_enhanced_targets_to_dataframe(
                        df,
                        price_col=price_col,
                        horizons=self.horizons,
                        barrier_sizes=self.barrier_sizes,
                        tp_sl_ratios=[(1.0, 0.5), (1.5, 0.75), (2.0, 1.0)]  # TP:SL ratios
                    )
                    
                    # 3) Sanity: enforce uniqueness again before write (defensive)
                    df = ensure_unique_columns(df)
                    
                    # Save to output directory with compression (matching optimized script)
                    # Create interval=5m/symbol=SYMBOL structure
                    interval_output_dir = self.output_dir / "interval=5m"
                    output_dir = interval_output_dir / f"symbol={symbol}"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    output_file = output_dir / parquet_file.name
                    df.to_parquet(output_file, index=False, compression='snappy')
                    processed_files += 1
                    
                except Exception as e:
                    logger.error(f"Error processing {parquet_file}: {e}")
                    continue
            
            if processed_files == 0:
                return {"symbol": symbol, "status": "error", "message": "No files processed successfully"}
            
            processing_time = time.time() - start_time
            return {
                "symbol": symbol, 
                "status": "success", 
                "files_processed": processed_files,
                "processing_time": processing_time,
                "rows_processed": len(df) if processed_files > 0 else 0
            }
            
        except Exception as e:
            return {"symbol": symbol, "status": "error", "message": str(e)}
    
    def process_symbols_parallel(self, symbols: List[str], batch_size: int = 20):
        """Process symbols in parallel with batching for memory management."""
        logger.info(f"Processing {len(symbols)} symbols with {self.n_workers} workers in batches of {batch_size}")
        
        # Process in batches to manage memory
        total_batches = (len(symbols) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(symbols))
            batch_symbols = symbols[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch_symbols)} symbols)")
            
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                # Submit batch tasks
                future_to_symbol = {
                    executor.submit(self.process_symbol_file, symbol): symbol 
                    for symbol in batch_symbols
                }
                
                # Process results as they complete
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    
                    try:
                        result = future.result()
                        
                        if result["status"] == "success":
                            self.completed_symbols.add(symbol)
                            self.stats["completed_symbols"] += 1
                            logger.info(f"✅ {symbol}: {result['files_processed']} files, {result['rows_processed']} rows in {result['processing_time']:.2f}s")
                        else:
                            self.failed_symbols.add(symbol)
                            self.stats["failed_symbols"] += 1
                            logger.error(f"❌ {symbol}: {result['message']}")
                        
                    except Exception as e:
                        self.failed_symbols.add(symbol)
                        self.stats["failed_symbols"] += 1
                        logger.error(f"❌ {symbol}: Exception - {e}")
                    
                    # Small delay to reduce CPU heat
                    time.sleep(self.throttle_delay)
            
            # Memory cleanup after each batch
            import gc
            gc.collect()
            
            # Save progress after each batch
            self.save_progress()
            self.print_progress()
            
            logger.info(f"Batch {batch_idx + 1} completed: {self.stats['completed_symbols']} successful, {self.stats['failed_symbols']} failed")
            
            # Longer delay between batches to let CPU cool down
            if batch_idx < total_batches - 1:  # Don't delay after last batch
                time.sleep(self.throttle_delay * 10)
                logger.info("💤 Cooling down CPU between batches...")
    
    def print_progress(self):
        """Print current progress."""
        elapsed = time.time() - self.stats["start_time"]
        completed = self.stats["completed_symbols"]
        failed = self.stats["failed_symbols"]
        skipped = self.stats["skipped_symbols"]
        total = self.stats["total_symbols"]
        
        if completed > 0:
            rate = completed / elapsed
            eta = (total - completed - skipped) / rate if rate > 0 else 0
        else:
            rate = 0
            eta = 0
        
        logger.info(f"📊 Progress: {completed + skipped}/{total} symbols "
                   f"({completed} completed, {failed} failed, {skipped} skipped) "
                   f"Rate: {rate:.2f} symbols/sec, ETA: {eta/60:.1f} min")
    
    def run(self, batch_size: int = 20):
        """Run the smart barrier processing."""
        logger.info("🚀 Starting Smart Barrier Processing")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Horizons: {self.horizons}")
        logger.info(f"Barrier sizes: {self.barrier_sizes}")
        logger.info(f"Workers: {self.n_workers}")
        
        # Get symbols to process
        symbols_to_process = self.get_symbols_to_process()
        
        if not symbols_to_process:
            logger.info("🎉 All symbols already have barrier targets!")
            return
        
        # Process symbols
        self.process_symbols_parallel(symbols_to_process, batch_size)
        
        # Final progress save
        self.save_progress()
        self.print_progress()
        
        # Summary
        logger.info("🏁 Processing complete!")
        logger.info(f"✅ Completed: {self.stats['completed_symbols']}")
        logger.info(f"❌ Failed: {self.stats['failed_symbols']}")
        logger.info(f"⏭️ Skipped: {self.stats['skipped_symbols']}")
        
        if self.stats["failed_symbols"] > 0:
            logger.warning(f"Failed symbols: {list(self.failed_symbols)}")
            logger.info("Run the script again to retry failed symbols")

def main():
    parser = argparse.ArgumentParser(description="Smart Barrier Processing")
    parser.add_argument("--data-dir", required=True, help="Input data directory")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--horizons", nargs="+", type=int, default=[5, 10, 15, 30, 60], 
                       help="Horizons to process")
    parser.add_argument("--barrier-sizes", nargs="+", type=float, default=[0.3, 0.5, 0.8],
                       help="Barrier sizes")
    parser.add_argument("--n-workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--batch-size", type=int, default=20, help="Process symbols in batches for memory management")
    parser.add_argument("--throttle-delay", type=float, default=0.2, help="Delay in seconds between operations to reduce CPU heat (default: 0.2)")
    parser.add_argument("--resume", action="store_true", help="Resume from previous run")
    parser.add_argument("--force", action="store_true", help="Force reprocessing of all symbols (ignore existing targets)")
    parser.add_argument("--clear-progress", action="store_true", help="Clear progress file and start fresh")
    
    args = parser.parse_args()
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Handle clear-progress flag
    if args.clear_progress:
        output_path = Path(args.output_dir)
        progress_file = output_path / "barrier_processing_progress.json"
        if progress_file.exists():
            progress_file.unlink()
            logger.info(f"🗑️  Cleared progress file: {progress_file}")
        else:
            logger.info("No progress file to clear")
    
    # Initialize processor
    processor = SmartBarrierProcessor(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        horizons=args.horizons,
        barrier_sizes=args.barrier_sizes,
        n_workers=args.n_workers,
        throttle_delay=args.throttle_delay,
        force=args.force
    )
    
    # Run processing
    processor.run(args.batch_size)

if __name__ == "__main__":
    main()
