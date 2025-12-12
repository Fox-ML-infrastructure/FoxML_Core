#!/usr/bin/env python3

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
Generate Versioned Labeled Dataset

Creates a new versioned dataset with corrected barrier targets (horizon unit fix).
Includes metadata tracking for traceability.
"""

import os
import sys
import json
import logging
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd

# Add project root to path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Import directly from barrier module to avoid __init__.py import issues
from DATA_PROCESSING.targets.barrier import (
    add_barrier_targets_to_dataframe,
    add_zigzag_targets_to_dataframe,
    add_mfe_mdd_targets_to_dataframe,
    add_enhanced_targets_to_dataframe
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_git_commit_hash() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
            check=True
        )
        return result.stdout.strip()
    except Exception as e:
        logger.warning(f"Could not get git commit hash: {e}")
        return "unknown"


def ensure_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate-named columns (keep first) and log what we removed."""
    if not df.columns.is_unique:
        dupes = df.columns[df.columns.duplicated(keep='first')]
        logger.warning(f"De-duplicating {dupes.size} duplicate column name(s); "
                       f"examples: {list(dict.fromkeys(dupes))[:10]}")
        df = df.loc[:, ~df.columns.duplicated(keep='first')].copy()
    return df


def drop_existing_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove any pre-existing target columns."""
    target_prefixes = (
        'will_peak_', 'will_valley_', 'y_will_', 'y_first_touch', 'p_up_', 'p_down_', 
        'barrier_up_', 'barrier_down_', 'vol_at_t_',
        'zigzag_peak_', 'zigzag_valley_', 'y_will_swing_',
        'mfe_', 'mdd_', 'max_return_', 'min_return_',
        'tth_', 'tth_abs_', 'hit_direction_', 'hit_asym_', 'tth_asym_',
        'ret_ord_', 'ret_zscore_', 'mfe_share_', 'time_in_profit_', 'flipcount_',
    )
    to_drop = [c for c in df.columns if any(c.startswith(p) for p in target_prefixes)]
    if to_drop:
        logger.info(f"Removing {len(to_drop)} pre-existing target cols before recompute")
        df = df.drop(columns=to_drop, errors='ignore')
    return df


def read_parquet_with_fallback(path: Path) -> pd.DataFrame:
    """Try fastparquet first, then pyarrow if available."""
    try:
        return pd.read_parquet(path, engine='fastparquet')
    except Exception as e_fast:
        logger.warning(f"fastparquet failed on {path.name}: {e_fast}. Trying pyarrow...")
        try:
            return pd.read_parquet(path, engine='pyarrow')
        except Exception as e_arrow:
            raise RuntimeError(f"Failed to read {path} with both engines. "
                               f"fastparquet: {e_fast}; pyarrow: {e_arrow}")


def process_symbol(
    symbol: str,
    input_dir: Path,
    output_dir: Path,
    horizons: List[int],
    barrier_sizes: List[float],
    interval_minutes: float = 5.0
) -> Dict[str, any]:
    """Process a single symbol and generate versioned labeled data."""
    try:
        parquet_files = list(input_dir.glob("*.parquet"))
        
        if not parquet_files:
            return {"symbol": symbol, "status": "error", "message": "No parquet files found"}
        
        processed_files = 0
        total_rows = 0
        
        for parquet_file in parquet_files:
            try:
                # Load data
                df = read_parquet_with_fallback(parquet_file)
                
                # Ensure column uniqueness
                df = ensure_unique_columns(df)
                
                # Drop existing targets
                df = drop_existing_target_columns(df)
                
                # Find price column
                price_col = 'close'
                if price_col not in df.columns:
                    for alt_col in ['vwap', 'mid', 'last']:
                        if alt_col in df.columns:
                            price_col = alt_col
                            break
                    else:
                        logger.warning(f"No suitable price column found in {parquet_file.name}")
                        continue
                
                # Add barrier targets (with interval_minutes for correct horizon conversion)
                df = add_barrier_targets_to_dataframe(
                    df, 
                    price_col=price_col,
                    horizons=horizons,
                    barrier_sizes=barrier_sizes,
                    interval_minutes=interval_minutes  # CRITICAL: Pass interval for correct conversion
                )
                
                # Add ZigZag targets
                df = add_zigzag_targets_to_dataframe(
                    df,
                    price_col=price_col,
                    horizons=horizons,
                    reversal_pcts=[0.05, 0.1, 0.2],
                    interval_minutes=interval_minutes  # CRITICAL: Pass interval
                )
                
                # Add MFE/MDD targets
                df = add_mfe_mdd_targets_to_dataframe(
                    df,
                    price_col=price_col,
                    horizons=horizons,
                    thresholds=[0.001, 0.002, 0.005],
                    interval_minutes=interval_minutes  # CRITICAL: Pass interval
                )
                
                # Add enhanced targets
                df = add_enhanced_targets_to_dataframe(
                    df,
                    price_col=price_col,
                    horizons=horizons,
                    barrier_sizes=barrier_sizes,
                    tp_sl_ratios=[(1.0, 0.5), (1.5, 0.75), (2.0, 1.0)],
                    interval_minutes=interval_minutes  # CRITICAL: Pass interval
                )
                
                # Ensure uniqueness again before write
                df = ensure_unique_columns(df)
                
                # Save to versioned output directory
                interval_output_dir = output_dir / "interval=5m"
                symbol_output_dir = interval_output_dir / f"symbol={symbol}"
                symbol_output_dir.mkdir(parents=True, exist_ok=True)
                
                output_file = symbol_output_dir / parquet_file.name
                df.to_parquet(output_file, index=False, compression='snappy')
                
                processed_files += 1
                total_rows += len(df)
                
                logger.info(f"  ✅ {symbol}: Processed {parquet_file.name} ({len(df)} rows)")
                
            except Exception as e:
                logger.error(f"  ❌ Error processing {parquet_file.name} for {symbol}: {e}")
                continue
        
        if processed_files == 0:
            return {"symbol": symbol, "status": "error", "message": "No files processed successfully"}
        
        return {
            "symbol": symbol,
            "status": "success",
            "files_processed": processed_files,
            "rows_processed": total_rows
        }
        
    except Exception as e:
        return {"symbol": symbol, "status": "error", "message": str(e)}


def create_metadata_file(
    output_dir: Path,
    symbols: List[str],
    horizons: List[int],
    barrier_sizes: List[float],
    interval_minutes: float,
    commit_hash: str
) -> None:
    """Create metadata file with version information."""
    metadata = {
        "barrier_version": 2,
        "horizon_units": "minutes",
        "interval_minutes": interval_minutes,
        "commit_hash": commit_hash,
        "generation_date": datetime.now().isoformat(),
        "symbols": symbols,
        "horizons": horizons,
        "barrier_sizes": barrier_sizes,
        "fixes": {
            "horizon_unit_bug": "Fixed horizon_minutes being used as horizon_bars in target computation",
            "commit": commit_hash,
            "description": "Horizon is now correctly converted from minutes to bars using interval_minutes"
        }
    }
    
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"📝 Created metadata file: {metadata_file}")
    logger.info(f"   Barrier version: {metadata['barrier_version']}")
    logger.info(f"   Commit hash: {commit_hash}")
    logger.info(f"   Interval: {interval_minutes}m")


def main():
    parser = argparse.ArgumentParser(description="Generate versioned labeled dataset with corrected barrier targets")
    
    # Try to load default data_dir from config
    default_data_dir = None
    try:
        from CONFIG.config_loader import get_cfg
        # Try multiple config paths (in order of preference)
        default_data_dir = get_cfg("system.paths.data_dir", default=None, config_name="system_config")
        if default_data_dir is None:
            default_data_dir = get_cfg("pipeline.paths.data_dir", default=None, config_name="pipeline_config")
        if default_data_dir:
            logger.info(f"📋 Loaded default data_dir from config: {default_data_dir}")
    except Exception as e:
        logger.debug(f"Could not load data_dir from config: {e}")
    
    parser.add_argument("--data-dir", 
                       default=default_data_dir,
                       required=default_data_dir is None,
                       help=f"Input data directory (default from config: {default_data_dir or 'not set'})")
    parser.add_argument("--output-dir", required=True, help="Output directory (e.g., data/data_labeled_v2)")
    parser.add_argument("--symbols", nargs="+", required=True, help="List of symbols to process (e.g., NVDA AAPL MSFT GOOGL TSLA)")
    parser.add_argument("--horizons", nargs="+", type=int, default=[5, 10, 15, 30, 60],
                       help="Horizons to process (default: 5 10 15 30 60)")
    parser.add_argument("--barrier-sizes", nargs="+", type=float, default=[0.3, 0.5, 0.8],
                       help="Barrier sizes (default: 0.3 0.5 0.8)")
    parser.add_argument("--interval-minutes", type=float, default=5.0,
                       help="Bar interval in minutes (default: 5.0)")
    
    args = parser.parse_args()
    
    # Validate data_dir was provided (either from config or command line)
    if not args.data_dir:
        parser.error("--data-dir is required (not found in config and not provided via command line)")
    
    # Get git commit hash
    commit_hash = get_git_commit_hash()
    logger.info(f"🔖 Git commit hash: {commit_hash}")
    
    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate input directory structure
    interval_dir = data_dir / "interval=5m"
    if not interval_dir.exists():
        logger.error(f"Input directory structure not found: {interval_dir}")
        logger.error("Expected: data_dir/interval=5m/symbol=SYMBOL/*.parquet")
        return
    
    logger.info(f"📂 Input directory: {data_dir}")
    logger.info(f"📂 Output directory: {output_dir}")
    logger.info(f"📊 Symbols to process: {args.symbols}")
    logger.info(f"⏱️  Horizons: {args.horizons} minutes")
    logger.info(f"📏 Barrier sizes: {args.barrier_sizes}")
    logger.info(f"🕐 Interval: {args.interval_minutes} minutes")
    
    # Process each symbol
    results = []
    for symbol in args.symbols:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {symbol}...")
        logger.info(f"{'='*60}")
        
        # Find symbol directory
        symbol_input_dir = interval_dir / f"symbol={symbol}"
        if not symbol_input_dir.exists():
            logger.warning(f"  ⚠️  Symbol directory not found: {symbol_input_dir}")
            results.append({"symbol": symbol, "status": "error", "message": "Symbol directory not found"})
            continue
        
        # Process symbol
        result = process_symbol(
            symbol=symbol,
            input_dir=symbol_input_dir,
            output_dir=output_dir,
            horizons=args.horizons,
            barrier_sizes=args.barrier_sizes,
            interval_minutes=args.interval_minutes
        )
        results.append(result)
        
        if result["status"] == "success":
            logger.info(f"  ✅ {symbol}: {result['files_processed']} files, {result['rows_processed']} rows")
        else:
            logger.error(f"  ❌ {symbol}: {result['message']}")
    
    # Create metadata file
    successful_symbols = [r["symbol"] for r in results if r["status"] == "success"]
    if successful_symbols:
        create_metadata_file(
            output_dir=output_dir,
            symbols=successful_symbols,
            horizons=args.horizons,
            barrier_sizes=args.barrier_sizes,
            interval_minutes=args.interval_minutes,
            commit_hash=commit_hash
        )
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("🏁 Processing Summary")
    logger.info(f"{'='*60}")
    successful = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - successful
    logger.info(f"✅ Successful: {successful}")
    logger.info(f"❌ Failed: {failed}")
    
    if failed > 0:
        failed_symbols = [r["symbol"] for r in results if r["status"] == "error"]
        logger.warning(f"Failed symbols: {failed_symbols}")


if __name__ == "__main__":
    main()
