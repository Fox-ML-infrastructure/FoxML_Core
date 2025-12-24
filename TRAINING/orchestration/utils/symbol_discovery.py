"""
Symbol Discovery Utility

Auto-discover symbols from data directory structure when symbols list is empty.
Supports batch selection with seeded randomization for reproducibility.
"""

import logging
import random
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def discover_symbols_from_data_dir(
    data_dir: Path,
    interval: Optional[str] = None
) -> Tuple[List[str], List[str]]:
    """
    Discover all available symbols from data directory structure.
    
    Supports:
    - New structure: data_dir/interval={interval}/symbol={symbol}/*.parquet
    - Legacy structure: data_dir/symbol={symbol}/*.parquet
    
    Args:
        data_dir: Directory containing symbol data
        interval: Optional interval string (e.g., "5m", "15m")
                  If data_dir already contains interval= in path, this is used
                  for validation only (interval mismatch raises error).
    
    Returns:
        Tuple of (symbols, search_paths):
        - symbols: Sorted list of valid symbol names
        - search_paths: List of paths that were searched (for error messages)
    
    Raises:
        ValueError: If data_dir contains interval=X but interval param is Y (mismatch)
    """
    data_path = Path(data_dir)
    search_paths = []
    
    # Path normalization: Check if data_dir already contains interval= in path
    path_interval = None
    for part in data_path.parts:
        if part.startswith("interval="):
            path_interval = part.split("=", 1)[1]
            break
    
    # Also check if the directory name itself is interval-scoped
    if path_interval is None and data_path.name.startswith("interval="):
        path_interval = data_path.name.split("=", 1)[1]
    
    # Interval mismatch guard
    if path_interval is not None and interval is not None:
        if path_interval != interval:
            raise ValueError(
                f"Interval mismatch: data_dir contains interval={path_interval} "
                f"but config specifies interval={interval}. "
                f"Use data_dir without interval= or set interval to match."
            )
        logger.debug(f"Interval matches: {interval}")
    
    # Determine normalized path for symbol search
    if path_interval is not None:
        # Already interval-scoped, use data_dir directly
        normalized_path = data_path
        logger.info(f"Using interval-scoped path: {data_path} (interval={path_interval})")
    elif interval is not None:
        # Append interval to path
        normalized_path = data_path / f"interval={interval}"
        logger.info(f"Appending interval={interval} to {data_path}")
    else:
        # No interval, try root-level symbol directories
        normalized_path = data_path
        logger.info(f"No interval in path, trying root-level symbol directories")
    
    symbols = []
    
    # Search order: Try new structure first, then fallback to legacy
    # 1. Try new structure: {normalized_path}/symbol=*/
    search_paths.append(str(normalized_path / "symbol=*"))
    logger.debug(f"Searching new structure: {normalized_path}/symbol=*/")
    
    symbol_dirs = list(normalized_path.glob("symbol=*"))
    logger.debug(f"Found {len(symbol_dirs)} symbol directories in new structure")
    
    # 2. Fallback to legacy if new structure yields no results
    if not symbol_dirs and normalized_path != data_path:
        search_paths.append(str(data_path / "symbol=*"))
        logger.debug(f"New structure empty, trying legacy: {data_path}/symbol=*/")
        symbol_dirs = list(data_path.glob("symbol=*"))
        logger.debug(f"Found {len(symbol_dirs)} symbol directories in legacy structure")
    
    # Extract and validate symbols
    valid_count = 0
    for symbol_dir in symbol_dirs:
        if not symbol_dir.is_dir():
            continue
        
        # Extract symbol name from directory name
        symbol = symbol_dir.name.split("=", 1)[1]
        
        # Parquet validation: check for any .parquet file
        # Try fast glob first, then rglob for deep trees
        has_parquet = any(symbol_dir.glob("*.parquet"))
        if not has_parquet:
            # Try recursive search for nested structures
            has_parquet = any(symbol_dir.rglob("*.parquet"))
        
        if has_parquet:
            symbols.append(symbol)
            valid_count += 1
        else:
            logger.debug(f"Skipping {symbol}: no .parquet files found under {symbol_dir}")
    
    total_count = len(symbol_dirs)
    logger.info(f"Validated {valid_count}/{total_count} symbols (found parquet files)")
    
    # Sort for deterministic behavior
    symbols = sorted(symbols)
    
    if symbols:
        preview = symbols[:5]
        suffix = "..." if len(symbols) > 5 else ""
        logger.info(
            f"Discovered {len(symbols)} valid symbols: {preview}{suffix} "
            f"(normalized path: {normalized_path})"
        )
    else:
        logger.warning(f"No valid symbols found in {normalized_path}")
    
    return symbols, search_paths


def select_symbol_batch(
    symbols: List[str],
    batch_size: Optional[int] = None,
    random_seed: Optional[int] = None
) -> List[str]:
    """
    Select a batch of symbols from the full list.
    
    If batch_size is None or >= len(symbols), returns all symbols.
    Otherwise, randomly samples batch_size symbols using a seeded RNG.
    
    Args:
        symbols: Full list of available symbols
        batch_size: Optional number of symbols to select
        random_seed: Optional seed for reproducible selection
    
    Returns:
        Sorted list of selected symbols
    """
    if not symbols:
        return []
    
    if batch_size is None or batch_size >= len(symbols):
        logger.info(f"Using all {len(symbols)} symbols (batch_size={batch_size})")
        return sorted(symbols)
    
    # Use local RNG to avoid global state mutation
    rng = random.Random(random_seed)
    selected = rng.sample(symbols, batch_size)
    
    logger.info(
        f"Selected {len(selected)}/{len(symbols)} symbols "
        f"(seed={random_seed})"
    )
    
    # Sort for consistent ordering
    return sorted(selected)

