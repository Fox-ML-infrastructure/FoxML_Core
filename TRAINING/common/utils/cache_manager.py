# MIT License - see LICENSE file

"""
Cache Manager Utilities

Generic cache management utilities for loading and saving cached results.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from TRAINING.common.utils.file_utils import write_atomic_json, read_atomic_json

logger = logging.getLogger(__name__)


def build_cache_key(*parts: str, separator: str = "_") -> str:
    """
    Build cache key from parts.
    
    Args:
        *parts: Key parts to join
        separator: Separator string (default: "_")
    
    Returns:
        Cache key string
    """
    return separator.join(str(part) for part in parts if part is not None)


def build_cache_key_with_symbol(target: str, config_hash: str, view: str, symbol: Optional[str] = None) -> str:
    """
    Build cache key for feature selection or similar operations.
    
    Handles SYMBOL_SPECIFIC view by appending symbol to key.
    
    Args:
        target: Target column name
        config_hash: Configuration hash
        view: View type (CROSS_SECTIONAL, SYMBOL_SPECIFIC, etc.)
        symbol: Optional symbol name (required for SYMBOL_SPECIFIC view)
    
    Returns:
        Cache key string
    """
    cache_key_parts = [target, config_hash, view]
    if view == "SYMBOL_SPECIFIC" and symbol:
        cache_key_parts.append(f"symbol={symbol}")
    return build_cache_key(*cache_key_parts)


def load_cache(
    cache_path: Path,
    config_hash: Optional[str] = None,
    verify_hash: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Load cache from file with optional hash verification.
    
    Args:
        cache_path: Path to cache file
        config_hash: Optional config hash to verify
        verify_hash: If True, verify config_hash matches (if provided)
    
    Returns:
        Cache data dictionary, or None if cache doesn't exist or verification fails
    """
    if not cache_path.exists():
        return None
    
    try:
        cache_data = read_atomic_json(cache_path)
        if cache_data is None:
            return None
        
        # Verify config hash if provided
        if verify_hash and config_hash is not None:
            cached_hash = cache_data.get('config_hash')
            if cached_hash != config_hash:
                logger.debug(f"Cache config hash mismatch (expected: {config_hash[:8]}, got: {cached_hash[:8] if cached_hash else None})")
                return None
        
        return cache_data
    except Exception as e:
        logger.warning(f"Failed to load cache from {cache_path}: {e}")
        return None


def save_cache(
    cache_path: Path,
    data: Dict[str, Any],
    config_hash: Optional[str] = None,
    include_timestamp: bool = True
) -> bool:
    """
    Save cache to file atomically.
    
    Args:
        cache_path: Path to cache file
        data: Data to cache
        config_hash: Optional config hash to include
        include_timestamp: If True, add timestamp to cache data
    
    Returns:
        True if save succeeded, False otherwise
    """
    try:
        cache_dir = cache_path.parent
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare cache data
        cache_data = data.copy()
        if config_hash is not None:
            cache_data['config_hash'] = config_hash
        if include_timestamp:
            cache_data['timestamp'] = time.time()
        
        # Save atomically
        write_atomic_json(cache_path, cache_data, default=str)
        logger.debug(f"💾 Saved cache: {cache_path}")
        return True
    except Exception as e:
        logger.warning(f"Failed to save cache to {cache_path}: {e}")
        return False


def get_cache_path(base_dir: Path, cache_type: str, cache_key: str, extension: str = ".json") -> Path:
    """
    Get cache file path.
    
    Args:
        base_dir: Base directory for cache
        cache_type: Cache type subdirectory (e.g., "feature_selection", "target_ranking")
        cache_key: Cache key (filename without extension)
        extension: File extension (default: ".json")
    
    Returns:
        Cache file path
    """
    cache_dir = base_dir / "cache" / cache_type
    return cache_dir / f"{cache_key}{extension}"


def invalidate_cache(cache_path: Path) -> bool:
    """
    Invalidate cache by deleting file.
    
    Args:
        cache_path: Path to cache file
    
    Returns:
        True if deletion succeeded, False otherwise
    """
    try:
        if cache_path.exists():
            cache_path.unlink()
            logger.debug(f"🗑️  Invalidated cache: {cache_path}")
            return True
        return False
    except Exception as e:
        logger.warning(f"Failed to invalidate cache {cache_path}: {e}")
        return False

