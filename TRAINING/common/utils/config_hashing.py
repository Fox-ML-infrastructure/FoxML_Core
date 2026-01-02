# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

"""
Config Hashing Utilities

Standardized config hashing for cache keys and reproducibility tracking.
Uses SHA256 of canonical JSON representation for consistency.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def compute_config_hash(config: Dict[str, Any], sort_keys: bool = True) -> str:
    """
    Compute deterministic hash of configuration dictionary.
    
    Uses SHA256 of canonical JSON representation for consistency across
    different Python versions and environments.
    
    Args:
        config: Configuration dictionary to hash
        sort_keys: If True, sort dictionary keys for deterministic output
    
    Returns:
        Hexadecimal hash string (64 characters)
    """
    # Convert to canonical JSON (sorted keys, no whitespace)
    json_str = json.dumps(config, sort_keys=sort_keys, separators=(',', ':'))
    
    # Compute SHA256 hash
    return hashlib.sha256(json_str.encode('utf-8')).hexdigest()


def compute_config_hash_from_values(**kwargs) -> str:
    """
    Compute hash from key-value pairs (convenience function).
    
    Args:
        **kwargs: Key-value pairs to include in hash
    
    Returns:
        Hexadecimal hash string
    """
    return compute_config_hash(kwargs)


def compute_config_hash_from_list(items: List[Any]) -> str:
    """
    Compute hash from list of items.
    
    Args:
        items: List of items to hash
    
    Returns:
        Hexadecimal hash string
    """
    # Convert list to dict with indices as keys for consistent hashing
    config = {str(i): item for i, item in enumerate(items)}
    return compute_config_hash(config)


def compute_string_hash(value: str) -> str:
    """
    Compute hash of a string value (for simple cases).
    
    Args:
        value: String to hash
    
    Returns:
        Hexadecimal hash string
    """
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def compute_config_hash_from_file(
    config_path: Path,
    short: bool = True
) -> str:
    """
    Compute hash from config file contents.
    
    SST (Single Source of Truth) for file-based config hashing.
    
    Args:
        config_path: Path to config file (YAML, JSON, etc.)
        short: If True, return short hash (8 chars). If False, return full hash.
    
    Returns:
        Hexadecimal hash string, or "unknown" if file cannot be read
    """
    try:
        with open(config_path, "rb") as f:
            content = f.read()
        full_hash = hashlib.sha256(content).hexdigest()
        return full_hash[:8] if short else full_hash
    except Exception:
        return "unknown"

