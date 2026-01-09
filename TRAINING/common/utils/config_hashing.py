# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Config Hashing Utilities

Standardized config hashing for cache keys and reproducibility tracking.
Uses SHA256 of canonical JSON representation for consistency.

This module is the SINGLE SOURCE OF TRUTH for canonicalization.
All fingerprinting code should import from here.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# =============================================================================
# CANONICALIZATION SST (Single Source of Truth)
# =============================================================================

def canonicalize(obj: Any) -> Any:
    """
    Recursively canonicalize object for deterministic JSON serialization.
    
    CRITICAL RULES:
    - Fails on unknown types (no str() fallback)
    - Drops None values AFTER canonicalization (not before)
    - Sorts sets by canonical_json of each element (handles non-orderable)
    - Detects numpy via module check (not hasattr)
    
    Args:
        obj: Object to canonicalize
        
    Returns:
        Canonicalized object suitable for JSON serialization
        
    Raises:
        TypeError: If object type is not supported
    """
    if obj is None:
        return None
    
    # Bool must be checked before int (bool is subclass of int in Python)
    if isinstance(obj, bool):
        return obj
    
    if isinstance(obj, int):
        return obj
    
    if isinstance(obj, float):
        # Handle special float values deterministically
        if obj != obj:  # NaN check (NaN != NaN)
            return "__NaN__"
        if obj == float('inf'):
            return "__Infinity__"
        if obj == float('-inf'):
            return "__NegInfinity__"
        # Normalize precision to avoid floating point representation differences
        return round(obj, 10)
    
    if isinstance(obj, str):
        return obj
    
    if isinstance(obj, (list, tuple)):
        return [canonicalize(x) for x in obj]
    
    if isinstance(obj, dict):
        # CRITICAL: Drop None AFTER canonicalization, not before
        out = {}
        for k, v in sorted(obj.items()):
            cv = canonicalize(v)
            if cv is not None:
                out[k] = cv
        return out
    
    if isinstance(obj, datetime):
        # Normalize to UTC ISO format for timezone stability
        if obj.tzinfo is None:
            # Assume UTC for naive datetimes
            obj = obj.replace(tzinfo=timezone.utc)
        return obj.astimezone(timezone.utc).isoformat()
    
    if isinstance(obj, (set, frozenset)):
        # CRITICAL: Sort by canonical_json to handle non-orderable elements
        canonicalized_elements = [canonicalize(x) for x in obj]
        return sorted(canonicalized_elements, key=lambda x: canonical_json(x))
    
    if isinstance(obj, Path):
        return str(obj)
    
    # Handle numpy types via module check (not hasattr - too broad)
    obj_module = getattr(obj.__class__, '__module__', '')
    if obj_module.startswith('numpy'):
        # Check for array first (has ndim attribute and ndim > 0)
        ndim = getattr(obj, 'ndim', None)
        if ndim is not None and ndim > 0:
            # numpy array - use tolist
            if hasattr(obj, 'tolist'):
                return canonicalize(obj.tolist())
        # numpy scalar (np.int64, np.float32, etc.) - has ndim=0 or is generic
        if hasattr(obj, 'item'):
            return canonicalize(obj.item())
    
    # Handle dataclasses
    if hasattr(obj, '__dataclass_fields__'):
        from dataclasses import asdict
        return canonicalize(asdict(obj))
    
    # Handle Decimal
    if type(obj).__name__ == 'Decimal':
        return str(obj)
    
    # FAIL on unknown types - never silently stringify
    raise TypeError(
        f"canonicalize() does not support type {type(obj).__name__} "
        f"(module: {obj_module}). Add explicit handling or convert to a "
        f"supported type before fingerprinting."
    )


def canonical_json(obj: Any) -> str:
    """
    Produce deterministic JSON string from object.
    
    Uses canonicalize() first, then json.dumps with sorted keys.
    
    Args:
        obj: Object to serialize
        
    Returns:
        Deterministic JSON string
    """
    return json.dumps(canonicalize(obj), sort_keys=True, separators=(',', ':'))


def sha256_full(s: str) -> str:
    """
    Compute full 64-character SHA256 hash.
    
    Use this for identity keys where collision resistance matters.
    
    Args:
        s: String to hash
        
    Returns:
        64-character hexadecimal hash
    """
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def sha256_short(s: str, n: int = 16) -> str:
    """
    Compute short SHA256 hash for debug/display purposes only.
    
    WARNING: Do not use for identity keys - truncation increases collision risk.
    
    Args:
        s: String to hash
        n: Number of characters to return (default 16)
        
    Returns:
        Truncated hexadecimal hash
    """
    return hashlib.sha256(s.encode('utf-8')).hexdigest()[:n]


# =============================================================================
# CONFIG HASHING (uses canonicalization SST)
# =============================================================================

def compute_config_hash(config: Dict[str, Any], sort_keys: bool = True) -> str:
    """
    Compute deterministic hash of configuration dictionary.
    
    Uses canonical_json() + sha256_full() for consistency across
    different Python versions and environments.
    
    Args:
        config: Configuration dictionary to hash
        sort_keys: If True, sort dictionary keys for deterministic output
                   (Note: canonicalize always sorts, this is for backward compat)
    
    Returns:
        Hexadecimal hash string (64 characters)
    """
    # Use canonical_json which handles nested structures, numpy, datetime, etc.
    return sha256_full(canonical_json(config))


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

