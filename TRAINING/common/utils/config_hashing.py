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
Config Hashing Utilities

Standardized config hashing for cache keys and reproducibility tracking.
Uses SHA256 of canonical JSON representation for consistency.
"""

import hashlib
import json
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

