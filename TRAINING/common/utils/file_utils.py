# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
File Utilities

Atomic file operations and file I/O utilities for safe, crash-consistent file operations.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional


def write_atomic_json(file_path: Path, data: Dict[str, Any], default: Any = None) -> None:
    """
    Write JSON file atomically using temp file + rename with full durability.
    
    This ensures crash consistency AND power-loss safety:
    1. Write to temp file
    2. fsync(tempfile) - ensure data is on disk
    3. os.replace() - atomic rename (POSIX: atomic, Windows: best-effort)
    4. fsync(directory) - ensure directory entry is on disk
    
    This pattern is required for "audit-ready" systems that must survive sudden power loss.
    
    Args:
        file_path: Target file path
        data: Data to write (must be JSON-serializable)
        default: Optional default function for JSON serialization (e.g., str for non-serializable types)
    
    Raises:
        IOError: If write fails
    """
    # SST: Sanitize data to normalize enums to strings before JSON serialization
    from enum import Enum
    try:
        import pandas as pd
        has_pandas = True
    except ImportError:
        has_pandas = False
    
    def _sanitize_for_json(obj):
        if has_pandas and isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, dict):
            return {k: _sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [_sanitize_for_json(v) for v in obj]
        else:
            return obj
    sanitized_data = _sanitize_for_json(data)
    
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temp_file = file_path.with_suffix('.tmp')
    
    try:
        # Write to temp file
        with open(temp_file, 'w') as f:
            if default is not None:
                json.dump(sanitized_data, f, indent=2, default=default)
            else:
                json.dump(sanitized_data, f, indent=2)
            f.flush()  # Ensure immediate write
            os.fsync(f.fileno())  # Force write to disk (durability)
        
        # Atomic rename (POSIX: atomic, Windows: best-effort)
        os.replace(temp_file, file_path)
        
        # Sync directory entry to ensure rename is durable
        # This is critical for power-loss safety
        try:
            dir_fd = os.open(file_path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)  # Sync directory entry
            finally:
                os.close(dir_fd)
        except (OSError, AttributeError):
            # Fallback: sync parent directory if available
            # Some systems don't support directory fsync
            pass
    except Exception as e:
        # Cleanup temp file on failure
        if temp_file.exists():
            try:
                temp_file.unlink()
            except Exception:
                pass
        raise IOError(f"Failed to write atomic JSON to {file_path}: {e}") from e


def read_atomic_json(file_path: Path) -> Optional[Dict[str, Any]]:
    """
    Read JSON file safely.
    
    Args:
        file_path: Path to JSON file
    
    Returns:
        Parsed JSON data, or None if file doesn't exist or read fails
    """
    if not file_path.exists():
        return None
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


# =============================================================================
# SST Helpers for JSON and Parquet Serialization
# =============================================================================

def sanitize_for_serialization(obj: Any) -> Any:
    """
    SST helper: Recursively sanitize data for JSON/Parquet serialization.
    
    Converts:
    - Enum objects → string values (View, Stage, etc.)
    - pd.Timestamp → ISO strings
    - Non-string dict keys → strings (for Parquet compatibility)
    
    This is the single source of truth for data sanitization.
    All JSON and Parquet writes should use this to handle enums.
    
    Args:
        obj: Object to sanitize (can be dict, list, tuple, Enum, Timestamp, etc.)
    
    Returns:
        Sanitized object with all enums converted to strings
    """
    from enum import Enum
    try:
        import pandas as pd
        has_pandas = True
    except ImportError:
        has_pandas = False
    
    if has_pandas and isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, Enum):
        # Convert enum to string value for serialization
        return obj.value
    elif isinstance(obj, dict):
        # For Parquet: also stringify keys (PyArrow doesn't support int keys)
        return {str(k): sanitize_for_serialization(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_serialization(v) for v in obj]
    else:
        return obj


def safe_json_dump(data: Any, file, **kwargs) -> None:
    """
    SST helper: Always-safe JSON dump that sanitizes enums and timestamps.
    
    This is the single source of truth for JSON writing. Always use this
    instead of json.dump() to ensure enums are converted to strings.
    
    Works with both file handles and Path objects.
    
    Args:
        data: Data to serialize (will be sanitized)
        file: File handle or Path object
        **kwargs: Passed to json.dump() (indent, default, etc.)
    
    Examples:
        # With file handle
        with open('data.json', 'w') as f:
            safe_json_dump(data, f, indent=2)
        
        # With Path object
        safe_json_dump(data, Path('data.json'), indent=2)
    """
    sanitized = sanitize_for_serialization(data)
    
    # Handle both file handles and Path objects
    if isinstance(file, Path):
        with open(file, 'w') as f:
            json.dump(sanitized, f, **kwargs)
    else:
        json.dump(sanitized, file, **kwargs)


def safe_dataframe_from_dict(data: Dict[str, Any]) -> 'pd.DataFrame':
    """
    SST helper: Create DataFrame from dict with enum sanitization.
    
    Use this instead of pd.DataFrame([data]) when data might contain enums.
    This ensures enums are converted to strings before DataFrame creation,
    preventing object dtype issues and parquet write failures.
    
    Args:
        data: Dictionary to convert (will be sanitized)
    
    Returns:
        pandas DataFrame with sanitized data
    
    Example:
        df = safe_dataframe_from_dict(metrics_data)
        df.to_parquet('metrics.parquet')
    """
    import pandas as pd
    sanitized = sanitize_for_serialization(data)
    return pd.DataFrame([sanitized])

