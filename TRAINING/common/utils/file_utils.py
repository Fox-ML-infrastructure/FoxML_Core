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
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temp_file = file_path.with_suffix('.tmp')
    
    try:
        # Write to temp file
        with open(temp_file, 'w') as f:
            if default is not None:
                json.dump(data, f, indent=2, default=default)
            else:
                json.dump(data, f, indent=2)
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

