# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Strict Mode Enforcement

Turns silent failures into hard errors when FOXML_STRICT_MODE=1
"""

import os
import logging
from typing import Any

logger = logging.getLogger(__name__)

STRICT_MODE = os.getenv("FOXML_STRICT_MODE", "0") == "1"

def strict_assert(condition: bool, message: str, exc_type: type = RuntimeError) -> None:
    """
    Assert condition is True. In strict mode, raises exception. Otherwise logs warning.
    
    Args:
        condition: Condition to check
        message: Error message if condition fails
        exc_type: Exception type to raise (default: RuntimeError)
    """
    if not condition:
        if STRICT_MODE:
            raise exc_type(f"[STRICT] {message}")
        else:
            logger.warning(f"[STRICT-WARN] {message}")

def strict_check_config_path(cfg: dict, path: str, default: Any = None) -> Any:
    """
    Get config value with strict validation.
    
    In strict mode, raises error if path doesn't exist.
    Otherwise returns default and logs warning.
    
    Args:
        cfg: Config dictionary
        path: Dot-separated path (e.g., "safety.leakage_detection.auto_fix_max_features_per_run")
        default: Default value if path not found
    
    Returns:
        Config value or default
    """
    keys = path.split(".")
    value = cfg
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            msg = f"Config path '{path}' not found. Missing key: '{key}' in {keys[:keys.index(key)]}"
            strict_assert(False, msg, ValueError)
            return default
    
    return value

def strict_check_type(value: Any, expected_type: type, name: str) -> None:
    """
    Check value type matches expected type.
    
    Args:
        value: Value to check
        expected_type: Expected type
        name: Name of value (for error message)
    """
    if not isinstance(value, expected_type):
        msg = f"{name} must be {expected_type.__name__}, got {type(value).__name__}: {value}"
        strict_assert(False, msg, TypeError)
