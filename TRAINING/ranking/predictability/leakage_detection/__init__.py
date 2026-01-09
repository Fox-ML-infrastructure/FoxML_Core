# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Leakage Detection Module

Modular components for leakage detection and feature analysis.
"""

from .feature_analysis import (
    find_near_copy_features,
    is_calendar_feature,
    detect_leaking_features
)
from .reporting import (
    save_feature_importances,
    log_suspicious_features
)

# Import from parent file (functions that weren't extracted yet)
# These are still in leakage_detection.py (parent file, not the folder)
import sys
from pathlib import Path
_parent_file = Path(__file__).parent.parent / "leakage_detection.py"
if _parent_file.exists():
    import importlib.util
    spec = importlib.util.spec_from_file_location("leakage_detection_main", _parent_file)
    leakage_detection_main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(leakage_detection_main)
    
    detect_leakage = leakage_detection_main.detect_leakage
    _save_feature_importances = leakage_detection_main._save_feature_importances
    _log_suspicious_features = leakage_detection_main._log_suspicious_features
    _detect_leaking_features = leakage_detection_main._detect_leaking_features
    _is_calendar_feature = leakage_detection_main._is_calendar_feature
else:
    raise ImportError(f"Could not find leakage_detection.py at {_parent_file}")

__all__ = [
    'find_near_copy_features',
    'is_calendar_feature',
    'detect_leaking_features',
    'save_feature_importances',
    'log_suspicious_features',
    # Functions still in main file
    'detect_leakage',
    '_save_feature_importances',
    '_log_suspicious_features',
    '_detect_leaking_features',
    '_is_calendar_feature',
]

