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

