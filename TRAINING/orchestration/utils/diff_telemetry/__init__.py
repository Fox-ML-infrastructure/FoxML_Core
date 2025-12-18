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
Diff Telemetry Module

Modular components for diff telemetry system.
"""

from .types import (
    ChangeSeverity,
    ComparabilityStatus,
    ResolvedRunContext,
    ComparisonGroup,
    NormalizedSnapshot,
    DiffResult,
    BaselineState
)

# Import from parent file (DiffTelemetry class still in main file)
import sys
from pathlib import Path
_parent_file = Path(__file__).parent.parent / "diff_telemetry.py"
if _parent_file.exists():
    import importlib.util
    spec = importlib.util.spec_from_file_location("diff_telemetry_main", _parent_file)
    diff_telemetry_main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diff_telemetry_main)
    
    DiffTelemetry = diff_telemetry_main.DiffTelemetry
    FINGERPRINT_SCHEMA_VERSION = getattr(diff_telemetry_main, 'FINGERPRINT_SCHEMA_VERSION', "1.0")
else:
    raise ImportError(f"Could not find diff_telemetry.py at {_parent_file}")

__all__ = [
    'ChangeSeverity',
    'ComparabilityStatus',
    'ResolvedRunContext',
    'ComparisonGroup',
    'NormalizedSnapshot',
    'DiffResult',
    'BaselineState',
    # Class still in main file
    'DiffTelemetry',
    'FINGERPRINT_SCHEMA_VERSION',
]

