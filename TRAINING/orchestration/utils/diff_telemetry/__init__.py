# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

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
    # Export run hash functions
    compute_full_run_hash = getattr(diff_telemetry_main, 'compute_full_run_hash', None)
    compute_run_hash_with_changes = getattr(diff_telemetry_main, 'compute_run_hash_with_changes', None)
    save_run_hash = getattr(diff_telemetry_main, 'save_run_hash', None)
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
    # Run hash functions
    'compute_full_run_hash',
    'compute_run_hash_with_changes',
    'save_run_hash',
]

