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
Orchestration-specific utilities.

Utilities used primarily by the orchestration module for checkpointing,
logging, run context, reproducibility tracking, and telemetry.
"""

# Re-export key utilities for convenience
from .checkpoint import CheckpointManager
from .logging_setup import setup_logging
from .run_context import RunContext
from .cohort_metadata_extractor import (
    extract_cohort_metadata,
    format_for_reproducibility_tracker
)
from .reproducibility_tracker import ReproducibilityTracker
from .diff_telemetry import ComparisonGroup

__all__ = [
    'CheckpointManager',
    'setup_logging',
    'RunContext',
    'extract_cohort_metadata',
    'format_for_reproducibility_tracker',
    'ReproducibilityTracker',
    'ComparisonGroup',
]

