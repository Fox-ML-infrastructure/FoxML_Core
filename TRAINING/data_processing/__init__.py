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
Backward compatibility wrapper for TRAINING.data_processing

This module has been moved to TRAINING.data.loading
All imports are re-exported here to maintain backward compatibility.
"""

# Re-export everything from the new location
from TRAINING.data.loading import (
    _load_mtf_data_pandas,
    strip_targets,
    collapse_identical_duplicate_columns,
    data_loader,
    data_utils,
)

# For backward compatibility
load_mtf_data_from_dir = _load_mtf_data_pandas
load_symbol_data = _load_mtf_data_pandas

__all__ = [
    '_load_mtf_data_pandas',
    'load_mtf_data_from_dir',
    'load_symbol_data',
    'strip_targets',
    'collapse_identical_duplicate_columns',
    'data_loader',
    'data_utils',
]

