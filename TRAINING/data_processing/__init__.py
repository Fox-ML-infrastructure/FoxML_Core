# MIT License - see LICENSE file

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

