# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""Data loading utilities"""

from .data_loader import (
    resolve_time_col,
    _pick_one,
    _load_mtf_data_pandas,
    _prepare_training_data_cross_sectional_pandas,
    load_mtf_data,
)
from .data_utils import (
    strip_targets,
    collapse_identical_duplicate_columns,
    prepare_sequence_cs,
)

# For backward compatibility, create module-level aliases
data_loader = _load_mtf_data_pandas
data_utils = type('DataUtils', (), {
    'strip_targets': strip_targets,
    'collapse_identical_duplicate_columns': collapse_identical_duplicate_columns,
})()

__all__ = [
    'resolve_time_col',
    '_pick_one',
    '_load_mtf_data_pandas',
    '_prepare_training_data_cross_sectional_pandas',
    'load_mtf_data',
    'strip_targets',
    'collapse_identical_duplicate_columns',
    'prepare_sequence_cs',
    'data_loader',
    'data_utils',
]
