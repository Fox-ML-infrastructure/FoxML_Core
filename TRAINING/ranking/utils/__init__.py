# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Ranking-specific utilities.

Utilities used primarily by the ranking module for feature selection,
target ranking, and cross-sectional analysis.
"""

# Re-export key utilities for convenience
from .cross_sectional_data import (
    load_mtf_data_for_ranking,
    prepare_cross_sectional_data_for_ranking,
    _compute_feature_fingerprint,
    _log_feature_set
)
from .leakage_filtering import (
    filter_features_for_target,
    _extract_horizon,
    _load_leakage_config,
    reload_feature_configs
)
from .leakage_budget import (
    compute_budget,
    compute_feature_lookback_max,
    infer_lookback_minutes
)
from .lookback_cap_enforcement import apply_lookback_cap
from .lookback_policy import assert_featureset_hash
from .target_validation import validate_target, check_cv_compatibility
from .target_utils import (
    is_classification_target,
    is_binary_classification_target
)
from .purged_time_series_split import PurgedTimeSeriesSplit
from .resolved_config import (
    create_resolved_config,
    compute_feature_lookback_max,
    derive_purge_embargo
)
from .data_interval import detect_interval_from_dataframe

__all__ = [
    # Cross-sectional data
    'load_mtf_data_for_ranking',
    'prepare_cross_sectional_data_for_ranking',
    '_compute_feature_fingerprint',
    '_log_feature_set',
    # Leakage filtering
    'filter_features_for_target',
    '_extract_horizon',
    '_load_leakage_config',
    'reload_feature_configs',
    # Leakage budget
    'compute_budget',
    'compute_feature_lookback_max',
    'infer_lookback_minutes',
    # Lookback enforcement
    'apply_lookback_cap',
    'assert_featureset_hash',
    # Target utilities
    'validate_target',
    'check_cv_compatibility',
    'is_classification_target',
    'is_binary_classification_target',
    # Time series
    'PurgedTimeSeriesSplit',
    # Config resolution
    'create_resolved_config',
    'derive_purge_embargo',
    # Data utilities
    'detect_interval_from_dataframe',
]

