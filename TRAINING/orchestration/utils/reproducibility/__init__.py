# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

"""
Reproducibility Module

Modular components for reproducibility tracking.
"""

from .utils import (
    collect_environment_info,
    compute_comparable_key,
    get_main_logger,
    _get_main_logger,  # Alias for backward compatibility
    make_tagged_scalar,
    make_tagged_not_applicable,
    make_tagged_per_target_feature,
    make_tagged_auto,
    make_tagged_not_computed,
    make_tagged_omitted,
    extract_scalar_from_tagged,
    extract_embargo_minutes,
    extract_folds,
    Stage,
    RouteType,
    TargetRankingView
)
from .config_loader import (
    load_thresholds,
    load_use_z_score,
    load_audit_mode,
    load_cohort_aware,
    load_n_ratio_threshold,
    load_cohort_config_keys
)

__all__ = [
    'collect_environment_info',
    'compute_comparable_key',
    'get_main_logger',
    '_get_main_logger',  # Alias for backward compatibility
    'make_tagged_scalar',
    'make_tagged_not_applicable',
    'make_tagged_per_target_feature',
    'make_tagged_auto',
    'make_tagged_not_computed',
    'make_tagged_omitted',
    'extract_scalar_from_tagged',
    'extract_embargo_minutes',
    'extract_folds',
    'Stage',
    'RouteType',
    'TargetRankingView',
    'load_thresholds',
    'load_use_z_score',
    'load_audit_mode',
    'load_cohort_aware',
    'load_n_ratio_threshold',
    'load_cohort_config_keys',
]

