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

