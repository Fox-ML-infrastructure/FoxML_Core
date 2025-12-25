# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

"""
Common/shared utilities.

Utilities used across multiple modules for core functionality,
data processing, validation, metrics, and analysis.
"""

# Re-export key utilities for convenience
from .core_utils import SYMBOL_COL, INTERVAL_TO_TARGET, create_time_aware_split
from .config_cleaner import clean_config_for_estimator
from .sklearn_safe import make_sklearn_dense_X
from .task_types import TaskType, TargetConfig
from .task_metrics import evaluate_by_task, compute_composite_score
from .duration_parser import (
    parse_duration,
    Duration,
    DurationLike,
    enforce_purge_audit_rule,
    format_duration
)
from .sst_contract import (
    resolve_target_horizon_minutes,
    normalize_family,
    tracker_input_adapter
)
from .validation import ValidationUtils
from .data_preprocessor import DataPreprocessor
from .target_resolver import TargetResolver
from .metrics import MetricsWriter, load_metrics_config, aggregate_metrics_facts
from .registry_validation import validate_all_registries

__all__ = [
    # Core utilities
    'SYMBOL_COL',
    'INTERVAL_TO_TARGET',
    'create_time_aware_split',
    # Config utilities
    'clean_config_for_estimator',
    # Sklearn utilities
    'make_sklearn_dense_X',
    # Task utilities
    'TaskType',
    'TargetConfig',
    'evaluate_by_task',
    'compute_composite_score',
    # Duration utilities
    'parse_duration',
    'Duration',
    'DurationLike',
    'enforce_purge_audit_rule',
    'format_duration',
    # SST contract
    'resolve_target_horizon_minutes',
    'normalize_family',
    'tracker_input_adapter',
    # Validation
    'ValidationUtils',
    # Data processing
    'DataPreprocessor',
    'TargetResolver',
    # Metrics
    'MetricsWriter',
    'load_metrics_config',
    'aggregate_metrics_facts',
    # Registry validation
    'validate_all_registries',
]

