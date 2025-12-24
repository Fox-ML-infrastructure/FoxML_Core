# MIT License - see LICENSE file

"""
Utility Components (Backward Compatibility Layer)

This module provides backward compatibility by re-exporting utilities
from their new locations. Utilities have been reorganized into:
- TRAINING/ranking/utils/ - Ranking-specific utilities
- TRAINING/orchestration/utils/ - Orchestration-specific utilities
- TRAINING/common/utils/ - Shared/common utilities

New code should import directly from the new locations.
"""

# Re-export from new locations for backward compatibility
# Ranking utilities
from TRAINING.ranking.utils.cross_sectional_data import (
    load_mtf_data_for_ranking,
    prepare_cross_sectional_data_for_ranking,
    _compute_feature_fingerprint,
    _log_feature_set
)
from TRAINING.ranking.utils.leakage_filtering import (
    filter_features_for_target,
    _extract_horizon,
    _load_leakage_config,
    reload_feature_configs
)
from TRAINING.ranking.utils import leakage_budget
# Import module for backward compatibility (some code imports leakage_budget as a module)
from TRAINING.ranking import utils as ranking_utils
leakage_budget = ranking_utils.leakage_budget

# Also import functions directly for convenience
from TRAINING.ranking.utils.leakage_budget import (
    compute_budget,
    compute_feature_lookback_max,
    infer_lookback_minutes
)
from TRAINING.ranking.utils.lookback_cap_enforcement import apply_lookback_cap
from TRAINING.ranking.utils.lookback_policy import assert_featureset_fingerprint
from TRAINING.ranking.utils.target_validation import validate_target, check_cv_compatibility
from TRAINING.ranking.utils.target_utils import (
    is_classification_target,
    is_binary_classification_target
)
from TRAINING.ranking.utils.purged_time_series_split import PurgedTimeSeriesSplit
from TRAINING.ranking.utils.resolved_config import (
    create_resolved_config,
    derive_purge_embargo
)
from TRAINING.ranking.utils.data_interval import detect_interval_from_dataframe

# Orchestration utilities
from TRAINING.orchestration.utils.checkpoint import CheckpointManager
from TRAINING.orchestration.utils.logging_setup import setup_logging
from TRAINING.orchestration.utils.run_context import RunContext
from TRAINING.orchestration.utils.cohort_metadata_extractor import (
    extract_cohort_metadata,
    format_for_reproducibility_tracker
)
from TRAINING.orchestration.utils.reproducibility_tracker import ReproducibilityTracker
from TRAINING.orchestration.utils.diff_telemetry import ComparisonGroup

# Common utilities
from TRAINING.common.utils.core_utils import SYMBOL_COL, INTERVAL_TO_TARGET, create_time_aware_split
from TRAINING.common.utils.config_cleaner import clean_config_for_estimator
from TRAINING.common.utils.sklearn_safe import make_sklearn_dense_X
from TRAINING.common.utils.task_types import TaskType, TargetConfig
from TRAINING.common.utils.task_metrics import evaluate_by_task, compute_composite_score
from TRAINING.common.utils.duration_parser import (
    parse_duration,
    Duration,
    DurationLike,
    enforce_purge_audit_rule,
    format_duration
)
from TRAINING.common.utils.sst_contract import (
    resolve_target_horizon_minutes,
    normalize_family,
    tracker_input_adapter
)
from TRAINING.common.utils.validation import ValidationUtils
from TRAINING.common.utils.data_preprocessor import DataPreprocessor
from TRAINING.common.utils.target_resolver import TargetResolver
from TRAINING.common.utils.metrics import MetricsWriter, load_metrics_config, aggregate_metrics_facts
from TRAINING.common.utils.registry_validation import validate_all_registries

# Legacy exports (for backward compatibility)
__all__ = [
    # Ranking utilities
    'load_mtf_data_for_ranking',
    'prepare_cross_sectional_data_for_ranking',
    '_compute_feature_fingerprint',
    '_log_feature_set',
    'filter_features_for_target',
    '_extract_horizon',
    '_load_leakage_config',
    'reload_feature_configs',
    'leakage_budget',  # Module export for backward compatibility
    'compute_budget',
    'compute_feature_lookback_max',
    'infer_lookback_minutes',
    'apply_lookback_cap',
    'assert_featureset_fingerprint',
    'validate_target',
    'check_cv_compatibility',
    'is_classification_target',
    'is_binary_classification_target',
    'PurgedTimeSeriesSplit',
    'create_resolved_config',
    'derive_purge_embargo',
    'detect_interval_from_dataframe',
    # Orchestration utilities
    'CheckpointManager',
    'setup_logging',
    'RunContext',
    'extract_cohort_metadata',
    'format_for_reproducibility_tracker',
    'ReproducibilityTracker',
    'ComparisonGroup',
    # Common utilities
    'SYMBOL_COL',
    'INTERVAL_TO_TARGET',
    'create_time_aware_split',
    'clean_config_for_estimator',
    'make_sklearn_dense_X',
    'TaskType',
    'TargetConfig',
    'evaluate_by_task',
    'compute_composite_score',
    'parse_duration',
    'Duration',
    'DurationLike',
    'enforce_purge_audit_rule',
    'format_duration',
    'resolve_target_horizon_minutes',
    'normalize_family',
    'tracker_input_adapter',
    'ValidationUtils',
    'DataPreprocessor',
    'TargetResolver',
    'MetricsWriter',
    'load_metrics_config',
    'aggregate_metrics_facts',
    'validate_all_registries',
]
