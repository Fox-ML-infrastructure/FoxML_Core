# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

"""
Target Ranking and Feature Selection Module

Extracted from SCRIPTS/rank_target_predictability.py and SCRIPTS/multi_model_feature_selection.py
to enable integration into the training pipeline while preserving leakage-free behavior.
"""

from .target_ranker import (
    TargetPredictabilityScore,
    evaluate_target_predictability,
    rank_targets,
    discover_targets,
    load_target_configs
)

from .feature_selector import (
    FeatureImportanceResult,
    select_features_for_target,
    rank_features_multi_model,
    load_multi_model_config
)

__all__ = [
    'TargetPredictabilityScore',
    'evaluate_target_predictability',
    'rank_targets',
    'discover_targets',
    'load_target_configs',
    'FeatureImportanceResult',
    'select_features_for_target',
    'rank_features_multi_model',
    'load_multi_model_config',
]

