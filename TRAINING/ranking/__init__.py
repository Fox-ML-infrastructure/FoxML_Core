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
Target Ranking and Feature Selection Module

Extracted from scripts/rank_target_predictability.py and scripts/multi_model_feature_selection.py
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

