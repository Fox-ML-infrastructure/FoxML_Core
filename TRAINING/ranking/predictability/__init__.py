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

"""Target predictability ranking - split from original large file for maintainability."""

# Re-export everything for backward compatibility
from TRAINING.ranking.predictability.scoring import (
    TargetPredictabilityScore,
)
from TRAINING.ranking.predictability.composite_score import (
    calculate_composite_score,
)

from TRAINING.ranking.predictability.data_loading import (
    load_target_configs,
    discover_all_targets,
    load_sample_data,
    prepare_features_and_target,
    load_multi_model_config,
    get_model_config,
)

from TRAINING.ranking.predictability.leakage_detection import (
    find_near_copy_features,
    detect_leakage,
    _save_feature_importances,
    _log_suspicious_features,
)

from TRAINING.ranking.predictability.model_evaluation import (
    train_and_evaluate_models,
    evaluate_target_predictability,
)

from TRAINING.ranking.predictability.reporting import (
    save_leak_report_summary,
    save_rankings,
    _get_recommendation,
)

from TRAINING.ranking.predictability.main import main

__all__ = [
    # Scoring
    'TargetPredictabilityScore',
    'calculate_composite_score',
    # Data loading
    'load_target_configs',
    'discover_all_targets',
    'load_sample_data',
    'prepare_features_and_target',
    'load_multi_model_config',
    'get_model_config',
    # Leakage detection
    'find_near_copy_features',
    'detect_leakage',
    '_save_feature_importances',
    '_log_suspicious_features',
    # Model evaluation
    'train_and_evaluate_models',
    'evaluate_target_predictability',
    # Reporting
    'save_leak_report_summary',
    'save_rankings',
    '_get_recommendation',
    # Main
    'main',
]
