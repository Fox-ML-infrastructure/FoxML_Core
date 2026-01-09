# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Target Predictability Ranking - backward compatibility wrapper.

This file has been split into modules in the predictability/ subfolder for better maintainability.
All imports are re-exported here to maintain backward compatibility.
"""

# Re-export everything from the predictability modules
from TRAINING.ranking.predictability import *

# Also export main directly for script execution
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
