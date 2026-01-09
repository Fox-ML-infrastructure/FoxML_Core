# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Specialized model classes - backward compatibility wrapper.

This file has been split into modules in the specialized/ subfolder for better maintainability.
All imports are re-exported here to maintain backward compatibility.
"""

# Re-export everything from the specialized modules
from TRAINING.models.specialized import *

# Also export main directly for script execution
from TRAINING.models.specialized.core import main

__all__ = [
    # Wrappers
    'TFSeriesRegressor',
    'GMMRegimeRegressor',
    'OnlineChangeHeuristic',
    # Predictors
    'GANPredictor',
    'ChangePointPredictor',
    # Trainers
    'train_changepoint_heuristic',
    'train_ftrl_proximal',
    'train_vae',
    'train_gan',
    'train_ensemble',
    'train_meta_learning',
    'train_multitask_temporal',
    'train_multi_task',
    'train_lightgbm_ranker',
    'train_xgboost_ranker',
    'safe_predict',
    'train_lightgbm',
    'train_xgboost',
    'train_mlp',
    'train_cnn1d_temporal',
    'train_tabcnn',
    'train_lstm_temporal',
    'train_tablstm',
    'train_transformer_temporal',
    'train_tabtransformer',
    'train_reward_based',
    'train_quantile_lightgbm',
    'train_ngboost',
    'train_gmm_regime',
    # Metrics
    'cs_metrics_by_time',
    # Core
    'train_model',
    'save_model',
    'train_with_strategy',
    'normalize_symbols',
    'setup_tf',
    'main',
    # Data utils
    'load_mtf_data',
    'get_common_feature_columns',
    'load_global_feature_list',
    'save_global_feature_list',
    'targets_for_interval',
    'cs_transform_live',
    'prepare_sequence_cs',
    'prepare_training_data_cross_sectional',
]
