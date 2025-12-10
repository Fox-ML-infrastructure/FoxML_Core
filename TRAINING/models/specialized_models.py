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
