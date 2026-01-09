# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""Specialized model classes - split from original large file for maintainability."""

# Import and re-export constants
from TRAINING.models.specialized.constants import (
    FAMILY_CAPS,
    assert_no_nan,
    tf_available,
    ngboost_available,
    USE_POLARS,
    TF_DEVICE,
    tf,
    STRATEGY_SUPPORT,
)

# Re-export everything for backward compatibility
from TRAINING.models.specialized.wrappers import (
    TFSeriesRegressor,
    GMMRegimeRegressor,
    OnlineChangeHeuristic,
)

from TRAINING.models.specialized.predictors import (
    GANPredictor,
    ChangePointPredictor,
)

from TRAINING.models.specialized.trainers import (
    train_changepoint_heuristic,
    train_ftrl_proximal,
    train_vae,
    train_gan,
    train_ensemble,
    train_meta_learning,
    train_multitask_temporal,
    train_multi_task,
    train_lightgbm_ranker,
    train_xgboost_ranker,
    safe_predict,
)

from TRAINING.models.specialized.trainers_extended import (
    train_lightgbm,
    train_xgboost,
    train_mlp,
    train_cnn1d_temporal,
    train_tabcnn,
    train_lstm_temporal,
    train_tablstm,
    train_transformer_temporal,
    train_tabtransformer,
    train_reward_based,
    train_quantile_lightgbm,
    train_ngboost,
    train_gmm_regime,
)

from TRAINING.models.specialized.metrics import (
    cs_metrics_by_time,
)

from TRAINING.models.specialized.core import (
    train_model,
    save_model,
    train_with_strategy,
    normalize_symbols,
    setup_tf,
    main,
)

from TRAINING.models.specialized.data_utils import (
    load_mtf_data,
    get_common_feature_columns,
    load_global_feature_list,
    save_global_feature_list,
    targets_for_interval,
    cs_transform_live,
    prepare_sequence_cs,
    prepare_training_data_cross_sectional,
)

__all__ = [
    # Constants
    'FAMILY_CAPS',
    'assert_no_nan',
    'tf_available',
    'ngboost_available',
    'USE_POLARS',
    'TF_DEVICE',
    'tf',
    'STRATEGY_SUPPORT',
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
    # Extended trainers
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
