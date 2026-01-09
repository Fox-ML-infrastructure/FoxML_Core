# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Training Strategy Execution Module

Execution code for training strategies (data preparation, family runners, training loops).
"""

from .main import main
from .training import train_models_for_interval_comprehensive, train_model_comprehensive, _legacy_train_fallback
from .data_preparation import prepare_training_data_cross_sectional
from .family_runners import _run_family_inproc, _run_family_isolated

# Import data functions from strategy_functions (not from data_preparation)
from TRAINING.training_strategies.strategy_functions import (
    load_mtf_data,
    discover_targets,
    prepare_training_data,
)

__all__ = [
    'main',
    'train_models_for_interval_comprehensive',
    'train_model_comprehensive',
    '_legacy_train_fallback',
    'load_mtf_data',
    'discover_targets',
    'prepare_training_data',
    'prepare_training_data_cross_sectional',
    '_run_family_inproc',
    '_run_family_isolated',
]

