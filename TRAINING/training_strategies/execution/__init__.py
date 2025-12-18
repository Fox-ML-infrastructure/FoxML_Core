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

