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
Training with Strategies - backward compatibility wrapper.

This file has been split into modules in the training_strategies/ subfolder for better maintainability.
All imports are re-exported here to maintain backward compatibility.
"""

# Re-export everything from the training_strategies modules
from TRAINING.training_strategies import *

# Also export main directly for script execution
from TRAINING.training_strategies.main import main

__all__ = [
    # Family runners
    '_run_family_inproc',
    '_run_family_isolated',
    # Utils
    'setup_logging',
    '_now',
    'safe_duration',
    '_pkg_ver',
    '_env_guard',
    'build_sequences_from_features',
    'tf_available',
    'ngboost_available',
    'pick_tf_device',
    # Data preparation
    'prepare_training_data_cross_sectional',
    'load_mtf_data',
    'discover_targets',
    'prepare_training_data',
    # Training
    'train_models_for_interval_comprehensive',
    'train_model_comprehensive',
    '_legacy_train_fallback',
    # Strategies
    'create_strategy_config',
    'train_with_strategy',
    'compare_strategies',
    # Main
    'main',
    # Constants
    'TF_FAMS',
    'TORCH_FAMS',
    'CPU_FAMS',
    'ALL_FAMILIES',
    'FAMILY_CAPS',
]
