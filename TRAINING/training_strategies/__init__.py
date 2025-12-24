# MIT License - see LICENSE file

"""Training strategies - split from original large file for maintainability."""

# Re-export everything for backward compatibility
from TRAINING.training_strategies.execution.family_runners import (
    _run_family_inproc,
    _run_family_isolated,
)

from TRAINING.training_strategies.utils import (
    setup_logging,
    _now,
    safe_duration,
    _pkg_ver,
    _env_guard,
    build_sequences_from_features,
    tf_available,
    ngboost_available,
    pick_tf_device,
)

from TRAINING.training_strategies.execution.data_preparation import (
    prepare_training_data_cross_sectional,
)

# Import strategy classes from strategies/ directory
from TRAINING.training_strategies.strategies import (
    BaseTrainingStrategy,
    SingleTaskStrategy,
    MultiTaskStrategy,
    CascadeStrategy,
)

# Import data/strategy functions from strategy_functions.py file
from TRAINING.training_strategies.strategy_functions import (
    load_mtf_data,
    discover_targets,
    prepare_training_data,
    create_strategy_config,
    train_with_strategy,
    compare_strategies,
)

from TRAINING.training_strategies.execution.training import (
    train_models_for_interval_comprehensive,
    train_model_comprehensive,
    _legacy_train_fallback,
)
from TRAINING.training_strategies.execution.data_preparation import (
    prepare_training_data_cross_sectional,
)

from TRAINING.training_strategies.execution.main import main

# Export constants
from TRAINING.training_strategies.execution.setup import (
    TF_FAMS,
    TORCH_FAMS,
    CPU_FAMS,
)
from TRAINING.training_strategies.utils import (
    ALL_FAMILIES,
)
# FAMILY_CAPS is in models.specialized.constants, not here
try:
    from TRAINING.models.specialized.constants import FAMILY_CAPS
except ImportError:
    FAMILY_CAPS = {}

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
