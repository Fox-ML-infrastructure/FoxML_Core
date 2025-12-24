# MIT License - see LICENSE file

"""
Target/Label Generation Module

Provides target generation for different trading strategies:
- barrier: Barrier/first-passage labels (will_peak, will_valley)
- excess_returns: Excess return labels and neutral band classification
- hft_forward: HFT forward return targets for short horizons
"""


from .barrier import (
    compute_barrier_targets,
    add_barrier_targets_to_dataframe,
    add_zigzag_targets_to_dataframe,
    add_mfe_mdd_targets_to_dataframe,
    add_enhanced_targets_to_dataframe
)
from .excess_returns import (
    rolling_beta,
    future_excess_return,
    compute_epsilon_train_only,
    label_excess_band,
    create_targets,
    validate_targets
)

__all__ = [
    # Barrier targets
    "compute_barrier_targets",
    
    # Time contract
    "TimeContract",
    "enforce_t_plus_one_boundary",
    "validate_feature_as_of_safety",
    "add_barrier_targets_to_dataframe",
    "add_zigzag_targets_to_dataframe",
    "add_mfe_mdd_targets_to_dataframe",
    "add_enhanced_targets_to_dataframe",
    # Excess return targets
    "rolling_beta",
    "future_excess_return",
    "compute_epsilon_train_only",
    "label_excess_band",
    "create_targets",
    "validate_targets",
]

