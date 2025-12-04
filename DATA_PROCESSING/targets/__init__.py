"""
Copyright (c) 2025 Fox ML Infrastructure

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
    compute_neutral_band,
    classify_excess_return
)

__all__ = [
    # Barrier targets
    "compute_barrier_targets",
    "add_barrier_targets_to_dataframe",
    "add_zigzag_targets_to_dataframe",
    "add_mfe_mdd_targets_to_dataframe",
    "add_enhanced_targets_to_dataframe",
    # Excess return targets
    "rolling_beta",
    "future_excess_return",
    "compute_neutral_band",
    "classify_excess_return",
]

