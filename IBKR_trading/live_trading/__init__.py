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
Live Trading System - IBKR Integration
=====================================

Complete live trading system integrating all trained models (tabular + sequential + multi-task)
across all horizons and strategies for IBKR trading.
"""


__version__ = "2.0.0"
__author__ = "Trading System Team"

# Import main components
from .main_loop import LiveTradingSystem, LiveTradingManager
from .model_predictor import ModelPredictor, ModelRegistry
from .horizon_blender import HorizonBlender, AdvancedBlender
from .barrier_gate import BarrierGate, AdvancedBarrierGate, BarrierProbabilityProvider
from .cost_arbitrator import CostArbitrator, CostModel, AdvancedCostModel
from .position_sizer import PositionSizer, AdvancedPositionSizer, PositionValidator

__all__ = [
    # Main system
    'LiveTradingSystem',
    'LiveTradingManager',
    
    # Core components
    'ModelPredictor',
    'ModelRegistry',
    'HorizonBlender',
    'AdvancedBlender',
    'BarrierGate',
    'AdvancedBarrierGate',
    'BarrierProbabilityProvider',
    'CostArbitrator',
    'CostModel',
    'AdvancedCostModel',
    'PositionSizer',
    'AdvancedPositionSizer',
    'PositionValidator',
]
