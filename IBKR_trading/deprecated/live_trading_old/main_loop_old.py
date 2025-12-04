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
Main Live Trading Loop
====================

Main live trading system integrating all components.
"""


import numpy as np
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import time

# Import our components
from .model_predictor import ModelPredictor, ModelRegistry
from .horizon_blender import HorizonBlender
from .barrier_gate import BarrierGate, BarrierProbabilityProvider
from .cost_arbitrator import CostArbitrator
from .position_sizer import PositionSizer, PositionValidator

logger = logging.getLogger(__name__)

class LiveTradingSystem:
    """Main live trading system integrating all components."""
    
    def __init__(self, config):
        self.config = config
        self.initialized = False
        
        # Initialize components
        self._initialize_components()
        
        # Initialize model registry and buffers
        self.model_registry = ModelRegistry(config)
        self.buffer_manager = self._initialize_buffers()
        
        # Initialize barrier probability provider
        self.barrier_provider = BarrierProbabilityProvider(config)
        
        # Initialize validators
        self.position_validator = PositionValidator(config)
        
        self.initialized = True
        logger.info("LiveTradingSystem initialized successfully")
    
    def _initialize_components(self):
        """Initialize all trading components."""
        try:
            # Model predictor
            self.model_predictor = ModelPredictor(
                model_registry=None,  # Will be set after model registry
                buffer_manager=None,  # Will be set after buffer manager
                config=self.config
            )
            
            # Horizon blender
            self.horizon_blender = HorizonBlender(self.config)
            
            # Barrier gate
            self.barrier_gate = BarrierGate(self.config)
            
            # Cost arbitrator
            self.cost_arbitrator = CostArbitrator(self.config)
            
            # Position sizer
            self.position_sizer = PositionSizer(self.config)
            
            logger.info("All components initialized")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    def _initialize_buffers(self):
        """Initialize sequence buffers for sequential models."""
        try:
            from live.seq_ring_buffer import SeqBufferManager
            
            buffer_manager = SeqBufferManager(
                T=self.config.get('lookback_T', 60),
                F=self.config.get('num_features', 50),
                ttl_seconds=self.config.get('ttl_seconds', 300)
            )
            
            logger.info("Buffer manager initialized")
            return buffer_manager
            
        except ImportError:
            logger.warning("SeqBufferManager not available, using fallback")
            return None
    
    def live_step(self, market_data: Dict, current_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Execute one live trading step.
        
        Args:
            market_data: Market data including features, spreads, volatility, etc.
            current_weights: Current portfolio weights
        
        Returns:
            Target weights for next rebalance
        """
        try:
            if not self.initialized:
                logger.error("System not initialized")
                return current_weights
            
            logger.info("Starting live trading step")
            start_time = time.time()
            
            # 1. Get predictions for all horizons
            alpha_by_horizon = self._get_horizon_predictions(market_data)
            
            if not alpha_by_horizon:
                logger.warning("No alpha generated for any horizon")
                return current_weights
            
            # 2. Get barrier probabilities
            peak_probs, valley_probs = self._get_barrier_probabilities(market_data)
            
            # 3. Arbitrate horizons
            alpha_arbitrated = self.cost_arbitrator.arbitrate_horizons(
                alpha_by_horizon, market_data
            )
            
            if not alpha_arbitrated:
                logger.warning("No alpha after horizon arbitration")
                return current_weights
            
            # 4. Apply barrier gate
            alpha_gated = self.barrier_gate.apply_gate(
                alpha_arbitrated, peak_probs, valley_probs
            )
            
            # 5. Size positions
            target_weights = self.position_sizer.size_positions(
                alpha_gated, market_data, current_weights
            )
            
            # 6. Validate positions
            validation_result = self.position_validator.validate_weights(target_weights)
            
            if not validation_result['valid']:
                logger.error(f"Position validation failed: {validation_result['errors']}")
                return current_weights
            
            if validation_result['warnings']:
                logger.warning(f"Position validation warnings: {validation_result['warnings']}")
            
            # Log performance metrics
            elapsed_time = time.time() - start_time
            logger.info(f"Live step completed in {elapsed_time:.2f}s")
            logger.info(f"Target weights: {len(target_weights)} symbols")
            
            return target_weights
            
        except Exception as e:
            logger.error(f"Error in live step: {e}")
            return current_weights
    
    def _get_horizon_predictions(self, market_data: Dict) -> Dict[str, Dict[str, float]]:
        """Get predictions for all horizons."""
        try:
            # Update model predictor with current components
            self.model_predictor.model_registry = self.model_registry
            self.model_predictor.buffer_manager = self.buffer_manager
            
            # Get predictions for all horizons
            predictions_by_horizon = self.model_predictor.predict_all_horizons(
                market_data['symbols'], market_data['features']
            )
            
            if not predictions_by_horizon:
                logger.warning("No predictions from any horizon")
                return {}
            
            # Blend predictions for each horizon
            alpha_by_horizon = {}
            for horizon, predictions in predictions_by_horizon.items():
                blended_alpha = self.horizon_blender.blend_horizon(horizon, predictions)
                if blended_alpha:
                    alpha_by_horizon[horizon] = blended_alpha
                    logger.info(f"Horizon {horizon}: {len(blended_alpha)} symbols")
            
            return alpha_by_horizon
            
        except Exception as e:
            logger.error(f"Error getting horizon predictions: {e}")
            return {}
    
    def _get_barrier_probabilities(self, market_data: Dict) -> tuple:
        """Get barrier probabilities."""
        try:
            symbols = market_data['symbols']
            features = market_data['features']
            
            # Get peak probabilities
            peak_probs = self.barrier_provider.get_peak_probabilities(symbols, features)
            
            # Get valley probabilities
            valley_probs = self.barrier_provider.get_valley_probabilities(symbols, features)
            
            logger.info(f"Got barrier probabilities for {len(peak_probs)} symbols")
            return peak_probs, valley_probs
            
        except Exception as e:
            logger.error(f"Error getting barrier probabilities: {e}")
            # Return default probabilities
            symbols = market_data.get('symbols', [])
            return {s: 0.5 for s in symbols}, {s: 0.5 for s in symbols}
    
    def update_features(self, symbol: str, features: np.ndarray, timestamp: Optional[datetime] = None):
        """Update features for a symbol."""
        try:
            if self.buffer_manager:
                success = self.buffer_manager.push_features(symbol, features, timestamp)
                if success:
                    logger.debug(f"Updated features for {symbol}")
                else:
                    logger.warning(f"Failed to update features for {symbol}")
            else:
                logger.warning("Buffer manager not available")
                
        except Exception as e:
            logger.error(f"Error updating features for {symbol}: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status information."""
        try:
            status = {
                'initialized': self.initialized,
                'components': {
                    'model_predictor': self.model_predictor is not None,
                    'horizon_blender': self.horizon_blender is not None,
                    'barrier_gate': self.barrier_gate is not None,
                    'cost_arbitrator': self.cost_arbitrator is not None,
                    'position_sizer': self.position_sizer is not None,
                    'buffer_manager': self.buffer_manager is not None,
                    'barrier_provider': self.barrier_provider is not None
                },
                'model_registry': {
                    'available_models': self.model_registry.list_available_models() if self.model_registry else {}
                },
                'buffer_status': self.buffer_manager.get_status() if self.buffer_manager else {}
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}
    
    def reset_system(self):
        """Reset the trading system."""
        try:
            if self.buffer_manager:
                self.buffer_manager.reset_all()
            
            logger.info("System reset completed")
            
        except Exception as e:
            logger.error(f"Error resetting system: {e}")

class LiveTradingManager:
    """Manager for live trading operations."""
    
    def __init__(self, config):
        self.config = config
        self.trading_system = LiveTradingSystem(config)
        self.is_running = False
        self.last_rebalance = None
        self.rebalance_frequency = config.get('rebalance_frequency', 300)  # 5 minutes
        
        logger.info("LiveTradingManager initialized")
    
    def start_trading(self):
        """Start live trading."""
        try:
            self.is_running = True
            logger.info("Live trading started")
            
        except Exception as e:
            logger.error(f"Error starting trading: {e}")
            self.is_running = False
    
    def stop_trading(self):
        """Stop live trading."""
        try:
            self.is_running = False
            logger.info("Live trading stopped")
            
        except Exception as e:
            logger.error(f"Error stopping trading: {e}")
    
    def should_rebalance(self) -> bool:
        """Check if it's time to rebalance."""
        try:
            if not self.is_running:
                return False
            
            if self.last_rebalance is None:
                return True
            
            time_since_rebalance = (datetime.now() - self.last_rebalance).total_seconds()
            return time_since_rebalance >= self.rebalance_frequency
            
        except Exception as e:
            logger.error(f"Error checking rebalance timing: {e}")
            return False
    
    def execute_rebalance(self, market_data: Dict, current_weights: Dict[str, float]) -> Dict[str, float]:
        """Execute a rebalance."""
        try:
            if not self.should_rebalance():
                return current_weights
            
            logger.info("Executing rebalance")
            
            # Get target weights
            target_weights = self.trading_system.live_step(market_data, current_weights)
            
            # Update last rebalance time
            self.last_rebalance = datetime.now()
            
            logger.info(f"Rebalance completed: {len(target_weights)} positions")
            return target_weights
            
        except Exception as e:
            logger.error(f"Error executing rebalance: {e}")
            return current_weights
    
    def update_market_data(self, market_data: Dict):
        """Update market data and features."""
        try:
            # Update features for sequential models
            features = market_data.get('features', {})
            for symbol, feature_data in features.items():
                if feature_data.ndim == 2:
                    # Latest row for sequential models
                    latest_features = feature_data[-1]
                    self.trading_system.update_features(symbol, latest_features)
                else:
                    # Single row
                    self.trading_system.update_features(symbol, feature_data)
            
            logger.debug("Market data updated")
            
        except Exception as e:
            logger.error(f"Error updating market data: {e}")
    
    def get_trading_status(self) -> Dict[str, Any]:
        """Get trading status."""
        try:
            return {
                'is_running': self.is_running,
                'last_rebalance': self.last_rebalance,
                'rebalance_frequency': self.rebalance_frequency,
                'system_status': self.trading_system.get_system_status()
            }
            
        except Exception as e:
            logger.error(f"Error getting trading status: {e}")
            return {'error': str(e)}
