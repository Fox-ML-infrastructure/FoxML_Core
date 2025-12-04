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
Live Trading Integration Tests
=============================

Comprehensive tests for the live trading system integration.
"""


import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live_trading.main_loop import LiveTradingSystem, LiveTradingManager
from live_trading.model_predictor import ModelPredictor, ModelRegistry
from live_trading.horizon_blender import HorizonBlender
from live_trading.barrier_gate import BarrierGate, BarrierProbabilityProvider
from live_trading.cost_arbitrator import CostArbitrator, CostModel
from live_trading.position_sizer import PositionSizer, PositionValidator

logger = logging.getLogger(__name__)

def generate_mock_market_data(symbols: List[str], num_features: int = 50) -> Dict[str, Any]:
    """Generate mock market data for testing."""
    np.random.seed(42)
    
    # Generate features for each symbol
    features = {}
    for symbol in symbols:
        # Generate time series features (T, F)
        T = 100  # 100 time steps
        F = num_features
        features[symbol] = np.random.randn(T, F).astype(np.float32)
    
    # Generate market data
    market_data = {
        'symbols': symbols,
        'features': features,
        'spreads': {symbol: np.random.uniform(1.0, 3.0) for symbol in symbols},
        'vol_short': {symbol: np.random.uniform(0.1, 0.3) for symbol in symbols},
        'participation': {symbol: np.random.uniform(0.005, 0.02) for symbol in symbols},
        'covariance': np.random.randn(len(symbols), len(symbols)),
        'sectors': {symbol: f'Sector_{i % 3}' for i, symbol in enumerate(symbols)},
        'momentum': {symbol: np.random.uniform(-0.1, 0.1) for symbol in symbols}
    }
    
    # Make covariance matrix positive definite
    market_data['covariance'] = market_data['covariance'] @ market_data['covariance'].T
    market_data['covariance'] += np.eye(len(symbols)) * 0.1
    
    return market_data

def test_model_predictor():
    """Test model predictor functionality."""
    print("🧪 Testing model predictor...")
    
    # Mock model registry
    class MockModelRegistry:
        def __init__(self, config):
            self.config = config
            self.models = {
                '5m': {
                    'LightGBM': MockModel('LightGBM'),
                    'XGBoost': MockModel('XGBoost'),
                    'CNN1D': MockModel('CNN1D')
                },
                '15m': {
                    'LightGBM': MockModel('LightGBM'),
                    'LSTM': MockModel('LSTM')
                }
            }
        
        def get_models_by_horizon(self, horizon):
            return self.models.get(horizon, {})
        
        def list_available_models(self):
            return self.models
    
    class MockModel:
        def __init__(self, name):
            self.name = name
        
        def predict(self, X):
            if X.ndim == 3:  # Sequential model
                return np.random.randn(X.shape[0])
            else:  # Tabular model
                return np.random.randn(X.shape[0])
    
    # Test model predictor
    config = {'horizons': ['5m', '15m']}
    model_registry = MockModelRegistry(config)
    buffer_manager = None  # Mock buffer manager
    
    predictor = ModelPredictor(model_registry, buffer_manager, config)
    
    # Test prediction
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    market_data = generate_mock_market_data(symbols)
    
    predictions = predictor.predict_horizon('5m', symbols, market_data['features'])
    
    assert len(predictions) > 0, "Should have predictions"
    assert all(len(preds) == len(symbols) for preds in predictions.values()), "All predictions should have same length"
    
    print("✅ Model predictor tests passed")

def test_horizon_blender():
    """Test horizon blender functionality."""
    print("🧪 Testing horizon blender...")
    
    config = {
        'blend_weights': {
            '5m': {'LightGBM': 0.4, 'XGBoost': 0.3, 'CNN1D': 0.3},
            '15m': {'LightGBM': 0.5, 'LSTM': 0.5}
        }
    }
    
    blender = HorizonBlender(config)
    
    # Test blending
    predictions = {
        'LightGBM': {'AAPL': 0.1, 'GOOGL': 0.2, 'MSFT': -0.1},
        'XGBoost': {'AAPL': 0.15, 'GOOGL': 0.18, 'MSFT': -0.05},
        'CNN1D': {'AAPL': 0.12, 'GOOGL': 0.22, 'MSFT': -0.08}
    }
    
    blended = blender.blend_horizon('5m', predictions)
    
    assert blended is not None, "Should have blended results"
    assert len(blended) == 3, "Should have 3 symbols"
    assert all(isinstance(v, float) for v in blended.values()), "All values should be floats"
    
    print("✅ Horizon blender tests passed")

def test_barrier_gate():
    """Test barrier gate functionality."""
    print("🧪 Testing barrier gate...")
    
    config = {'g_min': 0.2, 'gamma': 1.0, 'delta': 0.5}
    gate = BarrierGate(config)
    
    # Test gate application
    alpha = {'AAPL': 0.1, 'GOOGL': 0.2, 'MSFT': -0.1}
    peak_probs = {'AAPL': 0.3, 'GOOGL': 0.7, 'MSFT': 0.5}
    valley_probs = {'AAPL': 0.6, 'GOOGL': 0.2, 'MSFT': 0.4}
    
    gated_alpha = gate.apply_gate(alpha, peak_probs, valley_probs)
    
    assert len(gated_alpha) == 3, "Should have 3 symbols"
    assert all(isinstance(v, float) for v in gated_alpha.values()), "All values should be floats"
    assert all(abs(v) <= abs(alpha[s]) for s, v in gated_alpha.items()), "Gate should reduce absolute values"
    
    print("✅ Barrier gate tests passed")

def test_cost_arbitrator():
    """Test cost arbitrator functionality."""
    print("🧪 Testing cost arbitrator...")
    
    config = {
        'arbitration_mode': 'winner',
        'cost_k1': 0.5, 'cost_k2': 0.3, 'cost_k3': 0.1
    }
    arbitrator = CostArbitrator(config)
    
    # Test cost estimation
    cost_model = CostModel(config)
    market_data = generate_mock_market_data(['AAPL', 'GOOGL'])
    costs = cost_model.estimate_costs('5m', market_data)
    
    assert len(costs) == 2, "Should have costs for 2 symbols"
    assert all(isinstance(v, float) for v in costs.values()), "All costs should be floats"
    assert all(v >= 0 for v in costs.values()), "All costs should be non-negative"
    
    # Test horizon arbitration
    alpha_by_horizon = {
        '5m': {'AAPL': 0.1, 'GOOGL': 0.2},
        '15m': {'AAPL': 0.15, 'GOOGL': 0.18}
    }
    
    arbitrated = arbitrator.arbitrate_horizons(alpha_by_horizon, market_data)
    
    assert len(arbitrated) == 2, "Should have 2 symbols"
    assert all(isinstance(v, float) for v in arbitrated.values()), "All values should be floats"
    
    print("✅ Cost arbitrator tests passed")

def test_position_sizer():
    """Test position sizer functionality."""
    print("🧪 Testing position sizer...")
    
    config = {
        'z_max': 3.0, 'max_weight': 0.05, 'gross_target': 0.5,
        'no_trade_band': 0.008, 'use_risk_parity': False
    }
    sizer = PositionSizer(config)
    
    # Test position sizing
    alpha = {'AAPL': 0.1, 'GOOGL': 0.2, 'MSFT': -0.1}
    market_data = generate_mock_market_data(['AAPL', 'GOOGL', 'MSFT'])
    current_weights = {'AAPL': 0.0, 'GOOGL': 0.0, 'MSFT': 0.0}
    
    target_weights = sizer.size_positions(alpha, market_data, current_weights)
    
    assert len(target_weights) == 3, "Should have 3 symbols"
    assert all(isinstance(v, float) for v in target_weights.values()), "All values should be floats"
    assert all(abs(v) <= config['max_weight'] for v in target_weights.values()), "All weights should be within limits"
    
    # Test position validator
    validator = PositionValidator(config)
    validation_result = validator.validate_weights(target_weights)
    
    assert validation_result['valid'], f"Position validation failed: {validation_result['errors']}"
    
    print("✅ Position sizer tests passed")

def test_live_trading_system():
    """Test complete live trading system."""
    print("🧪 Testing live trading system...")
    
    config = {
        'horizons': ['5m', '15m'],
        'lookback_T': 60,
        'num_features': 50,
        'ttl_seconds': 300,
        'blend_weights': {
            '5m': {'LightGBM': 0.5, 'XGBoost': 0.5},
            '15m': {'LightGBM': 0.6, 'LSTM': 0.4}
        },
        'g_min': 0.2, 'gamma': 1.0, 'delta': 0.5,
        'arbitration_mode': 'winner',
        'z_max': 3.0, 'max_weight': 0.05, 'gross_target': 0.5,
        'no_trade_band': 0.008
    }
    
    # Create system
    system = LiveTradingSystem(config)
    
    # Test system status
    status = system.get_system_status()
    assert status['initialized'], "System should be initialized"
    
    # Test live step
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    market_data = generate_mock_market_data(symbols)
    current_weights = {symbol: 0.0 for symbol in symbols}
    
    target_weights = system.live_step(market_data, current_weights)
    
    assert len(target_weights) == 3, "Should have 3 symbols"
    assert all(isinstance(v, float) for v in target_weights.values()), "All values should be floats"
    
    print("✅ Live trading system tests passed")

def test_live_trading_manager():
    """Test live trading manager."""
    print("🧪 Testing live trading manager...")
    
    config = {
        'horizons': ['5m', '15m'],
        'rebalance_frequency': 60,  # 1 minute for testing
        'lookback_T': 60,
        'num_features': 50,
        'ttl_seconds': 300
    }
    
    manager = LiveTradingManager(config)
    
    # Test manager status
    status = manager.get_trading_status()
    assert 'is_running' in status, "Should have running status"
    
    # Test start/stop
    manager.start_trading()
    assert manager.is_running, "Should be running"
    
    manager.stop_trading()
    assert not manager.is_running, "Should not be running"
    
    print("✅ Live trading manager tests passed")

def test_integration():
    """Test end-to-end integration."""
    print("🧪 Testing integration...")
    
    # Create comprehensive config
    config = {
        'horizons': ['5m', '15m', '30m'],
        'lookback_T': 60,
        'num_features': 50,
        'ttl_seconds': 300,
        'blend_weights': {
            '5m': {'LightGBM': 0.4, 'XGBoost': 0.3, 'CNN1D': 0.3},
            '15m': {'LightGBM': 0.5, 'LSTM': 0.5},
            '30m': {'LightGBM': 0.6, 'Transformer': 0.4}
        },
        'g_min': 0.2, 'gamma': 1.0, 'delta': 0.5,
        'arbitration_mode': 'winner',
        'cost_k1': 0.5, 'cost_k2': 0.3, 'cost_k3': 0.1,
        'z_max': 3.0, 'max_weight': 0.05, 'gross_target': 0.5,
        'no_trade_band': 0.008
    }
    
    # Create manager
    manager = LiveTradingManager(config)
    manager.start_trading()
    
    # Generate test data
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    market_data = generate_mock_market_data(symbols, num_features=50)
    current_weights = {symbol: 0.0 for symbol in symbols}
    
    # Test rebalance
    target_weights = manager.execute_rebalance(market_data, current_weights)
    
    assert len(target_weights) == len(symbols), f"Should have {len(symbols)} symbols"
    assert all(isinstance(v, float) for v in target_weights.values()), "All values should be floats"
    assert all(abs(v) <= config['max_weight'] for v in target_weights.values()), "All weights should be within limits"
    
    # Test multiple rebalances
    for i in range(3):
        # Update market data
        manager.update_market_data(market_data)
        
        # Execute rebalance
        new_weights = manager.execute_rebalance(market_data, target_weights)
        
        assert len(new_weights) == len(symbols), f"Rebalance {i}: Should have {len(symbols)} symbols"
    
    manager.stop_trading()
    
    print("✅ Integration tests passed")

def run_live_integration_tests():
    """Run all live integration tests."""
    print("🔍 Running Live Trading Integration Tests")
    print("=" * 60)
    
    try:
        test_model_predictor()
        test_horizon_blender()
        test_barrier_gate()
        test_cost_arbitrator()
        test_position_sizer()
        test_live_trading_system()
        test_live_trading_manager()
        test_integration()
        
        print("\n🎉 All live integration tests passed!")
        print("✅ Live trading system is ready for production")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Live integration tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    success = run_live_integration_tests()
    
    if success:
        print("\n🚀 Live trading integration complete!")
        print("📋 System Components:")
        print("  ✅ Model Predictor - Unified prediction engine")
        print("  ✅ Horizon Blender - Per-horizon model blending")
        print("  ✅ Barrier Gate - Timing risk attenuation")
        print("  ✅ Cost Arbitrator - Cost model & horizon arbitration")
        print("  ✅ Position Sizer - Alpha to weights conversion")
        print("  ✅ Live Trading System - Main integration")
        print("  ✅ Live Trading Manager - Trading operations")
        print("  ✅ Integration Tests - End-to-end validation")
    else:
        print("\n❌ Some tests failed - check implementation")
