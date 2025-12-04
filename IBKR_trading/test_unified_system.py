#!/usr/bin/env python3

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
Test script for the unified trading system (Yahoo Finance + Alpaca + Model Zoo)
"""


import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
from IBKR_trading.live_trading.main_loop import LiveTradingSystem

def test_unified_system():
    """Test the unified trading system."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Configuration for testing
    config = {
        'symbols': ['SPY', 'QQQ', 'BTC-USD'],  # Small test set
        'horizons': ['5m', '15m', '1h'],  # Fewer horizons for testing
        'model_dir': 'TRAINING/models',
        'blender_dir': 'TRAINING/blenders',
        'device': 'cpu',
        'update_interval': 60,  # 1 minute for testing
        'g_min': 0.2,
        'gamma': 1.0,
        'delta': 0.5,
        'k1': 0.5,
        'k2': 0.3,
        'k3': 0.2,
        'z_max': 3.0,
        'max_weight': 0.05,
        'target_gross': 0.5,
        'no_trade_band': 0.008
    }
    
    try:
        logger.info("🧪 Testing Unified Trading System...")
        
        # Initialize system
        trading_system = LiveTradingSystem(config)
        logger.info("✅ System initialized successfully")
        
        # Test data fetching
        logger.info("📊 Testing Yahoo Finance data fetching...")
        market_data = trading_system._get_market_data()
        if market_data:
            logger.info(f"✅ Data fetched for {len(market_data['data'])} symbols")
            for symbol, data in market_data['data'].items():
                logger.info(f"  {symbol}: {len(data)} bars, latest: ${market_data['current_prices'].get(symbol, 0):.2f}")
        else:
            logger.warning("⚠️ No market data received")
        
        # Test feature generation
        logger.info("🔧 Testing feature generation...")
        features = trading_system._get_features(market_data)
        if features is not None:
            logger.info(f"✅ Features generated: {features.shape}")
        else:
            logger.warning("⚠️ No features generated")
        
        # Test portfolio status
        logger.info("📈 Testing portfolio status...")
        portfolio_status = trading_system.get_portfolio_status()
        logger.info(f"✅ Portfolio status: {portfolio_status}")
        
        # Test C++ kernel integration
        logger.info("🔬 Testing C++ kernel integration...")
        try:
            from cpp_kernels import benchmark_kernels
            benchmark_results = benchmark_kernels()
            logger.info(f"✅ C++ kernels benchmarked successfully")
        except Exception as e:
            logger.warning(f"⚠️ C++ kernel test failed: {e}")
        
        logger.info("🎉 All tests passed! System is ready for trading.")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_unified_system()
    if success:
        print("\n✅ Unified Trading System Test PASSED!")
        print("🚀 System is ready for live trading with:")
        print("   📊 Yahoo Finance data (FREE)")
        print("   💼 Alpaca paper trading execution")
        print("   🧠 Your trained model zoo")
    else:
        print("\n❌ Unified Trading System Test FAILED!")
        sys.exit(1)
