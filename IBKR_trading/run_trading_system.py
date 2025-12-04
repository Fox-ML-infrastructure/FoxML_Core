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
IBKR Trading System Runner

Main entry point for running the IBKR trading system.
This script orchestrates the complete trading pipeline.
"""


import sys
import os
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from IBKR_trading.live_trading.main_loop import LiveTradingSystem

def setup_logging():
    """Setup comprehensive logging for the trading system."""
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "ibkr_trading.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set specific loggers
    logging.getLogger("IBKR_trading").setLevel(logging.DEBUG)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)

def main():
    """Main entry point for IBKR trading system."""
    print("🚀 Starting IBKR Trading System...")
    print("=" * 50)
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Create configuration for unified trading system
        config = {
            'symbols': ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT', 'VTI', 'EFA', 'EEM', 'BTC-USD', 'ETH-USD'],
            'horizons': ['5m', '10m', '15m', '30m', '60m', '120m', '1d', '5d', '20d'],
            'model_dir': 'TRAINING/models',
            'blender_dir': 'TRAINING/blenders',
            'device': 'cpu',
            'update_interval': 300,  # 5 minutes
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
        
        logger.info("Initializing Unified Trading System (Yahoo Finance + Alpaca)...")
        
        # Initialize unified trading system
        trading_system = LiveTradingSystem(config)
        
        # Display system status
        print("📊 Unified Trading System")
        print("📈 Data: Yahoo Finance (FREE)")
        print("💼 Execution: Alpaca Paper Trading")
        print("🧠 Models: Your trained model zoo")
        print("=" * 50)
        
        # Start trading system
        logger.info("Starting main trading loop...")
        trading_system.start()
        
    except KeyboardInterrupt:
        print("\n🛑 Received keyboard interrupt, shutting down...")
        logger.info("Received keyboard interrupt, shutting down...")
        
    except Exception as e:
        print(f"❌ System error: {e}")
        logger.error(f"System error: {e}")
        sys.exit(1)
        
    finally:
        if 'trading_system' in locals():
            trading_system.stop()
        print("✅ IBKR trading system shutdown complete")

if __name__ == "__main__":
    main()
