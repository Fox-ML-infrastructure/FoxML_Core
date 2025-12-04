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
IBKR Integration Testing
Test IBKR connection and integration with the trading stack.
"""


import os
import sys
import time
import yaml
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

class IBKRIntegrationTester:
    """
    Test IBKR integration and connection.
    """
    
    def __init__(self, config_path: str = "IBKR_trading/config/ibkr_daily_test.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        self.logger = self.setup_logging()
        self.test_results = {}
        
    def load_config(self) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}
    
    def setup_logging(self) -> logging.Logger:
        """Setup logging for the tester."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # File handler
        fh = logging.FileHandler("logs/ibkr_integration_test.log")
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def test_ibkr_connection(self) -> bool:
        """Test IBKR connection."""
        self.logger.info("🔌 Testing IBKR connection...")
        
        try:
            # Check if IBKR TWS/Gateway is running
            import socket
            
            host = self.config.get('ibkr', {}).get('host', '127.0.0.1')
            port = self.config.get('ibkr', {}).get('port', 7497)
            
            # Test connection
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                self.logger.info(f"✅ IBKR connection successful: {host}:{port}")
                return True
            else:
                self.logger.error(f"❌ IBKR connection failed: {host}:{port}")
                self.logger.error("Make sure IBKR TWS or Gateway is running")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ IBKR connection test failed: {e}")
            return False
    
    def test_ibkr_api_import(self) -> bool:
        """Test if IBKR API can be imported."""
        self.logger.info("📦 Testing IBKR API import...")
        
        try:
            # Try to import ib_insync
            import ib_insync
            self.logger.info("✅ ib_insync imported successfully")
            
            # Try to create IB object
            from ib_insync import IB
            ib = IB()
            self.logger.info("✅ IB object created successfully")
            
            return True
            
        except ImportError as e:
            self.logger.error(f"❌ IBKR API import failed: {e}")
            self.logger.error("Install ib_insync: pip install ib_insync")
            return False
        except Exception as e:
            self.logger.error(f"❌ IBKR API test failed: {e}")
            return False
    
    def test_market_data_streaming(self) -> bool:
        """Test market data streaming."""
        self.logger.info("📊 Testing market data streaming...")
        
        try:
            from ib_insync import IB, Stock, util
            
            # Create IB connection
            ib = IB()
            
            # Connect to IBKR
            try:
                ib.connect('127.0.0.1', 7497, clientId=1)
                self.logger.info("✅ Connected to IBKR")
            except Exception as e:
                self.logger.error(f"❌ Failed to connect to IBKR: {e}")
                return False
            
            # Test market data for a simple symbol
            try:
                # Create contract
                contract = Stock('AAPL', 'SMART', 'USD')
                
                # Request market data
                ib.reqMktData(contract)
                
                # Wait for data
                time.sleep(2)
                
                # Check if we got data
                ticker = ib.ticker(contract)
                if ticker and ticker.last:
                    self.logger.info(f"✅ Market data received: AAPL = ${ticker.last}")
                    return True
                else:
                    self.logger.warning("⚠️ No market data received")
                    return False
                    
            except Exception as e:
                self.logger.error(f"❌ Market data streaming failed: {e}")
                return False
            finally:
                # Disconnect
                ib.disconnect()
                
        except Exception as e:
            self.logger.error(f"❌ Market data streaming test failed: {e}")
            return False
    
    def test_order_placement(self) -> bool:
        """Test order placement (paper trading)."""
        self.logger.info("📝 Testing order placement...")
        
        try:
            from ib_insync import IB, Stock, MarketOrder
            
            # Create IB connection
            ib = IB()
            
            # Connect to IBKR
            try:
                ib.connect('127.0.0.1', 7497, clientId=1)
                self.logger.info("✅ Connected to IBKR")
            except Exception as e:
                self.logger.error(f"❌ Failed to connect to IBKR: {e}")
                return False
            
            try:
                # Create contract
                contract = Stock('AAPL', 'SMART', 'USD')
                
                # Create market order (small quantity for testing)
                order = MarketOrder('BUY', 1)  # 1 share
                
                # Place order
                trade = ib.placeOrder(contract, order)
                
                # Wait for order status
                time.sleep(2)
                
                # Check order status
                if trade.orderStatus.status in ['Submitted', 'Filled']:
                    self.logger.info(f"✅ Order placed successfully: {trade.orderStatus.status}")
                    return True
                else:
                    self.logger.warning(f"⚠️ Order status: {trade.orderStatus.status}")
                    return False
                    
            except Exception as e:
                self.logger.error(f"❌ Order placement failed: {e}")
                return False
            finally:
                # Disconnect
                ib.disconnect()
                
        except Exception as e:
            self.logger.error(f"❌ Order placement test failed: {e}")
            return False
    
    def test_position_tracking(self) -> bool:
        """Test position tracking."""
        self.logger.info("📈 Testing position tracking...")
        
        try:
            from ib_insync import IB
            
            # Create IB connection
            ib = IB()
            
            # Connect to IBKR
            try:
                ib.connect('127.0.0.1', 7497, clientId=1)
                self.logger.info("✅ Connected to IBKR")
            except Exception as e:
                self.logger.error(f"❌ Failed to connect to IBKR: {e}")
                return False
            
            try:
                # Get positions
                positions = ib.positions()
                
                if positions:
                    self.logger.info(f"✅ Positions retrieved: {len(positions)} positions")
                    for pos in positions[:5]:  # Show first 5 positions
                        self.logger.info(f"   {pos.contract.symbol}: {pos.position} @ ${pos.avgCost}")
                else:
                    self.logger.info("✅ No positions found (empty portfolio)")
                
                return True
                
            except Exception as e:
                self.logger.error(f"❌ Position tracking failed: {e}")
                return False
            finally:
                # Disconnect
                ib.disconnect()
                
        except Exception as e:
            self.logger.error(f"❌ Position tracking test failed: {e}")
            return False
    
    def test_risk_management(self) -> bool:
        """Test risk management components."""
        self.logger.info("🛡️ Testing risk management...")
        
        try:
            # Test position limits
            max_positions = self.config.get('portfolio', {}).get('max_positions', 20)
            per_name_cap = self.config.get('portfolio', {}).get('per_name_cap', 0.05)
            
            self.logger.info(f"✅ Position limits: max={max_positions}, per_name={per_name_cap}")
            
            # Test drawdown limits
            max_drawdown = self.config.get('portfolio', {}).get('max_drawdown', 0.20)
            daily_loss_limit = self.config.get('portfolio', {}).get('daily_loss_limit', 0.03)
            
            self.logger.info(f"✅ Drawdown limits: max={max_drawdown}, daily={daily_loss_limit}")
            
            # Test kill switches
            kill_switches = self.config.get('safety', {}).get('kill_switches', {})
            if kill_switches:
                self.logger.info(f"✅ Kill switches configured: {list(kill_switches.keys())}")
            else:
                self.logger.warning("⚠️ No kill switches configured")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Risk management test failed: {e}")
            return False
    
    def test_model_compatibility(self) -> bool:
        """Test model compatibility."""
        self.logger.info("🧠 Testing model compatibility...")
        
        try:
            # Check model paths
            model_path = self.config.get('models', {}).get('daily', {}).get('model_path', 'models/daily_models/')
            
            if os.path.exists(model_path):
                self.logger.info(f"✅ Model path exists: {model_path}")
            else:
                self.logger.warning(f"⚠️ Model path not found: {model_path}")
            
            # Check model families
            families = self.config.get('models', {}).get('daily', {}).get('families', [])
            self.logger.info(f"✅ Model families: {families}")
            
            # Check horizons
            horizons = self.config.get('models', {}).get('daily', {}).get('horizons', [])
            self.logger.info(f"✅ Horizons: {horizons}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Model compatibility test failed: {e}")
            return False
    
    def test_configuration(self) -> bool:
        """Test configuration validity."""
        self.logger.info("⚙️ Testing configuration...")
        
        try:
            # Check required sections
            required_sections = ['models', 'portfolio', 'rebalancing', 'execution', 'safety', 'ibkr']
            
            for section in required_sections:
                if section in self.config:
                    self.logger.info(f"✅ {section} section present")
                else:
                    self.logger.error(f"❌ {section} section missing")
                    return False
            
            # Check critical settings
            critical_settings = [
                'portfolio.vol_target',
                'portfolio.max_positions',
                'rebalancing.schedule',
                'execution.participation_cap',
                'ibkr.host',
                'ibkr.port'
            ]
            
            for setting in critical_settings:
                keys = setting.split('.')
                value = self.config
                for key in keys:
                    if key in value:
                        value = value[key]
                    else:
                        self.logger.error(f"❌ Missing setting: {setting}")
                        return False
                
                self.logger.info(f"✅ {setting}: {value}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Configuration test failed: {e}")
            return False
    
    def run_all_tests(self) -> dict:
        """Run all IBKR integration tests."""
        self.logger.info("🚀 Starting IBKR integration testing...")
        
        tests = [
            ("Configuration", self.test_configuration),
            ("IBKR API Import", self.test_ibkr_api_import),
            ("IBKR Connection", self.test_ibkr_connection),
            ("Market Data Streaming", self.test_market_data_streaming),
            ("Order Placement", self.test_order_placement),
            ("Position Tracking", self.test_position_tracking),
            ("Risk Management", self.test_risk_management),
            ("Model Compatibility", self.test_model_compatibility)
        ]
        
        results = {}
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Running: {test_name}")
            self.logger.info(f"{'='*50}")
            
            try:
                result = test_func()
                results[test_name] = result
                if result:
                    passed += 1
                    self.logger.info(f"✅ {test_name}: PASSED")
                else:
                    self.logger.error(f"❌ {test_name}: FAILED")
            except Exception as e:
                self.logger.error(f"❌ {test_name}: ERROR - {e}")
                results[test_name] = False
        
        # Summary
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"TEST SUMMARY")
        self.logger.info(f"{'='*50}")
        self.logger.info(f"Passed: {passed}/{total}")
        self.logger.info(f"Success rate: {passed/total*100:.1f}%")
        
        if passed == total:
            self.logger.info("🎉 All IBKR integration tests PASSED!")
        else:
            self.logger.warning(f"⚠️ {total-passed} tests FAILED. Check logs for details.")
        
        return results

def main():
    """Main function to run IBKR integration tests."""
    print("🔌 IBKR Integration Testing")
    print("=" * 50)
    
    # Initialize tester
    tester = IBKRIntegrationTester()
    
    # Run all tests
    results = tester.run_all_tests()
    
    # Return results
    return results

if __name__ == "__main__":
    results = main()
