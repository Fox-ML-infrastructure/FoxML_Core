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
Test IBKR Trading System with Daily Models
This script tests the IBKR stack using your current daily models before switching to intraday.
"""


import os
import sys
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

# Import IBKR components
from IBKR_trading.live_trading.ibkr_rebalancer_integration import IBKRRebalancerIntegration
from IBKR_trading.live_trading.rotation_engine import RotationEngine, RotationConfig
from IBKR_trading.live_trading.intraday_rebalancer import IntradayRebalancer, RebalanceConfig

class DailyModelTester:
    """
    Test the IBKR trading system with daily models.
    """
    
    def __init__(self, config_path: str = "IBKR_trading/config/ibkr_daily_test.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        self.logger = self.setup_logging()
        
        # Initialize components
        self.rebalancer = None
        self.rotation_engine = None
        self.integration = None
        
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
        fh = logging.FileHandler("logs/daily_model_test.log")
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
    
    def load_daily_models(self) -> dict:
        """
        Load your existing daily models.
        This is a placeholder - replace with your actual model loading logic.
        """
        self.logger.info("Loading daily models...")
        
        # Placeholder for your daily models
        # Replace this with your actual model loading
        models = {
            "momentum": {
                "model": "dummy_momentum_model",
                "features": ["returns", "volatility", "volume"],
                "horizon": 1
            },
            "mean_reversion": {
                "model": "dummy_mean_reversion_model", 
                "features": ["returns", "volatility"],
                "horizon": 1
            },
            "volatility": {
                "model": "dummy_volatility_model",
                "features": ["volatility", "volume"],
                "horizon": 1
            },
            "volume": {
                "model": "dummy_volume_model",
                "features": ["volume", "returns"],
                "horizon": 1
            }
        }
        
        self.logger.info(f"Loaded {len(models)} daily models")
        return models
    
    def get_daily_data(self, symbols: list, lookback_days: int = 252) -> pd.DataFrame:
        """
        Get daily data for testing.
        This is a placeholder - replace with your actual data loading.
        """
        self.logger.info(f"Loading daily data for {len(symbols)} symbols...")
        
        # Placeholder for your daily data loading
        # Replace this with your actual data loading logic
        dates = pd.date_range(end=datetime.now(), periods=lookback_days, freq='D')
        
        data = []
        for symbol in symbols:
            for date in dates:
                data.append({
                    'symbol': symbol,
                    'date': date,
                    'open': 100 + np.random.normal(0, 5),
                    'high': 105 + np.random.normal(0, 3),
                    'low': 95 + np.random.normal(0, 3),
                    'close': 100 + np.random.normal(0, 5),
                    'volume': 1000000 + np.random.randint(0, 500000),
                    'returns': np.random.normal(0, 0.02),
                    'volatility': np.random.uniform(0.01, 0.05),
                    'momentum': np.random.normal(0, 0.1),
                    'mean_reversion': np.random.normal(0, 0.1)
                })
        
        df = pd.DataFrame(data)
        self.logger.info(f"Loaded {len(df)} daily records")
        return df
    
    def generate_daily_signals(self, models: dict, data: pd.DataFrame, symbols: list) -> dict:
        """
        Generate daily signals using your models.
        This is a placeholder - replace with your actual signal generation.
        """
        self.logger.info("Generating daily signals...")
        
        signals = {}
        for symbol in symbols:
            symbol_data = data[data['symbol'] == symbol].copy()
            if len(symbol_data) == 0:
                continue
            
            # Placeholder signal generation
            # Replace this with your actual model inference
            signal = {
                'momentum': np.random.normal(0, 0.1),
                'mean_reversion': np.random.normal(0, 0.1),
                'volatility': np.random.normal(0, 0.1),
                'volume': np.random.normal(0, 0.1)
            }
            
            # Aggregate signals (replace with your actual aggregation)
            aggregated_score = np.mean(list(signal.values()))
            signals[symbol] = {
                'score': aggregated_score,
                'signals': signal,
                'volatility': symbol_data['volatility'].iloc[-1],
                'volume': symbol_data['volume'].iloc[-1]
            }
        
        self.logger.info(f"Generated signals for {len(signals)} symbols")
        return signals
    
    def initialize_components(self):
        """Initialize IBKR trading components."""
        self.logger.info("Initializing IBKR components...")
        
        # Initialize rebalancer
        rebalance_config = RebalanceConfig(
            portfolio_vol_target=self.config['portfolio']['vol_target'],
            no_trade_threshold=self.config['rebalancing']['no_trade_threshold'],
            max_participation=self.config['execution']['participation_cap']
        )
        self.rebalancer = IntradayRebalancer(rebalance_config)
        
        # Initialize rotation engine
        rotation_config = RotationConfig(
            K=self.config['rebalancing']['rotation']['K'],
            K_buy=self.config['rebalancing']['rotation']['K_buy'],
            K_sell=self.config['rebalancing']['rotation']['K_sell'],
            z_keep=self.config['rebalancing']['rotation']['z_keep'],
            z_cut=self.config['rebalancing']['rotation']['z_cut'],
            delta_z_min=self.config['rebalancing']['rotation']['delta_z_min']
        )
        self.rotation_engine = RotationEngine(rotation_config)
        
        # Initialize integration
        self.integration = IBKRRebalancerIntegration(self.config)
        
        self.logger.info("IBKR components initialized")
    
    def run_daily_test(self, symbols: list = None, test_days: int = 30):
        """
        Run a daily model test.
        
        Args:
            symbols: List of symbols to test (default: top 20 S&P 500)
            test_days: Number of days to test
        """
        if symbols is None:
            symbols = [
                "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "ADBE", "CRM",
                "ORCL", "INTC", "AMD", "PYPL", "UBER", "SQ", "ZM", "SHOP", "ROKU", "PTON"
            ]
        
        self.logger.info(f"Starting daily model test with {len(symbols)} symbols for {test_days} days")
        
        # Load models and data
        models = self.load_daily_models()
        data = self.get_daily_data(symbols, test_days + 50)  # Extra data for features
        
        # Initialize components
        self.initialize_components()
        
        # Run simulation
        results = []
        current_weights = {symbol: 0.0 for symbol in symbols}
        
        for day in range(test_days):
            self.logger.info(f"Processing day {day + 1}/{test_days}")
            
            # Get data for this day
            day_data = data[data['date'] <= data['date'].max() - timedelta(days=test_days-day-1)]
            
            # Generate signals
            signals = self.generate_daily_signals(models, day_data, symbols)
            
            if not signals:
                continue
            
            # Prepare data for rebalancing
            scores = {symbol: signals[symbol]['score'] for symbol in signals}
            returns = {symbol: signals[symbol]['signals']['momentum'] for symbol in signals}  # Use momentum as return proxy
            volatilities = {symbol: signals[symbol]['volatility'] for symbol in signals}
            
            # Calculate correlation matrix (simplified)
            correlation_matrix = np.eye(len(symbols))
            
            # Calculate costs (simplified)
            costs = {symbol: 0.001 for symbol in symbols}  # 10 bps cost
            
            # Run rebalancing
            try:
                target_weights = self.integration.run_rebalancing_cycle(
                    symbols=symbols,
                    horizons=[1]  # Daily horizon
                )
                
                # Calculate performance metrics
                portfolio_return = sum(current_weights.get(symbol, 0) * returns.get(symbol, 0) for symbol in symbols)
                turnover = sum(abs(target_weights.get(symbol, 0) - current_weights.get(symbol, 0)) for symbol in symbols)
                
                results.append({
                    'day': day + 1,
                    'portfolio_return': portfolio_return,
                    'turnover': turnover,
                    'target_weights': target_weights.copy(),
                    'current_weights': current_weights.copy()
                })
                
                # Update current weights
                current_weights.update(target_weights)
                
                self.logger.info(f"Day {day + 1}: Return={portfolio_return:.4f}, Turnover={turnover:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error on day {day + 1}: {e}")
                continue
        
        # Calculate final results
        total_return = sum(r['portfolio_return'] for r in results)
        avg_turnover = np.mean([r['turnover'] for r in results])
        
        self.logger.info(f"Daily model test completed:")
        self.logger.info(f"Total return: {total_return:.4f}")
        self.logger.info(f"Average daily turnover: {avg_turnover:.4f}")
        
        return results
    
    def switch_to_intraday(self):
        """
        Switch configuration to intraday models.
        Call this when your intraday models are trained.
        """
        self.logger.info("Switching to intraday models...")
        
        # Update configuration
        self.config['models']['daily']['enabled'] = False
        self.config['models']['intraday']['enabled'] = True
        self.config['model_switching']['current_mode'] = 'intraday'
        
        # Update rebalancing schedule
        self.config['rebalancing']['schedule'] = ["09:35", "10:30", "12:00", "14:30", "15:50"]
        
        # Update horizons
        self.config['models']['intraday']['horizons'] = [5, 10, 15, 30, 60]
        
        # Save updated config
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        self.logger.info("Configuration updated for intraday models")
        self.logger.info("Ready to use intraday models when they're trained!")

def main():
    """Main function to run the daily model test."""
    print("🚀 IBKR Daily Model Tester")
    print("=" * 50)
    
    # Initialize tester
    tester = DailyModelTester()
    
    # Run test
    print("\n📊 Running daily model test...")
    results = tester.run_daily_test()
    
    # Show results
    print("\n📈 Test Results:")
    print(f"Total return: {sum(r['portfolio_return'] for r in results):.4f}")
    print(f"Average turnover: {np.mean([r['turnover'] for r in results]):.4f}")
    
    # Show how to switch to intraday
    print("\n🔄 To switch to intraday models when ready:")
    print("tester.switch_to_intraday()")
    
    return results

if __name__ == "__main__":
    results = main()
