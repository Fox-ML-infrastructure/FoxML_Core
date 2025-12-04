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
Crypto Model Training Script
============================

Trains all 20 model families on cryptocurrency data.
Supports 24/7 crypto markets with high-frequency data.
"""


# CRITICAL: Import determinism FIRST before any ML libraries
from TRAINING.common.determinism import set_global_determinism, stable_seed_from, seed_for, get_deterministic_params, log_determinism_info

# Set global determinism immediately - OPTIMIZED FOR PERFORMANCE
BASE_SEED = set_global_determinism(
    base_seed=42,
    threads=None,  # Auto-detect optimal thread count
    deterministic_algorithms=False,  # Allow parallel algorithms
    prefer_cpu_tree_train=False,  # Use GPU when available
    tf_on=True,  # Enable TensorFlow GPU
    strict_mode=False  # Allow optimizations
)

import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import sys
import os
import warnings
import joblib
from datetime import datetime, timedelta
import glob

# Polars optimization
DEFAULT_THREADS = str(max(1, (os.cpu_count() or 2) - 1))
USE_POLARS = os.getenv("USE_POLARS", "1") == "1"
if USE_POLARS:
    try:
        os.environ.setdefault("POLARS_MAX_THREADS", DEFAULT_THREADS)
        import polars as pl
        pl.enable_string_cache()
        print("🚀 Polars optimization enabled")
    except ImportError:
        USE_POLARS = False
        print("⚠️ Polars not available, using pandas")

# Import training components
from strategies.single_task import SingleTaskStrategy
from strategies.multi_task import MultiTaskStrategy
from strategies.cascade import CascadeStrategy
from models.factory import ModelFactory
from models.registry import ModelRegistry

# Import crypto data provider
sys.path.append(str(Path(__file__).parent.parent))
from scripts.data.providers.crypto_provider import CryptoDataProvider

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Create logs directory
Path('logs').mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/crypto_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CryptoTrainer:
    """Crypto model trainer with all 20 model families."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.crypto_provider = CryptoDataProvider()
        self.model_factory = ModelFactory()
        self.model_registry = ModelRegistry()
        
        # Crypto symbols
        self.crypto_symbols = config.get('crypto_symbols', [
            'BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'SOL-USD',
            'MATIC-USD', 'DOT-USD', 'AVAX-USD', 'ATOM-USD', 'LINK-USD'
        ])
        
        # Timeframes for crypto
        self.timeframes = config.get('timeframes', ['5m', '15m', '1h', '1d'])
        
        # Training parameters
        self.max_samples_per_symbol = config.get('max_samples_per_symbol', 1000)
        self.epochs = config.get('epochs', 50)
        self.batch_size = config.get('batch_size', 32)
        
        logger.info(f"🚀 Crypto trainer initialized with {len(self.crypto_symbols)} symbols")
    
    def load_crypto_data(self, symbol: str, timeframe: str = "1h") -> pd.DataFrame:
        """Load crypto data for a symbol from collected data."""
        try:
            # Try to load from collected data first
            data_file = Path(f"../crypto_data_2years/{symbol}_{timeframe}_5years.parquet")
            if data_file.exists():
                data = pd.read_parquet(data_file)
                logger.info(f"📊 Loaded {len(data)} crypto bars from {data_file}")
            else:
                # Fallback to live data
                logger.info(f"📊 No collected data for {symbol} ({timeframe}), fetching live data...")
                data = self.crypto_provider.get_crypto_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=self.max_samples_per_symbol
                )
            
            if len(data) == 0:
                logger.warning(f"No crypto data for {symbol} ({timeframe})")
                return pd.DataFrame()
            
            # Add crypto-specific features if not already present
            if 'Returns' not in data.columns:
                data = self.crypto_provider.get_crypto_features(data)
            
            # Add symbol and timeframe columns
            data['symbol'] = symbol
            data['timeframe'] = timeframe
            data['asset_type'] = 'crypto'
            
            logger.info(f"📊 Loaded {len(data)} crypto bars for {symbol} ({timeframe})")
            return data
            
        except Exception as e:
            logger.error(f"Error loading crypto data for {symbol}: {e}")
            return pd.DataFrame()
    
    def create_crypto_targets(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create crypto-specific targets."""
        if len(data) == 0:
            return data
        
        df = data.copy()
        
        # Forward returns (same as stocks)
        for horizon in [5, 10, 15, 30, 60, 120, 240, 480, 1440, 7200]:  # 5m to 5d
            df[f'fwd_ret_{horizon}m'] = df['Close'].shift(-horizon) / df['Close'] - 1
        
        # Crypto-specific targets
        # Volatility targets
        for window in [5, 10, 20, 60, 240]:  # 5m to 20h
            df[f'vol_{window}m'] = df['Returns'].rolling(window=window).std()
        
        # Momentum targets
        for window in [5, 10, 20, 60, 240]:
            df[f'mom_{window}m'] = df['Close'] / df['Close'].shift(window) - 1
        
        # Volume targets
        for window in [5, 10, 20, 60, 240]:
            df[f'vol_ratio_{window}m'] = df['Volume'] / df['Volume'].rolling(window=window).mean()
        
        # Crypto-specific barrier targets
        for horizon in [5, 10, 15, 30, 60, 120, 240, 480, 1440]:
            # Will peak/valley targets
            df[f'will_peak_{horizon}m'] = (df['Close'].shift(-horizon) > df['Close'].rolling(window=horizon).max()).astype(int)
            df[f'will_valley_{horizon}m'] = (df['Close'].shift(-horizon) < df['Close'].rolling(window=horizon).min()).astype(int)
            
            # Maximum drawdown/runup
            df[f'mdd_{horizon}m'] = df['Close'].rolling(window=horizon).min() / df['Close'] - 1
            df[f'mru_{horizon}m'] = df['Close'].rolling(window=horizon).max() / df['Close'] - 1
        
        # RSI targets
        df['rsi_14'] = self.crypto_provider._calculate_rsi(df['Close'], 14)
        df['rsi_30'] = self.crypto_provider._calculate_rsi(df['Close'], 30)
        
        # Crypto volatility regime
        df['vol_regime'] = (df['Volatility_20'] > df['Volatility_20'].rolling(window=100).quantile(0.8)).astype(int)
        
        logger.info(f"🎯 Created {len([col for col in df.columns if col.startswith(('fwd_ret_', 'vol_', 'mom_', 'will_', 'mdd_', 'mru_', 'rsi_', 'vol_regime'))])} crypto targets")
        return df
    
    def train_crypto_models(self, symbol: str, timeframe: str = "5m") -> Dict[str, Any]:
        """Train all models for a crypto symbol."""
        logger.info(f"🚀 Training crypto models for {symbol} ({timeframe})")
        
        # Load crypto data
        data = self.load_crypto_data(symbol, timeframe)
        if len(data) == 0:
            logger.warning(f"No data for {symbol} ({timeframe})")
            return {}
        
        # Create targets
        data = self.create_crypto_targets(data)
        
        # Get all available targets
        target_columns = [col for col in data.columns if col.startswith(('fwd_ret_', 'vol_', 'mom_', 'will_', 'mdd_', 'mru_', 'rsi_', 'vol_regime'))]
        
        if len(target_columns) == 0:
            logger.warning(f"No targets found for {symbol}")
            return {}
        
        logger.info(f"🎯 Found {len(target_columns)} crypto targets: {target_columns[:5]}...")
        
        # Train models for each target
        results = {}
        
        for target in target_columns:
            try:
                logger.info(f"🎯 Training models for {target}")
                
                # Prepare data
                feature_cols = [col for col in data.columns if col not in target_columns + ['symbol', 'timeframe', 'asset_type', 'timestamp']]
                X = data[feature_cols].fillna(0)
                y = data[target].fillna(0)
                
                # Remove rows with NaN targets
                valid_mask = ~(y.isna() | np.isinf(y))
                X = X[valid_mask]
                y = y[valid_mask]
                
                if len(X) == 0:
                    logger.warning(f"No valid data for {target}")
                    continue
                
                # Train single-task models
                strategy = SingleTaskStrategy({})
                strategy.train_models_for_target(X, y, target, symbol, timeframe)
                
                results[target] = {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'target': target,
                    'samples': len(X),
                    'features': len(X.columns),
                    'status': 'success'
                }
                
                logger.info(f"✅ {target} trained successfully")
                
            except Exception as e:
                logger.error(f"Error training {target}: {e}")
                results[target] = {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'target': target,
                    'status': 'error',
                    'error': str(e)
                }
        
        return results
    
    def train_all_crypto_models(self) -> Dict[str, Any]:
        """Train models for all crypto symbols and timeframes."""
        logger.info("🚀 Starting crypto model training...")
        
        all_results = {}
        
        for symbol in self.crypto_symbols:
            logger.info(f"\n{'='*50}")
            logger.info(f"📊 Training {symbol}")
            logger.info(f"{'='*50}")
            
            symbol_results = {}
            
            for timeframe in self.timeframes:
                try:
                    logger.info(f"⏰ Timeframe: {timeframe}")
                    results = self.train_crypto_models(symbol, timeframe)
                    symbol_results[timeframe] = results
                    
                except Exception as e:
                    logger.error(f"Error training {symbol} ({timeframe}): {e}")
                    symbol_results[timeframe] = {'error': str(e)}
            
            all_results[symbol] = symbol_results
        
        # Summary
        total_models = sum(len(results) for symbol_results in all_results.values() 
                          for results in symbol_results.values() 
                          if isinstance(results, dict))
        
        logger.info(f"\n🎉 Crypto training completed!")
        logger.info(f"📊 Total models trained: {total_models}")
        logger.info(f"📊 Symbols: {len(self.crypto_symbols)}")
        logger.info(f"📊 Timeframes: {len(self.timeframes)}")
        
        return all_results

def main():
    """Main entry point for crypto training."""
    parser = argparse.ArgumentParser(description='Train crypto models')
    parser.add_argument('--crypto-symbols', nargs='+', 
                       default=['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'SOL-USD'],
                       help='Crypto symbols to train on')
    parser.add_argument('--timeframes', nargs='+', 
                       default=['5m', '15m', '1h', '1d'],
                       help='Timeframes to train on')
    parser.add_argument('--max-samples-per-symbol', type=int, default=1000,
                       help='Maximum samples per symbol')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs for neural networks')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--output-dir', type=str, default='crypto_models',
                       help='Output directory for models')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    Path('logs').mkdir(exist_ok=True)
    
    # Configuration
    config = {
        'crypto_symbols': args.crypto_symbols,
        'timeframes': args.timeframes,
        'max_samples_per_symbol': args.max_samples_per_symbol,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'output_dir': args.output_dir
    }
    
    # Initialize trainer
    trainer = CryptoTrainer(config)
    
    # Train models
    results = trainer.train_all_crypto_models()
    
    # Save results
    results_file = Path(args.output_dir) / 'crypto_training_results.json'
    with open(results_file, 'w') as f:
        import json
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"💾 Results saved to {results_file}")

if __name__ == "__main__":
    main()
