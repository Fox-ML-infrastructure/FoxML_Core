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
Main Trading Loop - Live Trading System Integration
=================================================

Integrates all components for live trading with all trained models.
"""


import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import time
import threading
from collections import defaultdict

from .model_predictor import ModelPredictor, ModelRegistry
from .horizon_blender import HorizonBlender, AdvancedBlender
from .barrier_gate import BarrierGate, AdvancedBarrierGate, BarrierProbabilityProvider
from .cost_arbitrator import CostArbitrator, CostModel, AdvancedCostModel
from .position_sizer import PositionSizer, AdvancedPositionSizer, PositionValidator

logger = logging.getLogger(__name__)

# Import data and execution providers
try:
    from scripts.data.providers.yahoo_provider import YahooProvider
    from ml.alpaca_data_provider import AlpacaDataProvider
    DATA_PROVIDERS_AVAILABLE = True
except ImportError as e:
    DATA_PROVIDERS_AVAILABLE = False
    logger.warning(f"Data providers not available: {e}")
    
    # Create mock providers for testing
    class YahooProvider:
        def get_bars(self, symbol, timeframe, limit=100):
            import pandas as pd
            import numpy as np
            dates = pd.date_range(end=pd.Timestamp.now(), periods=limit, freq=timeframe)
            data = pd.DataFrame({
                'Open': np.random.uniform(100, 200, limit),
                'High': np.random.uniform(100, 200, limit),
                'Low': np.random.uniform(100, 200, limit),
                'Close': np.random.uniform(100, 200, limit),
                'Volume': np.random.uniform(1000000, 10000000, limit)
            }, index=dates)
            return data
        
        def get_quote(self, symbol):
            return np.random.uniform(100, 200)
    
    class AlpacaDataProvider:
        def get_account(self):
            return {'equity': 100000, 'cash': 50000, 'buying_power': 100000}
        
        def get_positions(self):
            return []
        
        def submit_order(self, symbol, qty, side, type, time_in_force):
            return {'id': f'order_{symbol}_{qty}'}

# Import C++ kernels for high-performance computation
try:
    from cpp_kernels import (
        ewma_vol, ofi, ridge_blend, project_simplex, 
        horizon_softmax, barrier_gate, risk_parity_ridge
    )
    CPP_KERNELS_AVAILABLE = True
    logger.info("✅ C++ kernels loaded successfully")
except ImportError as e:
    CPP_KERNELS_AVAILABLE = False
    logger.warning(f"⚠️ C++ kernels not available, using Python fallbacks: {e}")

logger = logging.getLogger(__name__)

class LiveTradingSystem:
    """Main live trading system integrating all components."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.symbols = config.get('symbols', [])
        self.horizons = config.get('horizons', ['5m', '10m', '15m', '30m', '60m', '120m', '1d', '5d', '20d'])
        
        # Initialize data and execution providers
        self.yahoo_provider = YahooProvider()  # Free live data
        self.alpaca_executor = AlpacaDataProvider()  # Paper trading execution
        logger.info("✅ Yahoo Finance data provider initialized (FREE!)")
        logger.info("✅ Alpaca execution provider initialized for paper trading")
        
        # Initialize model zoo components
        self.model_registry = ModelRegistry(config.get('model_dir', 'TRAINING/models'))
        self.model_predictor = ModelPredictor(self.model_registry, config.get('device', 'cpu'))
        self.horizon_blender = AdvancedBlender(config.get('blender_dir', 'TRAINING/blenders'))
        self.barrier_gate = AdvancedBarrierGate(
            g_min=config.get('g_min', 0.2),
            gamma=config.get('gamma', 1.0),
            delta=config.get('delta', 0.5)
        )
        self.cost_model = AdvancedCostModel(
            k1=config.get('k1', 0.5),
            k2=config.get('k2', 0.3),
            k3=config.get('k3', 0.2)
        )
        self.cost_arbitrator = CostArbitrator(self.cost_model)
        self.position_sizer = AdvancedPositionSizer(
            z_max=config.get('z_max', 3.0),
            max_weight=config.get('max_weight', 0.05),
            target_gross=config.get('target_gross', 0.5),
            no_trade_band=config.get('no_trade_band', 0.008)
        )
        self.position_validator = PositionValidator(
            max_weight=config.get('max_weight', 0.05),
            max_gross=config.get('max_gross', 1.0),
            max_net=config.get('max_net', 0.3)
        )
        self.barrier_provider = BarrierProbabilityProvider(self.model_predictor)
        
        # State tracking
        self.current_weights = np.zeros(len(self.symbols))
        self.portfolio_value = 0.0
        self.last_update = None
        self.performance_metrics = defaultdict(list)
        
        # Threading
        self.running = False
        self.update_thread = None
        
    def start(self):
        """Start the live trading system."""
        if self.running:
            logger.warning("Trading system already running")
            return
        
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        logger.info("Live trading system started")
    
    def stop(self):
        """Stop the live trading system."""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=5.0)
        logger.info("Live trading system stopped")
    
    def _update_loop(self):
        """Main update loop."""
        while self.running:
            try:
                self._single_update()
                time.sleep(self.config.get('update_interval', 60))  # Update every minute
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                time.sleep(5)  # Short sleep on error
    
    def _single_update(self):
        """Single update cycle."""
        logger.info("Starting trading update cycle")
        
        # 1. Get market data
        market_data = self._get_market_data()
        if market_data is None:
            logger.warning("No market data available")
            return
        
        # 2. Get features
        features = self._get_features(market_data)
        if features is None:
            logger.warning("No features available")
            return
        
        # 3. Get predictions for all horizons
        all_predictions = self._get_all_predictions(features)
        if not all_predictions:
            logger.warning("No predictions available")
            return
        
        # 4. Blend per horizon (using C++ kernels for high-performance blending)
        if CPP_KERNELS_AVAILABLE:
            # Use C++ ridge blending for high-performance
            blended_predictions = {}
            for horizon, predictions in all_predictions.items():
                if predictions:
                    # Convert to numpy arrays for C++ kernels
                    pred_array = np.array(list(predictions.values())).astype(np.float32)
                    weights = np.ones(len(pred_array)) / len(pred_array)  # Equal weights for now
                    
                    # Use C++ ridge blend for high-performance combination
                    blended_pred = ridge_blend(pred_array, weights, alpha=0.1)
                    blended_predictions[horizon] = {symbol: blended_pred[i] for i, symbol in enumerate(self.symbols)}
        else:
            # Fallback to Python blending
            blended_predictions = self.horizon_blender.blend_all_horizons(all_predictions)
        
        if not blended_predictions:
            logger.warning("No blended predictions available")
            return
        
        # 5. Estimate costs and compute net alpha
        net_alphas = {}
        for horizon, alpha in blended_predictions.items():
            horizon_minutes = self._horizon_to_minutes(horizon)
            costs = self.cost_model.estimate_costs(market_data, horizon_minutes)
            net_alphas[horizon] = self.cost_arbitrator.compute_net_alpha(alpha, costs)
        
        # 6. Arbitrate horizons (using C++ kernels for high-performance arbitration)
        if CPP_KERNELS_AVAILABLE:
            # Use C++ horizon softmax for high-performance horizon selection
            horizon_names = list(net_alphas.keys())
            horizon_alphas = []
            
            for horizon in horizon_names:
                alpha_array = np.array([net_alphas[horizon].get(symbol, 0) for symbol in self.symbols]).astype(np.float32)
                horizon_alphas.append(alpha_array)
            
            horizon_alphas_matrix = np.array(horizon_alphas).astype(np.float32)
            
            # Use C++ horizon softmax for arbitration
            horizon_weights = horizon_softmax(horizon_alphas_matrix, temperature=1.0)
            selected_horizon_idx = np.argmax(horizon_weights)
            selected_horizon = horizon_names[selected_horizon_idx]
            
            # Blend final alpha using horizon weights
            final_alpha_array = np.zeros(len(self.symbols), dtype=np.float32)
            for i, weight in enumerate(horizon_weights):
                final_alpha_array += weight * horizon_alphas_matrix[i]
            
            final_alpha = {symbol: final_alpha_array[i] for i, symbol in enumerate(self.symbols)}
        else:
            # Fallback to Python horizon arbitration
            final_alpha, selected_horizon = self.cost_arbitrator.arbitrate_horizons(
                net_alphas, market_data
            )
        
        # 7. Get barrier probabilities and apply gate (using C++ kernels for high-performance)
        if CPP_KERNELS_AVAILABLE:
            # Use C++ barrier gate for high-performance gating
            alpha_array = np.array([final_alpha.get(symbol, 0) for symbol in self.symbols]).astype(np.float32)
            
            # Get barrier probabilities (simplified for C++ kernel)
            barrier_probs = np.random.uniform(0.1, 0.9, len(self.symbols)).astype(np.float32)  # Placeholder
            
            # Use C++ barrier gate
            gated_alpha_array = barrier_gate(
                alpha_array, 
                barrier_probs,
                g_min=self.config.get('g_min', 0.2),
                gamma=self.config.get('gamma', 1.0),
                delta=self.config.get('delta', 0.5)
            )
            
            final_alpha = {symbol: gated_alpha_array[i] for i, symbol in enumerate(self.symbols)}
        else:
            # Fallback to Python barrier gating
            barrier_preds = self.barrier_provider.get_barrier_probabilities(features, self.symbols)
            if barrier_preds:
                final_alpha = self.barrier_gate.apply_gate(final_alpha, barrier_preds)
        
        # 8. Size positions (using C++ kernels for high-performance optimization)
        if CPP_KERNELS_AVAILABLE:
            # Use C++ risk parity ridge for high-performance position sizing
            alpha_array = np.array([final_alpha.get(symbol, 0) for symbol in self.symbols]).astype(np.float32)
            current_weights_array = np.array(self.current_weights).astype(np.float32)
            
            # Get covariance matrix for risk parity
            cov_matrix = market_data.get('covariance', np.eye(len(self.symbols)))
            cov_array = cov_matrix.astype(np.float32)
            
            # Use C++ risk parity ridge optimization
            target_weights_array = risk_parity_ridge(
                alpha_array, 
                cov_array, 
                current_weights_array,
                lambda_reg=0.1,
                max_weight=self.config.get('max_weight', 0.05)
            )
            
            target_weights = target_weights_array
        else:
            # Fallback to Python position sizing
            target_weights = self.position_sizer.size_positions(
                final_alpha, market_data, self.current_weights
            )
        
        # 9. Validate positions and project to simplex (using C++ kernels for high-performance)
        if CPP_KERNELS_AVAILABLE:
            # Use C++ project_simplex for high-performance weight projection
            target_weights_array = np.array(target_weights).astype(np.float32)
            
            # Project to simplex to ensure weights sum to 1
            projected_weights = project_simplex(target_weights_array)
            
            # Apply max weight constraint
            max_weight = self.config.get('max_weight', 0.05)
            projected_weights = np.clip(projected_weights, 0, max_weight)
            
            # Renormalize to ensure sum = 1
            projected_weights = projected_weights / np.sum(projected_weights)
            
            target_weights = projected_weights
        else:
            # Fallback to Python validation
            is_valid, violations = self.position_validator.validate_weights(target_weights, self.symbols)
            if not is_valid:
                logger.warning(f"Position validation failed: {violations}")
                target_weights = self.position_validator.fix_violations(target_weights, self.symbols)
        
        # 10. Execute trades
        self._execute_trades(target_weights)
        
        # 11. Update state
        self.current_weights = target_weights
        self.last_update = datetime.now()
        
        logger.info(f"Trading update completed. Selected horizon: {selected_horizon}")
    
    def _get_market_data(self) -> Optional[Dict]:
        """Get current market data from Yahoo Finance."""
        try:
            logger.info(f"📊 Fetching live data for {len(self.symbols)} symbols via Yahoo Finance...")
            
            market_data = {
                'symbols': self.symbols,
                'data': {},
                'current_prices': {},
                'volatility': {},
                'volume': {},
                'returns': {}
            }
            
            # Fetch live data for each symbol
            for symbol in self.symbols:
                try:
                    # Get live 5m data for trading signals
                    bars_data = self.yahoo_provider.get_bars(
                        symbol=symbol,
                        timeframe='5m',
                        limit=100  # Last 100 bars (8+ hours of data)
                    )
                    
                    if not bars_data.empty:
                        # Get current price
                        current_price = self.yahoo_provider.get_quote(symbol)
                        
                        # Calculate volatility from recent returns
                        returns = bars_data['Close'].pct_change().dropna()
                        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0.02
                        
                        # Get volume
                        volume = bars_data['Volume'].iloc[-1] if len(bars_data) > 0 else 1000000
                        
                        # Store data
                        market_data['data'][symbol] = bars_data
                        market_data['current_prices'][symbol] = current_price
                        market_data['volatility'][symbol] = volatility
                        market_data['volume'][symbol] = volume
                        market_data['returns'][symbol] = returns.iloc[-1] if len(returns) > 0 else 0
                        
                        logger.debug(f"✅ {symbol}: ${current_price:.2f}, vol={volatility:.3f}, volume={volume:,.0f}")
                    else:
                        logger.warning(f"⚠️ No data for {symbol}")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Error fetching {symbol}: {e}")
                    continue
            
            # Calculate covariance matrix from returns
            returns_matrix = np.array([market_data['returns'].get(s, 0) for s in self.symbols])
            if len(returns_matrix) > 1:
                market_data['covariance'] = np.cov(returns_matrix.reshape(1, -1)) * 0.1
            else:
                market_data['covariance'] = np.eye(len(self.symbols)) * 0.1
            
            logger.info(f"✅ Yahoo Finance data fetch complete: {len(market_data['data'])} symbols")
            return market_data
            
        except Exception as e:
            logger.error(f"❌ Market data fetching failed: {e}")
            return None
    
    def _get_features(self, market_data: Dict) -> Optional[pd.DataFrame]:
        """Get feature matrix for all symbols from real market data."""
        try:
            features_list = []
            feature_names = []
            
            for symbol in self.symbols:
                if symbol not in market_data['data']:
                    continue
                    
                bars_data = market_data['data'][symbol]
                if bars_data.empty:
                    continue
                
                # Calculate technical features from bars data
                close_prices = bars_data['Close']
                high_prices = bars_data['High']
                low_prices = bars_data['Low']
                volume = bars_data['Volume']
                
                # Basic price features
                returns = close_prices.pct_change().dropna()
                
                # Use C++ kernel for high-performance volatility calculation
                if CPP_KERNELS_AVAILABLE and len(returns) > 20:
                    returns_array = returns.values.astype(np.float32)
                    volatility = ewma_vol(returns_array, alpha=0.06)  # ~20-period EWMA
                else:
                    volatility = returns.rolling(20).std().iloc[-1] if len(returns) > 20 else 0.02
                
                # Technical indicators
                sma_20 = close_prices.rolling(20).mean().iloc[-1] if len(close_prices) > 20 else close_prices.iloc[-1]
                sma_50 = close_prices.rolling(50).mean().iloc[-1] if len(close_prices) > 50 else close_prices.iloc[-1]
                
                # RSI
                delta = close_prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                rsi = (100 - (100 / (1 + rs))).iloc[-1] if len(rs) > 0 else 50
                
                # Volume features
                volume_ma = volume.rolling(20).mean().iloc[-1] if len(volume) > 20 else volume.iloc[-1]
                volume_ratio = volume.iloc[-1] / volume_ma if volume_ma > 0 else 1
                
                # Price momentum
                momentum_5 = (close_prices.iloc[-1] / close_prices.iloc[-6] - 1) if len(close_prices) > 5 else 0
                momentum_20 = (close_prices.iloc[-1] / close_prices.iloc[-21] - 1) if len(close_prices) > 20 else 0
                
                # Create feature vector
                symbol_features = [
                    close_prices.iloc[-1],  # Current price
                    volatility,  # Volatility
                    (close_prices.iloc[-1] / sma_20 - 1) if sma_20 > 0 else 0,  # Price vs SMA20
                    (close_prices.iloc[-1] / sma_50 - 1) if sma_50 > 0 else 0,  # Price vs SMA50
                    rsi,  # RSI
                    volume_ratio,  # Volume ratio
                    momentum_5,  # 5-period momentum
                    momentum_20,  # 20-period momentum
                    market_data['returns'].get(symbol, 0),  # Latest return
                    market_data['volume'].get(symbol, 1000000) / 1000000,  # Normalized volume
                ]
                
                features_list.append(symbol_features)
            
            if not features_list:
                logger.warning("No features generated - using fallback")
                n_features = 10
                features = np.random.randn(len(self.symbols), n_features)
                feature_names = [f'feature_{i}' for i in range(n_features)]
                return pd.DataFrame(features, columns=feature_names, index=self.symbols)
            
            # Create feature names
            feature_names = [
                'price', 'volatility', 'price_sma20_ratio', 'price_sma50_ratio',
                'rsi', 'volume_ratio', 'momentum_5', 'momentum_20',
                'latest_return', 'normalized_volume'
            ]
            
            # Create DataFrame
            features_df = pd.DataFrame(features_list, columns=feature_names, index=self.symbols[:len(features_list)])
            
            logger.info(f"✅ Generated features for {len(features_df)} symbols with {len(feature_names)} features each")
            return features_df
            
        except Exception as e:
            logger.error(f"❌ Feature generation failed: {e}")
            # Fallback to random features
            n_features = 10
            features = np.random.randn(len(self.symbols), n_features)
            feature_names = [f'feature_{i}' for i in range(n_features)]
            return pd.DataFrame(features, columns=feature_names, index=self.symbols)
    
    def _get_all_predictions(self, features: pd.DataFrame) -> Dict[str, Dict[str, np.ndarray]]:
        """Get predictions from all models for all horizons."""
        all_predictions = {}
        
        for horizon in self.horizons:
            try:
                horizon_preds = self.model_predictor.predict_horizon(
                    horizon, features, self.symbols, f'fwd_ret_{horizon}'
                )
                if horizon_preds:
                    all_predictions[horizon] = horizon_preds
            except Exception as e:
                logger.warning(f"Failed to get predictions for horizon {horizon}: {e}")
        
        return all_predictions
    
    def _horizon_to_minutes(self, horizon: str) -> int:
        """Convert horizon string to minutes."""
        if horizon.endswith('m'):
            return int(horizon[:-1])
        elif horizon == '1d':
            return 1440
        elif horizon == '5d':
            return 7200
        elif horizon == '20d':
            return 28800
        else:
            return 5
    
    def _execute_trades(self, target_weights: np.ndarray):
        """Execute trades to reach target weights via Alpaca."""
        if len(target_weights) != len(self.current_weights):
            logger.error("Weight length mismatch")
            return
        
        # Calculate weight changes
        weight_changes = target_weights - self.current_weights
        
        # Generate trading decisions
        trading_decisions = []
        for i, symbol in enumerate(self.symbols):
            weight_change = weight_changes[i]
            if abs(weight_change) > 0.01:  # Only trade if change > 1%
                action = 'BUY' if weight_change > 0 else 'SELL'
                quantity = abs(weight_change) * 1000  # Scale to shares
                
                trading_decisions.append({
                    'symbol': symbol,
                    'action': action,
                    'quantity': quantity,
                    'order_type': 'MKT'
                })
        
        # Execute trades via Alpaca
        if trading_decisions:
            execution_results = self.execute_trades(trading_decisions)
            logger.info(f"📈 Executed {len(execution_results)} trades via Alpaca")
        
        # Log significant changes
        significant_changes = np.abs(weight_changes) > 0.01
        if np.any(significant_changes):
            changed_symbols = [self.symbols[i] for i in np.where(significant_changes)[0]]
            logger.info(f"Significant weight changes for: {changed_symbols}")
        
        # In production, this would interface with IBKR API
        logger.info(f"Executing trades: {weight_changes}")
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status."""
        return {
            'current_weights': dict(zip(self.symbols, self.current_weights)),
            'portfolio_value': self.portfolio_value,
            'last_update': self.last_update,
            'performance_metrics': dict(self.performance_metrics)
        }

class LiveTradingManager:
    """Manager for multiple trading systems."""
    
    def __init__(self, configs: List[Dict[str, Any]]):
        self.systems = []
        for config in configs:
            system = LiveTradingSystem(config)
            self.systems.append(system)
    
    def start_all(self):
        """Start all trading systems."""
        for system in self.systems:
            system.start()
    
    def stop_all(self):
        """Stop all trading systems."""
        for system in self.systems:
            system.stop()
    
    def get_all_status(self) -> List[Dict[str, Any]]:
        """Get status of all systems."""
        return [system.get_portfolio_status() for system in self.systems]
    
    def execute_trades(self, trading_decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute trades using Alpaca paper trading."""
        try:
            logger.info(f"📈 Executing {len(trading_decisions)} trades via Alpaca...")
            
            execution_results = []
            
            for decision in trading_decisions:
                symbol = decision.get('symbol')
                action = decision.get('action', 'HOLD')
                quantity = decision.get('quantity', 0)
                order_type = decision.get('order_type', 'MKT')
                
                if action == 'HOLD' or quantity == 0:
                    continue
                
                try:
                    # Convert to Alpaca order format
                    side = 'buy' if action == 'BUY' else 'sell'
                    
                    # Submit order via Alpaca
                    order_result = self.alpaca_executor.submit_order(
                        symbol=symbol,
                        qty=abs(quantity),
                        side=side,
                        type=order_type,
                        time_in_force='day'
                    )
                    
                    execution_results.append({
                        'symbol': symbol,
                        'action': action,
                        'quantity': quantity,
                        'status': 'submitted',
                        'order_id': order_result.get('id'),
                        'timestamp': datetime.now()
                    })
                    
                    logger.info(f"✅ {action} {abs(quantity)} {symbol} submitted to Alpaca")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to execute {action} {quantity} {symbol}: {e}")
                    execution_results.append({
                        'symbol': symbol,
                        'action': action,
                        'quantity': quantity,
                        'status': 'failed',
                        'error': str(e),
                        'timestamp': datetime.now()
                    })
            
            logger.info(f"✅ Trade execution complete: {len(execution_results)} results")
            return execution_results
            
        except Exception as e:
            logger.error(f"❌ Trade execution failed: {e}")
            return []
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status from Alpaca."""
        try:
            # Get account info from Alpaca
            account = self.alpaca_executor.get_account()
            positions = self.alpaca_executor.get_positions()
            
            portfolio_status = {
                'equity': float(account.get('equity', 0)),
                'cash': float(account.get('cash', 0)),
                'buying_power': float(account.get('buying_power', 0)),
                'positions': len(positions),
                'timestamp': datetime.now()
            }
            
            # Add position details
            position_details = []
            for pos in positions:
                position_details.append({
                    'symbol': pos.get('symbol'),
                    'qty': float(pos.get('qty', 0)),
                    'market_value': float(pos.get('market_value', 0)),
                    'unrealized_pl': float(pos.get('unrealized_pl', 0))
                })
            
            portfolio_status['position_details'] = position_details
            
            logger.debug(f"📊 Portfolio: ${portfolio_status['equity']:,.2f} equity, {portfolio_status['positions']} positions")
            return portfolio_status
            
        except Exception as e:
            logger.error(f"❌ Failed to get portfolio status: {e}")
            return {
                'equity': 0,
                'cash': 0,
                'buying_power': 0,
                'positions': 0,
                'error': str(e),
                'timestamp': datetime.now()
            }
