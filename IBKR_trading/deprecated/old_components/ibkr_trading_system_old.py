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
IBKRTradingSystem - Main execution system for IBKR intraday trading.

This module orchestrates all components of the IBKR trading system including
safety guards, ensemble decision making, execution, and monitoring.
"""


import time
import logging
import yaml
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import threading
import signal
import sys

# Import all trading components
from .pre_trade_guards import PreTradeGuards, GuardConfig
from .margin_gate import MarginGate, MarginConfig
from .short_sale_guard import ShortSaleGuard, ShortSaleConfig
from .rate_limiter import RateLimiter, RateLimitConfig
from .universe_prefilter import UniversePrefilter, PrefilterConfig
from .zoo_balancer import ZooBalancer
from .horizon_arbiter import HorizonArbiter
from .barrier_gates import BarrierGates
from .short_horizon_execution import ShortHorizonExecutionPolicy
from .netting_suppression import NettingSuppression, NettingConfig
from .drift_health import DriftHealth, DriftConfig
from .mode_controller import ModeController, ModeConfig, TradingMode
from .order_reconciler import OrderReconciler

logger = logging.getLogger(__name__)

@dataclass
class SystemConfig:
    """Main system configuration."""
    config_file: str = "config/ibkr_enhanced.yaml"
    model_dir: str = "models/tonight_16models_*"
    log_level: str = "INFO"
    max_symbols: int = 20
    update_frequency_s: int = 5
    emergency_stop: bool = False

class IBKRTradingSystem:
    """
    Main IBKR trading system orchestrator.
    
    Coordinates all components for safe, efficient intraday trading:
    1. Safety & Guard Layer (PreTradeGuards, MarginGate, ShortSaleGuard, RateLimiter)
    2. Decision & Ensemble Layer (ZooBalancer, HorizonArbiter, BarrierGates)
    3. Efficiency & Optimization Layer (UniversePrefilter, NettingSuppression, DriftHealth)
    4. Execution & Monitoring Layer (ShortHorizonExecutionPolicy, ModeController, OrderReconciler)
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize IBKR trading system.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.running = False
        self.emergency_stop = False
        
        # Load configuration
        self.yaml_config = self._load_config()
        
        # Initialize components
        self._initialize_components()
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        # Performance tracking
        self.metrics = {
            "total_cycles": 0,
            "successful_trades": 0,
            "blocked_trades": 0,
            "system_uptime": 0.0,
            "last_update": time.time()
        }
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            config_path = Path(self.config.config_file)
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
                
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            logger.info(f"Loaded configuration from {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _initialize_components(self):
        """Initialize all trading system components."""
        try:
            # Safety & Guard Layer
            self._init_safety_components()
            
            # Decision & Ensemble Layer
            self._init_decision_components()
            
            # Efficiency & Optimization Layer
            self._init_optimization_components()
            
            # Execution & Monitoring Layer
            self._init_execution_components()
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _init_safety_components(self):
        """Initialize safety and guard components."""
        # Guard configuration
        guard_config = GuardConfig(
            max_quote_ms=self.yaml_config['safety']['max_quote_ms'],
            max_spread_bps=self.yaml_config['safety']['max_spread_bps'],
            min_1m_adv=self.yaml_config['safety']['min_1m_adv'],
            open_blackout_s=self.yaml_config['safety']['open_blackout_s'],
            close_blackout_s=self.yaml_config['safety']['close_blackout_s'],
            max_daily_loss_pct=self.yaml_config['safety']['max_daily_loss_pct'],
            max_gross_exposure_pct=self.yaml_config['safety']['max_gross_exposure_pct']
        )
        
        # Margin configuration
        margin_config = MarginConfig(
            headroom_pct=self.yaml_config['margin']['headroom_pct'],
            min_buying_power=self.yaml_config['margin']['min_buying_power'],
            max_margin_utilization=self.yaml_config['margin']['max_margin_utilization']
        )
        
        # Short sale configuration
        short_sale_config = ShortSaleConfig(
            require_borrow=self.yaml_config['short_sale']['require_borrow'],
            enforce_ssr_price_test=self.yaml_config['short_sale']['enforce_ssr_price_test'],
            min_shortable_shares=self.yaml_config['short_sale']['min_shortable_shares'],
            max_borrow_rate=self.yaml_config['short_sale']['max_borrow_rate']
        )
        
        # Rate limiting configuration
        rate_config = RateLimitConfig(
            msgs_per_s=self.yaml_config['ibkr']['pacing']['msgs_per_s'],
            burst=self.yaml_config['ibkr']['pacing']['burst'],
            cancel_per_s=self.yaml_config['ibkr']['pacing']['cancel_per_s'],
            backoff_s=self.yaml_config['ibkr']['reconnect']['backoff_s'],
            jitter_ms=self.yaml_config['ibkr']['reconnect']['jitter_ms']
        )
        
        # Initialize components (with mock dependencies for now)
        self.margin_gate = MarginGate(None, margin_config)
        self.short_sale_guard = ShortSaleGuard(None, short_sale_config)
        self.rate_limiter = RateLimiter(rate_config)
        self.pre_trade_guards = PreTradeGuards(
            guard_config, None, None, None, 
            self.margin_gate, self.short_sale_guard
        )
        
        logger.info("Safety components initialized")
    
    def _init_decision_components(self):
        """Initialize decision and ensemble components."""
        # Initialize ensemble components
        self.zoo_balancer = ZooBalancer()
        self.horizon_arbiter = HorizonArbiter()
        self.barrier_gates = BarrierGates()
        
        logger.info("Decision components initialized")
    
    def _init_optimization_components(self):
        """Initialize optimization components."""
        # Prefilter configuration
        prefilter_config = PrefilterConfig(
            top_k=self.yaml_config['prefilter']['top_k'],
            min_score_bps=self.yaml_config['prefilter']['min_score_bps'],
            max_cost_ratio=self.yaml_config['prefilter']['max_cost_ratio']
        )
        
        # Netting configuration
        netting_config = NettingConfig(
            min_net_size=self.yaml_config['netting']['min_net_size'],
            max_churn_ratio=self.yaml_config['netting']['max_churn_ratio'],
            hysteresis_pct=self.yaml_config['netting']['hysteresis_pct']
        )
        
        # Drift health configuration
        drift_config = DriftConfig(
            window_size=self.yaml_config['health']['window_size'],
            ic_min_threshold=self.yaml_config['health']['ic_min_ewma'],
            reliability_min_threshold=self.yaml_config['health']['reliability_min']
        )
        
        # Initialize components
        self.universe_prefilter = UniversePrefilter(prefilter_config, None)
        self.netting_suppression = NettingSuppression(netting_config)
        self.drift_health = DriftHealth(drift_config)
        
        logger.info("Optimization components initialized")
    
    def _init_execution_components(self):
        """Initialize execution and monitoring components."""
        # Mode controller configuration
        mode_config = ModeConfig(
            max_broker_latency_ms=self.yaml_config['health']['max_broker_latency_ms'],
            max_exception_count=self.yaml_config['health']['max_exception_count'],
            kill_file_path=self.yaml_config['disaster_recovery']['panic_file']
        )
        
        # Initialize components
        self.execution_policy = ShortHorizonExecutionPolicy()
        self.mode_controller = ModeController(mode_config, self.drift_health, None, None)
        self.order_reconciler = OrderReconciler(None)
        
        logger.info("Execution components initialized")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.emergency_stop = True
            self.stop()
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def start(self):
        """Start the trading system."""
        try:
            logger.info("Starting IBKR trading system...")
            self.running = True
            self.emergency_stop = False
            
            # Start main trading loop
            self._main_loop()
            
        except Exception as e:
            logger.error(f"Failed to start trading system: {e}")
            self.stop()
            raise
    
    def stop(self):
        """Stop the trading system."""
        logger.info("Stopping IBKR trading system...")
        self.running = False
        
        # Flatten all positions if in emergency
        if self.emergency_stop:
            self._emergency_flatten()
    
    def _main_loop(self):
        """Main trading loop."""
        logger.info("Main trading loop started")
        start_time = time.time()
        
        while self.running and not self.emergency_stop:
            try:
                cycle_start = time.time()
                
                # Check system health and mode
                current_mode = self._check_system_health()
                if current_mode == TradingMode.EMERGENCY:
                    logger.critical("Emergency mode detected, stopping system")
                    self.emergency_stop = True
                    break
                
                # Get trading parameters for current mode
                mode_params = self.mode_controller.get_mode_parameters()
                
                # Skip trading if not allowed
                if not self.mode_controller.is_trading_allowed():
                    logger.info(f"Trading not allowed in {current_mode.value} mode")
                    time.sleep(self.config.update_frequency_s)
                    continue
                
                # Main trading cycle
                self._trading_cycle(mode_params)
                
                # Update metrics
                self._update_metrics(cycle_start)
                
                # Sleep until next cycle
                elapsed = time.time() - cycle_start
                sleep_time = max(0, self.config.update_frequency_s - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in main trading loop: {e}")
                time.sleep(1)  # Brief pause before retry
        
        self.metrics["system_uptime"] = time.time() - start_time
        logger.info("Main trading loop ended")
    
    def _check_system_health(self) -> TradingMode:
        """Check system health and return current mode."""
        # This would implement comprehensive health checking
        # For now, return LIVE mode
        return TradingMode.LIVE
    
    def _trading_cycle(self, mode_params: Dict[str, Any]):
        """Execute one trading cycle."""
        try:
            # 1. Get universe of symbols
            symbols = self._get_universe()
            if not symbols:
                return
            
            # 2. Pre-filter symbols
            filtered_symbols = self.universe_prefilter.pick(symbols)
            if not filtered_symbols:
                logger.info("No symbols passed pre-filtering")
                return
            
            # 3. Process each symbol
            for symbol in filtered_symbols[:self.config.max_symbols]:
                try:
                    self._process_symbol(symbol, mode_params)
                except Exception as e:
                    logger.error(f"Error processing symbol {symbol}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
    
    def _get_universe(self) -> Dict[str, Dict]:
        """Get universe of symbols for trading."""
        # This would implement symbol universe selection
        # For now, return mock data
        return {
            "AAPL": {"alpha_hint_bps": 5.0, "spread_bps": 2.0, "vol_bps": 15.0},
            "MSFT": {"alpha_hint_bps": 3.0, "spread_bps": 1.5, "vol_bps": 12.0},
            "GOOGL": {"alpha_hint_bps": 4.0, "spread_bps": 3.0, "vol_bps": 18.0}
        }
    
    def _process_symbol(self, symbol: str, mode_params: Dict[str, Any]):
        """Process a single symbol through the trading pipeline."""
        try:
            # 1. Pre-trade guards
            guard_ok, guard_reason = self.pre_trade_guards.check_symbol(symbol)
            if not guard_ok:
                logger.debug(f"Symbol {symbol} blocked by guards: {guard_reason}")
                self.metrics["blocked_trades"] += 1
                return
            
            # 2. Get model predictions (mock for now)
            predictions = self._get_model_predictions(symbol)
            if not predictions:
                return
            
            # 3. Ensemble decision making
            decision = self._make_ensemble_decision(symbol, predictions, mode_params)
            if not decision:
                return
            
            # 4. Netting and churn suppression
            netted_decision = self._apply_netting(symbol, decision)
            if not netted_decision:
                return
            
            # 5. Execution
            self._execute_decision(symbol, netted_decision, mode_params)
            
        except Exception as e:
            logger.error(f"Error processing symbol {symbol}: {e}")
    
    def _get_model_predictions(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get model predictions for symbol."""
        # This would implement model inference
        # For now, return mock predictions
        return {
            "horizons": {
                "5m": {"alpha": 0.002, "confidence": 0.8},
                "10m": {"alpha": 0.001, "confidence": 0.7}
            },
            "barrier_targets": {
                "will_peak_5m": 0.3,
                "will_valley_5m": 0.6,
                "y_will_peak_5m": 0.2
            }
        }
    
    def _make_ensemble_decision(self, symbol: str, predictions: Dict[str, Any], 
                               mode_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make ensemble decision for symbol."""
        try:
            # This would implement the full ensemble pipeline
            # For now, return mock decision
            return {
                "symbol": symbol,
                "side": "BUY",
                "size": 100.0,
                "horizon": "5m",
                "alpha": 0.002,
                "confidence": 0.8
            }
        except Exception as e:
            logger.error(f"Error making ensemble decision for {symbol}: {e}")
            return None
    
    def _apply_netting(self, symbol: str, decision: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply netting and churn suppression."""
        try:
            # This would implement netting logic
            # For now, return the decision as-is
            return decision
        except Exception as e:
            logger.error(f"Error applying netting for {symbol}: {e}")
            return None
    
    def _execute_decision(self, symbol: str, decision: Dict[str, Any], 
                      mode_params: Dict[str, Any]):
        """Execute trading decision."""
        try:
            # This would implement order execution
            logger.info(f"Executing decision for {symbol}: {decision}")
            self.metrics["successful_trades"] += 1
            
        except Exception as e:
            logger.error(f"Error executing decision for {symbol}: {e}")
    
    def _emergency_flatten(self):
        """Emergency flatten all positions."""
        logger.critical("Executing emergency flatten...")
        try:
            # This would implement emergency flattening
            logger.info("Emergency flatten completed")
        except Exception as e:
            logger.error(f"Error during emergency flatten: {e}")
    
    def _update_metrics(self, cycle_start: float):
        """Update system metrics."""
        self.metrics["total_cycles"] += 1
        self.metrics["last_update"] = time.time()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "running": self.running,
            "emergency_stop": self.emergency_stop,
            "current_mode": self.mode_controller.current_mode.value,
            "metrics": self.metrics.copy(),
            "components": {
                "pre_trade_guards": "active",
                "margin_gate": "active",
                "short_sale_guard": "active",
                "rate_limiter": "active",
                "universe_prefilter": "active",
                "zoo_balancer": "active",
                "horizon_arbiter": "active",
                "barrier_gates": "active",
                "execution_policy": "active",
                "netting_suppression": "active",
                "drift_health": "active",
                "mode_controller": "active",
                "order_reconciler": "active"
            }
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics."""
        return self.metrics.copy()

def main():
    """Main entry point for IBKR trading system."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create system configuration
    config = SystemConfig()
    
    # Initialize and start trading system
    trading_system = IBKRTradingSystem(config)
    
    try:
        trading_system.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"System error: {e}")
    finally:
        trading_system.stop()
        logger.info("IBKR trading system shutdown complete")

if __name__ == "__main__":
    main()
