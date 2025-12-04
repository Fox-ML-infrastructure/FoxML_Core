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
Enhanced Decision Pipeline - Integration of all pressure test upgrades
Shows how to wire together conformal gates, horizon arbiter 2.0, execution micro-planner, etc.
"""


import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np

# Import our new components
from conformal_gate import ConformalGate, ConformalConfig
from horizon_arbiter_2 import HorizonArbiter2, HorizonArbiter2Config
from execution_microplanner import ExecutionMicroPlanner, MicroPlannerConfig
from staleness_guard import StalenessGuard, StalenessConfig
from verification_checklist import VerificationChecklist

logger = logging.getLogger(__name__)

class EnhancedDecisionPipeline:
    """
    Enhanced decision pipeline integrating all pressure test upgrades.
    
    Features:
    1. Conformal gates for alpha retention
    2. Horizon arbitration 2.0 with meta-learning
    3. Execution micro-planner with queue awareness
    4. Staleness guard with TTL enforcement
    5. Verification checklist for monitoring
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize components
        self.conformal_gate = ConformalGate(ConformalConfig())
        self.horizon_arbiter = HorizonArbiter2(HorizonArbiter2Config())
        self.execution_planner = ExecutionMicroPlanner(MicroPlannerConfig())
        self.staleness_guard = StalenessGuard(StalenessConfig())
        self.verification = VerificationChecklist()
        
        # State tracking
        self.cycle_count = 0
        self.session_start = time.time()
        self.current_weights = {}
        
    def process_symbol(self, 
                      symbol: str, 
                      market_data: Dict[str, Any],
                      model_predictions: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single symbol through the enhanced pipeline.
        
        Args:
            symbol: Symbol to process
            market_data: Market data for the symbol
            model_predictions: Model predictions for the symbol
            
        Returns:
            Execution plan or None if blocked
        """
        self.cycle_count += 1
        
        try:
            # 1. Staleness check
            fresh, reason = self.staleness_guard.check_data_freshness(
                symbol=symbol,
                quote_data=market_data.get('quote'),
                bar_data=market_data.get('bar'),
                prediction_data=model_predictions
            )
            
            if not fresh:
                logger.debug(f"Symbol {symbol} blocked by staleness: {reason}")
                return None
            
            # 2. Get conformal quantiles
            horizon = model_predictions.get('horizon', '5m')
            q_lo, q_hi = self.conformal_gate.get_quantiles(symbol, horizon)
            
            # 3. Horizon arbitration 2.0
            alpha_by_h = model_predictions.get('horizon_alphas', {})
            h_star, alpha_star, arbiter_scores = self.horizon_arbiter.choose(
                alpha_by_h, market_data, self.config.get('thresholds', {})
            )
            
            if not h_star:
                logger.debug(f"Symbol {symbol} blocked by horizon arbiter")
                return None
            
            # 4. Conformal gating
            est_cost = self._estimate_costs(market_data, h_star)
            allowed, conf_reason = self.conformal_gate.allow(
                alpha_point_bps=alpha_star,
                q_lo_bps=q_lo,
                q_hi_bps=q_hi,
                est_cost_bps=est_cost,
                symbol=symbol,
                horizon=h_star
            )
            
            if not allowed:
                logger.debug(f"Symbol {symbol} blocked by conformal gate: {conf_reason}")
                return None
            
            # 5. Enhanced barrier gating (if available)
            barrier_predictions = model_predictions.get('barrier_targets', {})
            if barrier_predictions:
                barrier_ok, barrier_reason = self._check_barrier_gates(
                    barrier_predictions, alpha_star
                )
                if not barrier_ok:
                    logger.debug(f"Symbol {symbol} blocked by barrier gates: {barrier_reason}")
                    return None
            
            # 6. Execution planning
            side = 'BUY' if alpha_star > 0 else 'SELL'
            total_qty = self._calculate_position_size(alpha_star, market_data)
            
            if total_qty <= 0:
                logger.debug(f"Symbol {symbol} has zero/negative position size")
                return None
            
            execution_steps = self.execution_planner.plan(
                side=side,
                total_qty=total_qty,
                tif_seconds=self._get_tif_for_horizon(h_star),
                px_ref=market_data.get('mid_price', 100.0),
                spread_bps=market_data.get('spread_bps', 5.0),
                symbol=symbol
            )
            
            if not execution_steps:
                logger.debug(f"Symbol {symbol} blocked by execution planner")
                return None
            
            # 7. Log cycle metrics for verification
            cycle_metrics = {
                'conformal_coverage': self._calculate_conformal_coverage(symbol, horizon),
                'horizon_weights': arbiter_scores.get('weights', {}),
                'execution_metrics': {
                    'turnover': 0.0,  # Would be calculated from actual trades
                    'no_trade_band': self.config.get('no_trade_threshold', 0.008),
                    'rate_limit_hit': False  # Would be tracked by execution planner
                },
                'errors': []
            }
            self.verification.log_cycle_metrics(cycle_metrics)
            
            # 8. Update conformal calibration
            self._update_conformal_calibration(symbol, horizon, alpha_star)
            
            # 9. Update horizon arbiter training
            self._update_arbiter_training(symbol, market_data, h_star, alpha_star)
            
            return {
                'symbol': symbol,
                'horizon': h_star,
                'alpha': alpha_star,
                'side': side,
                'quantity': total_qty,
                'execution_steps': execution_steps,
                'conformal_reason': conf_reason,
                'arbiter_scores': arbiter_scores
            }
            
        except Exception as e:
            logger.error(f"Error processing symbol {symbol}: {e}")
            self.verification.log_cycle_metrics({'errors': [str(e)]})
            return None
    
    def _estimate_costs(self, market_data: Dict[str, Any], horizon: str) -> float:
        """Estimate trading costs for a horizon."""
        spread_bps = market_data.get('spread_bps', 5.0)
        vol_bps = market_data.get('vol_5m_bps', 20.0)
        participation_rate = market_data.get('participation_rate', 0.01)
        
        # Horizon-specific cost scaling
        h_minutes = int(horizon[:-1]) if horizon.endswith('m') else 5
        horizon_factor = np.sqrt(h_minutes / 5.0)
        
        # Cost components
        spread_cost = spread_bps
        vol_cost = 0.15 * vol_bps * horizon_factor
        impact_cost = 1.0 * (participation_rate ** 0.6)
        
        return spread_cost + vol_cost + impact_cost
    
    def _check_barrier_gates(self, barrier_predictions: Dict[str, float], alpha: float) -> Tuple[bool, str]:
        """Check barrier gates."""
        # Simple barrier gate logic
        will_peak = barrier_predictions.get('will_peak_5m', 0.0)
        will_valley = barrier_predictions.get('will_valley_5m', 0.0)
        
        # Block long entry if peak probability too high
        if alpha > 0 and will_peak > 0.6:
            return False, "peak_risk"
        
        # Block short entry if valley probability too high
        if alpha < 0 and will_valley > 0.6:
            return False, "valley_risk"
        
        return True, "ok"
    
    def _calculate_position_size(self, alpha: float, market_data: Dict[str, Any]) -> int:
        """Calculate position size based on alpha and risk parameters."""
        portfolio_value = market_data.get('portfolio_value', 100000.0)
        max_position_pct = self.config.get('max_position_pct', 0.05)
        
        # Base position size
        base_size = abs(alpha) * portfolio_value * 0.1  # 10% of alpha
        
        # Apply risk limits
        max_size = portfolio_value * max_position_pct
        position_size = min(base_size, max_size)
        
        # Round to lot size
        lot_size = market_data.get('lot_size', 100)
        return int(position_size // lot_size) * lot_size
    
    def _get_tif_for_horizon(self, horizon: str) -> int:
        """Get time-in-force for a horizon."""
        tif_map = {
            '5m': 15,
            '10m': 25,
            '15m': 30,
            '30m': 45,
            '60m': 60
        }
        return tif_map.get(horizon, 30)
    
    def _calculate_conformal_coverage(self, symbol: str, horizon: str) -> float:
        """Calculate conformal coverage for a symbol/horizon."""
        quality = self.conformal_gate.get_calibration_quality(symbol, horizon)
        return quality.get('coverage', 0.0)
    
    def _update_conformal_calibration(self, symbol: str, horizon: str, predicted_alpha: float):
        """Update conformal calibration with new prediction."""
        # In practice, this would be called after trade execution with realized return
        # For now, we'll simulate with a small random return
        realized_return = predicted_alpha + np.random.normal(0, 0.001)
        self.conformal_gate.update_calibration(symbol, horizon, predicted_alpha, realized_return)
    
    def _update_arbiter_training(self, symbol: str, market_data: Dict[str, Any], 
                                chosen_horizon: str, realized_alpha: float):
        """Update horizon arbiter training data."""
        # Extract state features
        state_features = self.horizon_arbiter._extract_state_features(market_data)
        
        # Get all available horizons
        all_horizons = list(market_data.get('horizon_alphas', {}).keys())
        
        # Update training data
        self.horizon_arbiter.update_training_data(
            state_features, chosen_horizon, realized_alpha, all_horizons
        )
        
        # Fit meta-learner periodically
        if self.cycle_count % 100 == 0:  # Every 100 cycles
            self.horizon_arbiter.fit_meta_learner()
    
    def run_verification(self) -> Dict[str, Any]:
        """Run verification checklist."""
        results = self.verification.run_full_verification(self.current_weights)
        self.verification.log_verification_results(results)
        return results
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get session summary with all metrics."""
        summary = self.verification.get_session_summary()
        
        # Add component-specific metrics
        summary.update({
            'conformal_gate_stats': self.conformal_gate.get_coverage_stats(),
            'horizon_arbiter_stats': self.horizon_arbiter.get_meta_learner_stats(),
            'execution_metrics': self.execution_planner.get_execution_metrics(),
            'staleness_summary': self.staleness_guard.get_staleness_summary()
        })
        
        return summary
    
    def reset_session(self):
        """Reset session state."""
        self.cycle_count = 0
        self.session_start = time.time()
        self.current_weights = {}
        self.execution_planner.reset_session()
        self.staleness_guard.reset_model_health()
        logger.info("Enhanced decision pipeline session reset")

# Example usage
def run_enhanced_pipeline_example():
    """Example of running the enhanced decision pipeline."""
    pipeline = EnhancedDecisionPipeline({
        'thresholds': {'enter_bps': 2.0, 'hold_bps': 1.0},
        'max_position_pct': 0.05,
        'no_trade_threshold': 0.008
    })
    
    # Simulate processing a symbol
    symbol = 'AAPL'
    market_data = {
        'quote': {'bid': 150.0, 'ask': 150.1, 'bid_size': 1000, 'ask_size': 1000, 'timestamp': time.time()},
        'bar': {'open': 149.5, 'high': 150.2, 'low': 149.0, 'close': 150.0, 'volume': 1000000, 'timestamp': time.time()},
        'mid_price': 150.05,
        'spread_bps': 6.7,
        'vol_5m_bps': 25.0,
        'portfolio_value': 100000.0,
        'lot_size': 100
    }
    
    model_predictions = {
        'horizon_alphas': {'5m': 0.003, '10m': 0.002, '15m': 0.001},
        'barrier_targets': {'will_peak_5m': 0.3, 'will_valley_5m': 0.6},
        'horizon': '5m'
    }
    
    # Process symbol
    result = pipeline.process_symbol(symbol, market_data, model_predictions)
    
    if result:
        logger.info(f"Execution plan for {symbol}: {result}")
    else:
        logger.info(f"Symbol {symbol} blocked by pipeline")
    
    # Run verification
    verification_results = pipeline.run_verification()
    
    # Get session summary
    summary = pipeline.get_session_summary()
    logger.info(f"Session summary: {summary}")
    
    return result, verification_results

if __name__ == "__main__":
    run_enhanced_pipeline_example()
