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
Unit tests for optimization engine.
Tests both greedy and QP optimization patterns.
"""


import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live_trading.optimization_engine import (
    OptimizationConfig, GreedyRotationEngine, QPOptimizationEngine,
    OptimizationEngineFactory
)

class TestOptimizationEngine(unittest.TestCase):
    """Test cases for optimization engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = OptimizationConfig(
            lambda_risk=0.1,
            gamma_cost=0.05,
            max_gross_exposure=1.5,
            per_name_cap=0.1,
            K=5, K_buy=4, K_sell=6,  # Smaller portfolio for testing
            z_keep=0.8, z_cut=0.2, delta_z_min=0.25
        )
        
        # Test universe
        self.universe = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        self.n = len(self.universe)
        
        # Test data
        self.z_scores = np.array([0.8, 0.6, 0.4, 0.2, 0.1])
        self.correlation_matrix = np.eye(self.n)
        self.costs = np.array([10, 10, 10, 10, 10])  # 10 bps each
        self.current_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    
    def test_greedy_rotation_engine(self):
        """Test greedy rotation engine."""
        engine = GreedyRotationEngine(self.config)
        
        # Test optimization
        target_weights, info = engine.optimize(
            self.z_scores, self.correlation_matrix, self.costs,
            self.current_weights, self.universe
        )
        
        # Check that weights sum to 1
        self.assertAlmostEqual(np.sum(target_weights), 1.0, places=6)
        
        # Check that weights are non-negative
        self.assertTrue(np.all(target_weights >= 0))
        
        # Check that per-name caps are respected
        self.assertTrue(np.all(target_weights <= self.config.per_name_cap))
        
        # Check optimization info
        self.assertEqual(info['method'], 'greedy_rotation')
        self.assertIn('changes', info)
        self.assertIn('reason_codes', info)
    
    def test_greedy_cut_low_performers(self):
        """Test cutting low performers in greedy engine."""
        engine = GreedyRotationEngine(self.config)
        
        # Set up data with clear losers
        z_scores = np.array([0.8, 0.6, 0.4, -0.5, -0.8])  # Clear losers
        pnl_since_entry = np.array([100, 50, 0, -200, -300])  # Big losers
        atr = np.array([0.02, 0.02, 0.02, 0.02, 0.02])
        
        target_weights, info = engine.optimize(
            z_scores, self.correlation_matrix, self.costs,
            self.current_weights, self.universe,
            pnl_since_entry=pnl_since_entry, atr=atr
        )
        
        # Should cut the big losers
        self.assertEqual(target_weights[3], 0.0)  # AMZN (big loser)
        self.assertEqual(target_weights[4], 0.0)  # TSLA (big loser)
        
        # Should have cut reasons
        self.assertIn("CUT_LOSER", str(info['reason_codes'].values()))
    
    def test_greedy_rotate_green_to_better(self):
        """Test rotating from green to better name in greedy engine."""
        engine = GreedyRotationEngine(self.config)
        
        # Set up data with clear alpha gap
        z_scores = np.array([0.8, 0.6, 0.4, 0.2, 0.1])  # Clear ranking
        current_weights = np.array([0.0, 0.0, 0.3, 0.3, 0.4])  # Hold lower-ranked
        
        target_weights, info = engine.optimize(
            z_scores, self.correlation_matrix, self.costs,
            current_weights, self.universe
        )
        
        # Should rotate to higher-ranked names
        self.assertGreater(target_weights[0], 0)  # AAPL should get position
        self.assertGreater(target_weights[1], 0)  # MSFT should get position
        
        # Should have rotation reasons
        self.assertTrue(any("ROTATE_BETTER" in str(reason) for reason in info['reason_codes'].values()))
    
    def test_greedy_no_trade_band(self):
        """Test no-trade band in greedy engine."""
        engine = GreedyRotationEngine(self.config)
        
        # Set up data with small changes
        z_scores = np.array([0.1, 0.1, 0.1, 0.1, 0.1])  # All similar scores
        current_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        
        target_weights, info = engine.optimize(
            z_scores, self.correlation_matrix, self.costs,
            current_weights, self.universe
        )
        
        # Should have minimal changes due to no-trade band
        L1_drift = np.sum(np.abs(target_weights - current_weights))
        self.assertLess(L1_drift, self.config.portfolio_drift_threshold + 0.001)
    
    def test_greedy_utility_threshold(self):
        """Test utility threshold in greedy engine."""
        engine = GreedyRotationEngine(self.config)
        
        # Set up data with high costs
        z_scores = np.array([0.8, 0.6, 0.4, 0.2, 0.1])
        current_weights = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        high_costs = np.array([1000, 1000, 1000, 1000, 1000])  # Very high costs
        
        target_weights, info = engine.optimize(
            z_scores, self.correlation_matrix, high_costs,
            current_weights, self.universe
        )
        
        # Should have utility failures
        self.assertTrue(any("UTILITY_FAIL" in str(reason) for reason in info['reason_codes'].values()))
    
    def test_qp_optimization_engine(self):
        """Test QP optimization engine."""
        try:
            engine = QPOptimizationEngine(self.config)
            
            # Test optimization
            target_weights, info = engine.optimize(
                self.z_scores, self.correlation_matrix, self.costs,
                self.current_weights, self.universe
            )
            
            # Check that weights sum to 1
            self.assertAlmostEqual(np.sum(target_weights), 1.0, places=6)
            
            # Check that weights are non-negative
            self.assertTrue(np.all(target_weights >= 0))
            
            # Check that per-name caps are respected
            self.assertTrue(np.all(target_weights <= self.config.per_name_cap))
            
            # Check optimization info
            self.assertEqual(info['method'], 'qp_optimization')
            self.assertIn('status', info)
            self.assertIn('objective_value', info)
            
        except ImportError:
            self.skipTest("cvxpy not available")
    
    def test_qp_with_beta_constraints(self):
        """Test QP optimization with beta constraints."""
        try:
            engine = QPOptimizationEngine(self.config)
            
            # Set up beta exposures
            beta_exposures = np.array([1.2, 0.8, 1.0, 1.1, 0.9])  # Different betas
            
            target_weights, info = engine.optimize(
                self.z_scores, self.correlation_matrix, self.costs,
                self.current_weights, self.universe,
                beta_exposures=beta_exposures
            )
            
            # Check that portfolio beta is close to zero
            portfolio_beta = np.sum(beta_exposures * target_weights)
            self.assertLess(abs(portfolio_beta), self.config.beta_tolerance + 0.001)
            
        except ImportError:
            self.skipTest("cvxpy not available")
    
    def test_qp_with_sector_constraints(self):
        """Test QP optimization with sector constraints."""
        try:
            engine = QPOptimizationEngine(self.config)
            
            # Set up sector exposures (2 sectors)
            sector_exposures = np.array([
                [1, 0],  # AAPL in sector 0
                [1, 0],  # MSFT in sector 0
                [0, 1],  # GOOGL in sector 1
                [0, 1],  # AMZN in sector 1
                [0, 1]   # TSLA in sector 1
            ])
            
            target_weights, info = engine.optimize(
                self.z_scores, self.correlation_matrix, self.costs,
                self.current_weights, self.universe,
                sector_exposures=sector_exposures
            )
            
            # Check that sector exposures are within limits
            for sector in range(sector_exposures.shape[1]):
                sector_exposure = np.sum(sector_exposures[:, sector] * target_weights)
                self.assertLessEqual(sector_exposure, self.config.sector_cap + 0.001)
            
        except ImportError:
            self.skipTest("cvxpy not available")
    
    def test_optimization_engine_factory(self):
        """Test optimization engine factory."""
        # Test greedy engine
        greedy_engine = OptimizationEngineFactory.create_engine("greedy", self.config)
        self.assertIsInstance(greedy_engine, GreedyRotationEngine)
        
        # Test QP engine (if available)
        try:
            qp_engine = OptimizationEngineFactory.create_engine("qp", self.config)
            self.assertIsInstance(qp_engine, QPOptimizationEngine)
        except ImportError:
            pass  # cvxpy not available
        
        # Test unknown engine type
        with self.assertRaises(ValueError):
            OptimizationEngineFactory.create_engine("unknown", self.config)
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test valid config
        config = OptimizationConfig(
            lambda_risk=0.1,
            gamma_cost=0.05,
            max_gross_exposure=1.5,
            per_name_cap=0.1
        )
        self.assertEqual(config.lambda_risk, 0.1)
        self.assertEqual(config.gamma_cost, 0.05)
        
        # Test default values
        default_config = OptimizationConfig()
        self.assertEqual(default_config.lambda_risk, 0.1)
        self.assertEqual(default_config.gamma_cost, 0.05)
        self.assertEqual(default_config.max_gross_exposure, 1.5)
        self.assertEqual(default_config.per_name_cap, 0.1)
    
    def test_performance_comparison(self):
        """Test performance comparison between engines."""
        # Test greedy engine performance
        greedy_engine = GreedyRotationEngine(self.config)
        
        start_time = datetime.now()
        target_weights_greedy, info_greedy = greedy_engine.optimize(
            self.z_scores, self.correlation_matrix, self.costs,
            self.current_weights, self.universe
        )
        greedy_time = (datetime.now() - start_time).total_seconds()
        
        # Greedy should be very fast
        self.assertLess(greedy_time, 0.1)  # Less than 100ms
        
        # Test QP engine performance (if available)
        try:
            qp_engine = QPOptimizationEngine(self.config)
            
            start_time = datetime.now()
            target_weights_qp, info_qp = qp_engine.optimize(
                self.z_scores, self.correlation_matrix, self.costs,
                self.current_weights, self.universe
            )
            qp_time = (datetime.now() - start_time).total_seconds()
            
            # QP should be reasonably fast
            self.assertLess(qp_time, 1.0)  # Less than 1 second
            
        except ImportError:
            self.skipTest("cvxpy not available")

if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
