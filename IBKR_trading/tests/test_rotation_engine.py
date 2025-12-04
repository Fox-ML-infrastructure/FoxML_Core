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
Unit tests for rotation engine.
Tests rotation logic with synthetic data to verify triggers fire correctly.
"""


import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live_trading.rotation_engine import RotationEngine, RotationConfig

class TestRotationEngine(unittest.TestCase):
    """Test cases for rotation engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = RotationConfig(
            K=5, K_buy=4, K_sell=6,  # Smaller portfolio for testing
            z_keep=0.8, z_cut=0.2, delta_z_min=0.25,
            tau_name_bps=12, tau_L1=0.01,
            gamma_cost=1.0, tau_U=0.0,
            pmax=0.07, k_ATR=1.2, Tmax_min=120
        )
        
        self.engine = RotationEngine(self.config)
        
        # Test universe
        self.universe = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        self.n = len(self.universe)
        
        # Current time
        self.current_time = datetime.now()
    
    def test_compute_z_scores(self):
        """Test z-score computation."""
        scores = np.array([0.5, 0.3, 0.1, -0.2, -0.4])
        sigma = np.array([0.02, 0.03, 0.01, 0.025, 0.015])
        
        z_scores = self.engine.compute_z_scores(scores, sigma)
        
        # Check clipping
        self.assertTrue(np.all(z_scores >= -self.config.zmax))
        self.assertTrue(np.all(z_scores <= self.config.zmax))
        
        # Check scaling
        expected_z = scores / sigma
        np.testing.assert_array_almost_equal(z_scores, np.clip(expected_z, -self.config.zmax, self.config.zmax))
    
    def test_get_ranks(self):
        """Test ranking by z-score."""
        z_scores = np.array([0.5, 0.3, 0.1, -0.2, -0.4])
        ranks = self.engine.get_ranks(z_scores)
        
        # Should be sorted by z-score (high to low)
        expected_ranks = np.array([0, 1, 2, 3, 4])  # AAPL, MSFT, GOOGL, AMZN, TSLA
        np.testing.assert_array_equal(ranks, expected_ranks)
    
    def test_cut_low_performers(self):
        """Test cutting low performers."""
        # Set up test data
        scores = np.array([0.5, 0.3, 0.1, -0.2, -0.4])
        sigma = np.array([0.02, 0.03, 0.01, 0.025, 0.015])
        pnl_since_entry_bps = np.array([100, 50, -200, -300, -400])  # TSLA is big loser
        atr = np.array([0.02, 0.03, 0.01, 0.025, 0.015])
        current_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        
        # Set up entry times for holding age test
        self.engine.entry_times = {
            "AAPL": self.current_time - timedelta(minutes=30),
            "MSFT": self.current_time - timedelta(minutes=30),
            "GOOGL": self.current_time - timedelta(minutes=30),
            "AMZN": self.current_time - timedelta(minutes=30),
            "TSLA": self.current_time - timedelta(minutes=30)
        }
        
        # Run rotation
        costs_bps_round = np.array([10, 10, 10, 10, 10])
        one_min_vol_shares = np.array([100000, 100000, 100000, 100000, 100000])
        spreads_bps = np.array([5, 5, 5, 5, 5])
        
        target_weights, reason_codes = self.engine.rotation_targets(
            self.universe, scores, sigma, pnl_since_entry_bps, atr,
            current_weights, costs_bps_round, one_min_vol_shares,
            spreads_bps, self.current_time
        )
        
        # TSLA should be cut (big loser)
        self.assertEqual(target_weights[4], 0.0)  # TSLA index
        self.assertEqual(reason_codes["TSLA"], "CUT_LOSER")
    
    def test_rotate_green_to_better(self):
        """Test rotating from green to better name."""
        # Set up test data with clear alpha gap
        scores = np.array([0.8, 0.6, 0.4, 0.2, 0.1])  # Clear ranking
        sigma = np.array([0.02, 0.02, 0.02, 0.02, 0.02])
        pnl_since_entry_bps = np.array([100, 50, 0, -50, -100])
        atr = np.array([0.02, 0.02, 0.02, 0.02, 0.02])
        
        # Start with positions in lower-ranked names
        current_weights = np.array([0.0, 0.0, 0.3, 0.3, 0.4])  # Hold GOOGL, AMZN, TSLA
        
        # Set up entry times
        self.engine.entry_times = {
            "AAPL": self.current_time - timedelta(minutes=30),
            "MSFT": self.current_time - timedelta(minutes=30),
            "GOOGL": self.current_time - timedelta(minutes=30),
            "AMZN": self.current_time - timedelta(minutes=30),
            "TSLA": self.current_time - timedelta(minutes=30)
        }
        
        # Run rotation
        costs_bps_round = np.array([5, 5, 5, 5, 5])  # Low costs
        one_min_vol_shares = np.array([100000, 100000, 100000, 100000, 100000])
        spreads_bps = np.array([2, 2, 2, 2, 2])  # Low spreads
        
        target_weights, reason_codes = self.engine.rotation_targets(
            self.universe, scores, sigma, pnl_since_entry_bps, atr,
            current_weights, costs_bps_round, one_min_vol_shares,
            spreads_bps, self.current_time
        )
        
        # Should rotate from lower-ranked to higher-ranked
        # AAPL and MSFT should get positions
        self.assertGreater(target_weights[0], 0)  # AAPL
        self.assertGreater(target_weights[1], 0)  # MSFT
        
        # Should have rotation reasons
        self.assertIn("AAPL", reason_codes)
        self.assertIn("MSFT", reason_codes)
    
    def test_no_trade_band(self):
        """Test no-trade band prevents unnecessary trading."""
        # Set up test data with small changes
        scores = np.array([0.1, 0.1, 0.1, 0.1, 0.1])  # All similar scores
        sigma = np.array([0.02, 0.02, 0.02, 0.02, 0.02])
        pnl_since_entry_bps = np.array([0, 0, 0, 0, 0])
        atr = np.array([0.02, 0.02, 0.02, 0.02, 0.02])
        current_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        
        # Set up entry times
        self.engine.entry_times = {
            "AAPL": self.current_time - timedelta(minutes=30),
            "MSFT": self.current_time - timedelta(minutes=30),
            "GOOGL": self.current_time - timedelta(minutes=30),
            "AMZN": self.current_time - timedelta(minutes=30),
            "TSLA": self.current_time - timedelta(minutes=30)
        }
        
        # Run rotation
        costs_bps_round = np.array([10, 10, 10, 10, 10])
        one_min_vol_shares = np.array([100000, 100000, 100000, 100000, 100000])
        spreads_bps = np.array([5, 5, 5, 5, 5])
        
        target_weights, reason_codes = self.engine.rotation_targets(
            self.universe, scores, sigma, pnl_since_entry_bps, atr,
            current_weights, costs_bps_round, one_min_vol_shares,
            spreads_bps, self.current_time
        )
        
        # Should have minimal changes due to no-trade band
        L1_drift = np.sum(np.abs(target_weights - current_weights))
        self.assertLess(L1_drift, self.config.tau_L1 + 0.001)  # Allow small tolerance
    
    def test_liquidity_filter(self):
        """Test liquidity filter prevents trading in illiquid names."""
        # Set up test data
        scores = np.array([0.8, 0.6, 0.4, 0.2, 0.1])
        sigma = np.array([0.02, 0.02, 0.02, 0.02, 0.02])
        pnl_since_entry_bps = np.array([0, 0, 0, 0, 0])
        atr = np.array([0.02, 0.02, 0.02, 0.02, 0.02])
        current_weights = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Set up entry times
        self.engine.entry_times = {}
        
        # Run rotation with poor liquidity
        costs_bps_round = np.array([10, 10, 10, 10, 10])
        one_min_vol_shares = np.array([1000, 1000, 1000, 1000, 1000])  # Very low volume
        spreads_bps = np.array([100, 100, 100, 100, 100])  # Very wide spreads
        
        target_weights, reason_codes = self.engine.rotation_targets(
            self.universe, scores, sigma, pnl_since_entry_bps, atr,
            current_weights, costs_bps_round, one_min_vol_shares,
            spreads_bps, self.current_time
        )
        
        # Should have liquidity failures
        self.assertTrue(any("LIQ_FAIL" in str(reason) for reason in reason_codes.values()))
    
    def test_utility_threshold(self):
        """Test utility threshold prevents unprofitable rotations."""
        # Set up test data with high costs
        scores = np.array([0.8, 0.6, 0.4, 0.2, 0.1])
        sigma = np.array([0.02, 0.02, 0.02, 0.02, 0.02])
        pnl_since_entry_bps = np.array([0, 0, 0, 0, 0])
        atr = np.array([0.02, 0.02, 0.02, 0.02, 0.02])
        current_weights = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Set up entry times
        self.engine.entry_times = {}
        
        # Run rotation with very high costs
        costs_bps_round = np.array([1000, 1000, 1000, 1000, 1000])  # Very high costs
        one_min_vol_shares = np.array([100000, 100000, 100000, 100000, 100000])
        spreads_bps = np.array([5, 5, 5, 5, 5])
        
        target_weights, reason_codes = self.engine.rotation_targets(
            self.universe, scores, sigma, pnl_since_entry_bps, atr,
            current_weights, costs_bps_round, one_min_vol_shares,
            spreads_bps, self.current_time
        )
        
        # Should have utility failures
        self.assertTrue(any("UTILITY_FAIL" in str(reason) for reason in reason_codes.values()))
    
    def test_holding_age_timeout(self):
        """Test holding age timeout for dead money."""
        # Set up test data
        scores = np.array([0.1, 0.1, 0.1, 0.1, 0.1])  # Low scores
        sigma = np.array([0.02, 0.02, 0.02, 0.02, 0.02])
        pnl_since_entry_bps = np.array([0, 0, 0, 0, 0])
        atr = np.array([0.02, 0.02, 0.02, 0.02, 0.02])
        current_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        
        # Set up old entry times (beyond timeout)
        old_time = self.current_time - timedelta(minutes=150)  # Beyond Tmax_min=120
        self.engine.entry_times = {
            "AAPL": old_time,
            "MSFT": old_time,
            "GOOGL": old_time,
            "AMZN": old_time,
            "TSLA": old_time
        }
        
        # Run rotation
        costs_bps_round = np.array([10, 10, 10, 10, 10])
        one_min_vol_shares = np.array([100000, 100000, 100000, 100000, 100000])
        spreads_bps = np.array([5, 5, 5, 5, 5])
        
        target_weights, reason_codes = self.engine.rotation_targets(
            self.universe, scores, sigma, pnl_since_entry_bps, atr,
            current_weights, costs_bps_round, one_min_vol_shares,
            spreads_bps, self.current_time
        )
        
        # Should cut dead money positions
        self.assertTrue(any("CUT_DEAD_MONEY" in str(reason) for reason in reason_codes.values()))

if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
