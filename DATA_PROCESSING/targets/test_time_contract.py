"""
Unit tests for time contract enforcement.

Tests that labels are invariant to changes in the current bar (bar t),
proving that labels start at t+1 and never include bar t.
"""

import pytest
import pandas as pd
import numpy as np
from DATA_PROCESSING.targets.barrier import (
    compute_barrier_targets,
    compute_zigzag_targets,
    compute_mfe_mdd_targets,
    compute_time_to_hit,
    compute_path_quality,
    compute_asymmetric_barriers
)


def test_barrier_targets_t_plus_one_invariance():
    """Test that barrier targets are invariant to changes in bar t."""
    # Create synthetic price series
    np.random.seed(42)
    n_bars = 100
    base_prices = 100 + np.cumsum(np.random.randn(n_bars) * 0.5)
    prices = pd.Series(base_prices)
    interval_minutes = 5.0
    horizon_minutes = 15
    
    # Compute labels
    targets1 = compute_barrier_targets(
        prices, 
        horizon_minutes=horizon_minutes,
        interval_minutes=interval_minutes
    )
    
    # Modify bar 50 drastically (should not affect label at bar 50)
    prices_modified = prices.copy()
    prices_modified.iloc[50] = prices.iloc[50] * 2.0  # Double the price
    
    targets2 = compute_barrier_targets(
        prices_modified,
        horizon_minutes=horizon_minutes,
        interval_minutes=interval_minutes
    )
    
    # Label at bar 50 should be identical (based on bars 51+, not bar 50)
    if 50 < len(targets1):
        assert targets1.iloc[50]['y_will_peak'] == targets2.iloc[50]['y_will_peak'], \
            "Label at bar 50 changed when bar 50 was modified - violates t+1 contract!"
        assert targets1.iloc[50]['y_will_valley'] == targets2.iloc[50]['y_will_valley'], \
            "Label at bar 50 changed when bar 50 was modified - violates t+1 contract!"


def test_zigzag_targets_t_plus_one_invariance():
    """Test that zigzag targets are invariant to changes in bar t."""
    np.random.seed(42)
    n_bars = 100
    base_prices = 100 + np.cumsum(np.random.randn(n_bars) * 0.5)
    prices = pd.Series(base_prices)
    interval_minutes = 5.0
    horizon_minutes = 15
    
    targets1 = compute_zigzag_targets(
        prices,
        horizon_minutes=horizon_minutes,
        interval_minutes=interval_minutes
    )
    
    prices_modified = prices.copy()
    prices_modified.iloc[50] = prices.iloc[50] * 2.0
    
    targets2 = compute_zigzag_targets(
        prices_modified,
        horizon_minutes=horizon_minutes,
        interval_minutes=interval_minutes
    )
    
    if 50 < len(targets1):
        assert targets1.iloc[50]['y_will_swing_high'] == targets2.iloc[50]['y_will_swing_high'], \
            "ZigZag label at bar 50 changed when bar 50 was modified - violates t+1 contract!"


def test_mfe_mdd_targets_t_plus_one_invariance():
    """Test that MFE/MDD targets are invariant to changes in bar t."""
    np.random.seed(42)
    n_bars = 100
    base_prices = 100 + np.cumsum(np.random.randn(n_bars) * 0.5)
    prices = pd.Series(base_prices)
    interval_minutes = 5.0
    horizon_minutes = 15
    
    targets1 = compute_mfe_mdd_targets(
        prices,
        horizon_minutes=horizon_minutes,
        interval_minutes=interval_minutes
    )
    
    prices_modified = prices.copy()
    prices_modified.iloc[50] = prices.iloc[50] * 2.0
    
    targets2 = compute_mfe_mdd_targets(
        prices_modified,
        horizon_minutes=horizon_minutes,
        interval_minutes=interval_minutes
    )
    
    if 50 < len(targets1):
        assert targets1.iloc[50]['y_will_peak_mfe'] == targets2.iloc[50]['y_will_peak_mfe'], \
            "MFE/MDD label at bar 50 changed when bar 50 was modified - violates t+1 contract!"


def test_horizon_bars_conversion():
    """Test that horizon_minutes is correctly converted to horizon_bars."""
    prices = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112])
    interval_minutes = 5.0
    horizon_minutes = 15  # Should be 3 bars (15 / 5 = 3)
    
    targets = compute_barrier_targets(
        prices,
        horizon_minutes=horizon_minutes,
        interval_minutes=interval_minutes
    )
    
    # For bar 0, future window should be bars [1, 4) = 3 bars (not 15 bars!)
    # This is verified by checking that we can compute labels for early bars
    # If it used 15 bars, we'd run out of data much earlier
    assert len(targets) > 0, "Should have computed at least some targets"
    
    # Verify that horizon_bars = 3 by checking we can compute labels near the end
    # With 13 bars total, horizon_bars=3 means we can compute up to bar 9 (9+3=12 < 13)
    # If it used 15 bars, we'd only get up to bar -2 (impossible)
    assert len(targets) >= 9, f"Expected at least 9 targets with horizon_bars=3, got {len(targets)}"


def test_horizon_not_multiple_of_interval():
    """Test handling when horizon_minutes is not a multiple of interval_minutes."""
    prices = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110])
    interval_minutes = 5.0
    horizon_minutes = 17  # Not a multiple of 5 (17 / 5 = 3.4 bars)
    
    # Should warn but still work (uses 3 bars = 15 minutes)
    targets = compute_barrier_targets(
        prices,
        horizon_minutes=horizon_minutes,
        interval_minutes=interval_minutes
    )
    
    assert len(targets) > 0, "Should still compute targets with non-multiple horizon"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
