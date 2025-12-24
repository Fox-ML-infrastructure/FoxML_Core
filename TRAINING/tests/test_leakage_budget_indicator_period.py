"""
Unit tests for indicator-period feature lookback inference.

Tests the fix for the bug where registry lag_bars=0 caused indicator-period
features (e.g., stoch_k_21, williams_r_21) to incorrectly resolve to 0.0m lookback
instead of period_bars * interval_minutes.
"""

import sys
from pathlib import Path
from unittest.mock import Mock

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from TRAINING.utils.leakage_budget import infer_lookback_minutes, _is_indicator_period_feature


class MockRegistry:
    """Mock registry that returns lag_bars=0 for testing."""
    
    def __init__(self, lag_bars=0):
        self.lag_bars = lag_bars
    
    def get_feature_metadata(self, feature_name):
        return {'lag_bars': self.lag_bars}


def test_indicator_period_feature_detection():
    """Test that _is_indicator_period_feature correctly identifies indicator-period features."""
    # Positive cases - simple patterns
    positive_simple = [
        'stoch_k_21', 'stoch_d_21', 'williams_r_21', 'rsi_30', 'rsi_21',
        'cci_30', 'mfi_21', 'sma_20', 'ema_50', 'ret_288', 'atr_14',
        'adx_14', 'macd_12', 'bb_20', 'mom_10', 'std_30', 'var_30', 'vol_20',
    ]
    for name in positive_simple:
        assert _is_indicator_period_feature(name), f"{name} should be detected as indicator-period feature"
    
    # Positive cases - compound patterns (from actual codebase)
    positive_compound = [
        'bb_upper_20', 'bb_lower_20', 'bb_width_20', 'bb_percent_b_20',
        'rsi_wilder_14', 'stoch_k_fast_21',  # Future variants
    ]
    for name in positive_compound:
        assert _is_indicator_period_feature(name), f"{name} should be detected as indicator-period feature"
    
    # Negative cases (false positives we want to avoid)
    negative_cases = [
        'hour_15', 'day_1', 'month_12', 'wd_1', '_day', '_month',
        'close', 'volume', 'day_of_week', 'is_weekend', 'trading_day',
        'macd_signal', 'macd_hist',  # No period suffix
    ]
    for name in negative_cases:
        assert not _is_indicator_period_feature(name), f"{name} should NOT be detected as indicator-period feature"


def test_stoch_k_21_with_registry_zero():
    """Test that stoch_k_21 correctly resolves to 105.0m (21 bars * 5m) even when registry returns lag_bars=0."""
    registry = MockRegistry(lag_bars=0)
    result = infer_lookback_minutes(
        feature_name='stoch_k_21',
        interval_minutes=5.0,
        registry=registry
    )
    assert result == 105.0, f"Expected 105.0m (21 bars * 5m), got {result}m"


def test_williams_r_21_with_registry_zero():
    """Test that williams_r_21 correctly resolves to 105.0m (21 bars * 5m) even when registry returns lag_bars=0."""
    registry = MockRegistry(lag_bars=0)
    result = infer_lookback_minutes(
        feature_name='williams_r_21',
        interval_minutes=5.0,
        registry=registry
    )
    assert result == 105.0, f"Expected 105.0m (21 bars * 5m), got {result}m"


def test_cci_30_with_registry_zero():
    """Test that cci_30 correctly resolves to 150.0m (30 bars * 5m) even when registry returns lag_bars=0."""
    registry = MockRegistry(lag_bars=0)
    result = infer_lookback_minutes(
        feature_name='cci_30',
        interval_minutes=5.0,
        registry=registry
    )
    assert result == 150.0, f"Expected 150.0m (30 bars * 5m), got {result}m"


def test_rsi_21_with_registry_zero():
    """Test that rsi_21 correctly resolves to 105.0m (21 bars * 5m) even when registry returns lag_bars=0."""
    registry = MockRegistry(lag_bars=0)
    result = infer_lookback_minutes(
        feature_name='rsi_21',
        interval_minutes=5.0,
        registry=registry
    )
    assert result == 105.0, f"Expected 105.0m (21 bars * 5m), got {result}m"


def test_mfi_21_with_registry_zero():
    """Test that mfi_21 correctly resolves to 105.0m (21 bars * 5m) even when registry returns lag_bars=0."""
    registry = MockRegistry(lag_bars=0)
    result = infer_lookback_minutes(
        feature_name='mfi_21',
        interval_minutes=5.0,
        registry=registry
    )
    assert result == 105.0, f"Expected 105.0m (21 bars * 5m), got {result}m"


def test_stoch_d_21_with_registry_zero():
    """Test that stoch_d_21 correctly resolves to 105.0m (21 bars * 5m) even when registry returns lag_bars=0."""
    registry = MockRegistry(lag_bars=0)
    result = infer_lookback_minutes(
        feature_name='stoch_d_21',
        interval_minutes=5.0,
        registry=registry
    )
    assert result == 105.0, f"Expected 105.0m (21 bars * 5m), got {result}m"


def test_stoch_k_21_scales_with_interval():
    """Test that stoch_k_21 lookback scales proportionally with interval_minutes."""
    registry = MockRegistry(lag_bars=0)
    
    # Test with interval=1m
    result_1m = infer_lookback_minutes(
        feature_name='stoch_k_21',
        interval_minutes=1.0,
        registry=registry
    )
    assert result_1m == 21.0, f"Expected 21.0m (21 bars * 1m), got {result_1m}m"
    
    # Test with interval=5m
    result_5m = infer_lookback_minutes(
        feature_name='stoch_k_21',
        interval_minutes=5.0,
        registry=registry
    )
    assert result_5m == 105.0, f"Expected 105.0m (21 bars * 5m), got {result_5m}m"
    
    # Verify scaling relationship
    assert result_5m == result_1m * 5, "Lookback should scale proportionally with interval"


def test_rsi_30_scales_with_interval():
    """Test that rsi_30 lookback scales proportionally with interval_minutes."""
    registry = MockRegistry(lag_bars=0)
    
    # Test with interval=1m
    result_1m = infer_lookback_minutes(
        feature_name='rsi_30',
        interval_minutes=1.0,
        registry=registry
    )
    assert result_1m == 30.0, f"Expected 30.0m (30 bars * 1m), got {result_1m}m"
    
    # Test with interval=5m
    result_5m = infer_lookback_minutes(
        feature_name='rsi_30',
        interval_minutes=5.0,
        registry=registry
    )
    assert result_5m == 150.0, f"Expected 150.0m (30 bars * 5m), got {result_5m}m"
    
    # Verify scaling relationship
    assert result_5m == result_1m * 5, "Lookback should scale proportionally with interval"


def test_stoch_k_21_with_spec_zero():
    """Test that stoch_k_21 correctly resolves even when spec_lookback_minutes=0.0."""
    result = infer_lookback_minutes(
        feature_name='stoch_k_21',
        interval_minutes=5.0,
        spec_lookback_minutes=0.0
    )
    assert result == 105.0, f"Expected 105.0m (21 bars * 5m), got {result}m"


def test_williams_r_21_with_spec_zero():
    """Test that williams_r_21 correctly resolves even when spec_lookback_minutes=0.0."""
    result = infer_lookback_minutes(
        feature_name='williams_r_21',
        interval_minutes=5.0,
        spec_lookback_minutes=0.0
    )
    assert result == 105.0, f"Expected 105.0m (21 bars * 5m), got {result}m"


def test_bb_upper_20_with_registry_zero():
    """Test that bb_upper_20 (compound indicator) correctly resolves even when registry returns lag_bars=0."""
    registry = MockRegistry(lag_bars=0)
    result = infer_lookback_minutes(
        feature_name='bb_upper_20',
        interval_minutes=5.0,
        registry=registry
    )
    assert result == 100.0, f"Expected 100.0m (20 bars * 5m), got {result}m"


def test_bb_lower_20_with_registry_zero():
    """Test that bb_lower_20 (compound indicator) correctly resolves even when registry returns lag_bars=0."""
    registry = MockRegistry(lag_bars=0)
    result = infer_lookback_minutes(
        feature_name='bb_lower_20',
        interval_minutes=5.0,
        registry=registry
    )
    assert result == 100.0, f"Expected 100.0m (20 bars * 5m), got {result}m"


if __name__ == '__main__':
    print("Running indicator-period feature lookback tests...")
    test_indicator_period_feature_detection()
    print("✓ Pattern detection tests passed")
    
    test_stoch_k_21_with_registry_zero()
    print("✓ stoch_k_21 with registry zero passed")
    
    test_williams_r_21_with_registry_zero()
    print("✓ williams_r_21 with registry zero passed")
    
    test_cci_30_with_registry_zero()
    print("✓ cci_30 with registry zero passed")
    
    test_rsi_21_with_registry_zero()
    print("✓ rsi_21 with registry zero passed")
    
    test_mfi_21_with_registry_zero()
    print("✓ mfi_21 with registry zero passed")
    
    test_stoch_d_21_with_registry_zero()
    print("✓ stoch_d_21 with registry zero passed")
    
    test_stoch_k_21_scales_with_interval()
    print("✓ stoch_k_21 interval scaling passed")
    
    test_rsi_30_scales_with_interval()
    print("✓ rsi_30 interval scaling passed")
    
    test_stoch_k_21_with_spec_zero()
    print("✓ stoch_k_21 with spec zero passed")
    
    test_williams_r_21_with_spec_zero()
    print("✓ williams_r_21 with spec zero passed")
    
    test_bb_upper_20_with_registry_zero()
    print("✓ bb_upper_20 with registry zero passed")
    
    test_bb_lower_20_with_registry_zero()
    print("✓ bb_lower_20 with registry zero passed")
    
    print("\n✅ All tests passed!")
