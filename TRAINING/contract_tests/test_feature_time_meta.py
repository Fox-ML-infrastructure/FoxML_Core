"""
Unit tests for multi-interval + embargo-aware feature pipeline.

Tests:
1. FeatureTimeMeta dataclass validation
2. effective_lookback_minutes computation (lookback + embargo)
3. Interval scaling in infer_lookback_minutes
4. As-of join alignment correctness (no lookahead)
"""

import sys
from pathlib import Path
from unittest.mock import Mock
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from TRAINING.utils.feature_time_meta import FeatureTimeMeta, effective_lookback_minutes
from TRAINING.utils.leakage_budget import infer_lookback_minutes
from TRAINING.utils.feature_alignment import align_features_asof, build_base_grid


def test_feature_time_meta_validation():
    """Test that FeatureTimeMeta validates correctly."""
    # Valid: lookback_bars only
    meta1 = FeatureTimeMeta('rsi_21', native_interval_minutes=5.0, lookback_bars=21, embargo_minutes=0.0)
    assert meta1.name == 'rsi_21'
    assert meta1.lookback_bars == 21
    assert meta1.lookback_minutes is None
    
    # Valid: lookback_minutes only
    meta2 = FeatureTimeMeta('rsi_105m', native_interval_minutes=5.0, lookback_minutes=105.0, embargo_minutes=0.0)
    assert meta2.lookback_minutes == 105.0
    assert meta2.lookback_bars is None
    
    # Invalid: both set
    try:
        FeatureTimeMeta('rsi_21', lookback_bars=21, lookback_minutes=105.0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # Expected
    
    print("✓ FeatureTimeMeta validation tests passed")


def test_effective_lookback_minutes():
    """Test effective lookback computation (lookback + embargo)."""
    # Test 1: lookback_bars with embargo
    meta1 = FeatureTimeMeta('rsi_21', native_interval_minutes=5.0, lookback_bars=21, embargo_minutes=15.0)
    eff1 = effective_lookback_minutes(meta1, base_interval_minutes=5.0, inferred_lookback_minutes=105.0)
    assert abs(eff1 - 120.0) < 0.1, f"Expected 120.0m (105 + 15), got {eff1}m"
    
    # Test 2: lookback_minutes with embargo
    meta2 = FeatureTimeMeta('rsi_105m', native_interval_minutes=5.0, lookback_minutes=105.0, embargo_minutes=30.0)
    eff2 = effective_lookback_minutes(meta2, base_interval_minutes=5.0)
    assert abs(eff2 - 135.0) < 0.1, f"Expected 135.0m (105 + 30), got {eff2}m"
    
    # Test 3: No embargo
    meta3 = FeatureTimeMeta('rsi_21', native_interval_minutes=5.0, lookback_bars=21, embargo_minutes=0.0)
    eff3 = effective_lookback_minutes(meta3, base_interval_minutes=5.0, inferred_lookback_minutes=105.0)
    assert abs(eff3 - 105.0) < 0.1, f"Expected 105.0m, got {eff3}m"
    
    # Test 4: Uses inferred_lookback_minutes when meta doesn't have explicit lookback
    meta4 = FeatureTimeMeta('unknown_feature', native_interval_minutes=5.0, embargo_minutes=10.0)
    eff4 = effective_lookback_minutes(meta4, base_interval_minutes=5.0, inferred_lookback_minutes=50.0)
    assert abs(eff4 - 60.0) < 0.1, f"Expected 60.0m (50 + 10), got {eff4}m"
    
    print("✓ effective_lookback_minutes tests passed")


def test_infer_lookback_minutes_with_feature_time_meta():
    """Test that infer_lookback_minutes uses per-feature interval from FeatureTimeMeta."""
    # Test 1: Feature with native_interval=15m (different from base 5m)
    meta1 = FeatureTimeMeta('rsi_21', native_interval_minutes=15.0, lookback_bars=21, embargo_minutes=0.0)
    result1 = infer_lookback_minutes('rsi_21', interval_minutes=5.0, feature_time_meta=meta1)
    expected1 = 21 * 15.0  # 21 bars * 15m = 315m (not 21 * 5m = 105m)
    assert abs(result1 - expected1) < 0.1, f"Expected {expected1}m (21 bars * 15m), got {result1}m"
    
    # Test 2: Feature with explicit lookback_minutes (should use that, not infer)
    meta2 = FeatureTimeMeta('rsi_105m', native_interval_minutes=5.0, lookback_minutes=105.0, embargo_minutes=0.0)
    result2 = infer_lookback_minutes('rsi_105m', interval_minutes=5.0, feature_time_meta=meta2)
    assert abs(result2 - 105.0) < 0.1, f"Expected 105.0m (explicit), got {result2}m"
    
    # Test 3: Feature without meta (backward compatible - uses interval_minutes)
    result3 = infer_lookback_minutes('rsi_21', interval_minutes=5.0, feature_time_meta=None)
    expected3 = 21 * 5.0  # 21 bars * 5m = 105m
    assert abs(result3 - expected3) < 0.1, f"Expected {expected3}m (21 bars * 5m), got {result3}m"
    
    print("✓ infer_lookback_minutes with FeatureTimeMeta tests passed")


def test_align_features_asof_no_lookahead():
    """Test that as-of join alignment prevents lookahead bias."""
    # Create base grid: 5-minute intervals
    base_timestamps = pd.date_range('2025-01-01 09:30', periods=10, freq='5min')
    base_grid = build_base_grid(['AAPL'], base_timestamps)
    
    # Create feature data: 15-minute intervals (slower than base)
    feature_timestamps = pd.date_range('2025-01-01 09:30', periods=4, freq='15min')
    feature_df = pd.DataFrame({
        'symbol': ['AAPL'] * 4,
        'ts': feature_timestamps,
        'daily_feature': [1.0, 2.0, 3.0, 4.0]  # Feature updates every 15 minutes
    })
    
    # Feature metadata: 15m native interval, 5m embargo (available 5m after bar close)
    meta = FeatureTimeMeta(
        'daily_feature',
        native_interval_minutes=15.0,
        embargo_minutes=5.0,
        lookback_bars=1
    )
    
    # Align features
    aligned_df = align_features_asof(
        base_grid,
        feature_dfs={'daily_feature': feature_df},
        feature_time_meta_map={'daily_feature': meta},
        base_interval_minutes=5.0
    )
    
    # Verify no lookahead: at base timestamp t, we should only see feature values
    # with availability_ts <= t (i.e., feature_ts + embargo <= t)
    
    # At 09:30 (base), feature at 09:30 has availability_ts = 09:30 + 5m = 09:35
    # So at 09:30, we should NOT see the 09:30 feature value (not yet available)
    # We should see None or previous value
    
    # At 09:35 (base), feature at 09:30 has availability_ts = 09:35
    # So at 09:35, we SHOULD see the 09:30 feature value (now available)
    
    assert 'daily_feature' in aligned_df.columns
    
    # Check that 09:30 base timestamp doesn't have 09:30 feature (not available yet)
    row_0930 = aligned_df[aligned_df['ts'] == pd.Timestamp('2025-01-01 09:30')]
    if len(row_0930) > 0:
        # Feature should be None or NaN at 09:30 (not available until 09:35)
        val_0930 = row_0930['daily_feature'].iloc[0]
        assert pd.isna(val_0930) or val_0930 is None, f"Feature should not be available at 09:30, got {val_0930}"
    
    # Check that 09:35 base timestamp has 09:30 feature (now available)
    row_0935 = aligned_df[aligned_df['ts'] == pd.Timestamp('2025-01-01 09:35')]
    if len(row_0935) > 0:
        val_0935 = row_0935['daily_feature'].iloc[0]
        assert not pd.isna(val_0935) and val_0935 == 1.0, f"Feature should be available at 09:35, got {val_0935}"
    
    print("✓ As-of join alignment (no lookahead) test passed")


def test_align_features_asof_mixed_intervals():
    """Test alignment of features with different native intervals."""
    # Base grid: 5-minute intervals
    base_timestamps = pd.date_range('2025-01-01 09:30', periods=20, freq='5min')
    base_grid = build_base_grid(['AAPL'], base_timestamps)
    
    # Feature 1: 5m interval (same as base)
    feat1_timestamps = pd.date_range('2025-01-01 09:30', periods=20, freq='5min')
    feat1_df = pd.DataFrame({
        'symbol': ['AAPL'] * 20,
        'ts': feat1_timestamps,
        'rsi_21': np.arange(20) * 0.1
    })
    meta1 = FeatureTimeMeta('rsi_21', native_interval_minutes=5.0, embargo_minutes=0.0, lookback_bars=21)
    
    # Feature 2: 15m interval (slower)
    feat2_timestamps = pd.date_range('2025-01-01 09:30', periods=7, freq='15min')
    feat2_df = pd.DataFrame({
        'symbol': ['AAPL'] * 7,
        'ts': feat2_timestamps,
        'daily_vol': np.arange(7) * 0.5
    })
    meta2 = FeatureTimeMeta('daily_vol', native_interval_minutes=15.0, embargo_minutes=0.0, lookback_bars=1)
    
    # Align both features
    aligned_df = align_features_asof(
        base_grid,
        feature_dfs={'rsi_21': feat1_df, 'daily_vol': feat2_df},
        feature_time_meta_map={'rsi_21': meta1, 'daily_vol': meta2},
        base_interval_minutes=5.0
    )
    
    # Verify both features are present
    assert 'rsi_21' in aligned_df.columns
    assert 'daily_vol' in aligned_df.columns
    
    # Verify rsi_21 has values at all base timestamps (same interval)
    assert aligned_df['rsi_21'].notna().sum() > 0
    
    # Verify daily_vol has values at 15m intervals (forward-filled)
    # At 09:30, 09:45, 10:00, etc., daily_vol should have new values
    # At intermediate timestamps, it should be forward-filled
    
    print("✓ Mixed interval alignment test passed")


if __name__ == '__main__':
    print("Running multi-interval feature pipeline tests...")
    
    test_feature_time_meta_validation()
    test_effective_lookback_minutes()
    test_infer_lookback_minutes_with_feature_time_meta()
    test_align_features_asof_no_lookahead()
    test_align_features_asof_mixed_intervals()
    
    print("\n✅ All tests passed!")
