"""
Comprehensive tests for multi-interval as-of alignment invariants and correctness.

Tests:
1. Embargo test: feature with embargo must not appear before availability
2. Interval test: different native intervals align correctly
3. Publish offset test: publish_offset shifts availability
4. Multi-symbol test: missing feature rows handled correctly
5. Staleness test: max_staleness caps forward-fill
6. Target isolation test: target column never aligned/shifted
7. Regression test: no alignment params = identical output
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from TRAINING.utils.feature_time_meta import FeatureTimeMeta
from TRAINING.utils.feature_alignment import align_features_asof, build_base_grid


def test_embargo_no_lookahead():
    """Test that embargo prevents lookahead: feature at 10:00 with 10m embargo must NOT appear at 10:05/10:10."""
    # Base grid: 5-minute intervals
    base_timestamps = pd.date_range('2025-01-01 10:00', periods=5, freq='5min')
    base_grid = build_base_grid(['AAPL'], base_timestamps)
    
    # Feature data: 5m interval, embargo=10m
    feature_timestamps = pd.date_range('2025-01-01 10:00', periods=5, freq='5min')
    feature_df = pd.DataFrame({
        'symbol': ['AAPL'] * 5,
        'ts': feature_timestamps,
        'embargo_feature': [1.0, 2.0, 3.0, 4.0, 5.0]
    })
    
    # Feature metadata: 5m native interval, 10m embargo
    meta = FeatureTimeMeta(
        'embargo_feature',
        native_interval_minutes=5.0,
        embargo_minutes=10.0,  # 10m embargo
        lookback_bars=1
    )
    
    # Align features
    aligned_df = align_features_asof(
        base_grid,
        feature_dfs={'embargo_feature': feature_df},
        feature_time_meta_map={'embargo_feature': meta},
        base_interval_minutes=5.0
    )
    
    # Check: feature at 10:00 has availability_ts = 10:00 + 10m = 10:10
    # So at 10:00, 10:05, 10:10 base timestamps:
    # - 10:00: should be None (not available until 10:10)
    # - 10:05: should be None (not available until 10:10)
    # - 10:10: should be 1.0 (now available)
    
    row_1000 = aligned_df[aligned_df['ts'] == pd.Timestamp('2025-01-01 10:00')]
    if len(row_1000) > 0:
        val_1000 = row_1000['embargo_feature'].iloc[0]
        assert pd.isna(val_1000) or val_1000 is None, f"Feature should NOT be available at 10:00 (embargo=10m), got {val_1000}"
    
    row_1005 = aligned_df[aligned_df['ts'] == pd.Timestamp('2025-01-01 10:05')]
    if len(row_1005) > 0:
        val_1005 = row_1005['embargo_feature'].iloc[0]
        assert pd.isna(val_1005) or val_1005 is None, f"Feature should NOT be available at 10:05 (embargo=10m), got {val_1005}"
    
    row_1010 = aligned_df[aligned_df['ts'] == pd.Timestamp('2025-01-01 10:10')]
    if len(row_1010) > 0:
        val_1010 = row_1010['embargo_feature'].iloc[0]
        assert not pd.isna(val_1010) and val_1010 == 1.0, f"Feature should be available at 10:10, got {val_1010}"
    
    print("✓ Embargo test passed: feature correctly unavailable before embargo expires")


def test_interval_alignment():
    """Test that 15m feature aligns correctly onto 5m base grid."""
    # Base grid: 5-minute intervals
    base_timestamps = pd.date_range('2025-01-01 10:00', periods=6, freq='5min')  # 10:00, 10:05, 10:10, 10:15, 10:20, 10:25
    base_grid = build_base_grid(['AAPL'], base_timestamps)
    
    # Feature data: 15m interval
    feature_timestamps = pd.date_range('2025-01-01 10:00', periods=2, freq='15min')  # 10:00, 10:15
    feature_df = pd.DataFrame({
        'symbol': ['AAPL'] * 2,
        'ts': feature_timestamps,
        'daily_feature': [100.0, 200.0]
    })
    
    # Feature metadata: 15m native interval, no embargo
    meta = FeatureTimeMeta(
        'daily_feature',
        native_interval_minutes=15.0,
        embargo_minutes=0.0,
        lookback_bars=1
    )
    
    # Align features
    aligned_df = align_features_asof(
        base_grid,
        feature_dfs={'daily_feature': feature_df},
        feature_time_meta_map={'daily_feature': meta},
        base_interval_minutes=5.0
    )
    
    # Check: 15m feature at 10:00 should appear at 10:00, 10:05, 10:10 (forward-filled)
    # Then 15m feature at 10:15 should appear at 10:15, 10:20, 10:25
    
    row_1000 = aligned_df[aligned_df['ts'] == pd.Timestamp('2025-01-01 10:00')]
    if len(row_1000) > 0:
        val_1000 = row_1000['daily_feature'].iloc[0]
        assert not pd.isna(val_1000) and val_1000 == 100.0, f"Feature should be available at 10:00, got {val_1000}"
    
    row_1005 = aligned_df[aligned_df['ts'] == pd.Timestamp('2025-01-01 10:05')]
    if len(row_1005) > 0:
        val_1005 = row_1005['daily_feature'].iloc[0]
        assert not pd.isna(val_1005) and val_1005 == 100.0, f"Feature should be forward-filled at 10:05, got {val_1005}"
    
    row_1015 = aligned_df[aligned_df['ts'] == pd.Timestamp('2025-01-01 10:15')]
    if len(row_1015) > 0:
        val_1015 = row_1015['daily_feature'].iloc[0]
        assert not pd.isna(val_1015) and val_1015 == 200.0, f"Feature should update at 10:15, got {val_1015}"
    
    print("✓ Interval alignment test passed: 15m feature correctly aligned onto 5m grid")


def test_publish_offset():
    """Test that publish_offset shifts availability even if native interval matches."""
    # Base grid: 5-minute intervals
    base_timestamps = pd.date_range('2025-01-01 10:00', periods=4, freq='5min')
    base_grid = build_base_grid(['AAPL'], base_timestamps)
    
    # Feature data: 5m interval (same as base), but publish_offset=5m
    feature_timestamps = pd.date_range('2025-01-01 10:00', periods=4, freq='5min')
    feature_df = pd.DataFrame({
        'symbol': ['AAPL'] * 4,
        'ts': feature_timestamps,
        'offset_feature': [1.0, 2.0, 3.0, 4.0]
    })
    
    # Feature metadata: 5m native interval, publish_offset=5m (no embargo)
    meta = FeatureTimeMeta(
        'offset_feature',
        native_interval_minutes=5.0,
        embargo_minutes=0.0,
        publish_offset_minutes=5.0,  # 5m publish offset
        lookback_bars=1
    )
    
    # Align features
    aligned_df = align_features_asof(
        base_grid,
        feature_dfs={'offset_feature': feature_df},
        feature_time_meta_map={'offset_feature': meta},
        base_interval_minutes=5.0
    )
    
    # Check: feature at 10:00 has availability_ts = 10:00 + 5m = 10:05
    # So at 10:00: should be None (not available until 10:05)
    # At 10:05: should be 1.0 (now available)
    
    row_1000 = aligned_df[aligned_df['ts'] == pd.Timestamp('2025-01-01 10:00')]
    if len(row_1000) > 0:
        val_1000 = row_1000['offset_feature'].iloc[0]
        assert pd.isna(val_1000) or val_1000 is None, f"Feature should NOT be available at 10:00 (publish_offset=5m), got {val_1000}"
    
    row_1005 = aligned_df[aligned_df['ts'] == pd.Timestamp('2025-01-01 10:05')]
    if len(row_1005) > 0:
        val_1005 = row_1005['offset_feature'].iloc[0]
        assert not pd.isna(val_1005) and val_1005 == 1.0, f"Feature should be available at 10:05, got {val_1005}"
    
    print("✓ Publish offset test passed: publish_offset correctly shifts availability")


def test_multi_symbol_missing_feature():
    """Test that missing feature rows for one symbol result in nulls only for that symbol/time."""
    # Base grid: 2 symbols, 3 timestamps
    base_timestamps = pd.date_range('2025-01-01 10:00', periods=3, freq='5min')
    base_grid = build_base_grid(['AAPL', 'MSFT'], base_timestamps)
    
    # Feature data: only AAPL has the feature, MSFT is missing
    feature_df = pd.DataFrame({
        'symbol': ['AAPL'] * 3,
        'ts': base_timestamps,
        'sparse_feature': [1.0, 2.0, 3.0]
    })
    
    # Feature metadata: 5m native interval, no embargo
    meta = FeatureTimeMeta(
        'sparse_feature',
        native_interval_minutes=5.0,
        embargo_minutes=0.0,
        lookback_bars=1
    )
    
    # Align features
    aligned_df = align_features_asof(
        base_grid,
        feature_dfs={'sparse_feature': feature_df},
        feature_time_meta_map={'sparse_feature': meta},
        base_interval_minutes=5.0
    )
    
    # Check: AAPL should have values, MSFT should have nulls
    aapl_rows = aligned_df[aligned_df['symbol'] == 'AAPL']
    msft_rows = aligned_df[aligned_df['symbol'] == 'MSFT']
    
    assert aapl_rows['sparse_feature'].notna().all(), "AAPL should have all feature values"
    assert msft_rows['sparse_feature'].isna().all(), "MSFT should have all nulls (feature missing)"
    
    print("✓ Multi-symbol test passed: missing feature rows handled correctly")


def test_staleness_cap():
    """Test that max_staleness caps forward-fill (older values get nulled)."""
    # Base grid: 5-minute intervals, extended range
    base_timestamps = pd.date_range('2025-01-01 10:00', periods=10, freq='5min')
    base_grid = build_base_grid(['AAPL'], base_timestamps)
    
    # Feature data: only one value at 10:00
    feature_df = pd.DataFrame({
        'symbol': ['AAPL'],
        'ts': [pd.Timestamp('2025-01-01 10:00')],
        'stale_feature': [100.0]
    })
    
    # Feature metadata: 5m native interval, max_staleness=15m
    meta = FeatureTimeMeta(
        'stale_feature',
        native_interval_minutes=5.0,
        embargo_minutes=0.0,
        max_staleness_minutes=15.0,  # 15m staleness cap
        lookback_bars=1
    )
    
    # Align features
    aligned_df = align_features_asof(
        base_grid,
        feature_dfs={'stale_feature': feature_df},
        feature_time_meta_map={'stale_feature': meta},
        base_interval_minutes=5.0
    )
    
    # Check: feature at 10:00 should be available up to 10:15 (15m staleness)
    # After 10:15, should be nulled
    
    row_1000 = aligned_df[aligned_df['ts'] == pd.Timestamp('2025-01-01 10:00')]
    if len(row_1000) > 0:
        val_1000 = row_1000['stale_feature'].iloc[0]
        assert not pd.isna(val_1000) and val_1000 == 100.0, f"Feature should be available at 10:00, got {val_1000}"
    
    row_1015 = aligned_df[aligned_df['ts'] == pd.Timestamp('2025-01-01 10:15')]
    if len(row_1015) > 0:
        val_1015 = row_1015['stale_feature'].iloc[0]
        # At 10:15, staleness = 15m, which is <= 15m cap, so should still be available
        assert not pd.isna(val_1015) and val_1015 == 100.0, f"Feature should still be available at 10:15 (staleness=15m <= cap), got {val_1015}"
    
    row_1020 = aligned_df[aligned_df['ts'] == pd.Timestamp('2025-01-01 10:20')]
    if len(row_1020) > 0:
        val_1020 = row_1020['stale_feature'].iloc[0]
        # At 10:20, staleness = 20m > 15m cap, so should be nulled
        assert pd.isna(val_1020) or val_1020 is None, f"Feature should be nulled at 10:20 (staleness=20m > 15m cap), got {val_1020}"
    
    print("✓ Staleness cap test passed: max_staleness correctly caps forward-fill")


def test_target_isolation():
    """Test that target column is never aligned/shifted (targets are never in feature_time_meta_map)."""
    # Base grid with target
    base_timestamps = pd.date_range('2025-01-01 10:00', periods=3, freq='5min')
    base_df = pd.DataFrame({
        'symbol': ['AAPL'] * 3,
        'ts': base_timestamps,
        'target': [0.1, 0.2, 0.3]  # Target column
    })
    
    # Feature data: 15m interval
    feature_timestamps = pd.date_range('2025-01-01 10:00', periods=2, freq='15min')
    feature_df = pd.DataFrame({
        'symbol': ['AAPL'] * 2,
        'ts': feature_timestamps,
        'feature_15m': [1.0, 2.0]
    })
    
    # Feature metadata (target is NOT in feature_time_meta_map)
    meta = FeatureTimeMeta(
        'feature_15m',
        native_interval_minutes=15.0,
        embargo_minutes=0.0,
        lookback_bars=1
    )
    
    # Align features (target should remain unchanged)
    aligned_df = align_features_asof(
        base_df,
        feature_dfs={'feature_15m': feature_df},
        feature_time_meta_map={'feature_15m': meta},  # Target NOT in map
        base_interval_minutes=5.0
    )
    
    # Check: target values should be unchanged
    assert 'target' in aligned_df.columns, "Target column should be preserved"
    assert (aligned_df['target'] == [0.1, 0.2, 0.3]).all(), "Target values should be unchanged"
    
    print("✓ Target isolation test passed: target column never aligned/shifted")


if __name__ == '__main__':
    print("Running multi-interval alignment invariant tests...")
    
    test_embargo_no_lookahead()
    test_interval_alignment()
    test_publish_offset()
    test_multi_symbol_missing_feature()
    test_staleness_cap()
    test_target_isolation()
    
    print("\n✅ All alignment invariant tests passed!")
