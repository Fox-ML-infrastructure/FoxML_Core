"""
Unit test for _Xd pattern inference in gatekeeper/sanitizer.

Tests that _Xd suffix features (e.g., price_momentum_60d, volatility_20d) are correctly
inferred with nonzero lookback values and are dropped/quarantined under the 240m cap.
"""
import sys
import re
sys.path.insert(0, '.')

from TRAINING.utils.leakage_budget import infer_lookback_minutes, compute_feature_lookback_max


def test_xd_pattern_inference():
    """Test that _Xd patterns are correctly inferred."""
    test_features = [
        'price_momentum_60d',      # Should be 86400m (60 * 1440)
        'volume_momentum_20d',     # Should be 28800m (20 * 1440)
        'volatility_3d',           # Should be 4320m (3 * 1440)
        'market_correlation_60d',  # Should be 86400m (60 * 1440)
        'rsi_30',                  # Should be 150m (30 bars * 5m interval)
        'cci_30',                  # Should be 150m (30 bars * 5m interval)
    ]
    
    print("Testing _Xd pattern inference:")
    all_passed = True
    
    for feat in test_features:
        result = infer_lookback_minutes(feat, 5.0, unknown_policy='drop')
        
        # Check if it's an _Xd pattern
        days_match = re.search(r'_(\d+(?:\.\d+)?)(d|D)(?!\d)', feat)
        if days_match:
            expected = float(days_match.group(1)) * 1440.0
            if abs(result - expected) < 1.0:
                print(f"  ✅ {feat:30} → {result:.1f}m (expected: {expected:.1f}m)")
            else:
                print(f"  ❌ {feat:30} → {result:.1f}m (expected: {expected:.1f}m) - FAILED")
                all_passed = False
        else:
            # Not an _Xd pattern, just check it's not 0.0
            if result == 0.0:
                print(f"  ⚠️  {feat:30} → {result:.1f}m (not _Xd, but 0.0 might be wrong)")
            else:
                print(f"  ✅ {feat:30} → {result:.1f}m (not _Xd pattern)")
    
    return all_passed


def test_canonical_map_includes_xd():
    """Test that canonical map includes _Xd features with correct lookback."""
    test_features = ['price_momentum_60d', 'volume_momentum_20d', 'volatility_3d', 'rsi_30']
    
    print("\nTesting canonical map includes _Xd features:")
    result = compute_feature_lookback_max(test_features, 5.0, stage='TEST')
    canonical_map = result.canonical_lookback_map if hasattr(result, 'canonical_lookback_map') else None
    
    if not canonical_map:
        print("  ❌ Canonical map not available - FAILED")
        return False
    
    all_passed = True
    from TRAINING.utils.leakage_budget import _feat_key
    
    for feat in test_features:
        normalized = _feat_key(feat)
        lookback = canonical_map.get(normalized, None)
        
        if lookback is None:
            print(f"  ❌ {feat:30} → NOT_FOUND in canonical map - FAILED")
            all_passed = False
        elif lookback == 0.0 and '_d' in feat.lower():
            print(f"  ❌ {feat:30} → {lookback} (BUG: _Xd feature has 0.0 lookback) - FAILED")
            all_passed = False
        else:
            print(f"  ✅ {feat:30} → {lookback:.1f}m")
    
    return all_passed


def test_gatekeeper_drops_xd_offenders():
    """Test that gatekeeper would drop _Xd features exceeding cap."""
    test_features = ['price_momentum_60d', 'volume_momentum_20d', 'volatility_3d', 'rsi_30']
    cap_minutes = 240.0  # 4 hours
    
    print(f"\nTesting gatekeeper would drop _Xd features exceeding cap ({cap_minutes}m):")
    result = compute_feature_lookback_max(test_features, 5.0, stage='TEST')
    canonical_map = result.canonical_lookback_map if hasattr(result, 'canonical_lookback_map') else None
    
    if not canonical_map:
        print("  ❌ Canonical map not available - FAILED")
        return False
    
    from TRAINING.utils.leakage_budget import _feat_key
    offenders = []
    
    for feat in test_features:
        normalized = _feat_key(feat)
        lookback = canonical_map.get(normalized, None)
        
        if lookback is not None and lookback > cap_minutes:
            offenders.append((feat, lookback))
            print(f"  ✅ {feat:30} → {lookback:.1f}m > {cap_minutes:.1f}m (would be dropped)")
        elif lookback is not None:
            print(f"  ✅ {feat:30} → {lookback:.1f}m <= {cap_minutes:.1f}m (would be kept)")
    
    expected_offenders = ['price_momentum_60d', 'volume_momentum_20d', 'volatility_3d']
    expected_offender_names = {f for f, _ in offenders}
    
    if all(f in expected_offender_names for f in expected_offenders):
        print(f"  ✅ All expected offenders would be dropped: {expected_offenders}")
        return True
    else:
        missing = set(expected_offenders) - expected_offender_names
        print(f"  ❌ Missing expected offenders: {missing} - FAILED")
        return False


if __name__ == '__main__':
    print("=" * 70)
    print("Unit tests for _Xd pattern inference")
    print("=" * 70)
    
    test1 = test_xd_pattern_inference()
    test2 = test_canonical_map_includes_xd()
    test3 = test_gatekeeper_drops_xd_offenders()
    
    print("\n" + "=" * 70)
    if test1 and test2 and test3:
        print("✅ All tests PASSED")
        sys.exit(0)
    else:
        print("❌ Some tests FAILED")
        sys.exit(1)
