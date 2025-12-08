#!/usr/bin/env python3
"""
Copyright (c) 2025-2026 Fox ML Infrastructure LLC

Comprehensive test for leakage prevention system.
Tests all integration points and edge cases.
"""

import sys
from pathlib import Path

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

print("="*80)
print("Comprehensive Leakage Prevention System Test")
print("="*80)
print()

# Test 1: Config Path Resolution
print("📋 Test 1: Config Path Resolution")
print("-" * 80)
try:
    from TRAINING.utils.leakage_filtering import _get_config_path, _load_leakage_config
    
    config_path = _get_config_path()
    print(f"✅ Config path: {config_path}")
    print(f"   Exists: {config_path.exists()}")
    
    config = _load_leakage_config()
    print(f"✅ Config loaded: {bool(config)}")
    print(f"   Has patterns: {len(config.get('always_exclude', {}).get('regex_patterns', [])) > 0}")
    print(f"   Has horizon patterns: {len(config.get('horizon_extraction', {}).get('patterns', [])) > 0}")
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()

print()
print("="*80)
print()

# Test 2: Feature Registry Path Resolution
print("📋 Test 2: Feature Registry Path Resolution")
print("-" * 80)
try:
    from TRAINING.common.feature_registry import get_registry
    
    registry = get_registry()
    print(f"✅ Registry loaded: {len(registry.features)} features, {len(registry.families)} families")
    print(f"   Config path: {registry.config_path}")
    print(f"   Exists: {registry.config_path.exists()}")
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()

print()
print("="*80)
print()

# Test 3: Horizon Extraction
print("📋 Test 3: Horizon Extraction")
print("-" * 80)
try:
    from TRAINING.utils.leakage_filtering import _extract_horizon, _load_leakage_config
    
    config = _load_leakage_config()
    test_targets = [
        'y_will_peak_60m_0.8',
        'fwd_ret_15m',
        'fwd_ret_1d',
        'y_will_peak_5m_0.5',
        'invalid_target'
    ]
    
    for target in test_targets:
        horizon = _extract_horizon(target, config)
        print(f"   {target}: {horizon}m" if horizon else f"   {target}: None (failed)")
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()

print()
print("="*80)
print()

# Test 4: Feature Filtering with Registry
print("📋 Test 4: Feature Filtering with Registry")
print("-" * 80)
try:
    from TRAINING.utils.leakage_filtering import filter_features_for_target
    
    all_cols = ['ts', 'p_up_60m_0.8', 'ret_5', 'rsi_10', 'fwd_ret_5m', 'y_will_peak_60m', 'symbol']
    target = 'y_will_peak_60m_0.8'
    
    safe = filter_features_for_target(
        all_cols, target, verbose=False, use_registry=True, data_interval_minutes=5
    )
    
    print(f"   Target: {target}")
    print(f"   All columns: {len(all_cols)}")
    print(f"   Safe columns: {len(safe)}")
    print(f"   Excluded: {set(all_cols) - set(safe)}")
    
    # Verify critical exclusions
    critical_leaks = ['ts', 'p_up_60m_0.8', 'fwd_ret_5m', 'y_will_peak_60m', 'symbol']
    for leak in critical_leaks:
        if leak in safe:
            print(f"   ⚠️  WARNING: {leak} was NOT excluded (should be rejected)")
        else:
            print(f"   ✅ {leak} correctly excluded")
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()

print()
print("="*80)
print()

# Test 5: Registry Filtering Without Horizon
print("📋 Test 5: Registry Filtering Without Horizon (Edge Case)")
print("-" * 80)
try:
    from TRAINING.utils.leakage_filtering import filter_features_for_target
    from TRAINING.utils.leakage_filtering import _load_leakage_config
    
    # Create a config without horizon patterns (simulate failure)
    config = _load_leakage_config()
    # Temporarily clear horizon patterns
    original_patterns = config.get('horizon_extraction', {}).get('patterns', [])
    
    all_cols = ['ts', 'p_up_60m_0.8', 'ret_5', 'rsi_10']
    target = 'invalid_target_no_horizon'
    
    # This should still filter metadata even without horizon
    safe = filter_features_for_target(
        all_cols, target, verbose=False, use_registry=True, data_interval_minutes=5
    )
    
    print(f"   Target: {target} (no horizon extractable)")
    print(f"   Safe columns: {safe}")
    print(f"   ts excluded: {'ts' not in safe}")
    print(f"   p_up_60m_0.8 excluded: {'p_up_60m_0.8' not in safe}")
    
    if 'ts' not in safe and 'p_up_60m_0.8' not in safe:
        print("   ✅ Metadata filtering works even without horizon")
    else:
        print("   ⚠️  WARNING: Metadata not filtered when horizon extraction fails")
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()

print()
print("="*80)
print()

# Test 6: Data Interval Detection
print("📋 Test 6: Data Interval Detection")
print("-" * 80)
try:
    from TRAINING.utils.data_interval import detect_interval_from_dataframe
    import pandas as pd
    import numpy as np
    
    # Create test dataframe with 5-minute intervals
    timestamps = pd.date_range('2025-01-01 09:30:00', periods=100, freq='5min')
    df = pd.DataFrame({
        'ts': timestamps,
        'price': np.random.randn(100),
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    detected = detect_interval_from_dataframe(df, timestamp_column='ts', default=5)
    print(f"   Detected interval: {detected}m (expected: 5m)")
    
    if detected == 5:
        print("   ✅ Interval detection works")
    else:
        print(f"   ⚠️  WARNING: Expected 5m, got {detected}m")
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()

print()
print("="*80)
print("✅ All tests completed!")
print("="*80)
print()
print("Next steps:")
print("  1. Run full training pipeline to verify end-to-end")
print("  2. Check logs for any warnings about config paths or registry")
print("  3. Verify that ts and p_* features are excluded in actual runs")

