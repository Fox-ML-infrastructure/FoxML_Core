#!/usr/bin/env python3
"""
Copyright (c) 2025-2026 Fox ML Infrastructure LLC

Quick test script for the Feature Registry system (all 4 phases).
Run this in a separate terminal to verify everything works.
"""

import sys
from pathlib import Path

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

print("="*80)
print("Feature Registry System Test")
print("="*80)
print()

# Test Phase 1: Feature Registry
print("📋 Phase 1: Feature Registry")
print("-" * 80)
try:
    from TRAINING.common.feature_registry import FeatureRegistry, get_registry
    
    registry = get_registry()
    print(f"✅ FeatureRegistry loads successfully")
    print(f"   Features: {len(registry.features)}")
    print(f"   Families: {len(registry.families)}")
    
    # Test auto-inference
    test_features = ['ret_5', 'rsi_10', 'tth_5m', 'fwd_ret_5m', 'unknown_feature']
    print(f"\n   Auto-inference test:")
    for feat in test_features:
        metadata = registry.auto_infer_metadata(feat)
        print(f"     {feat}: lag={metadata['lag_bars']}, rejected={metadata.get('rejected', False)}")
    
    # Test is_allowed
    print(f"\n   is_allowed test (horizon=12 bars):")
    for feat in test_features:
        allowed = registry.is_allowed(feat, 12)
        print(f"     {feat}: {'✅ allowed' if allowed else '❌ rejected'}")
    
except Exception as e:
    print(f"❌ Phase 1 failed: {e}")
    import traceback
    traceback.print_exc()

print()
print("="*80)
print()

# Test Phase 3: Leakage Sentinels
print("🔍 Phase 3: Leakage Sentinels")
print("-" * 80)
try:
    from TRAINING.common.leakage_sentinels import LeakageSentinel, SentinelResult
    
    sentinel = LeakageSentinel()
    print(f"✅ LeakageSentinel initializes successfully")
    print(f"   Thresholds: shifted={sentinel.shifted_target_threshold}, "
          f"symbol={sentinel.symbol_holdout_test_threshold}, "
          f"randomized={sentinel.randomized_time_threshold}")
    
    # Test SentinelResult
    result = SentinelResult(
        test_name="test",
        passed=True,
        score=0.3,
        threshold=0.5
    )
    print(f"✅ SentinelResult dataclass works: {result.test_name} (passed={result.passed})")
    
except Exception as e:
    print(f"❌ Phase 3 failed: {e}")
    import traceback
    traceback.print_exc()

print()
print("="*80)
print()

# Test Phase 4: Importance Diff Detector
print("📊 Phase 4: Importance Diff Detector")
print("-" * 80)
try:
    from TRAINING.common.importance_diff_detector import ImportanceDiffDetector, SuspiciousFeature
    
    detector = ImportanceDiffDetector(
        diff_threshold=0.1,
        relative_diff_threshold=0.5,
        min_importance_full=0.01
    )
    print(f"✅ ImportanceDiffDetector initializes successfully")
    print(f"   Thresholds: diff={detector.diff_threshold}, "
          f"relative={detector.relative_diff_threshold}, "
          f"min_importance={detector.min_importance_full}")
    
    # Test SuspiciousFeature
    feat = SuspiciousFeature(
        feature_name='test_feature',
        importance_full=0.8,
        importance_safe=0.1,
        importance_diff=0.7,
        relative_diff=0.875,
        reason='absolute_diff=0.7 > 0.1'
    )
    print(f"✅ SuspiciousFeature dataclass works: {feat.feature_name} (diff={feat.importance_diff:.2f})")
    
except Exception as e:
    print(f"❌ Phase 4 failed: {e}")
    import traceback
    traceback.print_exc()

print()
print("="*80)
print()

# Test Integration: Intelligent Trainer
print("🚀 Integration: Intelligent Trainer")
print("-" * 80)
try:
    from TRAINING.orchestration.intelligent_trainer import IntelligentTrainer
    
    # Just test that it imports and can be instantiated (without actually running)
    print(f"✅ IntelligentTrainer imports successfully")
    print(f"   (Full test requires data directory and symbols)")
    
except Exception as e:
    print(f"❌ Integration test failed: {e}")
    import traceback
    traceback.print_exc()

print()
print("="*80)
print("✅ All tests completed!")
print("="*80)
print()
print("Next steps:")
print("  1. Test with real data: python TRAINING/train.py --help")
print("  2. Enable leakage diagnostics: --run-leakage-diagnostics")
print("  3. Check docs/internal/planning/ for detailed documentation")

