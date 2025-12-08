#!/bin/bash

# Quick test command - just verify filtering works
# Fast verification that fixes are in place

set -e

echo "Quick Leakage Prevention Test"
echo "=============================="
echo ""

python -c "
import sys
sys.path.insert(0, '.')

print('1. Testing config path resolution...')
from scripts.utils.leakage_filtering import _get_config_path, _load_leakage_config
config_path = _get_config_path()
config = _load_leakage_config()
assert config_path.exists(), 'Config path not found!'
assert len(config.get('always_exclude', {}).get('regex_patterns', [])) > 0, 'No patterns loaded!'
print('   ✅ Config loaded successfully')

print()
print('2. Testing feature registry...')
from TRAINING.common.feature_registry import get_registry
registry = get_registry()
assert registry.config_path.exists(), 'Registry path not found!'
assert len(registry.features) > 0, 'No features in registry!'
print(f'   ✅ Registry loaded: {len(registry.features)} features')

print()
print('3. Testing feature filtering...')
from scripts.utils.leakage_filtering import filter_features_for_target

all_cols = ['ts', 'p_up_60m_0.8', 'ret_5', 'rsi_10', 'fwd_ret_5m', 'symbol']
target = 'y_will_peak_60m_0.8'

safe = filter_features_for_target(
    all_cols, target, verbose=False, use_registry=True, data_interval_minutes=5
)

critical_leaks = ['ts', 'p_up_60m_0.8', 'fwd_ret_5m', 'symbol']
all_excluded = all(leak not in safe for leak in critical_leaks)
assert all_excluded, f'Some leaks not excluded! Safe: {safe}'
print(f'   ✅ All critical leaks excluded: {critical_leaks}')
print(f'   ✅ Safe features: {safe}')

print()
print('4. Testing registry without horizon...')
safe_no_horizon = filter_features_for_target(
    ['ts', 'p_up_60m_0.8', 'ret_5'], 'invalid_target_no_horizon', 
    verbose=False, use_registry=True, data_interval_minutes=5
)
assert 'ts' not in safe_no_horizon, 'ts not excluded without horizon!'
assert 'p_up_60m_0.8' not in safe_no_horizon, 'p_up_60m_0.8 not excluded without horizon!'
print('   ✅ Metadata filtering works without horizon')

print()
print('=' * 50)
print('✅ ALL TESTS PASSED!')
print('=' * 50)
print()
print('The leakage prevention system is working correctly.')
print('You can now run the full training pipeline.')
"

