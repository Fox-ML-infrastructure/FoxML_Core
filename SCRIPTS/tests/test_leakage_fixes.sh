#!/bin/bash

# Test command for leakage prevention system fixes
# This verifies that ts and p_* features are properly excluded

set -e

echo "=" | tr -d '\n' | head -c 80
echo ""
echo "Testing Leakage Prevention System Fixes"
echo "=" | tr -d '\n' | head -c 80
echo ""
echo ""

# Test 1: Quick config and registry verification
echo "📋 Test 1: Config & Registry Path Resolution"
echo "----------------------------------------"
python -c "
import sys
sys.path.insert(0, '.')
from TRAINING.utils.leakage_filtering import _get_config_path, _load_leakage_config
from TRAINING.common.feature_registry import get_registry

config_path = _get_config_path()
config = _load_leakage_config()
registry = get_registry()

print(f'✅ Config path: {config_path.exists()}')
print(f'✅ Config loaded: {len(config.get(\"always_exclude\", {}).get(\"regex_patterns\", [])) > 0} patterns')
print(f'✅ Registry loaded: {len(registry.features)} features')
print(f'✅ Registry path: {registry.config_path.exists()}')
"

echo ""
echo "=" | tr -d '\n' | head -c 80
echo ""
echo ""

# Test 2: Feature filtering verification
echo "📋 Test 2: Feature Filtering (ts and p_* exclusion)"
echo "----------------------------------------"
python -c "
import sys
sys.path.insert(0, '.')
from TRAINING.utils.leakage_filtering import filter_features_for_target

all_cols = ['ts', 'p_up_60m_0.8', 'ret_5', 'rsi_10', 'fwd_ret_5m', 'y_will_peak_60m', 'symbol']
target = 'y_will_peak_60m_0.8'

safe = filter_features_for_target(
    all_cols, target, verbose=False, use_registry=True, data_interval_minutes=5
)

excluded = set(all_cols) - set(safe)
critical_leaks = ['ts', 'p_up_60m_0.8', 'fwd_ret_5m', 'y_will_peak_60m', 'symbol']

print(f'Target: {target}')
print(f'All columns: {len(all_cols)}')
print(f'Safe columns: {len(safe)}')
print(f'Excluded: {len(excluded)}')
print()

all_good = True
for leak in critical_leaks:
    if leak in safe:
        print(f'❌ {leak} was NOT excluded (SHOULD BE REJECTED)')
        all_good = False
    else:
        print(f'✅ {leak} correctly excluded')

if all_good:
    print()
    print('✅ All critical leaks properly excluded!')
else:
    print()
    print('❌ Some leaks not excluded - check configuration!')
    exit(1)
"

echo ""
echo "=" | tr -d '\n' | head -c 80
echo ""
echo ""

# Test 3: End-to-end training pipeline test (minimal)
echo "📋 Test 3: End-to-End Training Pipeline Test"
echo "----------------------------------------"
echo "Running intelligent training pipeline with leakage diagnostics..."
echo ""

DATA_DIR="data/data_labeled/interval=5m"
OUTPUT_DIR="test_leakage_fixes_output"
SYMBOLS="AAPL MSFT"

# Clean up previous output
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

python TRAINING/train.py \
    --data-dir "$DATA_DIR" \
    --symbols $SYMBOLS \
    --output-dir "$OUTPUT_DIR" \
    --auto-targets \
    --top-n-targets 2 \
    --auto-features \
    --top-m-features 30 \
    --run-leakage-diagnostics \
    --families LightGBM \
    --min-cs 5 \
    --max-rows-per-symbol 5000 \
    --max-rows-train 10000 \
    --force-refresh 2>&1 | tee "$OUTPUT_DIR/test.log" | grep -E "(LEAKAGE|excluded|ts|p_up|Filtered|registry|horizon)" || true

echo ""
echo "=" | tr -d '\n' | head -c 80
echo ""
echo "✅ Test completed!"
echo ""
echo "Check output in: $OUTPUT_DIR"
echo "Check logs for:"
echo "  - 'ts' should be excluded"
echo "  - 'p_up_60m_0.8' should be excluded"
echo "  - Feature registry messages"
echo "  - No 'Leakage config not found' warnings"
echo ""

