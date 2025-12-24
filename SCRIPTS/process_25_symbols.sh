#!/bin/bash
# Process data for 25 symbols with reworked barrier targets
# This script updates the experiment config to use 25 symbols

set -e

# Common symbols for testing (25 symbols)
SYMBOLS_25="AAPL MSFT GOOGL AMZN TSLA META NVDA JPM V JNJ WMT PG MA UNH HD DIS BAC ADBE PYPL NFLX CMCSA PFE KO AVGO COST"

echo "=========================================="
echo "Processing Data for 25 Symbols"
echo "=========================================="
echo ""
echo "This script will:"
echo "1. Update experiment config to use 25 symbols"
echo "2. Show you the command to run the training pipeline"
echo ""

# Update the experiment config
CONFIG_FILE="CONFIG/experiments/e2e_full_targets_test.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "📝 Updating $CONFIG_FILE with 25 symbols..."

# Create backup
cp "$CONFIG_FILE" "${CONFIG_FILE}.backup"

# Update symbols list (using Python for YAML safety)
python3 << PYTHON_EOF
import yaml
from pathlib import Path

config_file = Path("$CONFIG_FILE")
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

# Update symbols list
symbols_25 = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "JNJ",
    "WMT", "PG", "MA", "UNH", "HD", "DIS", "BAC", "ADBE", "PYPL", "NFLX",
    "CMCSA", "PFE", "KO", "AVGO", "COST"
]

config['data']['symbols'] = symbols_25

# Also update max_samples_per_symbol if it's too low for 25 symbols
if config['data'].get('max_samples_per_symbol', 0) < 2000:
    config['data']['max_samples_per_symbol'] = 2000
    config['data']['max_rows_per_symbol'] = 2000

with open(config_file, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f"✅ Updated config with {len(symbols_25)} symbols")
PYTHON_EOF

echo ""
echo "✅ Config updated successfully!"
echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo ""
echo "To run the training pipeline with 25 symbols:"
echo ""
echo "python -m TRAINING.orchestration.intelligent_trainer \\"
echo "    --experiment e2e_full_targets_test \\"
echo "    --auto-targets \\"
echo "    --auto-features \\"
echo "    --output-dir test_25_symbols_run"
echo ""
echo "Or if you need to process raw data first, check DATA_PROCESSING/ for"
echo "data processing scripts, or use the Python API directly."
echo ""
echo "To restore the original config:"
echo "  cp ${CONFIG_FILE}.backup $CONFIG_FILE"
echo ""

