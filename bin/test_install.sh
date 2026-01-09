#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

#
# Quick Test Script for FoxML Core Installation
#
# Verifies installation and runs a minimal test to ensure everything works.

set -e

echo "🧪 FoxML Core - Installation Test"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

# Parse environment name from environment.yml
ENV_NAME=$(grep -E "^name:" environment.yml | sed 's/name:[[:space:]]*//' | tr -d '"' | tr -d "'" | xargs)
if [[ -z "$ENV_NAME" ]]; then
    echo "❌ Could not determine environment name from environment.yml"
    exit 1
fi

# Check if environment is activated
if [[ -z "$CONDA_DEFAULT_ENV" ]] || [[ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
    echo "⚠️  Conda environment '$ENV_NAME' not activated."
    echo "   Activating now..."
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME" || {
        echo "❌ Failed to activate '$ENV_NAME' environment."
        echo "   Run: conda activate $ENV_NAME"
        exit 1
    }
fi

echo "✅ Environment: $CONDA_DEFAULT_ENV"
echo ""

# Test Python version
echo "📋 Checking Python version..."
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "   Python: $PYTHON_VERSION"

# Test critical imports
echo ""
echo "📦 Testing critical imports..."
python -c "import numpy; import pandas; import polars; import sklearn; print('✅ Core libraries OK')" || {
    echo "❌ Core library import failed"
    exit 1
}

# Test LightGBM (most common model)
echo ""
echo "🤖 Testing model libraries..."
python -c "import lightgbm; print('✅ LightGBM OK')" 2>/dev/null || echo "⚠️  LightGBM not available (optional)"

# Test pipeline import
echo ""
echo "🔧 Testing pipeline imports..."
python -c "from TRAINING.orchestration.intelligent_trainer import IntelligentTrainer; print('✅ Pipeline imports OK')" || {
    echo "❌ Pipeline import failed"
    exit 1
}

# Test config loading
echo ""
echo "⚙️  Testing config system..."
python -c "from CONFIG.config_loader import get_config_path; path = get_config_path('intelligent_training_config'); print(f'✅ Config system OK: {path.name}')" || {
    echo "❌ Config system failed"
    exit 1
}

echo ""
echo "✅ All tests passed! Installation is working correctly."
echo ""
echo "Next steps:"
echo "  1. Ensure you have data in: data/data_labeled/interval=5m/"
echo "  2. Run a quick test:"
echo "     python -m TRAINING.orchestration.intelligent_trainer --output-dir test_install"
echo ""
