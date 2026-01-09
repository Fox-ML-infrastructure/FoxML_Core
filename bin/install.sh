#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
#
# Quick Install Script for FoxML Core
#
# One-line install: bash <(curl -sL https://raw.githubusercontent.com/Fox-ML-infrastructure/FoxML_Core/main/bin/install.sh)
# Or: bash bin/install.sh

set -e

echo "🚀 FoxML Core - Quick Install"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Miniconda or Anaconda first."
    echo "   Download: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

echo "📦 Creating conda environment from environment.yml..."
conda env create -f environment.yml || {
    echo "⚠️  Environment creation failed. Trying to update existing environment..."
    conda env update -f environment.yml --prune
}

echo ""
echo "✅ Installation complete!"
echo ""
echo "To activate the environment:"
echo "  conda activate trader"
echo ""
echo "To verify installation, run:"
echo "  bash bin/test_install.sh"
echo ""
