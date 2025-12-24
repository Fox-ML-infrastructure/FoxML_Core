#!/bin/bash
# Clear test cache files before fresh runs
# Usage: SCRIPTS/SCRIPTS/clear_test_cache.sh

echo "🧹 Clearing test cache files..."

# Remove cache directories
find . -type d -name "cache" \( -path "*/test_*" -o -path "*/intelligent_output*" -o -path "*/test_e2e*" \) -exec rm -rf {} + 2>/dev/null

# Remove cache JSON files
find . -type f \( -name "*cache*.json" -o -name "*ranking*.json" \) \( -path "*/test_*" -o -path "*/intelligent_output*" -o -path "*/test_e2e*" \) -delete 2>/dev/null

echo "✅ Cache cleared!"
echo ""
echo "To also remove output directories (optional):"
echo "  rm -rf test_*output* intelligent_output* test_e2e*"
