#!/bin/bash
# Verification script for fix/training-pipeline-audit-fixes
# Run after training to verify all fixes are working

set -e

LOG_DIR="${1:-logs}"
RESULTS_DIR="${2:-RESULTS}"

echo "=========================================="
echo "Training Pipeline Fixes Verification"
echo "=========================================="
echo ""

ERRORS=0
WARNINGS=0

# A) Repro Tracker
echo "A) Reproducibility Tracker"
echo "---------------------------"
if grep -r "object has no attribute 'name'" "$LOG_DIR" 2>/dev/null | grep -v ".py:" > /dev/null; then
    echo "❌ FAIL: Found '.name' attribute errors"
    ERRORS=$((ERRORS + 1))
else
    echo "✅ PASS: No '.name' attribute errors"
fi

if grep -r "lightgbm\|xgboost" "$LOG_DIR" 2>/dev/null | grep -i "reproducibility\|tracker\|comparison" > /dev/null; then
    echo "✅ PASS: Tracker comparisons logged for lightgbm/xgboost"
else
    echo "⚠️  WARN: No tracker comparisons found (may be normal if no previous runs)"
    WARNINGS=$((WARNINGS + 1))
fi
echo ""

# B) Family Normalization + Registry
echo "B) Family Normalization + Registry"
echo "-----------------------------------"
if grep -r "Unknown family 'x_g_boost'" "$LOG_DIR" 2>/dev/null > /dev/null; then
    echo "❌ FAIL: Found 'x_g_boost' errors"
    ERRORS=$((ERRORS + 1))
else
    echo "✅ PASS: No 'x_g_boost' errors"
fi

if grep -r "TRAINER_MODULE_MAP, using fallback" "$LOG_DIR" 2>/dev/null > /dev/null; then
    echo "❌ FAIL: Found fallback warnings (family not in registry)"
    ERRORS=$((ERRORS + 1))
else
    echo "✅ PASS: No fallback warnings"
fi

# Check for invalid families being attempted (not skipped)
if grep -r "random_forest\|catboost\|neural_network\|lasso" "$LOG_DIR" 2>/dev/null | grep -v "skip\|SKIP\|skipped\|Skipped" | grep -i "train\|attempt\|fail" > /dev/null; then
    echo "❌ FAIL: Invalid families were attempted (should be skipped)"
    ERRORS=$((ERRORS + 1))
else
    echo "✅ PASS: Invalid families properly skipped"
fi
echo ""

# C) Plan + Routing Sanity
echo "C) Plan + Routing Sanity"
echo "------------------------"
if grep -r "Total jobs: 0" "$LOG_DIR" 2>/dev/null | grep -v "error\|ERROR\|Error" > /dev/null; then
    echo "❌ FAIL: Found 'Total jobs: 0' without error (should error)"
    ERRORS=$((ERRORS + 1))
else
    echo "✅ PASS: 0 jobs triggers error (or no 0-job plans)"
fi

# Check routing decision count consistency
ROUTING_COUNT=$(grep -r "Loaded routing decisions" "$LOG_DIR" 2>/dev/null | tail -1 | grep -oE "[0-9]+" | head -1 || echo "")
TARGET_COUNT=$(grep -r "Total targets:" "$LOG_DIR" 2>/dev/null | tail -1 | grep -oE "[0-9]+" | head -1 || echo "")
if [ -n "$ROUTING_COUNT" ] && [ -n "$TARGET_COUNT" ] && [ "$ROUTING_COUNT" != "$TARGET_COUNT" ]; then
    echo "⚠️  WARN: Routing count ($ROUTING_COUNT) != target count ($TARGET_COUNT)"
    WARNINGS=$((WARNINGS + 1))
else
    echo "✅ PASS: Routing count matches target count (or counts not found)"
fi

# Check CS: DISABLED is respected
if grep -r "CS: DISABLED" "$LOG_DIR" 2>/dev/null | grep -v "Skipping\|skipping\|Skip" > /dev/null; then
    echo "❌ FAIL: CS: DISABLED targets were not skipped"
    ERRORS=$((ERRORS + 1))
else
    echo "✅ PASS: CS: DISABLED targets properly skipped"
fi
echo ""

# D) Feature Integrity
echo "D) Feature Integrity"
echo "--------------------"
# Check for feature collapse errors
if grep -r "Feature collapse\|Feature schema mismatch" "$LOG_DIR" 2>/dev/null > /dev/null; then
    echo "⚠️  WARN: Feature collapse/schema mismatch detected (check if intentional)"
    WARNINGS=$((WARNINGS + 1))
else
    echo "✅ PASS: No feature collapse errors"
fi

# Check feature audit logs show pipeline stages
if grep -r "requested.*allowed.*present" "$LOG_DIR" 2>/dev/null > /dev/null; then
    echo "✅ PASS: Feature audit shows pipeline stages"
else
    echo "⚠️  WARN: Feature audit stages not found in logs"
    WARNINGS=$((WARNINGS + 1))
fi
echo ""

# E) Symbol-Specific Correctness
echo "E) Symbol-Specific Correctness"
echo "------------------------------"
# Check for single symbol in SYMBOL_SPECIFIC (should be multiple)
if grep -r "SYMBOL_SPECIFIC route.*1 symbol" "$LOG_DIR" 2>/dev/null > /dev/null; then
    echo "⚠️  WARN: SYMBOL_SPECIFIC route has only 1 symbol (may be intentional)"
    WARNINGS=$((WARNINGS + 1))
else
    echo "✅ PASS: SYMBOL_SPECIFIC route has multiple symbols (or not used)"
fi

# Check winner_symbols validation
if grep -r "winner_symbols" "$LOG_DIR" 2>/dev/null | grep -i "invalid\|missing\|fallback" > /dev/null; then
    echo "⚠️  WARN: winner_symbols validation warnings found"
    WARNINGS=$((WARNINGS + 1))
else
    echo "✅ PASS: winner_symbols validated correctly"
fi
echo ""

# Summary
echo "=========================================="
echo "Verification Summary"
echo "=========================================="
echo "Errors: $ERRORS"
echo "Warnings: $WARNINGS"
echo ""

if [ $ERRORS -eq 0 ]; then
    echo "✅ All critical checks passed!"
    if [ $WARNINGS -gt 0 ]; then
        echo "⚠️  $WARNINGS warnings found (review above)"
    fi
    exit 0
else
    echo "❌ $ERRORS critical errors found!"
    exit 1
fi

