#!/bin/bash

# Comprehensive IBKR Testing Script
# Runs all tests in sequence: C++ components, IBKR integration, and model compatibility

echo "🚀 IBKR Comprehensive Testing"
echo "============================="
echo ""

# Create logs directory
mkdir -p logs

# Function to run test and capture results
run_test() {
    local test_name="$1"
    local test_script="$2"
    local log_file="logs/${test_name}_test.log"
    
    echo "🧪 Running: $test_name"
    echo "📝 Log: $log_file"
    echo "----------------------------------------"
    
    # Run the test
    python "$test_script" > "$log_file" 2>&1
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ $test_name: PASSED"
    else
        echo "❌ $test_name: FAILED (exit code: $exit_code)"
    fi
    
    echo ""
    return $exit_code
}

# Function to check if IBKR TWS/Gateway is running
check_ibkr_running() {
    echo "🔍 Checking if IBKR TWS/Gateway is running..."
    
    # Check if port 7497 is open
    if nc -z 127.0.0.1 7497 2>/dev/null; then
        echo "✅ IBKR TWS/Gateway is running on port 7497"
        return 0
    else
        echo "❌ IBKR TWS/Gateway is not running on port 7497"
        echo "   Please start IBKR TWS or Gateway before running tests"
        return 1
    fi
}

# Function to check if Alpaca is running
check_alpaca_running() {
    echo "🔍 Checking if Alpaca is running..."
    
    # Check if Alpaca process is running
    if pgrep -f "alpaca" > /dev/null; then
        echo "✅ Alpaca is running"
        return 0
    else
        echo "⚠️ Alpaca is not running (optional for testing)"
        return 1
    fi
}

# Main testing sequence
echo "📋 Starting comprehensive testing sequence..."
echo ""

# Phase 1: C++ Component Testing
echo "🔨 Phase 1: C++ Component Testing"
echo "=================================="

# Build C++ components first
echo "🔨 Building C++ components..."
cd IBKR_trading/cpp_engine
if ./build.sh; then
    echo "✅ C++ components built successfully"
else
    echo "❌ C++ build failed"
    exit 1
fi
cd ../..

# Test C++ components
run_test "cpp_components" "IBKR_trading/test_cpp_components.py"
cpp_test_result=$?

# Phase 2: IBKR Integration Testing
echo "🔌 Phase 2: IBKR Integration Testing"
echo "====================================="

# Check if IBKR is running
check_ibkr_running
ibkr_running=$?

if [ $ibkr_running -eq 0 ]; then
    # Test IBKR integration
    run_test "ibkr_integration" "IBKR_trading/test_ibkr_integration.py"
    ibkr_test_result=$?
else
    echo "⚠️ Skipping IBKR integration tests (IBKR not running)"
    ibkr_test_result=1
fi

# Phase 3: Model Compatibility Testing
echo "🧠 Phase 3: Model Compatibility Testing"
echo "======================================="

# Check if Alpaca is running
check_alpaca_running
alpaca_running=$?

if [ $alpaca_running -eq 0 ]; then
    # Test model compatibility
    run_test "model_compatibility" "IBKR_trading/test_daily_models.py"
    model_test_result=$?
else
    echo "⚠️ Skipping model compatibility tests (Alpaca not running)"
    model_test_result=1
fi

# Phase 4: Performance Benchmarking
echo "⚡ Phase 4: Performance Benchmarking"
echo "===================================="

# Run performance benchmarks
run_test "performance_benchmark" "IBKR_trading/benchmark_performance.py"
performance_test_result=$?

# Final Results Summary
echo "📊 FINAL TEST RESULTS"
echo "===================="
echo ""

# C++ Components
if [ $cpp_test_result -eq 0 ]; then
    echo "✅ C++ Components: PASSED"
else
    echo "❌ C++ Components: FAILED"
fi

# IBKR Integration
if [ $ibkr_test_result -eq 0 ]; then
    echo "✅ IBKR Integration: PASSED"
else
    echo "❌ IBKR Integration: FAILED"
fi

# Model Compatibility
if [ $model_test_result -eq 0 ]; then
    echo "✅ Model Compatibility: PASSED"
else
    echo "❌ Model Compatibility: FAILED"
fi

# Performance Benchmark
if [ $performance_test_result -eq 0 ]; then
    echo "✅ Performance Benchmark: PASSED"
else
    echo "❌ Performance Benchmark: FAILED"
fi

echo ""

# Overall result
total_tests=4
passed_tests=0

[ $cpp_test_result -eq 0 ] && ((passed_tests++))
[ $ibkr_test_result -eq 0 ] && ((passed_tests++))
[ $model_test_result -eq 0 ] && ((passed_tests++))
[ $performance_test_result -eq 0 ] && ((passed_tests++))

echo "📈 Overall Results: $passed_tests/$total_tests tests passed"
echo "📊 Success rate: $((passed_tests * 100 / total_tests))%"

if [ $passed_tests -eq $total_tests ]; then
    echo ""
    echo "🎉 ALL TESTS PASSED!"
    echo "✅ IBKR trading system is ready for production"
    echo ""
    echo "🚀 Next steps:"
    echo "   1. Start Alpaca for comparison"
    echo "   2. Copy models to IBKR test environment"
    echo "   3. Run parallel testing"
    echo "   4. Compare results"
else
    echo ""
    echo "⚠️ SOME TESTS FAILED"
    echo "❌ Check logs for details and fix issues"
    echo ""
    echo "🔧 Troubleshooting:"
    echo "   1. Check C++ build: logs/cpp_components_test.log"
    echo "   2. Check IBKR connection: logs/ibkr_integration_test.log"
    echo "   3. Check model compatibility: logs/daily_model_test.log"
    echo "   4. Check performance: logs/performance_benchmark_test.log"
fi

echo ""
echo "📁 All logs saved in: logs/"
echo "📊 Check individual log files for detailed results"
