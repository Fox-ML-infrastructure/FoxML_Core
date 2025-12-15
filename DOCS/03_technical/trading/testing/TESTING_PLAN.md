# IBKR Testing Plan

## Testing Strategy

### Phase 1: C++ Component Validation (Today)
- Test C++ components in isolation
- Validate Python-C++ bindings
- Performance benchmarking
- Memory leak testing

### Phase 2: IBKR Integration Testing (Tomorrow)
- Run Alpaca in parallel for comparison
- Copy models to IBKR test environment
- Side-by-side performance validation
- Risk management validation

## Phase 1: C++ Component Testing

### 1. Build and Test C++ Components

```bash
# Build C++ components
cd IBKR_trading/cpp_engine
./build.sh

# Run C++ tests
./test_cpp_components.sh

# Run performance benchmarks
./benchmark_cpp_components.sh
```

### 2. Python-C++ Integration Testing

```bash
# Test Python bindings
python IBKR_trading/test_cpp_integration.py

# Test with sample data
python IBKR_trading/test_cpp_with_data.py
```

### 3. Memory and Performance Validation

```bash
# Memory leak testing
python IBKR_trading/test_memory_leaks.py

# Performance comparison
python IBKR_trading/benchmark_python_vs_cpp.py
```

## Phase 2: IBKR Integration Testing

### 1. Parallel Alpaca + IBKR Testing

```bash
# Start Alpaca (keep running for comparison)
python run_alpaca_trading.py --config config/alpaca_config.yaml

# Start IBKR testing (parallel)
python IBKR_trading/test_ibkr_integration.py --config config/ibkr_daily_test.yaml
```

### 2. Model Copying and Validation

```bash
# Copy models from Alpaca to IBKR
python IBKR_trading/copy_models_from_alpaca.py

# Validate model compatibility
python IBKR_trading/validate_model_compatibility.py
```

### 3. Side-by-Side Performance Comparison

```bash
# Run comparison test
python IBKR_trading/compare_alpaca_ibkr.py --duration 1h
```

## Testing Checklist

### C++ Components
- [ ] C++ builds successfully
- [ ] Python bindings work
- [ ] Performance is 2-5x faster than Python
- [ ] No memory leaks
- [ ] Handles edge cases (empty data, NaN, etc.)

### IBKR Integration
- [ ] Connects to IBKR TWS/Gateway
- [ ] Market data streaming works
- [ ] Order placement works
- [ ] Position tracking works
- [ ] Risk management works

### Model Compatibility
- [ ] Models load correctly
- [ ] Features are compatible
- [ ] Predictions are identical
- [ ] Performance is maintained

### Risk Management
- [ ] Position limits enforced
- [ ] Drawdown controls work
- [ ] Kill switches work
- [ ] Emergency flatten works

## Testing Commands

### Today (C++ Testing)

```bash
# 1. Build C++ components
cd IBKR_trading/cpp_engine
./build.sh

# 2. Test C++ components
python IBKR_trading/test_cpp_components.py

# 3. Benchmark performance
python IBKR_trading/benchmark_cpp.py

# 4. Test memory usage
python IBKR_trading/test_memory.py
```

### Tomorrow (IBKR Testing)

```bash
# 1. Start Alpaca (keep running)
python run_alpaca_trading.py --config config/alpaca_config.yaml &

# 2. Copy models to IBKR
python IBKR_trading/copy_models.py

# 3. Test IBKR connection
python IBKR_trading/test_ibkr_connection.py

# 4. Run parallel testing
python IBKR_trading/run_parallel_test.py --duration 2h

# 5. Compare results
python IBKR_trading/compare_results.py
```

## Success Criteria

### C++ Components
- **Performance**: 2-5x faster than Python equivalent
- **Memory**: No memory leaks after 24h continuous running
- **Accuracy**: Identical results to Python version
- **Stability**: No crashes with edge cases

### IBKR Integration
- **Connection**: Stable connection to IBKR
- **Data**: Real-time market data streaming
- **Orders**: Orders execute correctly
- **Risk**: All risk controls work
- **Performance**: Matches or exceeds Alpaca performance

### Model Compatibility
- **Accuracy**: Identical predictions to Alpaca
- **Speed**: Faster inference than Python
- **Memory**: Lower memory usage
- **Stability**: No crashes during inference

## Risk Mitigation

### Backup Strategy
- Keep Alpaca running as backup
- Copy all models before testing
- Save configuration backups
- Log all test results

### Rollback Plan
- If IBKR fails, fall back to Alpaca
- If C++ components fail, use Python fallback
- If models incompatible, use original models
- If performance degrades, revert changes

## Testing Schedule

### Today (C++ Testing)
- [ ] 09:00 - Build C++ components
- [ ] 10:00 - Test C++ components
- [ ] 11:00 - Benchmark performance
- [ ] 12:00 - Test memory usage
- [ ] 13:00 - Fix any issues
- [ ] 14:00 - Final C++ validation

### Tomorrow (IBKR Testing)
- [ ] 09:00 - Start Alpaca (keep running)
- [ ] 09:30 - Copy models to IBKR
- [ ] 10:00 - Test IBKR connection
- [ ] 10:30 - Run parallel testing
- [ ] 12:00 - Compare results
- [ ] 13:00 - Fix any issues
- [ ] 14:00 - Final IBKR validation

## Expected Outcomes

### C++ Components
- **2-5x performance improvement**
- **Lower memory usage**
- **Identical accuracy**
- **Production ready**

### IBKR Integration
- **Stable connection**
- **Real-time data**
- **Correct order execution**
- **Risk management working**
- **Performance matching Alpaca**

### Overall System
- **Ready for production**
- **Faster than current setup**
- **More robust risk management**
- **Better execution quality**

---

**Ready to start testing? Let's begin with C++ component validation!**
