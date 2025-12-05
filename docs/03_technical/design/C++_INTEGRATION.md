# C++ Integration

C++ high-performance components for hot path operations.

## Overview

The system uses C++ kernels for latency-critical operations while maintaining Python orchestration. This hybrid architecture provides Python flexibility with C++ performance.

## Implemented Kernels

### 1. Barrier Gate Operations

**Function**: `barrier_gate_batch`  
**Purpose**: Computes multiplicative gate from barrier probabilities  
**Optimization**: SIMD vectorized operations  
**Performance**: 2-4x faster than Python

### 2. Simplex Projection

**Function**: `project_simplex`  
**Purpose**: Projects weights to probability simplex  
**Optimization**: Efficient sorting and threshold computation  
**Performance**: 3-5x faster than Python

### 3. Risk Parity Ridge

**Function**: `risk_parity_ridge`  
**Purpose**: Solves ridge risk parity optimization  
**Optimization**: Eigen-based linear algebra  
**Performance**: 5-10x faster than Python

### 4. Horizon Softmax

**Function**: `horizon_softmax`  
**Purpose**: Softmax arbitration over horizons  
**Optimization**: Vectorized matrix operations  
**Performance**: 3-6x faster than Python

### 5. EWMA Volatility

**Function**: `ewma_vol`  
**Purpose**: Exponentially weighted moving average volatility  
**Optimization**: SIMD vectorized operations  
**Performance**: 4-8x faster than Python

### 6. Order Flow Imbalance

**Function**: `ofi_batch`  
**Purpose**: Computes OFI for batch of market data  
**Optimization**: Vectorized conditional operations  
**Performance**: 6-12x faster than Python

## Integration Pattern

### Python Fallback Strategy

All components include:

1. **C++ Detection**: Check if C++ engine is available
2. **C++ Execution**: Use C++ for large vectors (≥4 elements)
3. **Python Fallback**: Fall back to Python if C++ fails
4. **Error Handling**: Graceful degradation with logging

### Example

```python
if CPP_AVAILABLE and len(data) >= 4:
    try:
        result = cpp_engine.barrier_gate_batch(...)
    except Exception as e:
        logger.warning(f"C++ failed: {e}, using Python")
        result = python_fallback(...)
else:
    result = python_fallback(...)
```

## Performance Improvements

### Expected Gains

- **Decision Time**: 500-1000ms → 200-400ms (2-2.5x faster)
- **Throughput**: 10-20 symbols/sec → 50+ symbols/sec (2.5-5x faster)
- **Memory Efficiency**: 50% reduction in allocations
- **CPU Usage**: 2x more efficient

## Build Process

### Prerequisites

```bash
sudo apt-get install -y build-essential cmake libeigen3-dev libomp-dev
pip install pybind11 numpy
```

### Build

```bash
cd IBKR_trading/cpp_engine/python_bindings
./build_kernels.sh
```

## See Also

- [C++ Integration Summary](../../../IBKR_trading/live_trading/C++_INTEGRATION_SUMMARY.md) - Detailed summary
- [C++ Engine README](../../../IBKR_trading/cpp_engine/README.md) - Complete reference

