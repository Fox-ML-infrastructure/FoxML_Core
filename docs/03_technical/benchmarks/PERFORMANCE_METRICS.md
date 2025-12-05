# Performance Metrics

Performance benchmarks and metrics for the trading system.

## Decision Latency

### Target

- **Decision Time**: < 500ms per symbol
- **Throughput**: > 20 symbols/second

### Actual Performance

- **Decision Time**: 200-400ms (with C++ optimization)
- **Throughput**: 50+ symbols/second

## Model Training

### Training Time

- **LightGBM**: 2-5 minutes per model
- **XGBoost**: 3-6 minutes per model
- **Neural Networks**: 10-30 minutes per model

### Feature Selection

- **CPU**: 2-5 minutes per symbol (421 features)
- **GPU**: 10-30 seconds per symbol (7GB VRAM)

## Memory Usage

### Training

- **LightGBM/XGBoost**: 1-2GB
- **Neural Networks**: 2-4GB
- **Feature Selection**: 2-3GB

### Trading

- **IBKR System**: 1-1.5GB
- **Alpaca System**: 500MB-1GB

## See Also

- [Performance Optimization](../implementation/PERFORMANCE_OPTIMIZATION.md) - Optimization details
- [C++ Integration](../design/C++_INTEGRATION.md) - Performance improvements

