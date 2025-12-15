# Live Trading System - IBKR Integration

Complete live trading system integrating all trained models (tabular + sequential + multi-task) across all horizons and strategies for IBKR trading.

## Architecture

### Core Components

1. **ModelPredictor** - Unified model prediction engine
 - Handles prediction from all trained models across horizons and strategies
 - Supports tabular, sequential, and multi-task models
 - Implements caching and TTL for performance

2. **HorizonBlender** - Per-horizon model blending
 - Uses OOF-trained ridge regression + simplex projection
 - Blends all regression models per horizon into single alpha
 - Supports adaptive weighting based on performance

3. **BarrierGate** - Timing and risk attenuator
 - Uses barrier probabilities to scale final alpha multiplicatively
 - Implements calibrated barrier probability gating
 - Supports horizon-specific gates

4. **CostArbitrator** - Cost model and horizon arbitration
 - Estimates trading costs per symbol/horizon
 - Implements winner-takes-most and softmax arbitration
 - Supports session-aware and correlation-aware adjustments

5. **PositionSizer** - Alpha to weights conversion
 - Vol scaling with cross-sectional standardization
 - Risk parity optimization with ridge regression
 - No-trade band for turnover control

6. **LiveTradingSystem** - Main integration loop
 - Orchestrates all components in live trading loop
 - Handles market data, features, predictions, and execution
 - Implements error handling and state management

## Usage

### Basic Setup

```python
from live_trading import LiveTradingSystem

# Configuration
config = {
    'symbols': ['AAPL', 'MSFT', 'GOOGL'],
    'horizons': ['5m', '10m', '15m', '30m', '60m', '120m', '1d', '5d', '20d'],
    'model_dir': 'TRAINING/models',
    'blender_dir': 'TRAINING/blenders',
    'device': 'cpu',
    'update_interval': 60,
    'g_min': 0.2,
    'gamma': 1.0,
    'delta': 0.5,
    'z_max': 3.0,
    'max_weight': 0.05,
    'target_gross': 0.5,
    'no_trade_band': 0.008
}

# Create and start system
system = LiveTradingSystem(config)
system.start()

# Get portfolio status
status = system.get_portfolio_status()
print(status)

# Stop system
system.stop()
```

### Advanced Configuration

```python
from live_trading import LiveTradingSystem, LiveTradingManager

# Multiple systems
configs = [
    {
        'symbols': ['AAPL', 'MSFT'],
        'horizons': ['5m', '15m', '60m'],
        'target_gross': 0.3
    },
    {
        'symbols': ['GOOGL', 'AMZN'],
        'horizons': ['1d', '5d'],
        'target_gross': 0.7
    }
]

# Manager for multiple systems
manager = LiveTradingManager(configs)
manager.start_all()

# Get all status
all_status = manager.get_all_status()
```

## Trading Flow

1. **Market Data** - Fetch current market data (prices, spreads, volatility)
2. **Features** - Compute feature matrix for all symbols
3. **Predictions** - Get predictions from all models for all horizons
4. **Blending** - Blend models per horizon using OOF-trained weights
5. **Costs** - Estimate trading costs and compute net alpha
6. **Arbitration** - Select/weight horizons using cost-aware scoring
7. **Barrier Gating** - Apply barrier probability gate to final alpha
8. **Position Sizing** - Convert alpha to weights with risk management
9. **Validation** - Validate weights for risk limits
10. **Execution** - Execute trades to reach target weights

## Key Features

- **Multi-Horizon Support** - Handles 5m to 20d horizons simultaneously
- **Model Diversity** - Integrates all 20 model families across 3 strategies
- **Risk Management** - Vol scaling, correlation awareness, position limits
- **Cost Awareness** - Real-time cost estimation and horizon arbitration
- **Barrier Gating** - Uses barrier probabilities for timing control
- **Turnover Control** - No-trade band to reduce unnecessary trading
- **Error Handling** - Robust error handling and state management
- **Performance Tracking** - Built-in performance metrics and monitoring
- **C++ Hot Path** - SIMD-optimized kernels for latency-critical operations
- **Hybrid Architecture** - Python orchestration + C++ computation

## Configuration Options

### Model Settings
- `model_dir`: Directory containing trained models
- `blender_dir`: Directory for horizon blenders
- `device`: Device for model inference ('cpu' or 'cuda')

### Trading Parameters
- `symbols`: List of symbols to trade
- `horizons`: List of horizons to use
- `update_interval`: Update frequency in seconds

### Risk Management
- `g_min`: Minimum barrier gate value
- `gamma`: Peak probability scaling
- `delta`: Valley probability scaling
- `z_max`: Maximum z-score for vol scaling
- `max_weight`: Maximum individual position weight
- `target_gross`: Target gross exposure
- `no_trade_band`: Minimum weight change to trigger trade

### Cost Model
- `k1`: Spread cost coefficient
- `k2`: Volatility cost coefficient
- `k3`: Participation cost coefficient

## Performance Optimization

- **Caching** - Model predictions are cached with TTL
- **Batch Processing** - Efficient batch prediction across models
- **Memory Management** - Automatic cleanup of old cache entries
- **Threading** - Non-blocking update loop with error recovery

## Monitoring and Logging

- **Structured Logging** - Comprehensive logging of all operations
- **Performance Metrics** - Built-in tracking of key metrics
- **Error Reporting** - Detailed error reporting and recovery
- **Status Monitoring** - Real-time portfolio status and health

## Integration with IBKR

The system is designed to integrate with IBKR's API for:
- Market data feeds
- Order execution
- Portfolio management
- Risk monitoring

## Testing

Run the test suite to verify all components:

```bash
cd IBKR_trading
python -m pytest tests/test_live_integration.py -v
```

## Dependencies

### Python Dependencies
- numpy
- pandas
- scikit-learn
- torch (for sequential models)
- onnxruntime (for ONNX models)
- scipy (for optimization)
- joblib (for model persistence)
- pybind11 (for C++ integration)

### C++ Dependencies
- Eigen3 (linear algebra)
- pybind11 (Python bindings)
- OpenMP (parallel processing)
- BLAS/LAPACK (optimized linear algebra)

### Building C++ Kernels
```bash
cd IBKR_trading/cpp_engine/python_bindings
./build_kernels.sh
```

This will compile the C++ kernels and create the `ibkr_trading_engine_py` module for high-performance operations.

## License

This system is part of the trading infrastructure and follows the same licensing terms as the main project.
