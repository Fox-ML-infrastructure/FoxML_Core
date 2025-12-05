# Daily Testing

Procedures for daily model testing and validation.

## Overview

Daily testing validates that models are working correctly and ready for trading.

## Quick Start

```bash
cd IBKR_trading
python test_daily_models.py
```

## Test Process

### 1. Load Models

System loads all models from `models/` directory.

### 2. Generate Predictions

Models generate predictions for test symbols.

### 3. Validate Outputs

Check:
- Predictions are valid (not NaN, in expected range)
- Multi-horizon blending works
- Safety guards function

### 4. Check Performance

Monitor:
- Prediction latency
- Memory usage
- CPU utilization

## Configuration

Edit `IBKR_trading/config/ibkr_enhanced.yaml`:

```yaml
testing:
  symbols: ["AAPL", "MSFT", "GOOGL"]
  horizons: ["5m", "10m", "15m", "30m", "60m"]
```

## Expected Results

- All models load successfully
- Predictions generated for all symbols
- No errors or warnings
- Performance within limits

## Troubleshooting

### Models Not Loading

Check:
- Model files exist in `models/`
- Model format is correct
- Dependencies installed

### Prediction Errors

Check:
- Input data format
- Feature availability
- Model compatibility

## See Also

- [Daily Testing README](../../../IBKR_trading/DAILY_TESTING_README.md) - Complete guide
- [Testing Plan](TESTING_PLAN.md) - Testing strategy

