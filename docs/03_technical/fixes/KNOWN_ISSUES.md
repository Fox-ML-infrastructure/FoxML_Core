# Known Issues

Current known issues and limitations.

## Data Processing

### Short Fold Handling

**Issue**: Walk-forward validation may create folds shorter than `step_size`.

**Status**: Handled with `allow_truncated_final_fold` configuration.

**Workaround**: Set `allow_truncated_final_fold: true` in config, or skip short folds.

### Missing Data

**Issue**: Some features may have NaN values after feature engineering.

**Status**: Handled with `dropna()` and `min_history_bars` enforcement.

**Workaround**: Ensure sufficient history before using features.

## Model Training

### Overfitting in Tree Models

**Issue**: LightGBM/XGBoost may overfit without proper regularization.

**Status**: Fixed with conservative variants and early stopping.

**Solution**: Use `variant="conservative"` and enable early stopping.

### Dropout in Neural Networks

**Issue**: Dropout was inactive during training in some implementations.

**Status**: Fixed in MultiTaskStrategy.

**Solution**: Ensure `model.train()` is called during training.

## Trading Systems

### IBKR Connection Stability

**Issue**: IBKR API may disconnect during long-running sessions.

**Status**: Handled with RateLimiter and automatic reconnection.

**Workaround**: Monitor connection status, use reconnection logic.

### Alpaca Rate Limits

**Issue**: Alpaca API has rate limits that may be exceeded.

**Status**: Handled with rate limiting in broker implementation.

**Workaround**: Monitor API usage, implement backoff.

## Performance

### C++ Build Requirements

**Issue**: C++ components require specific build tools.

**Status**: Documented in build instructions.

**Workaround**: Follow build instructions, use Python fallback if needed.

## Reporting Issues

Report issues via GitHub issues or contact: jenn.lewis5789@gmail.com

## See Also

- [Bug Fixes](BUG_FIXES.md) - Fix history
- [Testing Plan](../testing/TESTING_PLAN.md) - Testing procedures

