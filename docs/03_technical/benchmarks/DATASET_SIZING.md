# Dataset Sizing

Strategies for sizing datasets for training and validation.

## Dataset Requirements

### Minimum Sizes

- **Training**: 120+ days (min_history_bars)
- **Validation**: 63+ days (1 quarter)
- **Testing**: 63+ days (1 quarter)

### Recommended Sizes

- **Training**: 252+ days (1 year)
- **Validation**: 63+ days (1 quarter)
- **Testing**: 63+ days (1 quarter)

## Walk-Forward Configuration

### Standard Configuration

```yaml
walkforward:
  fold_length: 252    # 1 year training
  step_size: 63       # 1 quarter step
  min_history_bars: 120
```

### Short-Fold Policy

If `test_len < step_size`:

```yaml
walkforward:
  allow_truncated_final_fold: true  # Set step_size = test_len
  # OR
  allow_truncated_final_fold: false  # Skip the fold
```

## Data Quality

### Coverage Requirements

- **Minimum**: 90% of days complete
- **Recommended**: 95%+ of days complete

### Bar Count

For 5-minute bars in RTH:
- **Expected**: 78 bars per day
- **Minimum**: 70 bars per day (90% coverage)

## Best Practices

1. **Use Walk-Forward**: Always use walk-forward validation
2. **Check Coverage**: Ensure sufficient data coverage
3. **Validate Temporally**: No temporal overlap
4. **Monitor Stability**: Check performance across folds

## See Also

- [Walk-Forward Validation](../../01_tutorials/training/WALKFORWARD_VALIDATION.md) - Validation guide
- [Data Sanity Rules](../../02_reference/data/DATA_SANITY_RULES.md) - Validation rules

