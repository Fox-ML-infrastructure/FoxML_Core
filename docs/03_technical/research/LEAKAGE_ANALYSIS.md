# Leakage Analysis

Analysis of data leakage issues and prevention measures.

## Overview

Data leakage occurs when information from the future is used to predict the past. This document summarizes leakage issues identified and fixed in the system.

## Types of Leakage

### 1. Target Leakage

**Issue**: Target columns included in feature sets.

**Detection**: Automatic detection via `CONFIG/excluded_features.yaml`.

**Fix**: `strip_targets()` function removes all target columns before training.

### 2. Forward Return Leakage

**Issue**: Forward returns calculated with look-ahead bias.

**Fix**: Corrected calculation to use only past data, no future information.

### 3. Temporal Overlap

**Issue**: Walk-forward validation had overlap between train and test sets.

**Fix**: Strict temporal separation enforced in fold builder.

## Prevention System

### Excluded Features Config

`CONFIG/excluded_features.yaml` lists features to exclude:

```yaml
excluded:
  - target_fwd_ret_5m
  - target_fwd_ret_15m
  - y_will_peak_60m_0.8
  # ... all target columns
```

### Automatic Filtering

All training functions automatically filter excluded features:

```python
from scripts.filter_leaking_features import filter_features

X_clean = filter_features(X, excluded_config="CONFIG/excluded_features.yaml")
```

## Impact

### Before Fixes

- R² scores artificially high (0.8+)
- Models appeared to perform well
- Real-world performance poor

### After Fixes

- R² scores realistic (0.1-0.3)
- Models perform as expected
- Real-world performance matches validation

## Best Practices

1. **Always strip targets**: Use `strip_targets()` before training
2. **Check excluded config**: Review `excluded_features.yaml` regularly
3. **Validate temporally**: Ensure no temporal overlap in validation
4. **Monitor metrics**: Unrealistic metrics may indicate leakage

## See Also

- [Leakage Fix README](../../../dep/LEAKAGE_FIX_README.md) - Detailed fix documentation
- [Safe Target Pattern](../../../TRAINING/SAFE_TARGET_PATTERN_IMPLEMENTATION.md) - Implementation details

