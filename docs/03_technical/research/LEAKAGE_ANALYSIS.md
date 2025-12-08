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

### Multi-Layer Protection

The system uses multiple layers to prevent data leakage:

1. **Feature Registry** (`CONFIG/feature_registry.yaml`) - Structural rules based on temporal metadata
2. **Excluded Features Config** (`CONFIG/excluded_features.yaml`) - Pattern-based exclusions
3. **Automated Leakage Detection** - Real-time detection during training
4. **Auto-Fixer** - Automatic detection and exclusion of leaking features

### Feature Registry

`CONFIG/feature_registry.yaml` defines temporal rules for features:
- `lag_bars`: How many bars back the feature is allowed to peek
- `allowed_horizons`: Which target horizons the feature can predict
- `source`: Where the feature comes from (price, volume, derived, etc.)

Features are automatically filtered based on target horizon to prevent leakage.

### Excluded Features Config

`CONFIG/excluded_features.yaml` lists patterns for features to exclude:

```yaml
always_exclude:
  regex_patterns:
    - "^tth_"      # time-to-hit
    - "^y_"        # ALL y_* targets
    - "^p_"        # ALL p_* probability features
  prefix_patterns:
    - "tth_"
    - "mfe_share_"
```

### Automatic Filtering

All training functions automatically filter excluded features:

```python
from TRAINING.utils.leakage_filtering import filter_features_for_target

safe_features = filter_features_for_target(
    all_features, 
    target_column="y_will_peak_60m_0.8",
    use_registry=True,  # Use feature registry
    data_interval_minutes=5
)
```

### Automated Leakage Detection & Auto-Fixer

The system automatically detects leakage during training and can auto-fix it:

**Detection Methods:**
- Perfect CV scores (≥99%)
- Perfect training accuracy (≥99.9%)
- Perfect R² scores (≥99.9%)
- Perfect correlation between predictions and targets

**Auto-Fixer:**
- Automatically identifies leaking features
- Updates `excluded_features.yaml` and `feature_registry.yaml`
- Re-runs training until no leakage is detected

**Configuration** (`CONFIG/training_config/safety_config.yaml`):

```yaml
safety:
  leakage_detection:
    auto_fix_thresholds:
      cv_score: 0.99              # CV score threshold (99%)
      training_accuracy: 0.999     # Training accuracy threshold (99.9%)
      training_r2: 0.999           # Training R² threshold (99.9%)
      perfect_correlation: 0.999   # Perfect correlation threshold (99.9%)
    auto_fix_min_confidence: 0.8   # Minimum confidence to auto-fix (80%)
    auto_fix_enabled: true         # Enable/disable auto-fixer
```

You can adjust these thresholds to be more or less sensitive to leakage detection.

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

