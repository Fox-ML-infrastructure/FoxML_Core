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

### Feature/Target Schema

`CONFIG/feature_target_schema.yaml` (new) provides an explicit schema for classifying columns:
- **Metadata columns**: `symbol`, `ts`, `interval`, `source`, etc.
- **Target patterns**: `^y_*`, `^fwd_ret_*`, `^barrier_*`, etc.
- **Feature families**: OHLCV, returns, volatility, moving averages, oscillators, etc.
- **Mode-specific rules**: Different rules for ranking vs. training modes

This schema ensures consistent classification and allows ranking mode to use more permissive rules (allowing basic OHLCV/TA features) while training mode uses strict rules.

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

**Backup System:**
- Config backups are automatically created in `CONFIG/backups/{target}/{timestamp}/` whenever auto-fix mode runs
- Backups are created even when no leaks are detected (to preserve state history)
- Each backup includes a manifest with git commit, timestamp, and file paths
- Automatic retention policy keeps the last N backups per target (configurable, default: 20)

**Auto-Fixer:**
- Automatically identifies leaking features
- Updates `excluded_features.yaml` and `feature_registry.yaml`
- Re-runs training until no leakage is detected

**Configuration** (`training_config/safety_config.yaml` - see [Safety & Leakage Configs](../../02_reference/configuration/SAFETY_LEAKAGE_CONFIGS.md)):

```yaml
safety:
  leakage_detection:
    # Auto-fixer thresholds
    auto_fix_thresholds:
      cv_score: 0.99              # CV score threshold (99%)
      training_accuracy: 0.999     # Training accuracy threshold (99.9%)
      training_r2: 0.999           # Training R² threshold (99.9%)
      perfect_correlation: 0.999   # Perfect correlation threshold (99.9%)
    auto_fix_min_confidence: 0.8   # Minimum confidence to auto-fix (80%)
    auto_fix_max_features_per_run: 20  # Max features to fix per run
    auto_fix_enabled: true         # Enable/disable auto-fixer
    
    # Pre-training leak scan (catches obvious leaks before model training)
    pre_scan:
      min_match: 0.999             # Minimum match ratio for binary classification (99.9%)
      min_corr: 0.999               # Minimum correlation for regression (99.9%)
      min_valid_pairs: 10           # Minimum valid pairs needed for correlation check
    
    # Feature count requirements for ranking
    ranking:
      min_features_required: 2     # Minimum features after filtering
      min_features_for_model: 3     # Minimum features for model training
      min_features_after_leak_removal: 2  # Minimum after removing leaks
    
    # Leakage warning thresholds
    warning_thresholds:
      classification:
        high: 0.90                  # ROC-AUC/Accuracy > 0.90 is suspicious
        very_high: 0.95             # ROC-AUC/Accuracy > 0.95 is extremely suspicious
      regression:
        forward_return:
          high: 0.50                # R² > 0.50 is suspicious for forward returns
          very_high: 0.60           # R² > 0.60 is extremely suspicious
        barrier:
          high: 0.70                # R² > 0.70 is suspicious for barrier targets
          very_high: 0.80           # R² > 0.80 is extremely suspicious
    
    # Model-specific alert thresholds
    model_alerts:
      suspicious_score: 0.99       # Score >= 0.99 triggers leakage alert
```

You can adjust these thresholds to be more or less sensitive to leakage detection.

### Pre-Training Leak Scan

Before training models, the system performs a **pre-training leak scan** that detects features that are near-copies of the target:

- **Binary Classification**: Detects features that match the target (or 1 - target) with ≥99.9% accuracy
- **Regression**: Detects features with ≥99.9% correlation with the target
- **Automatic Removal**: Leaky features are automatically removed before model training

This catches obvious leaks (like a feature that's literally the target column) before wasting compute on model training.

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

- [Ranking and Selection Consistency](../../01_tutorials/training/RANKING_SELECTION_CONSISTENCY.md) - Unified pipeline behavior (includes sklearn preprocessing)
- [Intelligent Training Tutorial](../../01_tutorials/training/INTELLIGENT_TRAINING_TUTORIAL.md) - Complete pipeline guide
- [Safety & Leakage Configs](../../02_reference/configuration/SAFETY_LEAKAGE_CONFIGS.md) - Leakage detection configuration
- [Feature & Target Configs](../../02_reference/configuration/FEATURE_TARGET_CONFIGS.md) - Feature/target configuration guide
- [Safe Target Pattern](../implementation/SAFE_TARGET_PATTERN_IMPLEMENTATION.md) - Implementation details
- [Intelligence Layer Overview](INTELLIGENCE_LAYER.md) - Complete intelligence layer overview

