# Safety & Leakage Detection Configuration Guide

Complete guide to configuring numerical stability guards and leakage detection in FoxML Core.

## Overview

The safety configuration controls numerical stability (feature clipping, target capping, gradient clipping) and leakage detection (pre-scan, auto-fixer, auto-rerun).

## Configuration File

### `training_config/safety_config.yaml`

**Purpose:** Numerical stability guards and leakage detection.

**When to use:** When adjusting leakage detection sensitivity, auto-fixer behavior, or numerical stability guards.

---

## Numerical Stability Settings

### Feature Clipping

**Purpose:** Prevent extreme feature values from causing numerical issues.

**Settings:**
```yaml
numerical_stability:
  feature_clipping:
    enabled: true
    min_value: -10.0
    max_value: 10.0
```

**Example: Adjusting Clipping Range**

```yaml
numerical_stability:
  feature_clipping:
    enabled: true
    min_value: -5.0  # Narrower range
    max_value: 5.0
```

### Target Capping

**Purpose:** Prevent extreme target values.

**Settings:**
```yaml
numerical_stability:
  target_capping:
    enabled: true
    max_abs_value: 1.0
```

**Example: Adjusting Target Cap**

```yaml
numerical_stability:
  target_capping:
    enabled: true
    max_abs_value: 2.0  # Allow larger target values
```

### Gradient Clipping

**Purpose:** Prevent exploding gradients in neural networks.

**Settings:**
```yaml
numerical_stability:
  gradient_clipping:
    enabled: true
    max_norm: 1.0
```

---

## Leakage Detection Settings

### Over-Budget Action

**Purpose:** Control what happens when features exceed the lookback budget.

**Settings:**
```yaml
leakage_detection:
  over_budget_action: drop  # drop | hard_stop | warn
```

**Options:**
- `drop`: Gatekeeper removes violating features (default for dev/testing)
- `hard_stop`: Fail the run if any violating feature exists (recommended for prod)
- `warn`: Allow but log violations (useful only for debugging)

**Example: Production Configuration**
```yaml
leakage_detection:
  over_budget_action: hard_stop  # Fail fast in production
```

### Lookback Budget Computation

**Purpose:** Control how the required lookback budget is computed.

**Settings:**
```yaml
leakage_detection:
  lookback_budget_minutes: auto  # auto | <number>
  lookback_buffer_minutes: 5.0   # Safety buffer in minutes
```

**Options:**
- `lookback_budget_minutes: auto`: Compute from actual final feature set (post-prune or chosen stage)
- `lookback_budget_minutes: <number>`: Force a cap for experiments (but still log what the true max was)

**Example: Experiment Configuration**
```yaml
leakage_detection:
  lookback_budget_minutes: 240  # Cap at 4 hours for experiment
  lookback_buffer_minutes: 10.0  # 10 minute safety buffer
```

### Cross-Validation Purge/Embargo Settings

**Purpose:** Control purge and embargo computation for CV splits.

**Settings:**
```yaml
leakage_detection:
  cv:
    purge_minutes: auto      # auto | <number>
    embargo_minutes: auto     # auto | <number>
    embargo_extra_bars: 5    # Extra bars for embargo safety margin
```

**Options:**
- `purge_minutes: auto`: Compute from feature lookback + buffer
- `embargo_minutes: auto`: Compute from horizon + extra bars
- `embargo_extra_bars`: Number of extra bars added to horizon for embargo (default: 5)

**Enforcement:**
- `purge >= feature_lookback_budget + buffer`
- `embargo >= horizon + extra`

**Example: Custom Embargo**
```yaml
leakage_detection:
  cv:
    embargo_extra_bars: 10  # 10 bars = 50 minutes for 5m bars
```

### Pre-Training Leak Scan

**Purpose:** Detect near-copy features before model training.

**Settings:**
```yaml
leakage_detection:
  pre_scan:
    min_match: 0.999  # Binary classification: 99.9% match threshold
    min_corr: 0.999   # Regression: 99.9% correlation threshold
    min_valid_pairs: 10  # Minimum valid pairs for correlation check
```

**How It Works:**
- **Binary Classification:** Detects features matching target with ≥99.9% accuracy
- **Regression:** Detects features with ≥99.9% correlation with target
- Automatically removes detected leaky features before training

**Example: Making Pre-Scan More Sensitive**

```yaml
leakage_detection:
  pre_scan:
    min_match: 0.95  # Lower threshold (detect at 95% instead of 99.9%)
    min_corr: 0.95
```

### Feature Count Requirements

**Purpose:** Minimum feature requirements for ranking and training.

**Settings:**
```yaml
leakage_detection:
  ranking:
    min_features_required: 2  # Minimum for ranking
    min_features_for_model: 3  # Minimum for model training
    min_features_after_leak_removal: 2  # Minimum after removing leaks
```

**Example: Adjusting Requirements**

```yaml
leakage_detection:
  ranking:
    min_features_required: 5  # Require more features
    min_features_for_model: 10
```

### Warning Thresholds

**Purpose:** Thresholds for logging warnings (not auto-fix).

**Settings:**
```yaml
leakage_detection:
  warning_thresholds:
    classification:
      high_auc: 0.95  # Warn if AUC > 0.95
    regression:
      forward_return:
        high_r2: 0.90  # Warn if R² > 0.90 for forward returns
      barrier:
        high_r2: 0.95  # Warn if R² > 0.95 for barrier targets
```

**Example: Adjusting Warning Thresholds**

```yaml
leakage_detection:
  warning_thresholds:
    classification:
      high_auc: 0.90  # Lower threshold (warn more often)
```

### Auto-Fix Thresholds

**Purpose:** Thresholds that trigger automatic leakage fixing.

**Settings:**
```yaml
leakage_detection:
  auto_fix_thresholds:
    cv_score: 0.99  # Cross-validation score threshold (99%)
    training_accuracy: 0.999  # Training accuracy threshold (99.9%)
    training_r2: 0.999  # Training R² threshold (99.9%)
    perfect_correlation: 0.999  # Perfect correlation threshold (99.9%)
```

**How It Works:**
- When any threshold is exceeded, auto-fixer is triggered
- Auto-fixer detects leaking features and updates configs
- Lower thresholds = more sensitive detection

**Example: Making Detection More Sensitive**

```yaml
leakage_detection:
  auto_fix_thresholds:
    cv_score: 0.95  # Lower from 0.99 (detect at 95% instead of 99%)
    training_accuracy: 0.98  # Lower from 0.999
```

**Example: Making Detection Less Sensitive**

```yaml
leakage_detection:
  auto_fix_thresholds:
    cv_score: 0.999  # Higher threshold (only detect at 99.9%)
    training_accuracy: 0.9999  # Very high threshold
```

### Auto-Fixer Settings

**Purpose:** Control auto-fixer behavior.

**Settings:**
```yaml
leakage_detection:
  auto_fix_min_confidence: 0.8  # Minimum confidence to auto-fix (80%)
  auto_fix_max_features_per_run: 20  # Max features to fix per run
  auto_fix_enabled: true  # Enable/disable auto-fixer
```

**How It Works:**
- Auto-fixer only fixes features with confidence ≥ `auto_fix_min_confidence`
- Limits to top N features per run to prevent overly aggressive fixes
- Can be disabled entirely if you prefer manual fixing

**Example: Making Auto-Fixer More Aggressive**

```yaml
leakage_detection:
  auto_fix_min_confidence: 0.7  # Lower from 0.8 (fix with 70% confidence)
  auto_fix_max_features_per_run: 30  # Increase from 20
```

**Example: Disabling Auto-Fixer**

```yaml
leakage_detection:
  auto_fix_enabled: false  # Disable automatic fixing
```

### Auto-Rerun Settings

**Purpose:** Control automatic re-evaluation after fixes.

**Settings:**
```yaml
leakage_detection:
  auto_rerun:
    enabled: true  # Enable automatic rerun
    max_reruns: 3  # Maximum reruns per target
    rerun_on_perfect_train_acc: true  # Rerun on perfect training accuracy
    rerun_on_high_auc_only: false  # Rerun on high AUC alone (default: false)
```

**How It Works:**
- After auto-fixer modifies configs, target is automatically re-evaluated
- Continues until no leakage detected or `max_reruns` reached
- Only reruns if configs were actually modified

**Example: Increasing Max Reruns**

```yaml
leakage_detection:
  auto_rerun:
    enabled: true
    max_reruns: 5  # Increase from 3
```

**Example: Disabling Auto-Rerun**

```yaml
leakage_detection:
  auto_rerun:
    enabled: false  # Disable automatic rerun
```

### Active Sanitization (Ghost Buster)

**Purpose:** Proactively quarantine features with excessive lookback before training starts.

**Settings:**
```yaml
active_sanitization:
  enabled: true  # Enable active sanitization
  max_safe_lookback_minutes: 240.0  # Maximum safe lookback in minutes (default: 4 hours)
  pattern_quarantine:
    enabled: false  # Optional pattern-based quarantine (more aggressive)
    patterns: []  # List of regex patterns to match
```

**How It Works:**
- Scans features after all other filtering (registry, patterns, etc.)
- Computes lookback for each feature using same logic as auto-fix
- Quarantines features with lookback > `max_safe_lookback_minutes`
- Prevents "ghost feature" discrepancies where audit and auto-fix see different lookback values

**Example: Default Configuration (Recommended)**

```yaml
active_sanitization:
  enabled: true
  max_safe_lookback_minutes: 240.0  # 4 hours
  pattern_quarantine:
    enabled: false
```

**Example: More Permissive (Allow Longer Lookback)**

```yaml
active_sanitization:
  enabled: true
  max_safe_lookback_minutes: 480.0  # 8 hours
```

**Example: Pattern-Based Quarantine (Aggressive)**

```yaml
active_sanitization:
  enabled: true
  max_safe_lookback_minutes: 240.0
  pattern_quarantine:
    enabled: true
    patterns:
      - ".*_1d$"      # Ends in _1d
      - ".*_24h$"     # Ends in _24h
      - ".*daily.*"   # Contains "daily"
```

**Example: Disabling Active Sanitization**

```yaml
active_sanitization:
  enabled: false
```

**See Also:**
- [Active Sanitization Guide](../../03_technical/implementation/ACTIVE_SANITIZATION.md) - Complete documentation

---

## Common Scenarios

### Scenario 1: Making Leakage Detection More Sensitive

**Goal:** Catch more potential leakage cases.

**Steps:**
1. Lower auto-fix thresholds:
```yaml
leakage_detection:
  auto_fix_thresholds:
    cv_score: 0.95  # Lower from 0.99
    training_accuracy: 0.98  # Lower from 0.999
```

2. Lower auto-fixer confidence:
```yaml
  auto_fix_min_confidence: 0.7  # Lower from 0.8
```

3. Lower pre-scan thresholds:
```yaml
  pre_scan:
    min_match: 0.95  # Lower from 0.999
    min_corr: 0.95
```

### Scenario 2: Making Leakage Detection Less Sensitive

**Goal:** Reduce false positives.

**Steps:**
1. Raise auto-fix thresholds:
```yaml
leakage_detection:
  auto_fix_thresholds:
    cv_score: 0.999  # Higher threshold
    training_accuracy: 0.9999
```

2. Raise auto-fixer confidence:
```yaml
  auto_fix_min_confidence: 0.9  # Higher from 0.8
```

### Scenario 3: Disabling Auto-Fixer (Manual Control)

**Goal:** Manually review and fix leakage.

**Steps:**
1. Disable auto-fixer:
```yaml
leakage_detection:
  auto_fix_enabled: false
```

2. Review logs for leakage warnings
3. Manually update `excluded_features.yaml` or `feature_registry.yaml`

### Scenario 4: Adjusting Numerical Stability

**Goal:** Prevent numerical issues with extreme values.

**Steps:**
1. Enable feature clipping:
```yaml
numerical_stability:
  feature_clipping:
    enabled: true
    min_value: -10.0
    max_value: 10.0
```

2. Enable target capping:
```yaml
  target_capping:
    enabled: true
    max_abs_value: 1.0
```

3. Enable gradient clipping (for neural networks):
```yaml
  gradient_clipping:
    enabled: true
    max_norm: 1.0
```

---

## Best Practices

1. **Start conservative** - Use default thresholds, then adjust based on results
2. **Monitor auto-fixer** - Review logs to see what features were auto-excluded
3. **Review backups** - Check `RESULTS/{cohort_id}/{run_name}/backups/` (NEW: integrated) or `CONFIG/backups/` (legacy) to understand what changed
4. **Test threshold changes** - Verify new thresholds work as expected
5. **Balance sensitivity** - Too sensitive = false positives, too lenient = missed leaks
6. **Use pre-scan** - Pre-training scan catches obvious leaks early
7. **Enable auto-rerun** - Let the system automatically verify fixes

---

## Related Documentation

- [Configuration System Overview](README.md) - Main configuration overview
- [Feature & Target Configs](FEATURE_TARGET_CONFIGS.md) - Feature configuration
- [Training Pipeline Configs](TRAINING_PIPELINE_CONFIGS.md) - Training configuration
- [Usage Examples](USAGE_EXAMPLES.md) - Practical configuration examples
- [Intelligence Layer Overview](../../03_technical/research/INTELLIGENCE_LAYER.md) - How leakage detection works

