# Active Sanitization (Ghost Buster)

Complete guide to the active sanitization system that proactively quarantines problematic features before training starts.

## Overview

Active sanitization is a **proactive feature quarantine system** that automatically removes features with excessive lookback windows before training begins. This prevents "ghost feature" discrepancies where audit and auto-fix see different lookback values.

### The Problem It Solves

Previously, the system had a conflict:
- **Audit enforcer** detected 1440m lookback (from daily/24h features)
- **Auto-fix resolver** calculated 1000m lookback (from sma_200)
- **Result**: `AUDIT VIOLATION: purge_minutes (1010) < feature_lookback_max_minutes (1440)`

This happened because:
1. Daily/24h features (e.g., `rolling_max_1440`, `volatility_daily`) require 1440 minutes (24 hours) of history
2. The auto-fix logic calculated lookback from bar-based patterns (e.g., `sma_200` = 200 bars * 5m = 1000m)
3. The audit logic detected the daily features separately, seeing 1440m
4. The purge was auto-adjusted to 1010m (1000 * 1.01), but this was still < 1440m

### The Solution

Active sanitization **quarantines problematic features BEFORE lookback computation**, ensuring both audit and auto-fix see the same values:

1. Features are scanned after all other filtering (registry, patterns, etc.)
2. Lookback is computed for each feature
3. Features with lookback > `max_safe_lookback_minutes` are quarantined
4. Only safe features proceed to training
5. Both audit and auto-fix now see the same (reduced) lookback values

## Architecture

### Module: `TRAINING/utils/feature_sanitizer.py`

**Main Function: `auto_quarantine_long_lookback_features()`**

```python
def auto_quarantine_long_lookback_features(
    feature_names: List[str],
    interval_minutes: Optional[float] = None,
    max_safe_lookback_minutes: Optional[float] = None,
    enabled: Optional[bool] = None
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """
    Active sanitization: automatically quarantine features with excessive lookback.
    
    Returns:
        (safe_features, quarantined_features, quarantine_report) tuple
    """
```

**How It Works:**
1. Loads configuration from `safety_config.yaml`
2. Computes lookback for each feature using `compute_feature_lookback_max()` (same logic as auto-fix)
3. Separates features into safe vs. quarantined based on threshold
4. Logs quarantine details
5. Returns safe features, quarantined features, and detailed report

**Pattern-Based Quarantine: `quarantine_by_pattern()`**

Optional more aggressive approach that quarantines features based on naming patterns rather than computed lookback:

```python
def quarantine_by_pattern(
    feature_names: List[str],
    patterns: Optional[List[str]] = None,
    enabled: Optional[bool] = None
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """
    Quarantine features by regex patterns (for specific problematic patterns).
    """
```

### Integration: `TRAINING/utils/leakage_filtering.py`

Active sanitization is integrated into `filter_features_for_target()`, running **after all other filtering**:

```python
# After registry filtering, pattern filtering, etc.
try:
    from TRAINING.utils.feature_sanitizer import auto_quarantine_long_lookback_features
    
    sanitized_features, quarantined_features, quarantine_report = auto_quarantine_long_lookback_features(
        feature_names=safe_columns,
        interval_minutes=data_interval_minutes,
        max_safe_lookback_minutes=None,  # Loads from config
        enabled=None  # Loads from config
    )
    
    if quarantined_features:
        safe_columns = sanitized_features
        # Log quarantine details
except Exception as e:
    # Don't fail if sanitization unavailable - just log and continue
    logger.debug(f"Active sanitization unavailable: {e}")
```

## Configuration

### `CONFIG/training_config/safety_config.yaml`

```yaml
# Active Sanitization (Ghost Buster)
# Automatically quarantines features with excessive lookback before training starts
active_sanitization:
  enabled: true  # Enable active sanitization (default: enabled)
  max_safe_lookback_minutes: 240.0  # Maximum safe lookback in minutes (default: 4 hours)
  # Features with lookback > this threshold will be automatically quarantined
  # Set to null to disable lookback-based quarantine (only use pattern-based)
  
  # Pattern-based quarantine (more aggressive - quarantines by naming patterns)
  pattern_quarantine:
    enabled: false  # Default: disabled (more aggressive, use with caution)
    patterns: []  # List of regex patterns to match (e.g., [".*_1d$", ".*daily.*"])
    # If enabled, features matching these patterns will be quarantined regardless of computed lookback
```

### Configuration Options

**`enabled`** (default: `true`)
- Enable/disable active sanitization
- If disabled, all features pass through (no quarantine)

**`max_safe_lookback_minutes`** (default: `240.0`)
- Maximum safe lookback in minutes (default: 4 hours)
- Features with lookback > this threshold are quarantined
- Increase if you want to allow longer lookback features
- Set to `null` to disable lookback-based quarantine

**`pattern_quarantine.enabled`** (default: `false`)
- Enable pattern-based quarantine (more aggressive)
- If enabled, features matching patterns are quarantined regardless of computed lookback
- Use with caution - can be overly aggressive

**`pattern_quarantine.patterns`** (default: `[]`)
- List of regex patterns to match
- Examples: `[".*_1d$", ".*daily.*", ".*_1440m"]`
- Only used if `pattern_quarantine.enabled: true`

## Usage Examples

### Example 1: Default Configuration (Recommended)

```yaml
active_sanitization:
  enabled: true
  max_safe_lookback_minutes: 240.0  # 4 hours
  pattern_quarantine:
    enabled: false
```

**Behavior:**
- Features with lookback > 240m are quarantined
- Daily/24h features (1440m) are automatically quarantined
- Pattern-based quarantine is disabled

### Example 2: More Permissive (Allow Longer Lookback)

```yaml
active_sanitization:
  enabled: true
  max_safe_lookback_minutes: 480.0  # 8 hours
  pattern_quarantine:
    enabled: false
```

**Behavior:**
- Features with lookback > 480m are quarantined
- Daily/24h features (1440m) are still quarantined
- Allows features with up to 8 hours of lookback

### Example 3: Pattern-Based Quarantine (Aggressive)

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
      - ".*_1440m.*"  # Contains _1440m
```

**Behavior:**
- Features matching patterns are quarantined regardless of computed lookback
- More aggressive - catches known problematic patterns even if lookback calculation fails

### Example 4: Disable Active Sanitization

```yaml
active_sanitization:
  enabled: false
```

**Behavior:**
- All features pass through (no quarantine)
- Use if you want to handle feature filtering manually

## Logging

Active sanitization logs detailed information about quarantined features:

```
👻 ACTIVE SANITIZATION: Quarantined 2 feature(s) with lookback > 240.0m to prevent audit violations
   🚫 rolling_max_1440: lookback (1440.0m) exceeds safe threshold (240.0m)
   🚫 volatility_daily: lookback (1440.0m) exceeds safe threshold (240.0m)
   ✅ 42 safe features remaining
```

If no features are quarantined:
```
✅ Active sanitization: All 42 features passed (lookback <= 240.0m)
```

## Integration with Other Systems

### Feature Filtering Pipeline

Active sanitization runs **after** all other filtering:
1. Metadata exclusion (hardcoded safety net)
2. Pattern-based exclusion (`excluded_features.yaml`)
3. Feature registry filtering
4. Target-specific filtering
5. **Active sanitization** ← Runs here
6. Ranking mode safe feature inclusion (if applicable)

### Auto-Fix System

Active sanitization works **with** the auto-fix system:
- **Auto-fix**: Adjusts `purge_minutes` based on feature lookback
- **Active sanitization**: Quarantines features with excessive lookback before auto-fix runs
- **Result**: Both systems see the same (reduced) lookback values

### Audit System

Active sanitization prevents audit violations:
- **Before**: Audit saw 1440m, auto-fix saw 1000m → `AUDIT VIOLATION`
- **After**: Both see 1000m (daily features quarantined) → No violation

## Troubleshooting

### Issue: Too Many Features Quarantined

**Symptom**: Many features are being quarantined, leaving too few for training.

**Solution**: Increase `max_safe_lookback_minutes`:
```yaml
active_sanitization:
  max_safe_lookback_minutes: 480.0  # Increase from 240.0
```

### Issue: Daily Features Still Causing Issues

**Symptom**: Daily features are not being quarantined, still causing audit violations.

**Solution**: Enable pattern-based quarantine:
```yaml
active_sanitization:
  pattern_quarantine:
    enabled: true
    patterns:
      - ".*_1d$"
      - ".*_24h$"
      - ".*daily.*"
```

### Issue: Active Sanitization Not Running

**Symptom**: No quarantine logs, features with excessive lookback still present.

**Solution**: Check configuration:
1. Verify `active_sanitization.enabled: true` in `safety_config.yaml`
2. Check logs for "Active sanitization unavailable" errors
3. Verify `filter_features_for_target()` is being called

## Best Practices

1. **Start with defaults**: Use default `max_safe_lookback_minutes: 240.0` (4 hours)
2. **Monitor quarantine logs**: Review what features are being quarantined
3. **Adjust threshold carefully**: Only increase if you have a good reason
4. **Use pattern-based quarantine sparingly**: More aggressive, can be overly restrictive
5. **Keep enabled**: Active sanitization prevents "ghost feature" discrepancies

## Related Documentation

- [Safety & Leakage Config Guide](../../02_reference/configuration/SAFETY_LEAKAGE_CONFIGS.md) - Configuration reference
- [Feature Lookback Calculation](../../02_reference/configuration/FEATURE_TARGET_CONFIGS.md) - How lookback is computed
