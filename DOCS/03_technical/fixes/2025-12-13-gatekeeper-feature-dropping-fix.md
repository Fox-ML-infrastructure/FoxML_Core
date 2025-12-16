# Gatekeeper Feature Dropping and Stale Budget Fix

**Date**: 2025-12-13  
**Related**: [Lookback Fingerprint Tracking](2025-12-13-lookback-fingerprint-tracking.md) | [Leakage Validation Fix](2025-12-13-leakage-validation-fix.md)

## Problem

The system was showing contradictory behavior:

1. **Features not being dropped**: `volatility_60d`, `volume_*_20d` features were still in `MODEL_TRAIN_INPUT` despite having 86400m/28800m lookback
2. **Stale budget values**: Summary showed `max_feature_lookback_minutes: 70.0m` but actual POST_PRUNE max was `86400.0m`
3. **CV explosion**: `purge_minutes: 86405.0m` on a 16-day dataset, causing empty/invalid CV folds
4. **Invariant violations**: `budget.actual_max=70.0m but actual max from features=86400.0m`

## Root Causes

### 1. Gatekeeper checking patterns before lookback

**Location**: `TRAINING/ranking/predictability/model_evaluation.py::_enforce_final_safety_gate()`

**Issue**: The gatekeeper was checking `is_daily_name` (pattern matching) BEFORE checking `lookback_minutes > safe_lookback_max`. This meant:
- Features with explicit suffixes like `_60d`, `_20d` were correctly identified as having long lookback
- But the pattern check happened first, and these features don't match "daily pattern" (they have explicit suffixes)
- So they weren't being dropped

**Fix**: Reordered checks - now `lookback_minutes > safe_lookback_max` is checked FIRST, before pattern matching.

### 2. Gatekeeper using purge_limit when lookback_budget_minutes is "auto"

**Issue**: If `lookback_budget_minutes: auto` (no cap), gatekeeper falls back to `purge_limit * 0.99`. But:
- Purge is computed FROM feature lookback (if `purge_include_feature_lookback: true`)
- If features have 86400m lookback, purge becomes 86405m
- Gatekeeper threshold becomes huge, so nothing gets dropped
- Circular dependency: purge depends on lookback, but gatekeeper depends on purge

**Fix**: Gatekeeper now checks `lookback_budget_minutes` cap FIRST (if set). Only falls back to purge_limit if no cap.

### 3. Stale budget in summary

**Location**: `TRAINING/ranking/predictability/model_evaluation.py::_log_canonical_summary()`

**Issue**: Summary was using `resolved_config.feature_lookback_max_minutes` which might be stale if it wasn't updated correctly at POST_PRUNE.

**Fix**: Added validation - if `resolved_config.feature_lookback_max_minutes` doesn't match POST_PRUNE `computed_lookback`, use POST_PRUNE value (the truth).

### 4. Purge > data span causing CV explosion

**Location**: `TRAINING/ranking/predictability/model_evaluation.py::train_and_evaluate_models()`

**Issue**: No validation that purge doesn't exceed data span. With 86405m purge on 16-day dataset, CV folds become empty/invalid.

**Fix**: Added hard-stop validation - if `purge_minutes >= data_span_minutes`, raise `RuntimeError` with clear message.

## Solution

### 1. Gatekeeper checks lookback FIRST

```python
# OLD (wrong order):
if is_daily_name:
    should_drop = True
elif lookback_minutes > safe_lookback_max:
    should_drop = True

# NEW (correct order):
if lookback_minutes > safe_lookback_max:
    should_drop = True  # Catches 60d/20d features
elif is_daily_name:
    should_drop = True  # Catches pattern-based features
```

### 2. Gatekeeper uses lookback_budget_minutes cap

```python
# Load lookback_budget_minutes cap from config
lookback_budget_cap = get_cfg("safety.leakage_detection.lookback_budget_minutes", ...)

# Priority: 1) config cap, 2) purge-derived
if lookback_budget_cap is not None:
    safe_lookback_max = lookback_budget_cap  # Use explicit cap
else:
    safe_lookback_max = purge_limit * 0.99  # Fallback to purge
```

### 3. Hard-stop if purge > data span

```python
# Validate purge doesn't exceed data span
if time_vals is not None and len(time_vals) > 0:
    data_span_minutes = (time_series.max() - time_series.min()).total_seconds() / 60.0
    if purge_minutes_val >= data_span_minutes:
        raise RuntimeError(
            f"🚨 INVALID CV CONFIGURATION: purge_minutes ({purge_minutes_val:.1f}m) >= data_span ({data_span_minutes:.1f}m). "
            f"This will produce empty/invalid CV folds. "
            f"Either: 1) Set lookback_budget_minutes cap to drop long-lookback features, "
            f"2) Load more data (≥ {purge_minutes_val/1440:.1f} trading days), or "
            f"3) Disable purge_include_feature_lookback in config."
        )
```

### 4. Summary validates and uses POST_PRUNE value

```python
# Validate resolved_config matches POST_PRUNE computed value
if 'computed_lookback' in locals() and computed_lookback is not None:
    if abs(max_lookback_val - computed_lookback) > 1.0:
        logger.error(f"🚨 SUMMARY MISMATCH: ...")
        max_lookback_val = computed_lookback  # Use the truth
```

## Files Modified

- `TRAINING/ranking/predictability/model_evaluation.py`:
  - `_enforce_final_safety_gate()`: Reordered checks, added lookback_budget_cap support
  - `train_and_evaluate_models()`: Added purge > data span validation
  - Summary section: Added validation to use POST_PRUNE value
  - POST_PRUNE section: Fixed resolved_config update

## Expected Behavior

### With `lookback_budget_minutes: 240` (cap set)

1. Gatekeeper drops features with lookback > 240m (including `volatility_60d`, `volume_*_20d`)
2. Budget reflects actual feature set (max < 240m)
3. Purge computed from actual lookback (not 86400m)
4. CV folds are valid (purge < data span)
5. Summary shows correct lookback value

### With `lookback_budget_minutes: auto` (no cap)

1. Gatekeeper uses purge-derived limit
2. If features have 86400m lookback, purge becomes huge
3. **Hard-stop triggers**: "purge >= data_span" error with clear message
4. User must either: set cap, load more data, or disable `purge_include_feature_lookback`

## Validation

- ✅ Gatekeeper now drops `volatility_60d`, `volume_*_20d` if cap is set
- ✅ Hard-stop prevents CV explosion from purge > data span
- ✅ Summary shows correct lookback value (matches POST_PRUNE)
- ✅ No more "budget.actual_max=70m but actual=86400m" for same stage

## Configuration

For intraday ranking on small windows (recommended):

```yaml
leakage_detection:
  lookback_budget_minutes: 240  # Cap at 4 hours
  over_budget_action: drop      # Drop violating features
  purge_padding_minutes: 5      # Small padding for audit rule
```

This ensures:
- Features with lookback > 240m are dropped
- Purge stays reasonable (< data span)
- CV folds are valid
- Summary is accurate
