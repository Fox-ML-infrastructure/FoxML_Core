# Feature Selection Lookback Cap Integration

**Date**: 2025-12-13  
**Related**: [Single Source of Truth Fix](INTERNAL/SINGLE_SOURCE_OF_TRUTH_COMPLETE.md) | [SST Enforcement Design](INTERNAL/SST_ENFORCEMENT_DESIGN.md)

## Overview

Integrated the same single-source-of-truth + cap enforcement pipeline used in ranking into feature selection. This ensures consistent behavior and prevents split-brain between ranking and feature selection.

## Architecture

### Shared Function: `apply_lookback_cap()`

**File**: `TRAINING/utils/lookback_cap_enforcement.py`

Single function used by both ranking and feature selection:
- Builds canonical lookback map (or uses provided)
- Quarantines features exceeding cap
- Computes budget from safe features
- Validates invariants (hard-fail in strict mode)
- Returns safe features + metadata

**Pipeline**:
1. Build canonical lookback map (single source of truth)
2. Quarantine features exceeding cap
3. Compute budget from safe features
4. Validate invariants (hard-fail in strict mode)
5. Return safe features + metadata

### Integration Points in Feature Selection

**File**: `TRAINING/ranking/feature_selector.py`

#### 1. Pre-Selection Enforcement (FS_PRE)

**Location**: After `apply_cleaning_and_audit_checks()`, before `run_importance_producers()`

**Purpose**: Prevents selector from even seeing unsafe features (faster + safer)

**Stages**:
- `FS_PRE_CROSS_SECTIONAL` (for CROSS_SECTIONAL view)
- `FS_PRE_SYMBOL_SPECIFIC_{symbol}` (for SYMBOL_SPECIFIC view, per symbol)

**Behavior**:
- Quarantines features exceeding cap
- Updates `feature_names` to only include safe features
- Updates `X` matrix to remove quarantined feature columns
- Updates `resolved_config.feature_lookback_max_minutes` with new max

#### 2. Post-Selection Enforcement (FS_POST)

**Location**: After `_aggregate_multi_model_importance()`, before returning selected features

**Purpose**: Catches long-lookback features that selection surfaced (prevents "pruning surfaced long-lookback" bugs)

**Stage**: `FS_POST_{view}`

**Behavior**:
- Quarantines selected features exceeding cap
- Updates `selected_features` to only include safe features
- Updates `summary_df` to remove quarantined feature rows
- Hard-fails in strict mode if cap violation detected

## Logging Improvements

### Config Knob: `log_mode`

**Location**: `CONFIG/pipeline/training/safety.yaml`

```yaml
safety:
  leakage_detection:
    log_mode: "summary"  # "summary" | "debug"
```

**Modes**:
- **`summary`** (default): One-line summaries per stage
  - `📊 FS_PRE_CROSS_SECTIONAL: n_features=300 → safe=177 quarantined=123 cap=240.0m actual_max=150.0m`
  - `   Top offenders: macd_signal(1440m), vwap_dev_low(1440m), ...`
- **`debug`**: Per-feature inference traces, pattern matches, recompute details

### Logging Changes

**File**: `TRAINING/utils/leakage_budget.py`

- Per-feature inference logs (`infer_lookback_minutes(...)`) → `DEBUG` (was `INFO`)
- Pattern match logs → `DEBUG` (was `INFO`)
- Registry lookup logs → `DEBUG` (was `INFO`)
- Stage summaries → `INFO` (one-liner per stage)
- Safety events (cap violations, quarantines, unknown lookbacks) → `WARNING/ERROR`

**Result**: Default logs show 1-3 lines per stage (counts + top offenders), not hundreds of per-feature traces.

## Telemetry Updates

**File**: `TRAINING/ranking/feature_selector.py` (lines 1483-1490)

Lookback cap enforcement results are tracked in telemetry:

```python
lookback_cap_metadata = {
    'pre_selection': {
        'quarantine_count': 123,
        'actual_max_lookback': 150.0,
        'safe_features_count': 177,
        'quarantined_features_sample': ['macd_signal', 'vwap_dev_low', ...]
    },
    'post_selection': {
        'quarantine_count': 0,
        'actual_max_lookback': 150.0,
        'safe_features_count': 100,
        'quarantined_features_sample': []
    }
}
```

Stored in `additional_data['lookback_cap_enforcement']` for reproducibility tracking.

## Invariants

### 1. Oracle Consistency

**Check**: `budget.max_feature_lookback_minutes == actual_max_uncapped_from_map`

**Location**: `apply_lookback_cap()` (lines 150-165)

**Behavior**: Hard-fails in strict mode if violated

### 2. No Late Cap Violations

**Check**: After `FS_POST`, assert `actual_max <= cap`

**Location**: `apply_lookback_cap()` (lines 130-145)

**Behavior**: 
- Strict mode: Hard-fail with `RuntimeError`
- Warn mode: Log warning and continue (if allowed)

## View Support

### CROSS_SECTIONAL View

- Pre-selection: `FS_PRE_CROSS_SECTIONAL` (single enforcement for all symbols)
- Post-selection: `FS_POST_CROSS_SECTIONAL` (single enforcement after aggregation)

### SYMBOL_SPECIFIC View

- Pre-selection: `FS_PRE_SYMBOL_SPECIFIC_{symbol}` (per-symbol enforcement)
- Post-selection: `FS_POST_SYMBOL_SPECIFIC` (single enforcement after aggregation)

**Note**: Both views use the same `apply_lookback_cap()` function, ensuring consistent behavior.

## Files Modified

1. **`TRAINING/utils/lookback_cap_enforcement.py`** (NEW)
   - Shared function for lookback cap enforcement
   - Used by both ranking and feature selection

2. **`TRAINING/ranking/feature_selector.py`**
   - Pre-selection enforcement (FS_PRE) - lines 305-368, 508-575
   - Post-selection enforcement (FS_POST) - lines 660-870
   - Telemetry tracking - lines 1483-1490

3. **`TRAINING/utils/leakage_budget.py`**
   - Per-feature logs → DEBUG (respects log_mode config)
   - Stage summaries → INFO

4. **`CONFIG/pipeline/training/safety.yaml`**
   - Added `log_mode: "summary"` config knob

## Testing

### Unit Tests

**File**: `TRAINING/utils/test_xd_pattern_inference.py`

Tests verify:
- `_Xd` pattern inference
- Canonical map includes all features
- Gatekeeper drops offenders correctly

### Integration Tests

Run feature selection with `lookback_budget_minutes: 240` and verify:

1. **Pre-selection**:
   ```
   📊 FS_PRE_CROSS_SECTIONAL: n_features=300 → safe=177 quarantined=123 cap=240.0m actual_max=150.0m
      Top offenders: macd_signal(1440m), vwap_dev_low(1440m), ...
   ```

2. **Post-selection**:
   ```
   📊 FS_POST_CROSS_SECTIONAL: n_features=100 → safe=100 quarantined=0 cap=240.0m actual_max=150.0m
   ```

3. **No late cap violations**: Selected features never exceed cap

4. **Telemetry**: `lookback_cap_enforcement` metadata in reproducibility logs

## Expected Behavior

### Before Integration
- Feature selection could select features exceeding cap
- No enforcement before/after selection
- Split-brain possible (ranking enforces, feature selection doesn't)

### After Integration
- Pre-selection: Unsafe features quarantined before selector sees them
- Post-selection: Selected features validated against cap
- Hard-fail in strict mode if cap violation detected
- Consistent behavior with ranking (same oracle, same enforcement)

## Verification Checklist

- [x] Shared function `apply_lookback_cap()` created
- [x] Pre-selection enforcement integrated (FS_PRE)
- [x] Post-selection enforcement integrated (FS_POST)
- [x] Works with CROSS_SECTIONAL view
- [x] Works with SYMBOL_SPECIFIC view
- [x] Logging respects log_mode config (summary vs debug)
- [x] Per-feature logs demoted to DEBUG
- [x] Stage summaries at INFO level
- [x] Telemetry tracks lookback cap enforcement
- [x] Invariants enforced (hard-fail in strict mode)
- [x] No syntax errors

## Related Documentation

- [Single Source of Truth Fix Summary](SINGLE_SOURCE_OF_TRUTH_COMPLETE.md)
- [POST_PRUNE Invariant Check](POST_PRUNE_INVARIANT_CHECK.md)
- [XD Pattern Fix Summary](XD_PATTERN_FIX_SUMMARY.md)

