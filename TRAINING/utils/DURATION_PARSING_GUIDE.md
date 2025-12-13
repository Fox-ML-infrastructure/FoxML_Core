# Duration Parsing and Canonicalization Guide

## Overview

The duration parsing system provides a **generalized, agnostic** solution for handling time periods across the entire codebase. It eliminates unit ambiguity and string comparison issues by enforcing a single invariant:

> **Everything becomes a `Duration` (canonical representation) before comparison or computation.**

## Architecture

### Core Invariant

1. **Parse** input → `Duration` (canonical representation)
2. **Enforce rules** using `Duration` arithmetic
3. **Format** `Duration` → string only at the edge (logging/UI)

### Key Components

- **`Duration` class**: Stores durations as integer microseconds for stable arithmetic
- **`parse_duration()`**: Universal parser for strings, timedeltas, numbers
- **`parse_duration_bars()`**: Explicit bar-based parsing (removes ambiguity)
- **`enforce_purge_audit_rule()`**: Generalized audit rule enforcement
- **`ceil_to_interval()`**: Interval-aware rounding

## Improvements Made

### 1. Fail-Closed Policy ✅

**Before**: Silent fallback to numeric comparison if parsing failed (footgun)

**After**: 
- **Strict mode**: Raises `ValueError` immediately
- **Warn mode**: Logs loud error, marks run as `non_auditable`, adds violation

This prevents "bad config" from becoming "quietly wrong leakage logic."

### 2. Internal Type Consistency ✅

**Before**: Converted to float minutes immediately, undermining the invariant

**After**:
- Core logic uses `Duration` objects internally
- Conversion to float minutes happens only at the boundary (for backward compatibility)
- Documented in `ResolvedConfig` that float minutes are a conversion boundary

### 3. Interval-Aware Strictness (Primary Mechanism) ✅

**Before**: Buffer fraction was the only mechanism

**After**:
- **Primary**: `min_purge = ceil_to_interval(lookback_max, interval) + interval`
  - Guarantees `purge > lookback_max` at data resolution
  - This is the **correct general solution**
- **Fallback**: `min_purge = lookback_max * (1 + buffer_frac)`
  - Only used when interval is unknown
  - Less precise but safe

### 4. Explicit Bar Parsing (Removes Ambiguity) ✅

**Before**: `assume_bars_if_number=True` created ambiguity ("is this seconds or bars?")

**After**:
- **`parse_duration_bars()`**: Explicit function for bar-based durations
- Supports: `20`, `"20b"`, `"20bars"` (all unambiguous)
- Best practice: Use this instead of `assume_bars_if_number=True`

**Future**: Config schema should support explicit formats like `lookback: "20b"` or separate fields (`lookback_bars` vs `lookback_duration`).

### 5. Policy Documentation ✅

Added clear documentation that:
- The rule "purge must exceed feature_lookback_max" is a **conservative policy**
- It's not a mathematical necessity if features are strictly past-only
- Purge/embargo is primarily driven by label horizon / overlap of label windows
- This policy prevents rolling window leakage as a safety measure

### 6. Float Rounding (Future Consideration)

Currently uses float → microseconds conversion. For bulletproof audit boundaries, consider using `Decimal` for parsing numeric tokens. This is a future enhancement if precision issues arise.

## Usage Examples

### Basic Duration Parsing

```python
from TRAINING.utils.duration_parser import parse_duration, format_duration

# Parse various formats
d1 = parse_duration("85.0m")
d2 = parse_duration("1h30m")
d3 = parse_duration("2d")
d4 = parse_duration("1h30m15s")

# All become Duration objects (canonical)
assert d1 < d2  # Comparison is unambiguous
```

### Bar-Based Lookbacks (Explicit, No Ambiguity)

```python
from TRAINING.utils.duration_parser import parse_duration_bars

# Explicit bar parsing (preferred)
lookback = parse_duration_bars(20, "5m")  # 20 bars at 5m = 100m
lookback2 = parse_duration_bars("20b", "5m")  # Same, explicit format
lookback3 = parse_duration_bars("20bars", "5m")  # Same

# All unambiguous - no "is this seconds or bars?" question
```

### Enforce Audit Rule

```python
from TRAINING.utils.duration_parser import enforce_purge_audit_rule

# Interval-aware strictness (primary mechanism)
purge_out, min_purge, changed = enforce_purge_audit_rule(
    "85.0m",
    "100.0m",
    interval="5m",  # Primary: interval-aware rounding
    buffer_frac=0.01,  # Fallback: only used if interval=None
    strict_greater=True
)

# Result: purge_out = 105m (ceil_to_interval(100m, 5m) + 5m)
```

### Fallback When Interval Unknown

```python
# When interval is unknown, falls back to buffer
purge_out, min_purge, changed = enforce_purge_audit_rule(
    "85.0m",
    "100.0m",
    interval=None,  # No interval = fallback to buffer
    buffer_frac=0.01,
    strict_greater=True
)

# Result: purge_out = 101m (100m * 1.01)
```

## Generalization Across Codebase

### Where This Applies

This system is **agnostic** and can be used anywhere durations are compared or computed:

1. **Purge/Embargo/Lookback** (already implemented)
   - `TRAINING/utils/resolved_config.py`
   - `TRAINING/utils/audit_enforcer.py`

2. **Feature Lookback Calculations**
   - `TRAINING/utils/resolved_config.py::compute_feature_lookback_max()`
   - Can be extended to use `parse_duration_bars()` for bar-based lookbacks

3. **Data Interval Detection**
   - `TRAINING/utils/data_interval.py`
   - Can normalize all interval strings to `Duration` before comparison

4. **Time Series Splits**
   - `TRAINING/utils/purged_time_series_split.py`
   - Can use `Duration` for all time-based calculations

5. **Feature Sanitization**
   - `TRAINING/utils/feature_sanitizer.py`
   - Can use `Duration` for lookback comparisons

### Migration Path

For existing code that uses float minutes:

1. **Keep backward compatibility**: Convert `Duration` → float at boundaries
2. **Gradually migrate**: Use `Duration` internally, convert only at edges
3. **New code**: Use `Duration` directly, avoid float minutes

### Integration Points

- **Config loading**: Can parse duration strings directly from YAML
- **Logging**: Use `format_duration()` for consistent formatting
- **Validation**: Use `Duration` comparisons for all time-based rules
- **Serialization**: Convert to float minutes for JSON/metadata (boundary conversion)

## Testing

All improvements are tested and verified:

- ✅ Explicit bar parsing works correctly
- ✅ Interval-aware strictness is primary mechanism
- ✅ Buffer fallback works when interval unknown
- ✅ Fail-closed behavior raises errors (no silent fallbacks)
- ✅ Integration with existing `resolved_config` works

## Future Enhancements

1. **Config Schema**: Add explicit bar format support (`"20b"`, `"20bars"`)
2. **Decimal Precision**: Use `Decimal` for parsing if precision issues arise
3. **ResolvedConfig Migration**: Consider storing `Duration` internally (larger refactor)
4. **Wider Adoption**: Migrate more modules to use `Duration` directly

## Summary

The duration parsing system is now:
- ✅ **Generalized**: Works with any time period format
- ✅ **Agnostic**: Can be used anywhere durations are needed
- ✅ **Fail-closed**: No silent fallbacks that hide errors
- ✅ **Interval-aware**: Primary mechanism uses data resolution
- ✅ **Unambiguous**: Explicit bar parsing removes ambiguity
- ✅ **Well-documented**: Policy vs mathematical necessity clarified

This provides a solid foundation for handling durations across the entire codebase.
