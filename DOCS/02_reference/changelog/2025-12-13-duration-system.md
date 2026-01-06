# Changelog — 2025-12-13 (Duration System & Audit Fixes)

**Generalized Duration Parsing System, Lookback Detection Precedence Fix, Documentation Review**

For a quick overview, see the [root changelog](../../../CHANGELOG.md).  
For other dates, see the [changelog index](README.md).

---

## Added

### Generalized Duration Parsing System

**Duration Parsing System**
- **New Module**: `TRAINING/utils/duration_parser.py` - Duration parsing and canonicalization for time periods
- **Duration Class**: Stores durations as integer microseconds for stable arithmetic
- **Universal Parser**: Handles strings (`"85.0m"`, `"1h30m"`, `"2d"`), timedeltas, numbers, and bar-based durations
- **Explicit Bar Parsing**: `parse_duration_bars()` removes ambiguity between seconds and bars
- **Interval-Aware Strictness**: Primary mechanism uses data resolution for purge/lookback enforcement
- **Files**:

**Key Features**:
- ✅ Canonical representation: Everything becomes `Duration` before comparison
- ✅ Fail-closed policy: No silent fallbacks on parsing errors
- ✅ Domain constraints: Negative durations, zero intervals rejected
- ✅ Interval-aware strictness: `min_purge = ceil_to_interval(lookback_max, interval) + interval`
- ✅ Explicit bar parsing: `parse_duration_bars()` removes ambiguity

### Non-Auditable Status Markers

**Sticky Markers for Audit Status**
- **Metadata Flags**: `is_auditable: False`, `audit_status: "non_auditable"` in run metadata
- **Prominent Output**: Large banner when run is non-auditable
- **Summary Lines**: `❌ AUDIT STATUS: NON-AUDITABLE` in training output
- **Downstream Helpers**: `is_auditable()`, `require_auditable()` functions for components
- **Files**:
- **Review Objectives**: Accuracy, realistic statements, removal of marketing language
- **Key Changes**: Model count corrections, qualified language, consistency fixes
- **Status**: Initial review completed, ongoing standards established

---

## Changed

### Lookback Detection Precedence Fix

**Fixed False Positives in Feature Lookback Detection**
- **Problem**: Features like `intraday_seasonality_15m` were incorrectly tagged as 1440m because keyword heuristics ran before explicit suffixes
- **Fix**: Reordered precedence to check explicit time suffixes (`_15m`, `_30m`, `_1d`, `_24h`) before keyword heuristics
- **Files Modified**:
  - `TRAINING/utils/resolved_config.py` - `compute_feature_lookback_max()`
  - `TRAINING/ranking/predictability/model_evaluation.py` - `_enforce_final_safety_gate()`
  - `TRAINING/utils/target_conditional_exclusions.py` - `compute_feature_lookback_minutes()`
- **New Precedence**:
  1. Registry metadata (most reliable)
  2. Explicit time suffixes (`_15m`, `_30m`, `_1d`, `_24h`) - **checked first**
  3. Keyword heuristics (`.*day.*`) - **fallback only**
- **Impact**: Eliminates false positives for features with explicit short lookbacks

### Audit Rule Enforcement

**Generalized and Hardened**
- **Interval-Aware Strictness**: Primary mechanism uses `ceil_to_interval(lookback_max, interval) + interval`
- **Buffer Fallback**: Only used when interval is unknown
- **Domain Constraints**: Negative durations, zero intervals rejected (fail fast)
- **Strict Mode Requirement**: `strict_greater=True` requires `interval` to be provided
- **Files**:
  - `TRAINING/utils/resolved_config.py` - Uses `enforce_purge_audit_rule()`
  - `TRAINING/utils/duration_parser.py` - Core enforcement logic

### Duplicate Warning Prevention

**Reduced Log Noise**
- **Fix**: Added caching to prevent duplicate "audit violation prevention" warnings
- **Implementation**: Cache key based on (purge, lookback, interval) combination
- **File**: `TRAINING/utils/resolved_config.py`

---

## Technical Details

### Duration Parsing Architecture

**Core Invariant**: Everything becomes a `Duration` (canonical representation) before comparison or computation.

**Flow**:
1. Parse input → `Duration` (canonical)
2. Validate domain constraints (fail fast)
3. Enforce rules using `Duration` arithmetic
4. Format `Duration` → string only at edge (logging/UI)

### Test Coverage

**Comprehensive Unit Tests**:
- ✅ Basic parsing (strings, timedeltas, floats)
- ✅ Compound strings (`"1h30m"`, `"90s"`)
- ✅ Bar-based parsing (`20b`, `20bars`)
- ✅ Interval-aware strictness
- ✅ Unaligned lookback rounding (102m → 110m)
- ✅ Strictness at equality (100m → 105m)
- ✅ Domain constraints (negative, zero, missing interval)
- ✅ Irregular interval handling

**All Tests Passing**: ✅ Verified

### Lookback Detection Test Results

**Before Fix**:
- ❌ `intraday_seasonality_15m` → 1440m (incorrect)
- ❌ `intraday_seasonality_30m` → 1440m (incorrect)

**After Fix**:
- ✅ `intraday_seasonality_15m` → 15.0m (correct)
- ✅ `intraday_seasonality_30m` → 30.0m (correct)
- ✅ `day_of_week` → 1440.0m (correct, no suffix)
- ✅ `mom_1d` → 1440.0m (correct, explicit suffix)

---

## Documentation

### New Documentation Files

- `DOCUMENTATION_REVIEW.md` - Documentation review statement

### Updated Documentation

- `TRAINING/utils/resolved_config.py` - Added `@deprecated` markers on float minute fields
- `TRAINING/utils/audit_enforcer.py` - Enhanced documentation for non-auditable status

---

## Migration Notes

### For New Code

- **Use Duration objects directly**: Prefer `Duration` over float minutes in new code
- **Explicit bar parsing**: Use `parse_duration_bars()` instead of `assume_bars_if_number=True`
- **Check audit status**: Use `is_auditable()` or `require_auditable()` in downstream components

### For Existing Code

- **Backward compatible**: Float minutes still work (converted at boundary)
- **Gradual migration**: Can migrate to `Duration` objects over time
- **Legacy fields**: Float minute fields marked `@deprecated` but still functional

---

## Related Issues

- Fixed false positives in feature lookback detection
- Eliminated unit ambiguity in duration comparisons
- Removed silent fallbacks that could hide configuration errors
- Established fail-closed policy for audit violations
