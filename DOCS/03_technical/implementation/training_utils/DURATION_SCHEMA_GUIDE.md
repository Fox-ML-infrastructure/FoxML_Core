# Duration Schema Guide

## Bar-Based Duration Encoding

To avoid ambiguity between "seconds" and "bars", configs should explicitly encode bar-based durations.

## Recommended Schema Patterns

### Pattern 1: Explicit String Format (Preferred)

Use explicit suffixes in duration strings:

```yaml
feature_lookback: "20b"      # 20 bars
feature_lookback: "20bars"    # 20 bars (alternative)
purge_buffer: "5b"            # 5 bars
```

**Parser Support**: Use `parse_duration_bars()` which accepts:
- `"20b"` → 20 bars
- `"20bars"` → 20 bars
- `"20"` → 20 bars (when passed to `parse_duration_bars()`)

### Pattern 2: Separate Fields (Alternative)

Use separate fields for bars vs duration:

```yaml
# Option A: Explicit bar field
feature_lookback_bars: 20
data_interval: "5m"

# Option B: Explicit duration field
feature_lookback_duration: "100m"  # Explicit time duration
```

**Implementation**: Config loader should handle both patterns and convert to Duration internally.

### Pattern 3: Context-Aware Parsing (Not Recommended)

Avoid relying on `assume_bars_if_number=True` at callsites. This creates ambiguity:

```python
# ❌ BAD: Ambiguous - is 20 seconds or bars?
parse_duration(20, interval="5m", assume_bars_if_number=True)

# ✅ GOOD: Explicit
parse_duration_bars(20, "5m")
parse_duration_bars("20b", "5m")
```

## Config Schema Example

```yaml
# Recommended: Explicit bar format
data:
  bar_interval: "5m"
  
features:
  max_lookback: "20b"  # Explicit: 20 bars
  
training:
  purge_buffer: "5b"   # Explicit: 5 bars
  default_purge: "85m"  # Explicit: 85 minutes (time duration)
```

## Migration Path

1. **Current State**: Some configs use bare numbers (ambiguous)
2. **Migration**: Update configs to use explicit formats (`"20b"` or separate `lookback_bars` field)
3. **Parser**: Use `parse_duration_bars()` for all bar-based durations
4. **Validation**: Config schema should validate bar formats

## Parser Functions

- **`parse_duration()`**: For time-based durations (strings, timedeltas, seconds)
- **`parse_duration_bars()`**: For bar-based durations (explicit, no ambiguity)

## Best Practices

1. ✅ **Always use explicit formats** (`"20b"`, `"20bars"`) in configs
2. ✅ **Use `parse_duration_bars()`** for bar-based parsing
3. ✅ **Separate fields** (`lookback_bars` vs `lookback_duration`) if schema supports it
4. ❌ **Avoid** `assume_bars_if_number=True` in new code
5. ❌ **Never** rely on context to determine if a number is seconds or bars
