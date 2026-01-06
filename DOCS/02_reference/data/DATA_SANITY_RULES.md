# Data Sanity Rules

Data validation rules and quality checks.

## Validation Rules

### Time Series Requirements

1. **Sorted Index**: Data must be sorted by timestamp
2. **No Duplicates**: No duplicate timestamps allowed
3. **Continuous**: Gaps should be minimal (< 5% missing bars)
4. **Timezone**: All timestamps in UTC

### Price Data Requirements

1. **Positive Values**: All prices must be > 0
2. **High >= Low**: High price must be >= low price
3. **High >= Open/Close**: High must be >= open and close
4. **Low <= Open/Close**: Low must be <= open and close
5. **Volume >= 0**: Volume must be non-negative

### Bar Count Requirements

For 5-minute bars in RTH:
- **Expected**: 78 bars per day (9:30 AM - 4:00 PM)
- **Minimum**: 70 bars per day (90% coverage)
- **Full Days**: At least 90% of days should be complete

## Validation Functions

### Normalize Interval

```python
from DATA_PROCESSING.pipeline import normalize_interval

df_clean = normalize_interval(df, interval="5m")
```

Validates and normalizes:
- Session alignment (RTH only)
- Grid correction (5-minute boundaries)
- Missing bar detection

### Assert Bars Per Day

```python
from DATA_PROCESSING.pipeline import assert_bars_per_day

assert_bars_per_day(df_clean, interval="5m", min_full_day_frac=0.90)
```

Checks:
- Bar count per day
- Minimum coverage fraction
- Raises error if insufficient

## Data Quality Checks

### Missing Data

```python
missing_pct = df.isnull().sum() / len(df) * 100
print(missing_pct[missing_pct > 5])  # Features with >5% missing
```

### Price Anomalies

```python
# Check for price anomalies
anomalies = df[
    (df['high'] < df['low']) |
    (df['high'] < df['open']) |
    (df['high'] < df['close']) |
    (df['low'] > df['open']) |
    (df['low'] > df['close'])
]
```

### Volume Checks

```python
# Check for zero or negative volume
zero_volume = df[df['volume'] <= 0]
```

## Configuration

Data sanity validation is configured in `CONFIG/data_sanity.yaml`:

```yaml
enabled: true
mode: enforce  # or "warn" or "off"

rules:
  min_bars_per_day: 70
  min_full_day_fraction: 0.90
  max_missing_fraction: 0.05
```

## See Also

- [Data Format Spec](DATA_FORMAT_SPEC.md) - Format specification
- [Column Reference](COLUMN_REFERENCE.md) - Column documentation

