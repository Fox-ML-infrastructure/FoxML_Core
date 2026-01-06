# Data Format Specification

Complete specification for data formats used in Fox-v1-infra.

## Raw Market Data Format

### File Location

```
data/data_labeled/interval=5m/
├── AAPL.parquet
├── MSFT.parquet
└── ...
```

### Required Columns

| Column | Type | Description | Required |
|--------|------|-------------|----------|
| `ts` | datetime64[ns] | Timestamp (UTC) | Yes |
| `open` | float64 | Opening price | Yes |
| `high` | float64 | Highest price | Yes |
| `low` | float64 | Lowest price | Yes |
| `close` | float64 | Closing price | Yes |
| `volume` | int64 | Trading volume | Yes |

### Data Requirements

- **Timeframe**: 5-minute bars
- **Timezone**: UTC timestamps
- **Coverage**: NYSE Regular Trading Hours (RTH) only
- **Index**: DatetimeIndex (sorted, no duplicates)
- **Missing Data**: NaN values allowed but should be minimal

### Example

```python
import pandas as pd

df = pd.read_parquet("data/data_labeled/interval=5m/AAPL.parquet")
print(df.head())
#                    ts    open    high     low   close  volume
# 2020-01-02 09:30:00  75.00  75.10  74.95  75.05  1000000
# 2020-01-02 09:35:00  75.05  75.20  75.00  75.15  1200000
```

## Labeled Dataset Format

### Structure

After processing, labeled datasets contain:

1. **OHLCV columns**: Base market data
2. **Feature columns**: 200+ engineered features
3. **Target columns**: Prediction labels (prefixed with `target_` or `y_`)

### Target Column Naming

- `target_fwd_ret_5m`: Forward return (5-minute horizon)
- `target_fwd_ret_15m`: Forward return (15-minute horizon)
- `y_will_peak_60m_0.8`: Barrier target (60m horizon, 0.8% barrier)

### Example

```python
labeled = pd.read_parquet("data/labeled/AAPL_labeled.parquet")
print(labeled.columns[:10])
# ['ts', 'open', 'high', 'low', 'close', 'volume', 
#  'ret_1m', 'ret_5m', 'vol_5m', 'target_fwd_ret_5m']
```

## Feature Format

### Feature Categories

- **Returns**: `ret_1m`, `ret_5m`, `ret_15m`, etc.
- **Volatility**: `vol_5m`, `vol_15m`, `volatility_20d`, etc.
- **Momentum**: `mom_1d`, `mom_5d`, `price_momentum_20d`, etc.
- **Volume**: `volume_ratio`, `vwap`, `turnover_5`, etc.
- **Technical Indicators**: `rsi_14`, `macd`, `bollinger_upper`, etc.

### Data Types

- **Numeric**: float64 (most features)
- **Boolean**: bool (regime indicators)
- **Categorical**: object/string (symbol, interval)

## See Also

- [Column Reference](COLUMN_REFERENCE.md) - Complete column documentation
- [Data Sanity Rules](DATA_SANITY_RULES.md) - Validation rules

