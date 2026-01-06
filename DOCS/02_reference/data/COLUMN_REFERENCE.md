# Column Reference

Documentation of columns in labeled datasets.

## Column Categories

1. OHLCV - Base market data
2. Features - Engineered features (~200+)
3. Targets - Prediction labels
4. Metadata - Timestamps, identifiers

## 1. OHLCV (Base Data)

### Core Columns
| Column | Type | Description |
|--------|------|-------------|
| `ts` | datetime | Timestamp (UTC, 5-minute bars) |
| `open` | float64 | Opening price |
| `high` | float64 | Highest price in period |
| `low` | float64 | Lowest price in period |
| `close` | float64 | Closing price |
| `volume` | int64 | Trading volume |

### Derived Columns
| Column | Type | Description |
|--------|------|-------------|
| `datetime` | datetime | Human-readable timestamp |
| `symbol` | string | Ticker symbol |
| `interval` | string | Bar interval (e.g., "5m") |

## 2. Features

### A. Returns Features
| Column | Window | Description |
|--------|--------|-------------|
| `ret_1m` | 1 bar | 1-bar (5min) return |
| `ret_5m` | 5 bars | 25-minute return |
| `ret_15m` | 15 bars | 75-minute return |
| `ret_30m` | 30 bars | 150-minute (2.5h) return |
| `ret_60m` | 60 bars | 300-minute (5h) return |
| `returns_1d` | 1 day | Daily return |
| `returns_5d` | 5 days | Weekly return |
| `returns_20d` | 20 days | Monthly return |

### B. Volatility Features
| Column | Window | Description |
|--------|--------|-------------|
| `vol_5m` | 5 bars | 25-minute realized volatility |
| `vol_15m` | 15 bars | 75-minute volatility |
| `vol_30m` | 30 bars | 2.5-hour volatility |
| `vol_60m` | 60 bars | 5-hour volatility |
| `volatility_20d` | 20 days | Monthly volatility |
| `volatility_60d` | 60 days | Quarterly volatility |
| `atr_14` | 14 bars | Average True Range |

### C. Momentum Features
| Column | Window | Description |
|--------|--------|-------------|
| `mom_1d` | 1 day | 1-day momentum |
| `mom_3d` | 3 days | 3-day momentum |
| `mom_5d` | 5 days | 5-day momentum |
| `mom_10d` | 10 days | 10-day momentum |
| `price_momentum_5d` | 5 days | 5-day price change |
| `price_momentum_20d` | 20 days | 20-day price change |
| `price_momentum_60d` | 60 days | 60-day price change |
| `roc` | 10 bars | Rate of change |

### D. Volume Features
| Column | Description |
|--------|-------------|
| `dollar_volume` | Price × Volume |
| `turnover_5` | 5-bar turnover |
| `turnover_20` | 20-bar turnover |
| `volume_ratio` | Current / Average volume |
| `vwap` | Volume-weighted average price |

### E. Moving Averages
| Column | Type | Window |
|--------|------|--------|
| `sma_5` | Simple | 5 bars |
| `sma_10` | Simple | 10 bars |
| `sma_20` | Simple | 20 bars |
| `sma_50` | Simple | 50 bars |
| `sma_200` | Simple | 200 bars |
| `ema_5` | Exponential | 5 bars |
| `ema_10` | Exponential | 10 bars |
| `ema_20` | Exponential | 20 bars |
| `ema_50` | Exponential | 50 bars |
| `hull_ma_5` | Hull | 5 bars |
| `hull_ma_10` | Hull | 10 bars |
| `kama_5` | Kaufman Adaptive | 5 bars |
| `tema_5` | Triple Exponential | 5 bars |
| `dema_5` | Double Exponential | 5 bars |
| `vwma_5` | Volume-Weighted | 5 bars |

### F. Oscillators
| Column | Description | Range |
|--------|-------------|-------|
| `rsi_14` | Relative Strength Index (14) | 0-100 |
| `rsi_7` | RSI (7) | 0-100 |
| `rsi_21` | RSI (21) | 0-100 |
| `macd` | MACD line | Unbounded |
| `macd_signal` | MACD signal line | Unbounded |
| `macd_hist` | MACD histogram | Unbounded |
| `stoch_k` | Stochastic %K | 0-100 |
| `stoch_d` | Stochastic %D | 0-100 |
| `williams_r` | Williams %R | -100-0 |
| `cci` | Commodity Channel Index | Unbounded |

### G. Bollinger Bands
| Column | Description |
|--------|-------------|
| `bollinger_upper` | Upper band (2σ) |
| `bollinger_middle` | Middle band (SMA) |
| `bollinger_lower` | Lower band (2σ) |
| `bollinger_width` | Band width |
| `bollinger_pct` | %B (position in bands) |

### H. Trend Indicators
| Column | Description |
|--------|-------------|
| `adx` | Average Directional Index |
| `plus_di` | Plus Directional Indicator |
| `minus_di` | Minus Directional Indicator |
| `aroon_up` | Aroon Up |
| `aroon_down` | Aroon Down |
| `psar` | Parabolic SAR |

### I. Advanced Features
| Column | Description |
|--------|-------------|
| `ichimoku_conv` | Ichimoku Conversion |
| `ichimoku_base` | Ichimoku Base |
| `donchian_upper` | Donchian Upper |
| `donchian_lower` | Donchian Lower |
| `keltner_upper` | Keltner Upper |
| `keltner_lower` | Keltner Lower |

### J. Cross-Sectional Features
| Column | Description |
|--------|-------------|
| `rank_return` | Cross-sectional return rank |
| `rank_volume` | Cross-sectional volume rank |
| `rank_volatility` | Cross-sectional volatility rank |
| `zscore_return` | Standardized return |
| `zscore_volume` | Standardized volume |

### K. Interaction Features (Optional)
| Column | Description |
|--------|-------------|
| `ret_1m_x_vol_5m` | Return × Volatility |
| `rsi_x_mom` | RSI × Momentum |
| `volume_x_price` | Volume × Price change |

## 3. Targets (Labels)

### A. Barrier Targets
| Column | Type | Description | Values |
|--------|------|-------------|--------|
| `y_first_touch` | int | Which barrier hit first | -1, 0, +1 |
| `y_will_peak` | int | Will hit upper barrier | 0, 1 |
| `y_will_valley` | int | Will hit lower barrier | 0, 1 |
| `p_up` | float | Probability of up barrier | 0.0-1.0 |
| `p_down` | float | Probability of down barrier | 0.0-1.0 |
| `barrier_up_{horizon}` | float | Upper barrier level | Price |
| `barrier_down_{horizon}` | float | Lower barrier level | Price |
| `vol_at_t_{horizon}` | float | Volatility at time t | % |

**Horizons:** `15m`, `30m`, `60m`, `120m`

### B. Zigzag Targets (Swing Labels)
| Column | Type | Description |
|--------|------|-------------|
| `zigzag_peak_{horizon}` | int | Local peak detected | 0, 1 |
| `zigzag_valley_{horizon}` | int | Local valley detected | 0, 1 |
| `y_will_swing_up_{horizon}` | int | Will swing up | 0, 1 |
| `y_will_swing_down_{horizon}` | int | Will swing down | 0, 1 |

### C. MFE/MDD Targets
| Column | Type | Description |
|--------|------|-------------|
| `mfe_{horizon}` | float | Max Favorable Excursion | % |
| `mdd_{horizon}` | float | Max Drawdown | % |
| `y_mfe_above_{threshold}` | int | MFE exceeds threshold | 0, 1 |
| `y_mdd_below_{threshold}` | int | MDD exceeds threshold | 0, 1 |

### D. Excess Return Targets
| Column | Type | Description |
|--------|------|-------------|
| `excess_ret_5d` | float | 5-day beta-adjusted return | % |
| `excess_ret_10d` | float | 10-day beta-adjusted return | % |
| `y_class` | int | 3-class label | -1, 0, +1 |
| `beta_60d` | float | Rolling 60-day beta | Unbounded |

### E. HFT Forward Returns
| Column | Type | Description |
|--------|------|-------------|
| `fwd_ret_15m` | float | 15-minute forward return | % |
| `fwd_ret_30m` | float | 30-minute forward return | % |
| `fwd_ret_60m` | float | 60-minute forward return | % |
| `fwd_ret_120m` | float | 120-minute forward return | % |

## 4. Metadata

### System Columns
| Column | Type | Description |
|--------|------|-------------|
| `symbol` | string | Ticker symbol (e.g., "AAPL") |
| `interval` | string | Bar interval (e.g., "5m") |
| `session` | string | Trading session |
| `processed_at` | datetime | Processing timestamp |
| `version` | string | Pipeline version |

## Column Naming Conventions

### Prefixes
- `ret_` - Returns
- `vol_` - Volatility
- `mom_` - Momentum
- `sma_` - Simple Moving Average
- `ema_` - Exponential Moving Average
- `rsi_` - Relative Strength Index
- `y_` - Target/label
- `p_` - Probability
- `fwd_` - Forward-looking

### Suffixes
- `_5m`, `_15m`, `_30m`, `_60m`, `_120m` - Time horizons (intraday)
- `_1d`, `_5d`, `_20d`, `_60d` - Time horizons (daily)
- `_5`, `_10`, `_14`, `_20`, `_50`, `_200` - Lookback windows

### Special Naming
- Cross-sectional: Prefix with `rank_` or `zscore_`
- Interactions: Use `_x_` separator (e.g., `ret_x_vol`)
- Regime: Prefix with `regime_`
- Latent: Prefix with `latent_` (from VAE/autoencoder)

## Data Types

| Type | Description | Example Columns |
|------|-------------|-----------------|
| `datetime[ns]` | Timestamp (UTC) | `ts`, `datetime` |
| `float64` | Continuous values | `close`, `volume`, features |
| `float32` | Memory-efficient floats | Some computed features |
| `int64` | Integer values | `volume` (original) |
| `int8` | Labels/classes | `y_first_touch`, `y_class` |
| `bool` | Binary flags | `is_market_open` |
| `string` | Text | `symbol`, `interval` |

## Missing Values

### Expected NaN Behavior
- First N bars: Features with lookback windows will have NaN
- Warmup period: Models require `min_history_bars` before prediction
- Sparse features: Some cross-sectional features may be NaN for illiquid symbols

### Handling
```python
# Remove warmup period
min_bars = 120  # From config
df_clean = df.dropna().iloc[min_bars:]

# Or impute
df_filled = df.fillna(method='ffill').fillna(0)
```

## Column Validation

### Check Required Columns
```python
from DATA_PROCESSING.utils import SchemaExpectations, validate_schema

required = {
    'ts', 'open', 'high', 'low', 'close', 'volume',
    'ret_1m', 'vol_5m', 'rsi_14',  # Key features
    'y_will_peak', 'y_will_valley'  # Key targets
}

schema = SchemaExpectations(required=required)
warnings = validate_schema(df.columns, schema)
```

### Verify Data Quality
```python
# Check for infinities
assert not df.select_dtypes(include=[np.number]).isin([np.inf, -np.inf]).any().any()

# Check value ranges
assert (df['rsi_14'] >= 0).all() and (df['rsi_14'] <= 100).all()
assert (df['volume'] >= 0).all()
```

## Feature Selection

### By Category
```python
# Get all return features
return_cols = [col for col in df.columns if col.startswith('ret_')]

# Get all volume features
volume_cols = [col for col in df.columns if 'volume' in col.lower() or 'turnover' in col]

# Get all target columns
target_cols = [col for col in df.columns if col.startswith('y_')]
```

### By Importance
```python
from TRAINING.model_fun import LightGBMTrainer

trainer = LightGBMTrainer()
trainer.train(X_train, y_train)
importance = trainer.get_feature_importance()

# Top 50 features
top_features = importance.head(50).index.tolist()
```

## Related Documentation

- `DATA_PROCESSING/` - Feature engineering code
- [Data Processing Walkthrough](../../01_tutorials/pipelines/DATA_PROCESSING_WALKTHROUGH.md) - Column generation and data pipeline
- [Model Training Guide](../../01_tutorials/training/MODEL_TRAINING_GUIDE.md) - Using columns in training
