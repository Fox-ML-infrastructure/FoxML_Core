# Data Processing Walkthrough

Complete guide to the data processing pipeline from raw market data to ML-ready labeled datasets.

## Pipeline Overview

```
Raw Market Data
    ↓
[1] Normalization → Session-aligned, grid-corrected data
    ↓
[2] Feature Engineering → 200+ technical features
    ↓
[3] Target Generation → Labels (barrier, excess returns, HFT)
    ↓
Labeled Dataset → Ready for model training
```

## Stage 1: Raw Data Acquisition

### Data Location

```
data/
└── data_labeled/
    └── interval=5m/
        ├── AAPL.parquet
        ├── MSFT.parquet
        └── ...
```

### Expected Format

- **Timeframe**: 5-minute bars
- **Columns**: `ts`, `open`, `high`, `low`, `close`, `volume`
- **Timezone**: UTC timestamps
- **Coverage**: NYSE Regular Trading Hours (RTH) only

### Quality Checks

```python
from DATA_PROCESSING.pipeline import normalize_interval, assert_bars_per_day

# Normalize to RTH and grid
df_clean = normalize_interval(df, interval="5m")

# Verify bar count per day
assert_bars_per_day(df_clean, interval="5m", min_full_day_frac=0.90)
```

## Stage 2: Feature Engineering

### Available Feature Builders

**SimpleFeatureBuilder** - Basic technical indicators (50+ features)
```python
from DATA_PROCESSING.features import SimpleFeatureBuilder

builder = SimpleFeatureBuilder()
features = builder.build(df_clean)
```

**ComprehensiveFeatureBuilder** - Extended feature set (200+ features)
```python
from DATA_PROCESSING.features import ComprehensiveFeatureBuilder

builder = ComprehensiveFeatureBuilder()
features = builder.build(df_clean)
```

**StreamingFeatureBuilder** - Real-time feature computation
```python
from DATA_PROCESSING.features import StreamingFeatureBuilder

builder = StreamingFeatureBuilder()
features = builder.build(df_clean)
```

### Feature Categories

- **Price Features**: Returns, volatility, momentum
- **Volume Features**: Volume ratios, VWAP, liquidity
- **Technical Indicators**: RSI, MACD, Bollinger Bands
- **Microstructure**: Bid-ask spread, order flow imbalance

## Stage 3: Target Generation

### Barrier Targets

```python
from DATA_PROCESSING.targets import BarrierTargetBuilder

builder = BarrierTargetBuilder()
targets = builder.build(df_clean, horizon="5m", barrier=0.001)
```

### Excess Returns

```python
from DATA_PROCESSING.targets import ExcessReturnsBuilder

builder = ExcessReturnsBuilder()
targets = builder.build(df_clean, horizon="5m")
```

### HFT Forward Returns

```python
from DATA_PROCESSING.targets import HFTForwardReturnsBuilder

builder = HFTForwardReturnsBuilder()
targets = builder.build(df_clean, horizon="1m")
```

## Complete Pipeline Example

```python
from DATA_PROCESSING.pipeline import normalize_interval
from DATA_PROCESSING.features import ComprehensiveFeatureBuilder
from DATA_PROCESSING.targets import BarrierTargetBuilder
import pandas as pd

# Load raw data
symbol = "AAPL"
df = pd.read_parquet(f"data/data_labeled/interval=5m/{symbol}.parquet")

# Stage 1: Normalize
df_clean = normalize_interval(df, interval="5m")

# Stage 2: Build features
feature_builder = ComprehensiveFeatureBuilder()
features = feature_builder.build(df_clean)

# Stage 3: Generate targets
target_builder = BarrierTargetBuilder()
targets = target_builder.build(df_clean, horizon="5m", barrier=0.001)

# Combine
labeled_data = pd.concat([features, targets], axis=1)

# Save
labeled_data.to_parquet(f"data/labeled/{symbol}_labeled.parquet")
```

## Batch Processing

Process multiple symbols:

```python
from pathlib import Path

data_dir = Path("data/data_labeled/interval=5m")
for parquet_file in data_dir.glob("*.parquet"):
    symbol = parquet_file.stem
    # Process each symbol...
```

## Next Steps

- [Feature Engineering Tutorial](FEATURE_ENGINEERING_TUTORIAL.md) - Advanced feature creation
- [Column Reference](../../02_reference/data/COLUMN_REFERENCE.md) - Complete column documentation
- [Data Format Spec](../../02_reference/data/DATA_FORMAT_SPEC.md) - Data format details

