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

**SimpleFeatureComputer** - Basic technical indicators (50+ features)
```python
from DATA_PROCESSING.features import SimpleFeatureComputer

computer = SimpleFeatureComputer()
features = computer.compute(df_clean)
```

**ComprehensiveFeatureBuilder** - Extended feature set (200+ features)
```python
from DATA_PROCESSING.features import ComprehensiveFeatureBuilder

builder = ComprehensiveFeatureBuilder(config_path="config/features.yaml")
# Note: build_features() processes files in batch
features = builder.build_features(input_paths, output_dir, universe_config)
```

**Note**: `StreamingFeatureBuilder` is not available as a class. Use functions from `DATA_PROCESSING.features.streaming_builder` for streaming processing.

### Feature Categories

- **Price Features**: Returns, volatility, momentum
- **Volume Features**: Volume ratios, VWAP, liquidity
- **Technical Indicators**: RSI, MACD, Bollinger Bands
- **Microstructure**: Bid-ask spread, order flow imbalance

## Stage 3: Target Generation

### Barrier Targets

```python
from DATA_PROCESSING.targets import add_barrier_targets_to_dataframe

# Functions, not classes
# NOTE: interval_minutes is REQUIRED for correct horizon conversion
df_clean = add_barrier_targets_to_dataframe(
    df_clean, 
    horizon_minutes=15, 
    barrier_size=0.5,
    interval_minutes=5.0  # REQUIRED: Bar interval in minutes (for horizon conversion)
)
```

**Important**: All barrier target functions now require `interval_minutes` to correctly convert `horizon_minutes` to `horizon_bars`. Without this, targets will use incorrect lookahead windows (e.g., 60 bars instead of 12 bars for 60m horizon on 5m data).

### Excess Returns

```python
from DATA_PROCESSING.targets import compute_neutral_band, classify_excess_return

# Functions, not classes
df_clean = compute_neutral_band(df_clean, horizon="5m")
df_clean = classify_excess_return(df_clean, horizon="5m")
```

### HFT Forward Returns

```python
from DATA_PROCESSING.targets.hft_forward import add_hft_targets

# Function for batch processing
add_hft_targets(data_dir="data/raw", output_dir="data/labeled")
```

## Complete Pipeline Example

```python
from DATA_PROCESSING.pipeline import normalize_interval
from DATA_PROCESSING.features import SimpleFeatureComputer
from DATA_PROCESSING.targets import add_barrier_targets_to_dataframe
import pandas as pd

# Load raw data
symbol = "AAPL"
df = pd.read_parquet(f"data/data_labeled/interval=5m/{symbol}.parquet")

# Stage 1: Normalize
df_clean = normalize_interval(df, interval="5m")

# Stage 2: Build features
feature_computer = SimpleFeatureComputer()
features = feature_computer.compute(df_clean)

# Stage 3: Generate targets (functions, not classes)
from DATA_PROCESSING.targets import add_barrier_targets_to_dataframe
df_with_targets = add_barrier_targets_to_dataframe(
    df_clean, horizon_minutes=15, barrier_size=0.5
)

# Combine features and targets
labeled_data = pd.concat([features, df_with_targets.filter(regex='target|will_')], axis=1)

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

## Module Documentation

For complete API reference and module details, see:
- **[DATA_PROCESSING Module](DATA_PROCESSING_README.md)** - Complete module documentation with API reference
- **[Feature Engineering Tutorial](FEATURE_ENGINEERING_TUTORIAL.md)** - Advanced feature creation
- **[Column Reference](../../02_reference/data/COLUMN_REFERENCE.md)** - Complete column documentation (if available)
- **[Data Format Spec](../../02_reference/data/DATA_FORMAT_SPEC.md)** - Data format details (if available)

## Module Structure

The `DATA_PROCESSING/` module is organized as follows:

```
DATA_PROCESSING/
├── features/          # Feature engineering (200+ features)
├── targets/           # Target/label generation
├── pipeline/          # Processing pipelines
├── utils/             # Shared utilities
└── README.md          # Complete module documentation
```

See `DATA_PROCESSING/README.md` (in repository root) for detailed API documentation and usage examples.

