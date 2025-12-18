# Pipeline Reference

Complete reference for data processing pipelines.

## Pipeline Overview

```
Raw Market Data
    ↓
[1] Normalization → Session-aligned, grid-corrected
    ↓
[2] Feature Engineering → 200+ technical features
    ↓
[3] Target Generation → Labels (barrier, excess returns)
    ↓
Labeled Dataset
```

## Normalization

### Normalize Interval

```python
from DATA_PROCESSING.pipeline import normalize_interval

df_clean = normalize_interval(df, interval="5m")
```

**What it does:**
- Aligns to trading session (RTH)
- Corrects to grid boundaries (5-minute intervals)
- Removes pre/post market data
- Validates bar count

## Feature Engineering

### Feature Builders

**SimpleFeatureComputer**: 50+ basic features
```python
from DATA_PROCESSING.features import SimpleFeatureComputer

computer = SimpleFeatureComputer()
features = computer.compute(df)
```

**ComprehensiveFeatureBuilder**: 200+ extended features
```python
from DATA_PROCESSING.features import ComprehensiveFeatureBuilder

builder = ComprehensiveFeatureBuilder(config_path="config/features.yaml")
# Note: build_features() processes files, not single DataFrames
features = builder.build_features(input_paths, output_dir, universe_config)
```

**Note**: `StreamingFeatureBuilder` is not available as a class. Use `DATA_PROCESSING.features.streaming_builder` functions for streaming processing.

## Target Generation

### Barrier Targets

```python
from DATA_PROCESSING.targets import add_barrier_targets_to_dataframe

# Functions, not classes
df = add_barrier_targets_to_dataframe(
    df, horizon_minutes=15, barrier_size=0.5
)
```

### Excess Returns

```python
from DATA_PROCESSING.targets import compute_neutral_band, classify_excess_return

# Functions, not classes
df = compute_neutral_band(df, horizon="5m")
df = classify_excess_return(df, horizon="5m")
```

### HFT Forward Returns

```python
from DATA_PROCESSING.targets.hft_forward import add_hft_targets

# Function for batch processing
add_hft_targets(data_dir="data/raw", output_dir="data/labeled")
```

## Batch Processing

Process multiple symbols:

```python
from pathlib import Path

data_dir = Path("data/data_labeled/interval=5m")
for parquet_file in data_dir.glob("*.parquet"):
    symbol = parquet_file.stem
    # Process...
```

## See Also

- [Data Processing Walkthrough](../../01_tutorials/pipelines/DATA_PROCESSING_WALKTHROUGH.md) - Detailed guide
- [First Pipeline Run](../../01_tutorials/pipelines/FIRST_PIPELINE_RUN.md) - Quick start
- [Data Processing README](../../01_tutorials/pipelines/DATA_PROCESSING_README.md) - Module overview

