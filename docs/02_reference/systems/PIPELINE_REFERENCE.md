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

**SimpleFeatureBuilder**: 50+ basic features
```python
from DATA_PROCESSING.features import SimpleFeatureBuilder
builder = SimpleFeatureBuilder()
features = builder.build(df)
```

**ComprehensiveFeatureBuilder**: 200+ extended features
```python
from DATA_PROCESSING.features import ComprehensiveFeatureBuilder
builder = ComprehensiveFeatureBuilder()
features = builder.build(df)
```

**StreamingFeatureBuilder**: Real-time computation
```python
from DATA_PROCESSING.features import StreamingFeatureBuilder
builder = StreamingFeatureBuilder()
features = builder.build(df)
```

## Target Generation

### Barrier Targets

```python
from DATA_PROCESSING.targets import BarrierTargetBuilder

builder = BarrierTargetBuilder()
targets = builder.build(df, horizon="5m", barrier=0.001)
```

### Excess Returns

```python
from DATA_PROCESSING.targets import ExcessReturnsBuilder

builder = ExcessReturnsBuilder()
targets = builder.build(df, horizon="5m")
```

### HFT Forward Returns

```python
from DATA_PROCESSING.targets import HFTForwardReturnsBuilder

builder = HFTForwardReturnsBuilder()
targets = builder.build(df, horizon="1m")
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
- [Data Processing README](../../../DATA_PROCESSING/README.md) - Module overview

