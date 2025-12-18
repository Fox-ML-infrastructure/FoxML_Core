# DATA_PROCESSING API Reference

Complete API reference for the DATA_PROCESSING module - feature engineering, target generation, and data processing pipelines.

## Overview

The `DATA_PROCESSING` module provides tools for transforming raw market data into ML-ready features and targets. It includes feature engineering (200+ features), target generation (barriers, excess returns, HFT), and processing pipelines.

**Module Location:** `DATA_PROCESSING/`

**Related Documentation:**
- **[Module README](../../01_tutorials/pipelines/DATA_PROCESSING_README.md)** - Complete module overview and quick start
- **[Data Processing Walkthrough](../../01_tutorials/pipelines/DATA_PROCESSING_WALKTHROUGH.md)** - Step-by-step tutorial
- **[Pipeline Reference](../systems/PIPELINE_REFERENCE.md)** - Pipeline workflows

---

## Features Module

### `SimpleFeatureComputer`

Basic feature computer with 50+ technical indicators.

```python
from DATA_PROCESSING.features import SimpleFeatureComputer

computer = SimpleFeatureComputer()
features = computer.compute(df)
```

**Methods:**
- `compute(df: pd.DataFrame) -> pd.DataFrame` - Compute all simple features

**Features Generated:**
- Price features: Returns, volatility, momentum
- Volume features: Volume ratios, VWAP
- Technical indicators: RSI, MACD, Bollinger Bands

### `ComprehensiveFeatureBuilder`

Extended feature builder with 200+ features for ranking pipeline.

```python
from DATA_PROCESSING.features import ComprehensiveFeatureBuilder

builder = ComprehensiveFeatureBuilder(config_path="config/features.yaml")
features = builder.build_features(
    input_paths=["data/AAPL.parquet", "data/MSFT.parquet"],
    output_dir="data/processed/",
    universe_config={"symbols": ["AAPL", "MSFT"]}
)
```

**Methods:**
- `build_features(input_paths: List[str], output_dir: str, universe_config: Dict) -> None` - Build features for multiple files

**Configuration:**
- Loads from `config/features.yaml` (or specified path)
- Supports universe-level feature engineering

### Streaming Features

For memory-efficient processing of large datasets:

```python
from DATA_PROCESSING.features.streaming_builder import (
    compute_features_streaming,
    process_file_streaming
)

# Process single file with streaming
process_file_streaming(
    input_path="data/large_file.parquet",
    output_path="data/processed/features.parquet",
    chunk_size=100000
)

# Compute features in streaming mode
features = compute_features_streaming(df_lazy, chunk_size=50000)
```

**Functions:**
- `compute_features_streaming(df: pl.LazyFrame, chunk_size: int) -> pl.LazyFrame` - Compute features in chunks
- `process_file_streaming(input_path: str, output_path: str, chunk_size: int) -> None` - Process file with streaming

### Regime Features

Regime detection and regime-specific features:

```python
from DATA_PROCESSING.features.regime_features import (
    detect_regime,
    compute_regime_features
)

# Detect market regime
regime = detect_regime(df, window=20)

# Compute regime-specific features
regime_features = compute_regime_features(df, regime)
```

---

## Targets Module

### Barrier Targets

Generate barrier/first-passage targets (will_peak, will_valley).

```python
from DATA_PROCESSING.targets import add_barrier_targets_to_dataframe

df = add_barrier_targets_to_dataframe(
    df,
    horizon_minutes=15,
    barrier_size=0.5,
    upper_barrier=True,
    lower_barrier=True
)
```

**Function:**
- `add_barrier_targets_to_dataframe(df: pd.DataFrame, horizon_minutes: int, barrier_size: float, upper_barrier: bool = True, lower_barrier: bool = True) -> pd.DataFrame`

**Parameters:**
- `horizon_minutes`: Prediction horizon in minutes
- `barrier_size`: Barrier size as fraction (e.g., 0.5 = 50%)
- `upper_barrier`: Generate upper barrier targets (will_peak)
- `lower_barrier`: Generate lower barrier targets (will_valley)

**Returns:** DataFrame with added target columns (`y_will_peak_*`, `y_will_valley_*`)

### Excess Returns

Generate excess return targets (market-adjusted returns).

```python
from DATA_PROCESSING.targets import (
    compute_neutral_band,
    classify_excess_return
)

# Compute neutral band (market-adjusted)
df = compute_neutral_band(df, horizon="5m", market_col="SPY_return")

# Classify excess returns
df = classify_excess_return(df, horizon="5m", thresholds=[-0.01, 0.01])
```

**Functions:**
- `compute_neutral_band(df: pd.DataFrame, horizon: str, market_col: str = "SPY_return") -> pd.DataFrame` - Compute market-adjusted returns
- `classify_excess_return(df: pd.DataFrame, horizon: str, thresholds: List[float]) -> pd.DataFrame` - Classify excess returns into categories

### HFT Forward Returns

Generate short-horizon forward return targets (15m-120m).

```python
from DATA_PROCESSING.targets.hft_forward import add_hft_targets

# Batch process directory
add_hft_targets(
    data_dir="data/raw",
    output_dir="data/labeled",
    horizons=[15, 30, 60, 120]  # minutes
)
```

**Function:**
- `add_hft_targets(data_dir: str, output_dir: str, horizons: List[int]) -> None` - Add HFT forward return targets to all files in directory

**Horizons:** List of minutes (e.g., [15, 30, 60, 120])

---

## Pipeline Module

### Normalization

Normalize market data to trading session and grid boundaries.

```python
from DATA_PROCESSING.pipeline import normalize_interval, assert_bars_per_day

# Normalize to 5-minute intervals, RTH only
df_clean = normalize_interval(
    df,
    interval="5m",
    session="RTH",  # Regular Trading Hours
    timezone="America/New_York"
)

# Verify bar count per day
assert_bars_per_day(
    df_clean,
    interval="5m",
    min_full_day_frac=0.90  # Require 90% of expected bars
)
```

**Functions:**
- `normalize_interval(df: pd.DataFrame, interval: str, session: str = "RTH", timezone: str = "America/New_York") -> pd.DataFrame` - Normalize data to interval grid
- `assert_bars_per_day(df: pd.DataFrame, interval: str, min_full_day_frac: float = 0.90) -> None` - Assert minimum bar count per day

**What normalization does:**
- Aligns timestamps to grid boundaries (e.g., 9:30, 9:35, 9:40 for 5m)
- Filters to Regular Trading Hours (RTH) only
- Removes pre/post market data
- Validates data quality

### Barrier Pipeline

Smart barrier processing with resumability and parallelization.

```python
from DATA_PROCESSING.pipeline.barrier_pipeline import process_barriers

# Process barriers for multiple symbols
process_barriers(
    input_dir="data/processed",
    output_dir="data/labeled",
    horizons=[15, 30, 60],  # minutes
    barrier_sizes=[0.5, 1.0],  # fractions
    parallel=True,
    resume=True  # Resume from checkpoint
)
```

**Function:**
- `process_barriers(input_dir: str, output_dir: str, horizons: List[int], barrier_sizes: List[float], parallel: bool = True, resume: bool = True) -> None`

**Features:**
- Parallel processing across symbols
- Resumable (checkpoint-based)
- Progress tracking
- Error recovery

---

## Utils Module

### Memory Management

Monitor and manage memory usage during processing.

```python
from DATA_PROCESSING.utils import MemoryManager

mem_mgr = MemoryManager()

# Check memory before processing
mem_mgr.check_memory("Before feature engineering")

# Process data...

# Check memory after
mem_mgr.check_memory("After feature engineering")

# Get memory usage
usage = mem_mgr.get_memory_usage()
print(f"Memory: {usage['percent']}% used")
```

**Class:** `MemoryManager`

**Methods:**
- `check_memory(label: str) -> None` - Log current memory usage
- `get_memory_usage() -> Dict[str, Any]` - Get detailed memory stats
- `force_gc() -> None` - Force garbage collection

### Logging Setup

Centralized logging configuration.

```python
from DATA_PROCESSING.utils import CentralLoggingManager

log_mgr = CentralLoggingManager(config_path="config/logging_config.yaml")
log_mgr.setup_logging()

# Use in your code
import logging
logger = logging.getLogger(__name__)
logger.info("Processing started")
```

**Class:** `CentralLoggingManager`

**Methods:**
- `setup_logging(config_path: str = None) -> None` - Setup logging from config
- `get_logger(name: str) -> logging.Logger` - Get configured logger

### Schema Validation

Validate data schemas before processing.

```python
from DATA_PROCESSING.utils import SchemaValidator

validator = SchemaValidator()

# Validate required columns
validator.validate_columns(
    df,
    required=["ts", "open", "high", "low", "close", "volume"]
)

# Validate data types
validator.validate_types(
    df,
    types={"ts": "datetime64[ns]", "close": "float64"}
)
```

**Class:** `SchemaValidator`

**Methods:**
- `validate_columns(df: pd.DataFrame, required: List[str]) -> None` - Validate required columns exist
- `validate_types(df: pd.DataFrame, types: Dict[str, str]) -> None` - Validate column types

### I/O Helpers

Polars lazy loading and efficient I/O operations.

```python
from DATA_PROCESSING.utils import load_dataframe_lazy, save_dataframe

# Lazy load large file
df_lazy = load_dataframe_lazy("data/large_file.parquet")

# Process lazily
result = df_lazy.filter(pl.col("volume") > 1000).collect()

# Save efficiently
save_dataframe(result, "data/processed/filtered.parquet")
```

**Functions:**
- `load_dataframe_lazy(path: str) -> pl.LazyFrame` - Load parquet file lazily
- `save_dataframe(df: pd.DataFrame, path: str) -> None` - Save DataFrame efficiently

### Bootstrap

Exchange calendar loading and market session utilities.

```python
from DATA_PROCESSING.utils import load_exchange_calendar, get_trading_days

# Load NYSE calendar
cal = load_exchange_calendar("NYSE")

# Get trading days in range
trading_days = get_trading_days(
    cal,
    start="2024-01-01",
    end="2024-12-31"
)
```

**Functions:**
- `load_exchange_calendar(exchange: str = "NYSE") -> ExchangeCalendar` - Load exchange calendar
- `get_trading_days(cal: ExchangeCalendar, start: str, end: str) -> List[datetime]` - Get trading days in range

---

## Complete Example

```python
from DATA_PROCESSING.pipeline import normalize_interval
from DATA_PROCESSING.features import SimpleFeatureComputer
from DATA_PROCESSING.targets import add_barrier_targets_to_dataframe
from DATA_PROCESSING.utils import MemoryManager, SchemaValidator
import pandas as pd

# Initialize utilities
mem_mgr = MemoryManager()
validator = SchemaValidator()

# Load raw data
df = pd.read_parquet("data/raw/AAPL.parquet")

# Validate schema
validator.validate_columns(df, required=["ts", "open", "high", "low", "close", "volume"])

# Check memory
mem_mgr.check_memory("After load")

# Stage 1: Normalize
df_clean = normalize_interval(df, interval="5m")

# Stage 2: Build features
feature_computer = SimpleFeatureComputer()
features = feature_computer.compute(df_clean)

# Stage 3: Generate targets
df_labeled = add_barrier_targets_to_dataframe(
    df_clean,
    horizon_minutes=15,
    barrier_size=0.5
)

# Combine features and targets
final_data = pd.concat([features, df_labeled.filter(regex='y_')], axis=1)

# Save
final_data.to_parquet("data/labeled/AAPL_labeled.parquet")

mem_mgr.check_memory("After processing")
```

---

## See Also

- **[Module README](../../01_tutorials/pipelines/DATA_PROCESSING_README.md)** - Complete module documentation
- **[Data Processing Walkthrough](../../01_tutorials/pipelines/DATA_PROCESSING_WALKTHROUGH.md)** - Step-by-step tutorial
- **[Pipeline Reference](../systems/PIPELINE_REFERENCE.md)** - Pipeline workflows
- **[Feature Engineering Tutorial](../../01_tutorials/pipelines/FEATURE_ENGINEERING_TUTORIAL.md)** - Advanced feature creation
