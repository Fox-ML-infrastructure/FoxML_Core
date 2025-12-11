# DATA_PROCESSING Module

Data processing pipeline for market data feature engineering and target generation.

## Directory Structure

```
DATA_PROCESSING/
├── features/              # Feature engineering modules
│   ├── comprehensive_builder.py   # 200+ features for ranking
│   ├── simple_features.py         # Basic feature computation
│   └── streaming_builder.py       # Memory-efficient Polars streaming
│
├── targets/               # Target/label generation
│   ├── barrier.py                 # Barrier/first-passage labels
│   ├── excess_returns.py          # Excess return classification
│   └── hft_forward.py             # HFT forward return targets
│
├── pipeline/              # Processing pipelines
│   ├── barrier_pipeline.py        # Smart barrier processing
│   └── normalize.py               # Session normalization
│
├── utils/                 # Shared utilities
│   ├── memory_manager.py          # Memory management
│   ├── logging_setup.py           # Centralized logging
│   ├── schema_validator.py        # Schema validation
│   ├── io_helpers.py              # I/O utilities
│   └── bootstrap.py               # Calendar loading
│
└── data/                  # Data storage
    ├── raw/               # Raw unprocessed data
    ├── processed/         # Processed features
    └── labeled/           # Data with targets
```

## Quick Start

### Feature Engineering
```python
from DATA_PROCESSING.features import ComprehensiveFeatureBuilder, SimpleFeatureComputer

# Comprehensive features (200+)
builder = ComprehensiveFeatureBuilder(config_path="config/features.yaml")
features = builder.build_features(input_paths, output_dir, universe_config)

# Simple features
computer = SimpleFeatureComputer()
features = computer.get_all_features()
```

### Target Generation
```python
from DATA_PROCESSING.targets import compute_barrier_targets, future_excess_return

# Barrier targets (will_peak, will_valley)
targets = compute_barrier_targets(
    prices=df['close'],
    horizon_minutes=15,
    barrier_size=0.5
)

# Excess return targets
excess_ret = future_excess_return(asset_ret, mkt_ret, H=5)
```

### Processing Pipeline
```python
from DATA_PROCESSING.pipeline import normalize_interval

# Normalize session data
df_normalized = normalize_interval(df, interval="5m")
```

### Utilities
```python
from DATA_PROCESSING.utils import MemoryManager, CentralLoggingManager

# Memory management
mem_mgr = MemoryManager()
mem_mgr.check_memory("Before processing")

# Logging setup
log_mgr = CentralLoggingManager(config_path="config/logging_config.yaml")
```

## Module Purposes

### features/
Transform raw market data into engineered features for ML models.

Key features:
- Technical indicators (200+)
- Momentum and volatility features
- Cross-sectional rankings
- Memory-efficient streaming computation

### targets/
Generate prediction targets and labels for model training.

Target types:
- Barrier targets: Will price hit upper/lower barrier first?
- Excess returns: Future returns adjusted for market beta
- HFT targets: Short-horizon forward returns (15m-120m)

### pipeline/
End-to-end processing workflows for production data.

Key pipelines:
- Session normalization (RTH only, grid-aligned)
- Smart barrier processing (resumable, parallel)

### utils/
Common utilities shared across all processing modules.

Components:
- Memory management and monitoring
- Centralized logging configuration
- Schema validation
- I/O helpers (Polars lazy loading)
- Exchange calendar bootstrap

## Data Flow

```
Raw Data (data/raw/)
    ↓
[features/] → Engineer features
    ↓
Processed Features (data/processed/)
    ↓
[targets/] → Generate labels
    ↓
Labeled Data (data/labeled/)
    ↓
Ready for model training (TRAINING/)
```

## Best Practices

1. Memory Management: Use `MemoryManager` for large datasets
2. Streaming: Use `streaming_builder.py` for datasets that don't fit in memory
3. Schema Validation: Always validate schema before processing
4. Logging: Use `CentralLoggingManager` for consistent logging
5. Normalization: Always normalize session data before feature engineering

## Dependencies

- `polars` - Fast DataFrame operations
- `pandas` - Legacy compatibility
- `numpy` - Numerical computations
- `exchange_calendars` - Market calendar utilities
- `yaml` - Configuration loading
- `psutil` - Memory monitoring

## Related Documentation

- `TRAINING/` - Model training using processed features
- `CONFIG/` - Centralized configuration files
- `INFORMATION/` - Project documentation
