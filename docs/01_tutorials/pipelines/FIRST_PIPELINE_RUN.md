# First Pipeline Run

Run your first data processing pipeline from raw data to labeled dataset.

## Prerequisites

- Raw market data in `data/data_labeled/interval=5m/`
- Python environment configured (see [Environment Setup](../setup/ENVIRONMENT_SETUP.md))
- Dependencies installed

## Quick Start

### 1. Prepare Raw Data

Ensure your data is in the correct format:

```
data/data_labeled/interval=5m/
├── AAPL.parquet
├── MSFT.parquet
└── ...
```

Each file should contain:
- Columns: `ts`, `open`, `high`, `low`, `close`, `volume`
- Timezone: UTC timestamps
- Coverage: NYSE Regular Trading Hours (RTH) only

### 2. Run Normalization

```python
from DATA_PROCESSING.pipeline import normalize_interval
import pandas as pd

# Load raw data
df = pd.read_parquet("data/data_labeled/interval=5m/AAPL.parquet")

# Normalize to RTH and grid
df_clean = normalize_interval(df, interval="5m")

# Save normalized data
df_clean.to_parquet("data/data_labeled/interval=5m/AAPL_normalized.parquet")
```

### 3. Build Features

```python
from DATA_PROCESSING.features import SimpleFeatureBuilder

builder = SimpleFeatureBuilder()
features = builder.build(df_clean)

# Save features
features.to_parquet("data/features/AAPL_features.parquet")
```

### 4. Generate Targets

```python
from DATA_PROCESSING.targets import BarrierTargetBuilder

target_builder = BarrierTargetBuilder()
targets = target_builder.build(df_clean, horizon="5m")

# Save targets
targets.to_parquet("data/targets/AAPL_targets.parquet")
```

### 5. Combine Features and Targets

```python
# Combine for training
labeled_data = pd.concat([features, targets], axis=1)
labeled_data.to_parquet("data/labeled/AAPL_labeled.parquet")
```

## Complete Example

```python
from DATA_PROCESSING.pipeline import normalize_interval
from DATA_PROCESSING.features import SimpleFeatureBuilder
from DATA_PROCESSING.targets import BarrierTargetBuilder
import pandas as pd

# Load and normalize
df = pd.read_parquet("data/data_labeled/interval=5m/AAPL.parquet")
df_clean = normalize_interval(df, interval="5m")

# Build features
feature_builder = SimpleFeatureBuilder()
features = feature_builder.build(df_clean)

# Generate targets
target_builder = BarrierTargetBuilder()
targets = target_builder.build(df_clean, horizon="5m")

# Combine
labeled_data = pd.concat([features, targets], axis=1)
labeled_data.to_parquet("data/labeled/AAPL_labeled.parquet")

print(f"Created labeled dataset with {len(labeled_data)} rows and {len(labeled_data.columns)} columns")
```

## Verification

Check your labeled dataset:

```python
labeled = pd.read_parquet("data/labeled/AAPL_labeled.parquet")
print(labeled.info())
print(labeled.head())
```

## Next Steps

- [Data Processing Walkthrough](DATA_PROCESSING_WALKTHROUGH.md) - Detailed pipeline guide
- [Feature Engineering Tutorial](FEATURE_ENGINEERING_TUTORIAL.md) - Advanced features
- [Model Training Guide](../training/MODEL_TRAINING_GUIDE.md) - Train models on labeled data

