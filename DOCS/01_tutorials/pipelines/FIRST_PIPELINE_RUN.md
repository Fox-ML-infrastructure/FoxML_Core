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
from DATA_PROCESSING.features import SimpleFeatureComputer

computer = SimpleFeatureComputer()
features = computer.compute(df_clean)

# Save features
features.to_parquet("data/features/AAPL_features.parquet")
```

### 4. Generate Targets

```python
from DATA_PROCESSING.targets import add_barrier_targets_to_dataframe

# Functions, not classes
# NOTE: interval_minutes is REQUIRED for correct horizon conversion
df_with_targets = add_barrier_targets_to_dataframe(
    df_clean, 
    horizon_minutes=15, 
    barrier_size=0.5,
    interval_minutes=5.0  # REQUIRED: Bar interval in minutes
)

# Save targets (targets are added as columns to the dataframe)
df_with_targets.to_parquet("data/targets/AAPL_targets.parquet")
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
from DATA_PROCESSING.features import SimpleFeatureComputer
from DATA_PROCESSING.targets import add_barrier_targets_to_dataframe
import pandas as pd

# Load and normalize
df = pd.read_parquet("data/data_labeled/interval=5m/AAPL.parquet")
df_clean = normalize_interval(df, interval="5m")

# Build features
feature_computer = SimpleFeatureComputer()
features = feature_computer.compute(df_clean)

# Generate targets (functions add columns to dataframe)
# NOTE: interval_minutes is REQUIRED for correct horizon conversion
df_with_targets = add_barrier_targets_to_dataframe(
    df_clean, 
    horizon_minutes=15, 
    barrier_size=0.5,
    interval_minutes=5.0  # REQUIRED: Bar interval in minutes
)

# Combine features and targets
labeled_data = pd.concat([features, df_with_targets.filter(regex='target|will_')], axis=1)
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

