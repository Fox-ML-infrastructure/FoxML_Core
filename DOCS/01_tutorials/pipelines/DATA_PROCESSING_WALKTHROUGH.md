# Data Processing Walkthrough

Step-by-step guide to using the DATA_PROCESSING module.

## Overview

This walkthrough guides you through using the DATA_PROCESSING module to process raw market data, generate features, and create labeled datasets for machine learning.

## Prerequisites

- Raw market data in OHLCV format
- Python 3.8+
- Required packages installed
- Output directory configured

## Step 1: Prepare Raw Data

Place your raw market data in the input directory:

```
data/raw/
├── AAPL.csv
├── MSFT.csv
├── GOOGL.csv
└── ...
```

Data format should be CSV with columns:
- `timestamp` or `date`
- `open`, `high`, `low`, `close`
- `volume`
- `symbol` (optional, if multiple symbols in one file)

## Step 2: Run Data Processing Pipeline

### Option A: Full Pipeline

Run the complete data processing pipeline:

```python
from DATA_PROCESSING.pipeline import run_pipeline

run_pipeline(
    input_dir="data/raw",
    output_dir="RESULTS/processed",
    symbols=["AAPL", "MSFT", "GOOGL"]
)
```

### Option B: Step-by-Step

Process data in steps:

```python
from DATA_PROCESSING.pipeline import load_raw_data, process_data
from DATA_PROCESSING.features import generate_features
from DATA_PROCESSING.targets import generate_targets

# Step 1: Load raw data
raw_data = load_raw_data("data/raw")

# Step 2: Process data
processed_data = process_data(raw_data)

# Step 3: Generate features
features = generate_features(processed_data)

# Step 4: Generate targets
labeled_data = generate_targets(features)
```

## Step 3: Generate Features

Generate technical indicators and features:

```python
from DATA_PROCESSING.features import generate_features

features = generate_features(
    data_dir="RESULTS/processed",
    output_dir="RESULTS/features",
    feature_types=["technical", "statistical", "momentum"]
)
```

Available feature types:
- `technical` - Technical indicators (RSI, MACD, etc.)
- `statistical` - Statistical features (rolling means, std, etc.)
- `momentum` - Momentum and trend features
- `volatility` - Volatility features

## Step 4: Generate Targets

Create prediction targets:

```python
from DATA_PROCESSING.targets import generate_targets

targets = generate_targets(
    data_dir="RESULTS/features",
    output_dir="RESULTS/labeled",
    horizons=[5, 10, 15, 30, 60]  # minutes
)
```

## Step 5: Validate Output

Validate processed data:

```python
from DATA_PROCESSING.utils import validate_data

validation_results = validate_data(
    data_dir="RESULTS/labeled",
    check_missing=True,
    check_duplicates=True,
    check_targets=True
)

if validation_results["valid"]:
    print("✅ Data validation passed")
else:
    print(f"❌ Validation errors: {validation_results['errors']}")
```

## Output Structure

After processing, you'll have:

```
RESULTS/
├── processed/      # Processed OHLCV data
├── features/       # Generated features
└── labeled/       # Labeled data with targets
```

## Integration with TRAINING Pipeline

**Note**: The TRAINING pipeline uses data from `data/data_labeled_v2/` and does not depend on DATA_PROCESSING.

If you want to use DATA_PROCESSING output with TRAINING:
1. Copy processed data to `data/data_labeled_v2/`
2. Ensure data format matches TRAINING requirements
3. See [Data Format Spec](../../02_reference/data/DATA_FORMAT_SPEC.md) for format details

## Troubleshooting

### Common Issues

1. **Missing columns**: Ensure raw data has required OHLCV columns
2. **Date format**: Use standard date formats (YYYY-MM-DD or ISO format)
3. **Missing values**: Handle missing values before processing
4. **Memory issues**: Process data in batches for large datasets

### Validation Errors

If validation fails:
- Check data format matches specifications
- Verify no missing required columns
- Check for duplicate timestamps
- Ensure targets are valid (no NaN, no infinite values)

## Related Documentation

- [Data Processing README](DATA_PROCESSING_README.md) - Module overview
- [Data Processing API Reference](../../02_reference/api/DATA_PROCESSING_API.md) - Complete API docs
- [Data Format Spec](../../02_reference/data/DATA_FORMAT_SPEC.md) - Data format requirements
- [Column Reference](../../02_reference/data/COLUMN_REFERENCE.md) - Column documentation

