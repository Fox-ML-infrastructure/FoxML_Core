# Data Processing Module Overview

Overview and usage guide for the DATA_PROCESSING module.

## Overview

The DATA_PROCESSING module provides data preprocessing, feature engineering, and target generation functionality for preparing raw market data for machine learning training.

## Purpose

The DATA_PROCESSING module:
- Processes raw market data (OHLCV)
- Generates technical indicators and features
- Creates prediction targets
- Validates data quality
- Outputs processed data for training

## Module Structure

```
DATA_PROCESSING/
├── features/      # Feature engineering modules
│   └── __pycache__/
├── pipeline/      # Data pipeline orchestration
│   └── __pycache__/
├── targets/       # Target generation modules
│   └── __pycache__/
├── utils/         # Utility functions
│   └── __pycache__/
└── data/          # Data storage
    ├── raw/       # Raw market data
    ├── processed/ # Processed data
    ├── features/  # Generated features
    └── labeled/   # Labeled data with targets
```

## Key Components

### Features Module

Feature engineering functions for creating:
- Technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Statistical features (rolling means, std, correlations)
- Derived features (momentum, volatility, etc.)

### Pipeline Module

Data pipeline orchestration:
- End-to-end data processing workflows
- Batch processing support
- Error handling and validation
- Progress tracking

### Targets Module

Target generation functions:
- Forward return calculation
- Multi-horizon targets (5m, 10m, 15m, 30m, 60m, etc.)
- Target validation and quality checks

### Utils Module

Utility functions:
- Data loading and saving
- Format conversion
- Validation utilities

## Output Structure

Processed data is saved to:
- **Features**: `RESULTS/features/` (default)
- **Processed Data**: `RESULTS/processed/`
- **Labeled Data**: `RESULTS/labeled/`

## Integration with TRAINING Pipeline

**Important**: The DATA_PROCESSING module is **independent** of the TRAINING pipeline.

The TRAINING pipeline:
- Uses data from `data/data_labeled_v2/` directory
- Does not import from DATA_PROCESSING modules
- Has its own feature engineering pipeline

DATA_PROCESSING is a standalone module that can be used separately for data preparation tasks.

## Usage

### Standalone Usage

The DATA_PROCESSING module can be used independently:

```python
from DATA_PROCESSING.pipeline import run_pipeline
from DATA_PROCESSING.features import generate_features

# Run full pipeline
run_pipeline(
    input_dir="data/raw",
    output_dir="RESULTS/processed"
)

# Generate features only
generate_features(
    data_dir="data/processed",
    output_dir="RESULTS/features"
)
```

### Configuration

Default output paths can be configured:
- `RESULTS/features/` - Feature output (updated from `DATA_PROCESSING/data/features/`)
- `RESULTS/processed/` - Processed data output
- `RESULTS/labeled/` - Labeled data output

## Related Documentation

- [Data Processing API Reference](../../02_reference/api/DATA_PROCESSING_API.md) - Complete API documentation
- [Data Processing Walkthrough](DATA_PROCESSING_WALKTHROUGH.md) - Step-by-step tutorial
- [Data Format Spec](../../02_reference/data/DATA_FORMAT_SPEC.md) - Data format specifications
- [Column Reference](../../02_reference/data/COLUMN_REFERENCE.md) - Column documentation

## Status

**Note**: The DATA_PROCESSING module is a standalone module. The core TRAINING pipeline does not depend on it and uses data from `data/data_labeled_v2/` instead.

