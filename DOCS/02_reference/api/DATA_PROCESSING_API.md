# Data Processing API Reference

Complete API reference for the DATA_PROCESSING module.

## Overview

The DATA_PROCESSING module provides data preprocessing, feature engineering, and target generation functionality. This module is used for preparing raw market data for training and analysis.

## Module Structure

```
DATA_PROCESSING/
├── features/      # Feature engineering modules
├── pipeline/      # Data pipeline orchestration
├── targets/       # Target generation modules
├── utils/         # Utility functions
└── data/          # Data storage (raw, processed, labeled)
```

## Features Module

**Location**: `DATA_PROCESSING/features/`

Feature engineering functions for creating technical indicators, statistical features, and derived features from raw market data.

### Key Functions

- Feature calculation from OHLCV data
- Technical indicator generation
- Statistical feature extraction
- Feature normalization and scaling

## Pipeline Module

**Location**: `DATA_PROCESSING/pipeline/`

Data pipeline orchestration for processing raw data through feature engineering and target generation.

### Key Functions

- Pipeline execution and orchestration
- Data validation and quality checks
- Batch processing support
- Error handling and recovery

## Targets Module

**Location**: `DATA_PROCESSING/targets/`

Target generation functions for creating prediction targets from market data.

### Key Functions

- Forward return calculation
- Target label generation
- Multi-horizon target support
- Target validation

## Utils Module

**Location**: `DATA_PROCESSING/utils/`

Utility functions for data processing operations.

### Key Functions

- Data loading and saving
- Data format conversion
- Validation utilities
- Helper functions

## Data Directory Structure

```
DATA_PROCESSING/data/
├── raw/           # Raw market data
├── processed/     # Processed data
├── features/      # Generated features
└── labeled/       # Labeled data with targets
```

## Integration with TRAINING Pipeline

The DATA_PROCESSING module is independent of the TRAINING pipeline. The TRAINING pipeline uses data from `data/data_labeled_v2/` instead.

**Note**: DATA_PROCESSING is a standalone module. The TRAINING pipeline does not depend on it.

## Configuration

Default output paths:
- Features: `RESULTS/features/` (updated from `DATA_PROCESSING/data/features/`)
- Processed data: `RESULTS/processed/`
- Labeled data: `RESULTS/labeled/`

## Related Documentation

- [Data Processing Walkthrough](../../01_tutorials/pipelines/DATA_PROCESSING_WALKTHROUGH.md) - Step-by-step guide
- [Data Processing README](../../01_tutorials/pipelines/DATA_PROCESSING_README.md) - Overview and usage
- [Data Format Spec](../data/DATA_FORMAT_SPEC.md) - Data format specifications
- [Column Reference](../data/COLUMN_REFERENCE.md) - Column documentation

