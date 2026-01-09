# Getting Started

Installation, setup, and first pipeline run.

> **📊 Testing & Development:** All testing, validation, and development work is performed using **5-minute interval data**. The software supports various data intervals, but all tests, benchmarks, and development workflows use 5-minute bars as the standard reference.

## Installation

### Prerequisites

- Python 3.11+
- **RAM**: 
  - **Small experiments**: 16-32GB minimum (laptop-friendly for limited sample/universe sizes)
  - **Production/ideal**: 128GB+ minimum, 512GB-1TB recommended for best performance
  - Requirements scale with sample count, universe size, and feature count
- GPU optional (CUDA 11.8+ if using GPU)
- Conda or pip

**Note on Universe Batching:** The pipeline supports batching large universes across multiple runs. While batching works, running with as few batches as possible is ideal for best results (enables better cross-sectional analysis and comprehensive feature selection).

### Install

```bash
git clone <repository-url>
cd trader

conda env create -f environment.yml
conda activate trader

# Verify
python --version  # Should be 3.11+
python -c "import polars; print('Polars OK')"
```

See [Installation Guide](../01_tutorials/setup/INSTALLATION.md) for details.

## System Overview

FoxML Core provides:
- **3-stage intelligent pipeline**: Target ranking → feature selection → model training
- **Dual-view evaluation**: Both cross-sectional (pooled) and symbol-specific (per-symbol) training modes
- ML pipeline infrastructure
- Multi-model training systems
- Walk-forward validation

**Single Source of Truth (SST)**: All training parameters load from YAML configs (with fallback defaults for edge cases). Same config → same results across all pipeline stages (reproducibility ensured when using proper configs).

> **Note for existing users:** SST and determinism improvements (2025-12-10) were internal changes. Your existing code and configs continue to work unchanged - no migration required. See [Deterministic Training](DETERMINISTIC_TRAINING.md) for details.

### System Flow

```
Raw Data → Features → Targets → [Target Ranking → Feature Selection → Training] → Models → Evaluation
```

1. Raw Data: Panel/time-series data (currently tested with OHLCV market data)
2. Features: 200+ engineered features (returns, volatility, technical indicators, domain-specific)
3. Targets: Prediction labels (barrier, excess returns, forward returns, domain-specific)
4. **Pipeline Stages**: 
   - **Target Ranking**: Ranks targets by predictability (evaluates in both cross-sectional and symbol-specific views)
   - **Feature Selection**: Selects optimal features per target
   - **Training**: Trains models with automatic routing decisions (cross-sectional, symbol-specific, or both)
5. Models: 17+ model types (LightGBM, XGBoost, Deep Learning)
6. Evaluation: Walk-forward validation, performance metrics

See [Architecture Overview](ARCHITECTURE_OVERVIEW.md) for details.

## Data Preparation

### Data Location

**Note:** All testing and development uses 5-minute interval data. The examples below use `interval=5m` as the standard reference.

```
data/data_labeled/interval=5m/
├── AAPL.parquet
├── MSFT.parquet
└── ...
```

### Data Format

Each parquet file contains:
- `ts`: Timestamp (UTC)
- `open`, `high`, `low`, `close`: OHLC prices
- `volume`: Trading volume (market data)

### Verify Data

```python
import polars as pl

df = pl.read_parquet("data/data_labeled/interval=5m/AAPL.parquet")
print(df.head())
print(f"Columns: {df.columns}")
print(f"Rows: {len(df)}")
```

## Data Pipeline

### Feature Engineering

```bash
python DATA_PROCESSING/features/comprehensive_builder.py \
    --config config/features.yaml \
    --output-dir DATA_PROCESSING/data/processed/
```

Output: 200+ features from raw OHLCV data.

### Target Generation

```bash
python DATA_PROCESSING/pipeline/barrier_pipeline.py \
    --input-dir data/data_labeled/interval=5m/ \
    --output-dir DATA_PROCESSING/data/labeled/
```

Output: Prediction targets (barrier labels, excess returns).

See [Data Processing Walkthrough](../01_tutorials/pipelines/DATA_PROCESSING_WALKTHROUGH.md) for details.

## Model Training

```python
from TRAINING.model_fun import LightGBMTrainer
from CONFIG.config_loader import load_model_config
import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

# Load labeled data
df = pl.read_parquet("DATA_PROCESSING/data/labeled/AAPL_labeled.parquet")

# Prepare features and targets
feature_cols = [col for col in df.columns 
                if not col.startswith('y_') and col not in ['ts', 'symbol']]
X = df[feature_cols].to_pandas()
y = df['y_will_peak'].to_pandas()

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, shuffle=False  # No shuffle for time series
)

# Load config
config = load_model_config("lightgbm", variant="conservative")

# Train
trainer = LightGBMTrainer(config)
trainer.train(X_train, y_train, X_val, y_val)

# Results
importance = trainer.get_feature_importance()
if importance is not None:
    # Get top 20 features
    top_indices = np.argsort(importance)[-20:][::-1]
    print("Top 20 Features:")
    for idx in top_indices:
        print(f"  Feature {idx}: {importance[idx]:.4f}")
```

### Results

- Feature importance: ranking of predictive features
- Validation metrics: performance on held-out data
- Model file: saved to `models/` directory

See [Model Training Guide](../01_tutorials/training/MODEL_TRAINING_GUIDE.md) for details.

## Walk-Forward Validation

Walk-forward validation trains on past data and tests on future data, rolling forward in time. Prevents look-ahead bias and provides realistic performance estimates.

```python
# Note: TRAINING.walkforward is not yet implemented
# Use PurgedTimeSeriesSplit for temporal validation instead
from scripts.utils.purged_time_series_split import PurgedTimeSeriesSplit
import polars as pl

# Load data
df = pl.read_parquet("DATA_PROCESSING/data/labeled/AAPL_labeled.parquet")

# Create walk-forward engine
engine = WalkForwardEngine(
    data=df,
    train_days=252,  # 1 year training
    test_days=63,    # 1 quarter testing
    step_days=21     # Step forward 1 month
)

# Run validation
config = load_model_config("lightgbm", variant="conservative")
results = engine.run(
    model_name="lightgbm",
    config=config,
    metrics=["sharpe", "max_drawdown", "hit_rate"]
)

# View results
print(results.summary())
```

See [Walk-Forward Validation](../01_tutorials/training/WALKFORWARD_VALIDATION.md) for details.

## Configuration

### Config Files

All models use configs from `CONFIG/model_config/`:

```yaml
# CONFIG/model_config/lightgbm.yaml
model:
  n_estimators: 1500
  learning_rate: 0.03
  max_depth: 5
```

### Variants

Each model has 3 variants:
- `conservative`: High regularization, less overfitting
- `balanced`: Default
- `aggressive`: Faster training, lower regularization

```python
config = load_model_config("lightgbm", variant="conservative")
```

### Custom Configs

Create custom variant:

```yaml
# CONFIG/model_config/lightgbm_custom.yaml
model:
  n_estimators: 2000
  learning_rate: 0.02
```

See [Config Basics](../01_tutorials/configuration/CONFIG_BASICS.md) for details.

## Common Tasks

- Add new feature: [Feature Engineering Tutorial](../01_tutorials/pipelines/FEATURE_ENGINEERING_TUTORIAL.md)
- Add new model: [Model Training Guide](../01_tutorials/training/MODEL_TRAINING_GUIDE.md)
- Optimize performance: [Performance Optimization](../03_technical/implementation/PERFORMANCE_OPTIMIZATION.md)

## Troubleshooting

- Import errors: Check environment activation
- Out of memory: Use streaming builder or reduce batch size
- GPU issues: [GPU Setup](../01_tutorials/setup/GPU_SETUP.md)
- Known issues: [Known Issues](../03_technical/fixes/KNOWN_ISSUES.md)

## Related Documentation

- [Quick Start](QUICKSTART.md)
- [Architecture Overview](ARCHITECTURE_OVERVIEW.md)
- [Documentation Index](../INDEX.md)
- [API Reference](../02_reference/api/)
- [Research Notes](../03_technical/research/)
