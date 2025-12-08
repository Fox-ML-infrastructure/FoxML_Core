# Architecture Overview

System architecture for FoxML Core.

## System Purpose

FoxML Core is a research infrastructure system for ML pipelines, quantitative workflows, and reproducible experiments. Provides:

- Scalable ML workflow design
- Leakage-safe research architecture
- High-throughput data processing
- Multi-model training systems
- Hybrid C++/Python infrastructure
- HPC-compatible orchestration patterns

Infrastructure, not a trading bot. Provides architecture, not alpha.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Acquisition                         │
│              (Raw OHLCV → Normalized Data)                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  DATA_PROCESSING                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Features   │  │   Targets    │  │   Pipeline   │     │
│  │  Engineering │→ │  Generation  │→ │  Workflows   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  17+ Models  │  │  Walk-Forward │  │  Strategies  │     │
│  │   (LightGBM, │  │  Validation   │  │ (Single/Multi)│     │
│  │   XGBoost,   │  │               │  │              │     │
│  │   Deep Lrn)  │  │               │  │              │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Trading Integrations (Optional)                │
│  ┌──────────────┐              ┌──────────────┐           │
│  │  IBKR System │              │ Alpaca System │           │
│  │  (Paper/Prod)│              │  (Paper Only) │           │
│  └──────────────┘              └──────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. CONFIG (Configuration Management)

Centralized, version-controlled configuration.

Features:
- 17 model configs with 3 variants each (conservative/balanced/aggressive)
- Runtime overrides
- Environment variable support
- YAML-based

Location: `CONFIG/`

Usage:
```python
from CONFIG.config_loader import load_model_config
config = load_model_config("lightgbm", variant="conservative")
```

### 2. DATA_PROCESSING (ETL Pipeline)

Transforms raw data into ML-ready features and targets.

Modules:
- `features/`: Feature engineering (simple, comprehensive, streaming)
- `targets/`: Target generation (barrier, excess returns, HFT)
- `pipeline/`: End-to-end workflows
- `utils/`: Memory management, logging, validation

Output: Labeled datasets with 200+ features and multiple target types.

### 3. TRAINING (Model Training)

Trains and validates ML models.

**Intelligent Training Pipeline:**
- Automatic target ranking (multi-model consensus)
- Automatic feature selection (per target)
- Unified workflow: ranking → selection → training
- Caching for faster iterative development
- **Leakage detection & auto-fix**: Automatic detection and remediation of data leakage
- **Config backup system**: Automatic backups of config files before auto-fix modifications
- **Timestamped outputs**: Output directories automatically timestamped (format: `YYYYMMDD_HHMMSS`) for run tracking

**Available Models:**
- Core: LightGBM, XGBoost, Ensemble, MultiTask
- Deep Learning: MLP, Transformer, LSTM, CNN1D
- Feature Engineering: VAE, GAN, GMMRegime
- Probabilistic: NGBoost, QuantileLightGBM
- Advanced: ChangePoint, FTRL, RewardBased, MetaLearning

**Training Strategies:**
- Single-task (one model per target)
- Multi-task (shared model for correlated targets)
- Cascade (sequential dependencies)

**Validation:** Walk-forward analysis for realistic performance estimation.

### 4. Trading Integrations (Optional)

**IBKR System**:
- Multi-horizon trading (5m, 10m, 15m, 30m, 60m)
- Safety guards (margin, short-sale, rate limiting)
- Cost-aware decision making
- C++ inference engine for performance

**Alpaca System**:
- Paper trading only
- Simplified integration
- Research-focused

## Data Flow

### Stage 1: Raw Data
- Input: OHLCV data (5-minute bars)
- Format: Parquet files per symbol
- Location: `data/data_labeled/interval=5m/`

### Stage 2: Feature Engineering
- Input: Normalized OHLCV
- Output: 200+ engineered features
- Features: Returns, volatility, momentum, technical indicators
- Location: `DATA_PROCESSING/data/processed/`

### Stage 3: Target Generation
- Input: Processed features
- Output: Prediction labels
- Targets: Barrier labels, excess returns, forward returns
- Location: `DATA_PROCESSING/data/labeled/`

### Stage 4: Model Training
- Input: Labeled datasets
- Process: Intelligent training pipeline (ranking → selection → training)
- **Leakage Detection**: Pre-training leak scan + auto-fixer with config backups
- Output: Trained models + rankings + feature selections
- Models: Saved to `{output_dir}_YYYYMMDD_HHMMSS/training_results/`
- Rankings: Saved to `{output_dir}_YYYYMMDD_HHMMSS/target_rankings/`
- Feature Selections: Saved to `{output_dir}_YYYYMMDD_HHMMSS/feature_selections/`
- Config Backups: Saved to `CONFIG/backups/{target}/{timestamp}/` (when auto-fix runs)
- Configs: Versioned in `CONFIG/model_config/` (see [Configuration Reference](../02_reference/configuration/README.md))

### Stage 5: Evaluation
- Input: Trained models
- Output: Performance metrics
- Metrics: Sharpe, drawdown, hit rate, profit factor
- Location: `{output_dir}_YYYYMMDD_HHMMSS/training_results/`

## Design Principles

### 1. Configuration-Driven
- No hardcoded parameters
- All settings in YAML configs
- Easy experimentation and reproducibility

### 2. Leakage-Safe
- Strict temporal validation
- Walk-forward analysis
- No future information leakage
- **Pre-training leak scan**: Detects near-copy features before model training
- **Auto-fixer with backups**: Automatically detects and fixes leakage, with config backups for rollback
- **Config-driven safety**: All leakage thresholds configurable via `safety_config.yaml`

### 3. Modular Architecture
- Independent components
- Clear interfaces
- Easy to extend

### 4. Performance-Optimized
- C++ inference for trading
- Streaming builders for large datasets
- GPU support for training

### 5. Research-Focused
- Reproducible experiments
- Comprehensive logging
- Multiple model types

## Technology Stack

Languages:
- Python 3.11+ (primary)
- C++ (inference engine)

Key Libraries:
- Polars (data processing)
- LightGBM/XGBoost (tabular models)
- PyTorch (deep learning)
- scikit-learn (utilities)

Infrastructure:
- YAML configs
- Parquet data format
- JSON logging

## Directory Structure

```
trader/
├── CONFIG/              # Centralized configurations (see [Configuration Reference](../02_reference/configuration/README.md))
├── DATA_PROCESSING/     # ETL pipelines
├── TRAINING/            # Model training
├── IBKR_trading/        # IBKR integration (optional)
├── ALPACA_trading/      # Alpaca integration (optional)
├── data/                # Data storage
├── models/              # Trained models
├── results/             # Results and metrics
└── docs/                # Documentation
```

## Related Documentation

- [Getting Started](GETTING_STARTED.md)
- [Quick Start](QUICKSTART.md)
- [System Reference](../02_reference/systems/)
- [Architecture Deep Dive](../03_technical/design/ARCHITECTURE_DEEP_DIVE.md)
- [Module Reference](../02_reference/api/MODULE_REFERENCE.md)
- [Data Format Spec](../02_reference/data/DATA_FORMAT_SPEC.md)
- [Model Catalog](../02_reference/models/MODEL_CATALOG.md)
