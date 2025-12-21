# Architecture Overview

System architecture for FoxML Core.

## System Purpose

FoxML Core is a research infrastructure system for ML pipelines, cross-sectional data workflows, and reproducible experiments. Provides:

- Scalable ML workflow design
- Leakage-safe research architecture
- High-throughput data processing (verified stable with workloads up to 100 GB RAM)
- Multi-model training systems
- Hybrid C++/Python infrastructure
- HPC-compatible orchestration patterns (single-node optimized; distributed HPC planned/WIP)

**System Requirements**: See [System Requirements](SYSTEM_REQUIREMENTS.md) for hardware specifications. Verified stable up to 100 GB RAM; targeted for 1 TB+ institutional deployment.

Cross-sectional ML infrastructure. Provides architecture, not domain-specific applications.

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
│              Model Artifacts & Outputs                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   Trained    │  │  Predictions  │  │  Performance  │   │
│  │    Models    │  │   & Metrics  │  │   Reports     │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. CONFIG (Configuration Management)

**Single Source of Truth (SST)**: Centralized, version-controlled configuration. All training parameters load from configs (with fallback defaults for edge cases).

**Features:**
- **20 model families** all use config-driven hyperparameters (n_estimators, max_depth, learning_rate, etc.)
- **All train/test splits** use `preprocessing.validation.test_size` from config
- **All random seeds** use `BASE_SEED` from determinism system for full reproducibility
- **17 model configs** with 3 variants each (conservative/balanced/aggressive)
- **Runtime overrides** and environment variable support
- **YAML-based** configuration files

**Reproducibility**: Same config file → identical results across all pipeline stages (reproducibility ensured when using proper configs; external factors like library versions may affect results).

**Location:** `CONFIG/`

**Usage:**
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
- **Training routing & planning** (NEW - 2025-12-11): Config-driven routing decisions, automatic training plan generation
- **2-stage training pipeline** (NEW): CPU models first, then GPU models (all 20 model families)
- **One-command end-to-end flow** (NEW): Complete pipeline from ranking → feature selection → training plan → training execution
- Unified workflow: ranking → selection → routing → training
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

### 4. Model Output & Integration

**Model Artifacts**:
- Serialized trained models
- Performance metrics and validation reports
- Feature importance and model diagnostics

**Integration Points**:
- Models can be integrated with external systems and applications
- Standard formats for model serialization
- Performance tracking and monitoring interfaces

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
- Process: Intelligent training pipeline (ranking → selection → routing plan → training plan → training)
- **Training Routing** (NEW): Config-driven decisions about where to train (cross-sectional vs symbol-specific)
- **Training Plan** (NEW): Automatic generation of actionable training jobs with priorities and model families
- **2-Stage Training** (NEW): CPU models first (10 models), then GPU models (10 models: 4 TF + 6 Torch)
- **Leakage Detection**: Pre-training leak scan + auto-fixer with config backups
- Output: Trained models + rankings + feature selections + routing plans + training plans
- Models: Saved to `{output_dir}_YYYYMMDD_HHMMSS/training_results/`
- Rankings: Saved to `{output_dir}_YYYYMMDD_HHMMSS/target_rankings/`
- Feature Selections: Saved to `{output_dir}_YYYYMMDD_HHMMSS/feature_selections/`
- Routing Plans: Saved to `{output_dir}/METRICS/routing_plan/`
- Training Plans: Saved to `{output_dir}/globals/training_plan/` (master_training_plan.json) - primary location, `METRICS/training_plan/` supported as legacy fallback
- Config Backups: Saved to `RESULTS/{cohort_id}/{run_name}/backups/{target}/{timestamp}/` (NEW: integrated into run directory) or `CONFIG/backups/{target}/{timestamp}/` (legacy, backward compatible)
- Configs: Versioned in `CONFIG/model_config/` (see [Configuration Reference](../02_reference/configuration/README.md))

### Stage 5: Evaluation
- Input: Trained models
- Output: Performance metrics
- Metrics: Sharpe, drawdown, hit rate, profit factor
- Location: `{output_dir}_YYYYMMDD_HHMMSS/training_results/`

## Design Principles

### 1. Configuration-Driven
- All training parameters load from YAML configs (with fallback defaults for edge cases)
- Easy experimentation and reproducibility
- See `CONFIG/CONFIG_AUDIT.md` for remaining hardcoded thresholds in some modules

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
- C++ inference for high-performance operations
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
├── data/                # Data storage
├── models/              # Trained models
├── results/             # Results and metrics
└── docs/                # Documentation
```

## Related Documentation

- [Getting Started](GETTING_STARTED.md)
- [Quick Start](QUICKSTART.md)
- [Intelligent Training Tutorial](../01_tutorials/training/INTELLIGENT_TRAINING_TUTORIAL.md) - Automated training pipeline
- [Ranking and Selection Consistency](../01_tutorials/training/RANKING_SELECTION_CONSISTENCY.md) - Unified pipeline behavior
- [Modular Config System](../02_reference/configuration/MODULAR_CONFIG_SYSTEM.md) - Configuration system
- [System Reference](../02_reference/systems/)
- [Architecture Deep Dive](../03_technical/design/ARCHITECTURE_DEEP_DIVE.md)
- [Module Reference](../02_reference/api/MODULE_REFERENCE.md)
- [Data Format Spec](../02_reference/data/DATA_FORMAT_SPEC.md)
- [Model Catalog](../02_reference/models/MODEL_CATALOG.md)
