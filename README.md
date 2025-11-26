
Email: jtlewis204@gmail.com

# RESEARCH AND TRADING PLATFORM(defaults tuned for 5m data)

A comprehensive machine learning system for high-frequency trading, featuring temporal leakage prevention, cross-sectional feature engineering, and a model zoo with 17+ production-ready models.

## Quick Start

1. **Setup Environment**
   ```bash
   conda env create -f environment.yml
   conda activate trader_env
   ```

2. **Quick Start Guide**
 - See [INFORMATION/01_QUICK_START.md](INFORMATION/01_QUICK_START.md) for detailed setup instructions

3. **Key Features**
 - **Temporal Leakage Prevention**: Purged time series cross-validation
 - **Cross-Sectional Data Loading**: Multi-symbol feature engineering
 - **17+ Model Families**: LightGBM, XGBoost, Deep Learning, Probabilistic models
 - **Feature Selection**: IC-based and predictive power ranking
 - **Target Discovery**: Automated target predictability ranking
 - **Production-Ready**: Comprehensive validation and safety guards

## Table of Contents

### Getting Started

- **[INFORMATION/01_QUICK_START.md](INFORMATION/01_QUICK_START.md)** - Complete setup and quick start guide
- **[INFORMATION/07_PROJECT_OVERVIEW.md](INFORMATION/07_PROJECT_OVERVIEW.md)** - High-level architecture and workflow
- **[SETUP/QUICK_SETUP.sh](SETUP/QUICK_SETUP.sh)** - Automated setup script
- **[notes_from_creator.md](notes_from_creator.md)** - Notes from the project creator

### Core Documentation (INFORMATION/)

- **[INFORMATION/01_QUICK_START.md](INFORMATION/01_QUICK_START.md)** - Quick start guide
- **[INFORMATION/02_CONFIG_REFERENCE.md](INFORMATION/02_CONFIG_REFERENCE.md)** - Configuration file reference
- **[INFORMATION/03_MIGRATION_NOTES.md](INFORMATION/03_MIGRATION_NOTES.md)** - Migration and upgrade notes
- **[INFORMATION/04_DATA_PIPELINE.md](INFORMATION/04_DATA_PIPELINE.md)** - Data processing pipeline documentation
- **[INFORMATION/05_MODEL_TRAINING.md](INFORMATION/05_MODEL_TRAINING.md)** - Model training guide
- **[INFORMATION/06_COLUMN_REFERENCE.md](INFORMATION/06_COLUMN_REFERENCE.md)** - Column and feature reference
- **[INFORMATION/07_PROJECT_OVERVIEW.md](INFORMATION/07_PROJECT_OVERVIEW.md)** - Project overview and architecture
- **[INFORMATION/08_FEATURE_SELECTION.md](INFORMATION/08_FEATURE_SELECTION.md)** - Feature selection guide
- **[INFORMATION/MULTI_MODEL_FEATURE_SELECTION.md](INFORMATION/MULTI_MODEL_FEATURE_SELECTION.md)** - Multi-model feature selection

### Data Processing

- **[DATA_PROCESSING/README.md](DATA_PROCESSING/README.md)** - Data processing module documentation
- **[docs/DATASET_SIZING_STRATEGY.md](docs/DATASET_SIZING_STRATEGY.md)** - Dataset sizing recommendations

### Training

- **[TRAINING/FEATURE_SELECTION_GUIDE.md](TRAINING/FEATURE_SELECTION_GUIDE.md)** - Feature selection workflow
- **[TRAINING/FIRST_BATCH_SPECS_IMPLEMENTATION.md](TRAINING/FIRST_BATCH_SPECS_IMPLEMENTATION.md)** - First batch implementation guide
- **[TRAINING/SAFE_TARGET_PATTERN_IMPLEMENTATION.md](TRAINING/SAFE_TARGET_PATTERN_IMPLEMENTATION.md)** - Safe target pattern implementation
- **[TRAINING/TRAINING_OPTIMIZATION_GUIDE.md](TRAINING/TRAINING_OPTIMIZATION_GUIDE.md)** - Training optimization strategies
- **[TRAINING/IMPORT_AUDIT_AND_STRUCTURE.md](TRAINING/IMPORT_AUDIT_AND_STRUCTURE.md)** - Import structure and audit
- **[TRAINING/strategies/STRATEGY_UPDATES.md](TRAINING/strategies/STRATEGY_UPDATES.md)** - Training strategy updates

#### Training Experiments

- **[TRAINING/EXPERIMENTS/README.md](TRAINING/EXPERIMENTS/README.md)** - Experiments overview
- **[TRAINING/EXPERIMENTS/QUICK_START.md](TRAINING/EXPERIMENTS/QUICK_START.md)** - Quick start for experiments
- **[TRAINING/EXPERIMENTS/IMPLEMENTATION_SUMMARY.md](TRAINING/EXPERIMENTS/IMPLEMENTATION_SUMMARY.md)** - Implementation summary
- **[TRAINING/EXPERIMENTS/OPERATIONS_GUIDE.md](TRAINING/EXPERIMENTS/OPERATIONS_GUIDE.md)** - Operations guide
- **[TRAINING/EXPERIMENTS/phase1_feature_engineering/README.md](TRAINING/EXPERIMENTS/phase1_feature_engineering/README.md)** - Phase 1 feature engineering

### Feature Selection & Ranking

- **[docs/COMPREHENSIVE_FEATURE_RANKING.md](docs/COMPREHENSIVE_FEATURE_RANKING.md)** - Comprehensive feature ranking guide
- **[docs/ADDITIONAL_FEATURE_SELECTION_MODELS.md](docs/ADDITIONAL_FEATURE_SELECTION_MODELS.md)** - Additional feature selection models
- **[docs/ADDITIONAL_MODELS_QUICKSTART.md](docs/ADDITIONAL_MODELS_QUICKSTART.md)** - Quick start for additional models
- **[docs/ALL_MODELS_ENABLED.md](docs/ALL_MODELS_ENABLED.md)** - All models enabled documentation
- **[docs/COMPLETE_FEATURE_SELECTION_MODELS.md](docs/COMPLETE_FEATURE_SELECTION_MODELS.md)** - Complete feature selection models
- **[docs/FEATURE_IMPORTANCE_FIX.md](docs/FEATURE_IMPORTANCE_FIX.md)** - Feature importance fixes
- **[docs/IMPORTANCE_R2_WEIGHTING.md](docs/IMPORTANCE_R2_WEIGHTING.md)** - Importance and R² weighting
- **[docs/IMPORTANCE_SCORE_INTERPRETATION.md](docs/IMPORTANCE_SCORE_INTERPRETATION.md)** - Interpreting importance scores
- **[scripts/ranking.md](scripts/ranking.md)** - Ranking scripts documentation
- **[NOTES/QUICK_START_FEATURE_RANKING.md](NOTES/QUICK_START_FEATURE_RANKING.md)** - Quick start for feature ranking

### Validation & Leakage Prevention

- **[docs/VALIDATION_LEAK_AUDIT.md](docs/VALIDATION_LEAK_AUDIT.md)** - Validation leakage audit report
- **[docs/TARGET_LEAKAGE_CLARIFICATION.md](docs/TARGET_LEAKAGE_CLARIFICATION.md)** - Target leakage clarification
- **[docs/FWD_RET_20D_LEAKAGE_ANALYSIS.md](docs/FWD_RET_20D_LEAKAGE_ANALYSIS.md)** - Forward return leakage analysis
- **[docs/FIXES/leakage.md](docs/FIXES/leakage.md)** - Leakage fixes documentation
- **[docs/FIXES/DEEPER_LEAK_FIX.md](docs/FIXES/DEEPER_LEAK_FIX.md)** - Deeper leak fixes
- **[docs/FIXES/FINAL_LEAKAGE_SUMMARY.md](docs/FIXES/FINAL_LEAKAGE_SUMMARY.md)** - Final leakage summary
- **[docs/FIXES/LEAKAGE_FIXED_NEXT_STEPS.md](docs/FIXES/LEAKAGE_FIXED_NEXT_STEPS.md)** - Next steps after leakage fixes
- **[docs/FIXES/QUICK_START_CLEAN_BASELINE.md](docs/FIXES/QUICK_START_CLEAN_BASELINE.md)** - Quick start with clean baseline
- **[docs/FIXES/ROUND3_TEMPORAL_OVERLAP_FIX.md](docs/FIXES/ROUND3_TEMPORAL_OVERLAP_FIX.md)** - Round 3 temporal overlap fixes
- **[docs/FIXES/TARGET_IS_LEAKED.md](docs/FIXES/TARGET_IS_LEAKED.md)** - Target leakage identification
- **[leakage.md](leakage.md)** - Leakage documentation

### Target Discovery & Analysis

- **[docs/TARGET_DISCOVERY_UPDATE.md](docs/TARGET_DISCOVERY_UPDATE.md)** - Target discovery updates
- **[docs/TARGET_RECOMMENDATIONS.md](docs/TARGET_RECOMMENDATIONS.md)** - Target recommendations
- **[docs/TARGET_MODEL_PIPELINE_ANALYSIS.md](docs/TARGET_MODEL_PIPELINE_ANALYSIS.md)** - Target and model pipeline analysis
- **[docs/TARGET_TO_FEATURE_WORKFLOW.md](docs/TARGET_TO_FEATURE_WORKFLOW.md)** - Target to feature workflow

### Model & Performance

- **[docs/MODEL_ENABLING_RECOMMENDATIONS.md](docs/MODEL_ENABLING_RECOMMENDATIONS.md)** - Model enabling recommendations
- **[docs/ALPHA_ENHANCEMENT_ROADMAP.md](docs/ALPHA_ENHANCEMENT_ROADMAP.md)** - Alpha enhancement roadmap
- **[docs/GPU_SETUP_MULTI_MODEL.md](docs/GPU_SETUP_MULTI_MODEL.md)** - GPU setup for multi-model training
- **[docs/AVOID_LONG_RUNS.md](docs/AVOID_LONG_RUNS.md)** - Avoiding long-running processes

### Setup & Configuration

- **[SETUP/QUICK_SETUP.sh](SETUP/QUICK_SETUP.sh)** - Quick setup script
- **[SETUP/verify_data_directory_setup.sh](SETUP/verify_data_directory_setup.sh)** - Data directory verification

### Logging & Monitoring

- **[docs/JOURNALD_LOGGING.md](docs/JOURNALD_LOGGING.md)** - Journald logging setup
- **[docs/RESTORE_FROM_LOGS.md](docs/RESTORE_FROM_LOGS.md)** - Restoring from logs
- **[NOTES/journallog.md](NOTES/journallog.md)** - Journal logging notes

### Workflows & Next Steps

- **[docs/NEXT_STEPS_WORKFLOW.md](docs/NEXT_STEPS_WORKFLOW.md)** - Next steps workflow
- **[NOTES/WHAT_TO_DO_NEXT.md](NOTES/WHAT_TO_DO_NEXT.md)** - What to do next
- **[docs/CODE_REVIEW_BUGS.md](docs/CODE_REVIEW_BUGS.md)** - Code review and bugs

### Scripts Documentation

- **[scripts/OUTDATED_SCRIPTS.md](scripts/OUTDATED_SCRIPTS.md)** - List of outdated scripts
- **[scripts/ranking.md](scripts/ranking.md)** - Ranking scripts guide

### Updates & Changelog

- **[UPDATE/README.md](UPDATE/README.md)** - Updates overview
- Various update entries in `UPDATE/` directory

### ALPACA Paper Trading Integration

- **[ALPACA_trading/README.md](ALPACA_trading/README.md)** - ALPACA paper trading system overview
- **Core Components:**
  - **[ALPACA_trading/core/README.md](ALPACA_trading/core/README.md)** - Core trading engine components (regime detection, strategy selection, performance tracking, risk management)
- **Broker Integration:**
  - **[ALPACA_trading/brokers/README.md](ALPACA_trading/brokers/README.md)** - Broker interface and implementations (Alpaca paper trading, data providers)
- **Trading Strategies:**
  - **[ALPACA_trading/strategies/README.md](ALPACA_trading/strategies/README.md)** - Trading strategy implementations (regime-aware ensemble, factory pattern)
- **ML Integration:**
  - **[ALPACA_trading/ml/README.md](ALPACA_trading/ml/README.md)** - Machine learning model interface, registry, and runtime
- **Scripts & Tools:**
  - **[ALPACA_trading/scripts/README.md](ALPACA_trading/scripts/README.md)** - Executable scripts (paper runner, data fetching)
  - **[ALPACA_trading/cli/README.md](ALPACA_trading/cli/README.md)** - Command-line interface for paper trading
  - **[ALPACA_trading/utils/README.md](ALPACA_trading/utils/README.md)** - Utility functions and helpers
  - **[ALPACA_trading/tools/README.md](ALPACA_trading/tools/README.md)** - Development and debugging tools
- **Configuration:**
  - **[ALPACA_trading/config/README.md](ALPACA_trading/config/README.md)** - Configuration files and settings

### IBKR Trading Integration (Untested)

**️ Warning**: The IBKR trading code is **untested** and should not be used for live trading. See [notes_from_creator.md](notes_from_creator.md) for details.

- **[IBKR_trading/README.md](IBKR_trading/README.md)** - IBKR trading system overview
- **[IBKR_trading/DAILY_TESTING_README.md](IBKR_trading/DAILY_TESTING_README.md)** - Daily model testing guide
- **[IBKR_trading/live_trading/README.md](IBKR_trading/live_trading/README.md)** - Live trading system documentation
- **[IBKR_trading/live_trading/C++_INTEGRATION_SUMMARY.md](IBKR_trading/live_trading/C++_INTEGRATION_SUMMARY.md)** - C++ integration summary
- **[IBKR_trading/LIVE_TRADING_INTEGRATION.md](IBKR_trading/LIVE_TRADING_INTEGRATION.md)** - Live trading integration guide
- **[IBKR_trading/YAHOO_FINANCE_INTEGRATION.md](IBKR_trading/YAHOO_FINANCE_INTEGRATION.md)** - Yahoo Finance integration
- **[IBKR_trading/INTRADAY_TRADING_ANALYSIS.md](IBKR_trading/INTRADAY_TRADING_ANALYSIS.md)** - Intraday trading analysis
- **[IBKR_trading/ENHANCED_REBALANCING_TRADING_PLAN.md](IBKR_trading/ENHANCED_REBALANCING_TRADING_PLAN.md)** - Enhanced rebalancing plan
- **[IBKR_trading/TESTING_PLAN.md](IBKR_trading/TESTING_PLAN.md)** - Testing plan
- **[IBKR_trading/TESTING_SUMMARY.md](IBKR_trading/TESTING_SUMMARY.md)** - Testing summary
- **[IBKR_trading/IMPLEMENTATION_STATUS.md](IBKR_trading/IMPLEMENTATION_STATUS.md)** - Implementation status
- **[IBKR_trading/SYSTEMD_DEPLOYMENT_PLAN.md](IBKR_trading/SYSTEMD_DEPLOYMENT_PLAN.md)** - Systemd deployment plan
- **[IBKR_trading/PERFORMANCE_OPTIMIZATION_PLAN.md](IBKR_trading/PERFORMANCE_OPTIMIZATION_PLAN.md)** - Performance optimization plan
- **[IBKR_trading/OPTIMIZATION_ENGINE_ANALYSIS.md](IBKR_trading/OPTIMIZATION_ENGINE_ANALYSIS.md)** - Optimization engine analysis
- **[IBKR_trading/PRESSURE_TEST_IMPLEMENTATION_ROADMAP.md](IBKR_trading/PRESSURE_TEST_IMPLEMENTATION_ROADMAP.md)** - Pressure test roadmap
- **[IBKR_trading/PRESSURE_TEST_UPGRADES.md](IBKR_trading/PRESSURE_TEST_UPGRADES.md)** - Pressure test upgrades
- **[IBKR_trading/MATHEMATICAL_FOUNDATIONS.md](IBKR_trading/MATHEMATICAL_FOUNDATIONS.md)** - Mathematical foundations
- **[IBKR_trading/cpp_engine/README.md](IBKR_trading/cpp_engine/README.md)** - C++ engine documentation
- **[IBKR_trading/CS_DOCS/OPTIMIZATION_ARCHITECTURE.md](IBKR_trading/CS_DOCS/OPTIMIZATION_ARCHITECTURE.md)** - Optimization architecture
- **[IBKR_trading/deprecated/README.md](IBKR_trading/deprecated/README.md)** - Deprecated components
- **[IBKR_trading/deprecated/DEPRECATION_NOTICE.md](IBKR_trading/deprecated/DEPRECATION_NOTICE.md)** - Deprecation notice

## Project Structure

```
trader/
├── DATA_PROCESSING/          # Data processing pipeline
│   ├── features/             # Feature engineering modules
│   ├── targets/              # Target generation
│   ├── pipeline/             # Processing workflows
│   └── utils/                # Utilities
├── TRAINING/                 # Model training framework
│   ├── model_fun/            # 17+ model trainers
│   ├── strategies/           # Training strategies
│   ├── EXPERIMENTS/          # Experimental workflows
│   │   ├── phase1_feature_engineering/
│   │   ├── phase2_core_models/
│   │   └── phase3_sequential_models/
│   ├── common/               # Core utilities
│   ├── preprocessing/        # Data preprocessing
│   ├── processing/           # Cross-sectional processing
│   ├── blenders/             # Model blending
│   └── tools/                # Training tools
├── scripts/                  # Utility scripts
│   └── utils/                # Script utilities
├── INFORMATION/              # Core documentation
├── docs/                     # Additional documentation
│   └── FIXES/                # Leakage fixes documentation
├── NOTES/                    # Development notes
├── SETUP/                    # Setup scripts
├── UPDATE/                   # Update logs
│   └── YYYY-MM-DD/          # Date-organized updates
├── ALPACA_trading/          # ALPACA paper trading integration
│   ├── scripts/             # Paper trading scripts
│   │   └── data/            # Data fetching utilities
│   ├── core/                # Core trading engine
│   │   ├── risk/            # Risk management guardrails
│   │   └── telemetry/       # Performance telemetry
│   ├── brokers/             # Broker interface implementations
│   ├── strategies/          # Trading strategies
│   ├── ml/                  # Model interface and registry
│   ├── cli/                 # Command-line interface
│   ├── config/              # Configuration files
│   ├── utils/               # Utility functions
│   └── tools/               # Development tools
└── IBKR_trading/            # IBKR trading integration (untested)
    ├── live_trading/        # Live trading components
    ├── cpp_engine/          # C++ inference engine
    │   ├── src/             # C++ source files
    │   ├── include/         # Header files
    │   ├── python_bindings/ # Python bindings
    │   └── benchmarks/       # Performance benchmarks
    ├── CS_DOCS/             # Cross-sectional documentation
    ├── brokers/              # Broker integration
    ├── optimization/         # Optimization engine
    ├── scripts/              # Trading scripts
    ├── systemd/              # Systemd service files
    ├── tests/                # Test suite
    └── deprecated/           # Deprecated components
```

## Key Components

### Data Processing
- **Feature Engineering**: 200+ features with cross-sectional support
- **Target Generation**: Barrier targets, excess returns, HFT forward returns
- **Temporal Leakage Prevention**: Horizon-aware feature filtering

### Model Training
- **17+ Model Families**: LightGBM, XGBoost, MLP, Transformer, LSTM, CNN1D, VAE, GAN, NGBoost, and more
- **Unified Training Interface**: Consistent API across all models
- **Cross-Validation**: Purged time series splits to prevent leakage

### Feature Selection
- **IC-Based Ranking**: Information coefficient ranking
- **Predictive Power**: Multi-model feature importance
- **Cross-Sectional Support**: Multi-symbol feature engineering

### Target Discovery
- **Predictability Ranking**: Automated target evaluation
- **Composite Scoring**: R², consistency, and importance weighting
- **Leakage Detection**: Automatic temporal overlap detection

## License

This project is licensed under the **Academic Use Only License** - see the [LICENSE](LICENSE) file for details.

**IMPORTANT**: This software is for **ACADEMIC, EDUCATIONAL, AND RESEARCH USE ONLY**. Commercial use, including trading securities, generating revenue, or any for-profit use, is **STRICTLY PROHIBITED**.

## Mathematical Foundations & Credits

### Equations & Formulas

- **[EQUATIONS_LIST.md](EQUATIONS_LIST.md)** - Complete list of all mathematical equations and formulas used throughout the codebase, including:
  - IBKR Trading System equations (standardization, blending, arbitration, costs, optimization)
  - Target Ranking System equations (metrics, composite scores, purging)
  - Performance Metrics (R², IC, Sharpe, MDD, etc.)
  - Risk Management formulas (gates, thresholds)
  - Feature Pruning and Data Processing equations

**Total**: 40+ equations documented with file references and implementation details.

### Academic Credits

- **[CREDITS.md](CREDITS.md)** - Complete list of academic contributors whose foundational work underlies the mathematical components, including:
  - Statistics & Correlation: Karl Pearson, Charles Spearman
  - Optimization: Gauss & Legendre, Hoerl & Kennard, Ledoit & Wolf
  - Portfolio Theory: Harry Markowitz, Thierry Roncalli
  - Market Impact: Almgren & Chriss, Albert Kyle
  - Machine Learning: Leo Breiman, Friedman, López de Prado
  - Risk Metrics: William Sharpe, Burke, Magdon-Ismail & Atiya
  - And many more...

## ️ Development & Acknowledgments

**Independent Development**: This project was developed entirely on personal hardware without assistance, funding, or resources from the University of Alabama at Birmingham (UAB) or any other institution. All development, research, and computational work was conducted independently.

## Contributing

This is a research and development project. For questions or feedback, see [notes_from_creator.md](notes_from_creator.md).

## ️ Important Notes

- **ALPACA Paper Trading**: The ALPACA paper trading system is designed for paper trading only. Use Alpaca's paper trading API for safe testing.
- **IBKR Execution Code**: The IBKR execution code is untested. See [notes_from_creator.md](notes_from_creator.md) for details.
- **Temporal Leakage**: Always use `PurgedTimeSeriesSplit` for validation. Standard K-Fold is **fatal** for financial ML.
- **Data Requirements**: Ensure proper data directory setup before running training scripts.

## Repository Structure & Gitignored Directories

Several directories are excluded from version control via `.gitignore`. This section explains what's excluded and why.

### Excluded Directories

#### Data Directories
- **`data/`** - Primary data directory containing labeled datasets
- **`datasets/`** - Additional dataset storage
- **`DATA_PROCESSING/data/`** - Processed data from the pipeline
- **`results/`** - Training results and outputs

**Why excluded:** The dataset is **250-400 GB** in size, making it impractical to include in a Git repository. This includes:
- Historical market data (OHLCV bars)
- Processed features and targets
- Training outputs and model artifacts

**What you need to know:**
- You'll need to provide your own market data to use this system
- The code expects data in Parquet format following the schema documented in `INFORMATION/06_COLUMN_REFERENCE.md`
- See `DATA_PROCESSING/README.md` for data processing pipeline details

#### Model & Configuration Directories
- **`models/`** - Trained model files (`.pkl`, weights, etc.)
- **`CONFIG/`** - Configuration files (may exist locally but not tracked)
- **`config/`** - Additional configuration files

**Why excluded:**
- Model files can be large (hundreds of MB to GB per model)
- Configuration files may contain local paths or personal settings
- These are generated/configured locally and don't need to be version controlled

**What you need to know:**
- Configuration examples and templates are documented in `INFORMATION/02_CONFIG_REFERENCE.md`
- You'll need to create your own config files or adapt the examples
- Models are trained locally and saved to the `models/` directory

#### Other Excluded Items
- **`logs/`** - Log files generated during execution
- **`dep/`** - Deprecated code and scripts
- **`catboost_info/`** - CatBoost training artifacts
- **`.env`** - Environment variables (may contain secrets)
- **`venv/`** - Python virtual environment
- **`__pycache__/`** - Python bytecode cache
- **`*.csv`, `*.parquet`, `*.h5`, `*.pkl`** - Large data files

### What IS Included

The repository includes:
- **All source code** - Complete ML pipeline, training code, feature engineering
- **Documentation** - Comprehensive guides and references (including IBKR_trading documentation)
- **Scripts** - All utility and analysis scripts
- **Configuration examples** - Documented in INFORMATION/ directory
- **Setup scripts** - Environment setup and verification
- **IBKR_trading documentation** - Trading system documentation (code is untested)

### Getting Started Without the Dataset

1. **Obtain market data** - You'll need historical OHLCV data in the expected format
2. **Set up data directory** - Create `data/data_labeled/interval=5m/` structure
3. **Process your data** - Use the `DATA_PROCESSING/` pipeline to generate features
4. **Train models** - Use the `TRAINING/` framework with your processed data

See `INFORMATION/04_DATA_PIPELINE.md` for detailed data requirements and processing steps.

## Quick Reference

- **Start Here**: [INFORMATION/01_QUICK_START.md](INFORMATION/01_QUICK_START.md)
- **Feature Ranking**: [scripts/ranking.md](scripts/ranking.md)
- **Leakage Prevention**: [docs/VALIDATION_LEAK_AUDIT.md](docs/VALIDATION_LEAK_AUDIT.md)
- **Training Guide**: [INFORMATION/05_MODEL_TRAINING.md](INFORMATION/05_MODEL_TRAINING.md)

---

*Last Updated: Generated automatically from repository structure*

