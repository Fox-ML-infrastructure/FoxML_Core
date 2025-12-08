# Documentation Index

Complete navigation guide for FoxML Core documentation.

**Last Updated**: 2025-12-08  
**Recent Updates**: Added unified pipeline consistency guide, structured logging configuration, and shared utility modules documentation.

## Quick Navigation

- [Quick Start](00_executive/QUICKSTART.md) - Get running in 5 minutes
- [Architecture Overview](00_executive/ARCHITECTURE_OVERVIEW.md) - System at a glance
- [Getting Started](00_executive/GETTING_STARTED.md) - Onboarding guide

### Project Status & Licensing

- [Roadmap](../ROADMAP.md) - Current development focus and upcoming work
- [Changelog](../CHANGELOG.md) - Recent technical and compliance changes
- [Licensing & Subscriptions](../LEGAL/SUBSCRIPTIONS.md) - AGPL vs commercial usage and subscription tiers
- [Legal Documentation Index](LEGAL_INDEX.md) - Complete legal and compliance documentation

### Who Should Read What

- **Execs / PMs / Stakeholders** → README, Quick Start, Architecture Overview, Roadmap, Changelog
- **Quants / Researchers** → Getting Started → Pipelines & Training tutorials → Intelligence Layer Overview
- **Infra / MLOps** → Setup tutorials, Configuration Reference, Systems Reference, Operations
- **Trading / Desk Ops** → Trading tutorials, IBKR/Alpaca System Reference, System Documentation
- **Legal / Compliance** → [Legal Documentation Index](LEGAL_INDEX.md)

## Tier A: Executive / High-Level

First-time users start here.

- [README](../README.md) - Project overview, licensing, contact
- [Quick Start](00_executive/QUICKSTART.md) - Get running in 5 minutes
- [Architecture Overview](00_executive/ARCHITECTURE_OVERVIEW.md) - System architecture
- [Getting Started](00_executive/GETTING_STARTED.md) - Onboarding guide

## Tier B: Tutorials / Walkthroughs

Step-by-step guides for common tasks.

### Setup
- [Installation](01_tutorials/setup/INSTALLATION.md) - System installation
- [Environment Setup](01_tutorials/setup/ENVIRONMENT_SETUP.md) - Python environment
- [GPU Setup](01_tutorials/setup/GPU_SETUP.md) - GPU configuration

### Pipelines
- [First Pipeline Run](01_tutorials/pipelines/FIRST_PIPELINE_RUN.md) - Run your first pipeline
- [Data Processing Walkthrough](01_tutorials/pipelines/DATA_PROCESSING_WALKTHROUGH.md) - Data pipeline guide
- [Feature Engineering Tutorial](01_tutorials/pipelines/FEATURE_ENGINEERING_TUTORIAL.md) - Feature creation

### Training
- [Intelligent Training Tutorial](01_tutorials/training/INTELLIGENT_TRAINING_TUTORIAL.md) - Automated target ranking, feature selection, and training (includes timestamped outputs and backup system)
- [Ranking and Selection Consistency](01_tutorials/training/RANKING_SELECTION_CONSISTENCY.md) - **NEW**: Unified pipeline behavior (interval handling, sklearn preprocessing, CatBoost configuration)
- [Model Training Guide](01_tutorials/training/MODEL_TRAINING_GUIDE.md) - Manual training workflow (how to run it)
- [Walk-Forward Validation](01_tutorials/training/WALKFORWARD_VALIDATION.md) - Validation workflow
- [Feature Selection Tutorial](01_tutorials/training/FEATURE_SELECTION_TUTORIAL.md) - Manual feature selection
- [Experiments Workflow](01_tutorials/training/EXPERIMENTS_WORKFLOW.md) - 3-phase optimized training workflow
- [Experiments Quick Start](01_tutorials/training/EXPERIMENTS_QUICK_START.md) - Quick start guide
- [Experiments Operations](01_tutorials/training/EXPERIMENTS_OPERATIONS.md) - Step-by-step operations
- [Phase 1: Feature Engineering](01_tutorials/training/PHASE1_FEATURE_ENGINEERING.md) - Phase 1 documentation

### Trading
- [Paper Trading Setup](01_tutorials/trading/PAPER_TRADING_SETUP.md) - Paper trading
- [IBKR Integration](01_tutorials/trading/IBKR_INTEGRATION.md) - IBKR setup
- [Alpaca Integration](01_tutorials/trading/ALPACA_INTEGRATION.md) - Alpaca setup

### Configuration
- [Config Basics](01_tutorials/configuration/CONFIG_BASICS.md) - Configuration fundamentals
- [Config Examples](01_tutorials/configuration/CONFIG_EXAMPLES.md) - Example configs
- [Advanced Config](01_tutorials/configuration/ADVANCED_CONFIG.md) - Advanced configuration

## Tier C: Core Reference Docs

Complete technical reference for daily use.

### API Reference
- [Module Reference](02_reference/api/MODULE_REFERENCE.md) - Python API (includes `target_utils.py` and `sklearn_safe.py` utilities)
- [Intelligent Trainer API](02_reference/api/INTELLIGENT_TRAINER_API.md) - Intelligent training pipeline API reference
- [CLI Reference](02_reference/api/CLI_REFERENCE.md) - Command-line tools
- [Config Schema](02_reference/api/CONFIG_SCHEMA.md) - Configuration schema

### Data Reference
- [Data Format Spec](02_reference/data/DATA_FORMAT_SPEC.md) - Data formats
- [Column Reference](02_reference/data/COLUMN_REFERENCE.md) - Column documentation
- [Data Sanity Rules](02_reference/data/DATA_SANITY_RULES.md) - Validation rules

### Models Reference
- [Model Catalog](02_reference/models/MODEL_CATALOG.md) - All available models
- [Model Config Reference](02_reference/models/MODEL_CONFIG_REFERENCE.md) - Model configurations
- [Training Parameters](02_reference/models/TRAINING_PARAMETERS.md) - Training settings

### Systems Reference
- [IBKR System Reference](02_reference/systems/IBKR_SYSTEM_REFERENCE.md) - IBKR integration
- [Alpaca System Reference](02_reference/systems/ALPACA_SYSTEM_REFERENCE.md) - Alpaca integration
- [Pipeline Reference](02_reference/systems/PIPELINE_REFERENCE.md) - Data pipelines

### Configuration Reference
- **[Modular Config System](02_reference/configuration/MODULAR_CONFIG_SYSTEM.md)** - Complete guide to modular configs, experiment configs, typed configs, migration (includes `logging_config.yaml`)
- [Configuration System Overview](02_reference/configuration/README.md) - Centralized configuration system overview (includes `logging_config.yaml` documentation)
- [Feature & Target Configs](02_reference/configuration/FEATURE_TARGET_CONFIGS.md) - Feature/target configuration guide
- [Training Pipeline Configs](02_reference/configuration/TRAINING_PIPELINE_CONFIGS.md) - System resources and training behavior
- [Safety & Leakage Configs](02_reference/configuration/SAFETY_LEAKAGE_CONFIGS.md) - Leakage detection and numerical stability
- [Model Configuration](02_reference/configuration/MODEL_CONFIGURATION.md) - Model hyperparameters and variants
- [Usage Examples](02_reference/configuration/USAGE_EXAMPLES.md) - Practical configuration examples (includes interval config and CatBoost examples)
- [Config Loader API](02_reference/configuration/CONFIG_LOADER_API.md) - Programmatic config loading (includes logging config utilities)
- [Config Overlays](02_reference/configuration/CONFIG_OVERLAYS.md) - Overlay system for environment-specific configs
- [Environment Variables](02_reference/configuration/ENVIRONMENT_VARIABLES.md) - Environment-based configuration

## Tier D: Deep Technical Appendices

Research notes, design rationale, advanced topics.

### Research
- [Intelligence Layer Overview](03_technical/research/INTELLIGENCE_LAYER.md) - Complete overview of intelligent training pipeline decision-making and automation
- [Leakage Analysis](03_technical/research/LEAKAGE_ANALYSIS.md) - Leakage research
- [Feature Importance Methodology](03_technical/research/FEATURE_IMPORTANCE_METHODOLOGY.md) - Feature importance research
- [Target Discovery](03_technical/research/TARGET_DISCOVERY.md) - Target research
- [Validation Methodology](03_technical/research/VALIDATION_METHODOLOGY.md) - Validation research

### Design
- [Architecture Deep Dive](03_technical/design/ARCHITECTURE_DEEP_DIVE.md) - System architecture
- [Mathematical Foundations](03_technical/design/MATHEMATICAL_FOUNDATIONS.md) - Math background
- [Optimization Engine](03_technical/design/OPTIMIZATION_ENGINE.md) - Optimization design
- [C++ Integration](03_technical/design/C++_INTEGRATION.md) - C++ components

### Benchmarks
- [Performance Metrics](03_technical/benchmarks/PERFORMANCE_METRICS.md) - Performance data
- [Model Comparisons](03_technical/benchmarks/MODEL_COMPARISONS.md) - Model benchmarks
- [Dataset Sizing](03_technical/benchmarks/DATASET_SIZING.md) - Dataset strategies

### Fixes
- [Known Issues](03_technical/fixes/KNOWN_ISSUES.md) - Current issues
- [Bug Fixes](03_technical/fixes/BUG_FIXES.md) - Fix history
- [Migration Notes](03_technical/fixes/MIGRATION_NOTES.md) - Migration guide
- [TensorFlow Executable Stack Fix](03_technical/fixes/TENSORFLOW_EXECUTABLE_STACK_FIX.md) - Fix for libtensorflow_cc.so executable stack error

### Roadmaps
- [Alpha Enhancement Roadmap](03_technical/roadmaps/ALPHA_ENHANCEMENT_ROADMAP.md) - Enhancement plan
- [Future Work](03_technical/roadmaps/FUTURE_WORK.md) - Planned features

### Implementation
- [Feature Selection Implementation](03_technical/implementation/FEATURE_SELECTION_GUIDE.md) - Feature selection implementation details (see also [Ranking and Selection Consistency](01_tutorials/training/RANKING_SELECTION_CONSISTENCY.md) for unified pipeline behavior)
- [Training Optimization](03_technical/implementation/TRAINING_OPTIMIZATION_GUIDE.md) - Training optimization guide
- [Safe Target Pattern](03_technical/implementation/SAFE_TARGET_PATTERN_IMPLEMENTATION.md) - Safe target pattern implementation
- [First Batch Specs](03_technical/implementation/FIRST_BATCH_SPECS_IMPLEMENTATION.md) - First batch specifications
- [Strategy Updates](03_technical/implementation/STRATEGY_UPDATES.md) - Training strategy updates
- [Experiments Implementation](03_technical/implementation/EXPERIMENTS_IMPLEMENTATION.md) - Experiments implementation summary
- [IBKR Status](03_technical/implementation/IBKR_STATUS.md) - IBKR implementation
- [Pressure Test Plan](03_technical/implementation/PRESSURE_TEST_PLAN.md) - Testing plan
- [Performance Optimization](03_technical/implementation/PERFORMANCE_OPTIMIZATION.md) - Optimization work
- [Adding Proprietary Models](03_technical/implementation/ADDING_PROPRIETARY_MODELS.md) - Using BaseModelTrainer to add custom models

### Testing
- [Testing Plan](03_technical/testing/TESTING_PLAN.md) - Test strategy
- [Testing Summary](03_technical/testing/TESTING_SUMMARY.md) - Test results
- [Daily Testing](03_technical/testing/DAILY_TESTING.md) - Daily test procedures

### Operations
- [Journald Logging](03_technical/operations/JOURNALD_LOGGING.md) - Logging setup
- [Restore from Logs](03_technical/operations/RESTORE_FROM_LOGS.md) - Recovery procedures
- [Avoid Long Runs](03_technical/operations/AVOID_LONG_RUNS.md) - Performance tips
- [Systemd Deployment](03_technical/operations/SYSTEMD_DEPLOYMENT.md) - Deployment guide

## Additional Documentation

### System Specifications
- See [Ranking and Selection Consistency](01_tutorials/training/RANKING_SELECTION_CONSISTENCY.md) for unified pipeline behavior
- See [Feature Selection Implementation](03_technical/implementation/FEATURE_SELECTION_GUIDE.md) for implementation details

### System Documentation
- [IBKR Trading](../IBKR_trading/README.md) - IBKR live trading system
- [IBKR Live Trading Integration](../IBKR_trading/LIVE_TRADING_INTEGRATION.md) - IBKR integration guide
- [Alpaca Trading](../ALPACA_trading/README.md) - Alpaca paper trading system
- [IBKR C++ Engine](../IBKR_trading/cpp_engine/README.md) - C++ performance components

## Documentation Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the documentation structure and maintenance policy.
