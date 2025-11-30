email: jtlewis@uab.edu

Any use-case that involves money, forecasting, modeling, optimization, or strategy is prohibited 
unless the resulting benefit is directly reinvested into protection of marginalized groups or public service. 
There are no exceptions, reinterpretations, or edge-case allowances.

---

## Why This License Exists

Aurora was built through personal hardship and adversity. This license exists to ensure that:

- People who were cruel, dismissive, or harmful cannot exploit this work
- Institutions that protect marginalized people can benefit from it
- Tools are used to uplift, not exploit
- The communities this project aims to protect are materially supported

**Use of this software requires good-faith respect toward the creator and the communities it aims to protect.**

This project was built by someone who faced discrimination and hostility in technical spaces. 
This license exists so that people who reproduce that harm cannot exploit my work.

If you were part of creating that harm, you don't get to stand on my shoulders.

---

Disclaimer: I'm still learning this domain, so this may not be perfect. Please review and audit before any production use.

## Licensing Summary

![Minority Protective Use Only](https://img.shields.io/badge/Use-Minority%20Protective%20Only-purple?style=for-the-badge)

This project is licensed under the **GNU AGPL v3.0**.  
In addition to the AGPL, this project includes a **Universal Additional Permission**
(see `EXCEPTION.md`) granting expanded rights ONLY to nonprofit, academic, medical, public-health, 
and community-service organizations that maintain **comprehensive, enforceable protections for ALL marginalized groups**.

**Institutions cannot cherry-pick protections. They must protect ALL minorities, including but not limited to:**

**Sexual Orientation and Gender Identity:** LGBTQ+ individuals, non-binary, transgender, and gender-diverse people

**Race and Ethnicity:** Black, Latino/Latinx, Asian, Native American/Indigenous, Middle Eastern/North African, multiracial, and all people of color

**Religion:** Religious minorities (Muslims, Jews, Sikhs, Hindus, Buddhists, atheists, etc.)

**Disability:** Physical, intellectual, developmental, mental health, chronic illness, and neurodivergent individuals

**Gender:** Women, non-binary, transgender, and gender non-conforming individuals

**Immigration Status:** Immigrants, refugees, asylum seekers, and temporary residents

**Socioeconomic Status:** People experiencing poverty, homelessness, housing insecurity, or economic disadvantage

**Other Marginalized Groups:** People with criminal records, limited English proficiency, older adults facing discrimination, people with HIV/AIDS, sex workers, rural communities, and any other group facing systemic discrimination

**See EXCEPTION.md for the complete definition of marginalized groups.**

All institutional use of this software must:
1. materially benefit **ALL** marginalized people,  
2. materially protect **ALL** marginalized people, or  
3. directly support public-health, patient care, or community-service missions.

Institutions that do NOT maintain **comprehensive protections for ALL marginalized groups** are **not permitted** to use this software 
for any internal, operational, analytical, financial, forecasting, or strategic purpose.

**Protecting some minorities while discriminating against others is prohibited.**

Commercial entities, hedge funds, trading firms, for-profit finance groups, or any actor seeking 
commercial profit or financial advantage are **explicitly prohibited** from using this software 
beyond the baseline rights guaranteed by the AGPL. Financial modeling, forecasting, allocation, 
alpha research, market analysis, or strategic decision-making use by commercial or discriminatory 
organizations is strictly forbidden.

This project may not be used to harm, disadvantage, surveil, target, or restrict **ANY** marginalized people 
under ANY circumstances. No interpretive loopholes exist: if a use-case does not clearly support 
public good and comprehensive protection of ALL marginalized groups, it is prohibited.

## Warning Against Bad-Faith or Strategic Misuse

Any attempt to exploit, reinterpret, or narrow the meaning of this Additional Permission 
for purposes of bypassing protections for marginalized groups or enabling financial profit-seeking is 
considered a violation. 

**Institutions cannot cherry-pick protections. Protecting one group while discriminating against others is prohibited.**

If a use-case is not clearly and unambiguously aligned with comprehensive protection of **ALL** marginalized groups or public-good 
missions, it is prohibited.

## License Documentation & Compliance Tools

This repository includes comprehensive licensing documentation and compliance tools:

### Core License Documents
- **[EXCEPTION.md](EXCEPTION.md)** — Universal Additional Permission with full terms and conditions
- **[ENFORCEMENT.md](ENFORCEMENT.md)** — Enforcement policy, verification rights, and termination procedures
- **[FLOWCHART.md](FLOWCHART.md)** — Visual decision flowchart: "Can You Use Aurora?"

### Compliance Forms & Templates
- **[ATTESTATION_INSTITUTIONAL_USE.md](ATTESTATION_INSTITUTIONAL_USE.md)** — Institutional attestation form (required for institutional use)
- **[INTERNAL_USE_CERTIFICATION.md](INTERNAL_USE_CERTIFICATION.md)** — Department-level certification form
- **[DECLARATION_OF_COMPLIANT_USE.md](DECLARATION_OF_COMPLIANT_USE.md)** — Lawyer-facing notice letter template

### Automated Compliance
- **GitHub Actions Workflow** — Automatic license compliance warning on every push/PR (see `.github/workflows/license-warning.yml`)
- **Issue Template** — Forces license acknowledgment before opening issues
- **Pull Request Template** — Requires compliance confirmation for all contributions

### Compliance Badge
- **SVG Badge**: `lgbtq_protective_use_only.svg` — Machine-readable compliance indicator

### Values & Purpose
- **[VALUES.md](VALUES.md)** — Why this license exists and the values behind it

### Practical Safeguards
- **Compliance Checker**: `scripts/compliance_check.py` — Runtime license reminder (no tracking, just awareness)

**All institutional users must review and comply with these documents before using Aurora.**

## Quick Installation

**One-Command Install:**

```bash
curl -fsSL https://raw.githubusercontent.com/Aurora-Jennifer/Aurora-v2/main/install.sh | bash
```

This installation script will:
- Display and require agreement to the license terms
- Check Python 3.8+ installation
- Set up a virtual environment (recommended)
- Install all dependencies
- Run license compliance checks

**For detailed installation instructions, see [INSTALL.md](INSTALL.md)**

# Licensing FAQ

### Q: Can a hedge fund use Aurora?
**No.** Not for any financial, analytical, or trading purpose. AGPL allows private use only.

### Q: Can a university use Aurora for finance or budgeting?
Only if they maintain enforceable LGBTQ+ protections.  
If not, they are prohibited.

### Q: Can a finance department use Aurora to optimize revenue?
Only if all revenue is used to protect LGBTQ+ people or support public-interest missions.

### Q: Can someone claim their for-profit use “indirectly supports” LGBTQ+ people?
No. Indirect benefit does not qualify. The benefit must be **direct and material**.

### Q: Can a company claim they “aren’t anti-LGBTQ” as a loophole?
No. They must have explicit, documented nondiscrimination protections.

### Q: Can a bad actor reinterpret the text creatively?
No. See the No Loopholes Clause.

### Q: Can anyone weaken the meaning of LGBTQ+ protections?
No. Definitions are binding. 


# High-Frequency Trading (HFT) Machine Learning System

A comprehensive high-frequency trading machine learning platform featuring temporal leakage prevention, cross-sectional feature engineering, and a production-ready model zoo with 17+ model families optimized for HFT applications.

## Overview

This system is designed for high-frequency trading research, featuring:
- **Sub-second inference**: C++ inference engine for ultra-low latency
- **Temporal leakage prevention**: Purged time series cross-validation for financial ML
- **Cross-sectional processing**: Multi-symbol feature engineering and target generation
- **17+ Model Families**: LightGBM, XGBoost, Deep Learning, Probabilistic models
- **Production-ready validation**: Comprehensive safety guards and leakage detection

## Quick Start

### Option 1: One-Command Install (Recommended)

```bash
curl -fsSL https://raw.githubusercontent.com/Aurora-Jennifer/Aurora-v2/main/install.sh | bash
```

The install script will:
- Show license compliance notice and require agreement
- Check for Python 3.8+ and pip
- Create and activate a virtual environment (`aurora_env`)
- Install all dependencies from `requirements.txt`
- Run license compliance checks

After installation, activate the virtual environment:
```bash
source aurora_env/bin/activate
```

### Option 2: Manual Installation

1. **Setup Environment (Conda)**
   ```bash
   conda env create -f environment.yml
   conda activate trader_env
   ```

2. **Setup Environment (Virtual Environment)**
   ```bash
   python3 -m venv aurora_env
   source aurora_env/bin/activate
   pip install -r requirements.txt
   ```

3. **Quick Start Guide**
   - See [INFORMATION/01_QUICK_START.md](INFORMATION/01_QUICK_START.md) for detailed setup instructions

3. **Key HFT Features**
   - **Ultra-Low Latency**: C++ inference engine for sub-millisecond predictions
   - **Temporal Leakage Prevention**: Purged time series cross-validation
   - **Cross-Sectional Data Loading**: Multi-symbol feature engineering
   - **17+ Model Families**: Optimized for HFT signal generation
   - **Feature Selection**: IC-based and predictive power ranking
   - **Target Discovery**: Automated target predictability ranking

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

### Data Processing (DATA_PROCESSING/)

- **[DATA_PROCESSING/README.md](DATA_PROCESSING/README.md)** - Data processing module documentation

### Training (TRAINING/)

- **[TRAINING/FEATURE_SELECTION_GUIDE.md](TRAINING/FEATURE_SELECTION_GUIDE.md)** - Feature selection workflow
- **[TRAINING/FIRST_BATCH_SPECS_IMPLEMENTATION.md](TRAINING/FIRST_BATCH_SPECS_IMPLEMENTATION.md)** - First batch implementation guide
- **[TRAINING/SAFE_TARGET_PATTERN_IMPLEMENTATION.md](TRAINING/SAFE_TARGET_PATTERN_IMPLEMENTATION.md)** - Safe target pattern implementation
- **[TRAINING/TRAINING_OPTIMIZATION_GUIDE.md](TRAINING/TRAINING_OPTIMIZATION_GUIDE.md)** - Training optimization strategies
- **[TRAINING/IMPORT_AUDIT_AND_STRUCTURE.md](TRAINING/IMPORT_AUDIT_AND_STRUCTURE.md)** - Import structure and audit

#### Training Strategies (TRAINING/strategies/)

- **[TRAINING/strategies/STRATEGY_UPDATES.md](TRAINING/strategies/STRATEGY_UPDATES.md)** - Training strategy updates

#### Training Experiments (TRAINING/EXPERIMENTS/)

- **[TRAINING/EXPERIMENTS/README.md](TRAINING/EXPERIMENTS/README.md)** - Experiments overview
- **[TRAINING/EXPERIMENTS/QUICK_START.md](TRAINING/EXPERIMENTS/QUICK_START.md)** - Quick start for experiments
- **[TRAINING/EXPERIMENTS/IMPLEMENTATION_SUMMARY.md](TRAINING/EXPERIMENTS/IMPLEMENTATION_SUMMARY.md)** - Implementation summary
- **[TRAINING/EXPERIMENTS/OPERATIONS_GUIDE.md](TRAINING/EXPERIMENTS/OPERATIONS_GUIDE.md)** - Operations guide
- **[TRAINING/EXPERIMENTS/phase1_feature_engineering/README.md](TRAINING/EXPERIMENTS/phase1_feature_engineering/README.md)** - Phase 1 feature engineering

### Scripts (scripts/)

- **[scripts/OUTDATED_SCRIPTS.md](scripts/OUTDATED_SCRIPTS.md)** - List of outdated scripts
- **[scripts/ranking.md](scripts/ranking.md)** - Ranking scripts guide

### Documentation (docs/)

#### Feature Selection & Ranking

- **[docs/COMPREHENSIVE_FEATURE_RANKING.md](docs/COMPREHENSIVE_FEATURE_RANKING.md)** - Comprehensive feature ranking guide
- **[docs/ADDITIONAL_FEATURE_SELECTION_MODELS.md](docs/ADDITIONAL_FEATURE_SELECTION_MODELS.md)** - Additional feature selection models
- **[docs/ADDITIONAL_MODELS_QUICKSTART.md](docs/ADDITIONAL_MODELS_QUICKSTART.md)** - Quick start for additional models
- **[docs/ALL_MODELS_ENABLED.md](docs/ALL_MODELS_ENABLED.md)** - All models enabled documentation
- **[docs/COMPLETE_FEATURE_SELECTION_MODELS.md](docs/COMPLETE_FEATURE_SELECTION_MODELS.md)** - Complete feature selection models
- **[docs/FEATURE_IMPORTANCE_FIX.md](docs/FEATURE_IMPORTANCE_FIX.md)** - Feature importance fixes
- **[docs/IMPORTANCE_R2_WEIGHTING.md](docs/IMPORTANCE_R2_WEIGHTING.md)** - Importance and R² weighting
- **[docs/IMPORTANCE_SCORE_INTERPRETATION.md](docs/IMPORTANCE_SCORE_INTERPRETATION.md)** - Interpreting importance scores

#### Validation & Leakage Prevention

- **[docs/VALIDATION_LEAK_AUDIT.md](docs/VALIDATION_LEAK_AUDIT.md)** - Validation leakage audit report
- **[docs/TARGET_LEAKAGE_CLARIFICATION.md](docs/TARGET_LEAKAGE_CLARIFICATION.md)** - Target leakage clarification
- **[docs/FWD_RET_20D_LEAKAGE_ANALYSIS.md](docs/FWD_RET_20D_LEAKAGE_ANALYSIS.md)** - Forward return leakage analysis

#### Leakage Fixes (docs/FIXES/)

- **[docs/FIXES/leakage.md](docs/FIXES/leakage.md)** - Leakage fixes documentation
- **[docs/FIXES/DEEPER_LEAK_FIX.md](docs/FIXES/DEEPER_LEAK_FIX.md)** - Deeper leak fixes
- **[docs/FIXES/FINAL_LEAKAGE_SUMMARY.md](docs/FIXES/FINAL_LEAKAGE_SUMMARY.md)** - Final leakage summary
- **[docs/FIXES/LEAKAGE_FIXED_NEXT_STEPS.md](docs/FIXES/LEAKAGE_FIXED_NEXT_STEPS.md)** - Next steps after leakage fixes
- **[docs/FIXES/QUICK_START_CLEAN_BASELINE.md](docs/FIXES/QUICK_START_CLEAN_BASELINE.md)** - Quick start with clean baseline
- **[docs/FIXES/ROUND3_TEMPORAL_OVERLAP_FIX.md](docs/FIXES/ROUND3_TEMPORAL_OVERLAP_FIX.md)** - Round 3 temporal overlap fixes
- **[docs/FIXES/TARGET_IS_LEAKED.md](docs/FIXES/TARGET_IS_LEAKED.md)** - Target leakage identification

#### Target Discovery & Analysis

- **[docs/TARGET_DISCOVERY_UPDATE.md](docs/TARGET_DISCOVERY_UPDATE.md)** - Target discovery updates
- **[docs/TARGET_RECOMMENDATIONS.md](docs/TARGET_RECOMMENDATIONS.md)** - Target recommendations
- **[docs/TARGET_MODEL_PIPELINE_ANALYSIS.md](docs/TARGET_MODEL_PIPELINE_ANALYSIS.md)** - Target and model pipeline analysis
- **[docs/TARGET_TO_FEATURE_WORKFLOW.md](docs/TARGET_TO_FEATURE_WORKFLOW.md)** - Target to feature workflow

#### Model & Performance

- **[docs/MODEL_ENABLING_RECOMMENDATIONS.md](docs/MODEL_ENABLING_RECOMMENDATIONS.md)** - Model enabling recommendations
- **[docs/ALPHA_ENHANCEMENT_ROADMAP.md](docs/ALPHA_ENHANCEMENT_ROADMAP.md)** - Alpha enhancement roadmap
- **[docs/GPU_SETUP_MULTI_MODEL.md](docs/GPU_SETUP_MULTI_MODEL.md)** - GPU setup for multi-model training
- **[docs/AVOID_LONG_RUNS.md](docs/AVOID_LONG_RUNS.md)** - Avoiding long-running processes
- **[docs/DATASET_SIZING_STRATEGY.md](docs/DATASET_SIZING_STRATEGY.md)** - Dataset sizing recommendations

#### Logging & Monitoring

- **[docs/JOURNALD_LOGGING.md](docs/JOURNALD_LOGGING.md)** - Journald logging setup
- **[docs/RESTORE_FROM_LOGS.md](docs/RESTORE_FROM_LOGS.md)** - Restoring from logs

#### Workflows & Next Steps

- **[docs/NEXT_STEPS_WORKFLOW.md](docs/NEXT_STEPS_WORKFLOW.md)** - Next steps workflow
- **[docs/CODE_REVIEW_BUGS.md](docs/CODE_REVIEW_BUGS.md)** - Code review and bugs

### Notes (NOTES/)

- **[NOTES/journallog.md](NOTES/journallog.md)** - Journal logging notes
- **[NOTES/QUICK_START_FEATURE_RANKING.md](NOTES/QUICK_START_FEATURE_RANKING.md)** - Quick start for feature ranking
- **[NOTES/WHAT_TO_DO_NEXT.md](NOTES/WHAT_TO_DO_NEXT.md)** - What to do next

### Setup (SETUP/)

- **[SETUP/QUICK_SETUP.sh](SETUP/QUICK_SETUP.sh)** - Quick setup script
- **[SETUP/verify_data_directory_setup.sh](SETUP/verify_data_directory_setup.sh)** - Data directory verification

### Configuration (CONFIG/)

- **[CONFIG/README.md](CONFIG/README.md)** - Configuration documentation

### Data (data/)

- **[data/README.md](data/README.md)** - Data directory documentation

### Deprecated (dep/)

- **[dep/GPU_FEATURE_SELECTION_GUIDE.md](dep/GPU_FEATURE_SELECTION_GUIDE.md)** - GPU feature selection guide (deprecated)
- **[dep/LEAKAGE_FIX_README.md](dep/LEAKAGE_FIX_README.md)** - Leakage fix documentation (deprecated)
- **[dep/QUICK_START_GPU.md](dep/QUICK_START_GPU.md)** - Quick start GPU guide (deprecated)

### Root Documentation

- **[leakage.md](leakage.md)** - Leakage documentation
- **[EQUATIONS_LIST.md](EQUATIONS_LIST.md)** - Complete list of all mathematical equations and formulas
- **[CREDITS.md](CREDITS.md)** - Academic credits and contributors
- **[CREDIT.md](CREDIT.md)** - Additional credits

### ALPACA Paper Trading Integration (ALPACA_trading/)

- **[ALPACA_trading/README.md](ALPACA_trading/README.md)** - ALPACA paper trading system overview

#### Core Components (ALPACA_trading/core/)

- **[ALPACA_trading/core/README.md](ALPACA_trading/core/README.md)** - Core trading engine components (regime detection, strategy selection, performance tracking, risk management)

#### Broker Integration (ALPACA_trading/brokers/)

- **[ALPACA_trading/brokers/README.md](ALPACA_trading/brokers/README.md)** - Broker interface and implementations (Alpaca paper trading, data providers)

#### Trading Strategies (ALPACA_trading/strategies/)

- **[ALPACA_trading/strategies/README.md](ALPACA_trading/strategies/README.md)** - Trading strategy implementations (regime-aware ensemble, factory pattern)

#### ML Integration (ALPACA_trading/ml/)

- **[ALPACA_trading/ml/README.md](ALPACA_trading/ml/README.md)** - Machine learning model interface, registry, and runtime

#### Scripts & Tools (ALPACA_trading/scripts/)

- **[ALPACA_trading/scripts/README.md](ALPACA_trading/scripts/README.md)** - Executable scripts (paper runner, data fetching)

#### CLI (ALPACA_trading/cli/)

- **[ALPACA_trading/cli/README.md](ALPACA_trading/cli/README.md)** - Command-line interface for paper trading

#### Utilities (ALPACA_trading/utils/)

- **[ALPACA_trading/utils/README.md](ALPACA_trading/utils/README.md)** - Utility functions and helpers

#### Tools (ALPACA_trading/tools/)

- **[ALPACA_trading/tools/README.md](ALPACA_trading/tools/README.md)** - Development and debugging tools

#### Configuration (ALPACA_trading/config/)

- **[ALPACA_trading/config/README.md](ALPACA_trading/config/README.md)** - Configuration files and settings

### IBKR Trading Integration (IBKR_trading/)

**Warning**: The IBKR trading code is **untested** and should not be used for live trading. See [notes_from_creator.md](notes_from_creator.md) for details.

- **[IBKR_trading/README.md](IBKR_trading/README.md)** - IBKR trading system overview
- **[IBKR_trading/DAILY_TESTING_README.md](IBKR_trading/DAILY_TESTING_README.md)** - Daily model testing guide
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

#### Live Trading (IBKR_trading/live_trading/)

- **[IBKR_trading/live_trading/README.md](IBKR_trading/live_trading/README.md)** - Live trading system documentation
- **[IBKR_trading/live_trading/C++_INTEGRATION_SUMMARY.md](IBKR_trading/live_trading/C++_INTEGRATION_SUMMARY.md)** - C++ integration summary

#### C++ Engine (IBKR_trading/cpp_engine/)

- **[IBKR_trading/cpp_engine/README.md](IBKR_trading/cpp_engine/README.md)** - C++ engine documentation

#### Cross-Sectional Documentation (IBKR_trading/CS_DOCS/)

- **[IBKR_trading/CS_DOCS/OPTIMIZATION_ARCHITECTURE.md](IBKR_trading/CS_DOCS/OPTIMIZATION_ARCHITECTURE.md)** - Optimization architecture

#### Deprecated (IBKR_trading/deprecated/)

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

## Key HFT Components

### Data Processing
- **Feature Engineering**: 200+ features with cross-sectional support
- **Target Generation**: Barrier targets, excess returns, HFT forward returns
- **Temporal Leakage Prevention**: Horizon-aware feature filtering
- **Ultra-Low Latency**: Optimized data pipelines for HFT

### Model Training
- **17+ Model Families**: LightGBM, XGBoost, MLP, Transformer, LSTM, CNN1D, VAE, GAN, NGBoost, and more
- **Unified Training Interface**: Consistent API across all models
- **Cross-Validation**: Purged time series splits to prevent leakage
- **HFT Optimization**: Models optimized for sub-second inference

### Feature Selection
- **IC-Based Ranking**: Information coefficient ranking
- **Predictive Power**: Multi-model feature importance
- **Cross-Sectional Support**: Multi-symbol feature engineering
- **Real-Time Pruning**: Feature selection for low-latency inference

### Target Discovery
- **Predictability Ranking**: Automated target evaluation
- **Composite Scoring**: R², consistency, and importance weighting
- **Leakage Detection**: Automatic temporal overlap detection
- **HFT Targets**: Forward returns optimized for high-frequency trading

### C++ Inference Engine
- **Sub-Millisecond Latency**: C++ inference for ultra-low latency predictions
- **Python Bindings**: Seamless integration with Python training pipeline
- **Optimized Models**: ONNX and native C++ model support
- **Benchmarking**: Performance benchmarks and profiling tools

## License

This project is licensed under the **GNU Affero General Public License v3.0** - see the [LICENSE](LICENSE) file for details.

**IMPORTANT**: This software is for **ACADEMIC, EDUCATIONAL, AND RESEARCH USE ONLY**. Commercial use, including trading securities, generating revenue, or any for-profit use, may be subject to license restrictions.

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

## Development & Acknowledgments

**Independent Development**: This project was developed entirely on personal hardware without assistance, funding, or resources from the University of Alabama at Birmingham (UAB) or any other institution. All development, research, and computational work was conducted independently.

## Contributing

This is a research and development project. For questions or feedback, see [notes_from_creator.md](notes_from_creator.md).

## Important Notes

- **ALPACA Paper Trading**: The ALPACA paper trading system is designed for paper trading only. Use Alpaca's paper trading API for safe testing.
- **IBKR Execution Code**: The IBKR execution code is untested. See [notes_from_creator.md](notes_from_creator.md) for details.
- **Temporal Leakage**: Always use `PurgedTimeSeriesSplit` for validation. Standard K-Fold is **fatal** for financial ML.
- **Data Requirements**: Ensure proper data directory setup before running training scripts.
- **HFT Latency**: The C++ inference engine is optimized for sub-millisecond latency. See [IBKR_trading/cpp_engine/README.md](IBKR_trading/cpp_engine/README.md) for performance details.

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
- **C++ inference engine** - Source code and build system

### Getting Started Without the Dataset

1. **Obtain market data** - You'll need historical OHLCV data in the expected format
2. **Set up data directory** - Create `data/data_labeled/interval=5m/` structure
3. **Process your data** - Use the `DATA_PROCESSING/` pipeline to generate features
4. **Train models** - Use the `TRAINING/` framework with your processed data
5. **Deploy C++ engine** - Build and deploy the C++ inference engine for low-latency predictions

See `INFORMATION/04_DATA_PIPELINE.md` for detailed data requirements and processing steps.

## Quick Reference

- **Start Here**: [INFORMATION/01_QUICK_START.md](INFORMATION/01_QUICK_START.md)
- **Feature Ranking**: [scripts/ranking.md](scripts/ranking.md)
- **Leakage Prevention**: [docs/VALIDATION_LEAK_AUDIT.md](docs/VALIDATION_LEAK_AUDIT.md)
- **Training Guide**: [INFORMATION/05_MODEL_TRAINING.md](INFORMATION/05_MODEL_TRAINING.md)
- **C++ Engine**: [IBKR_trading/cpp_engine/README.md](IBKR_trading/cpp_engine/README.md)
- **HFT Performance**: [IBKR_trading/PERFORMANCE_OPTIMIZATION_PLAN.md](IBKR_trading/PERFORMANCE_OPTIMIZATION_PLAN.md)

---

*Last Updated: Generated automatically from repository structure*
