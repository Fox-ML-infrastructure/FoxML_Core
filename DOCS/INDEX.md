# Documentation Index

Complete navigation guide for FoxML Core documentation.

**Last Updated**: 2025-12-13  
**Recent Updates**: SST Enforcement Design - EnforcedFeatureSet contract eliminates split-brain across all training paths (target ranking and feature selection, both cross-sectional and symbol-specific views). Type boundary wiring ensures X matrix sliced immediately using enforced.features (no rediscovery). Boundary assertions validate featureset integrity at all key boundaries with auto-fix capability. Provably split-brain free system with immediate mis-wire detection. Active Sanitization (Ghost Buster) - Proactive feature quarantine system that automatically removes features with excessive lookback before training starts, preventing "ghost feature" discrepancies where audit and auto-fix see different lookback values. Decision-Making System (EXPERIMENTAL) - Automated decision policies with configurable thresholds for feature instability, route instability, feature explosion decline, and class balance drift. Bayesian Patch Policy - Thompson sampling over discrete patch templates for adaptive config tuning. All decision/stability thresholds now config-driven (SST compliance). Sample Size Binning System - RESULTS directory organized by sample size bins (`RESULTS/sample_25k-50k/{run_name}/`) for easy comparison of runs with similar cross-sectional sample sizes. Audit-grade binning with unambiguous boundaries, versioning, and metadata storage. Trend Analysis System - Automated trend analysis across target ranking, feature selection, and cross-sectional ranking with exponential decay weighting and regression detection. Trend metadata stored in `metadata.json` for audit-grade verification. Cohort-Aware Reproducibility System - Runs organized by data cohort with sample-adjusted drift detection. Integrated Config Backups - Backups stored in run directory, organized by sample size bin. Enhanced Metadata - `metadata.json` includes full symbols list, bin info, and data characteristics. Unified Metadata Extractor - Centralized utility for consistent cohort identification across all modules.

## Documentation Structure

FoxML Core documentation is organized into four tiers:

- **[00_executive/](00_executive/)** - High-level overviews and quick-start guides for executives, PMs, and stakeholders
- **[01_tutorials/](01_tutorials/)** - Step-by-step guides and walkthroughs for common tasks and workflows
- **[02_reference/](02_reference/)** - Complete technical reference for daily use, API documentation, and system specifications
- **[03_technical/](03_technical/)** - Deep technical appendices, research notes, design rationale, and advanced topics

See each directory's [README](00_executive/README.md) for detailed contents.

## Quick Navigation

- [Quick Start](00_executive/QUICKSTART.md) - Get running in 5 minutes
- [Architecture Overview](00_executive/ARCHITECTURE_OVERVIEW.md) - System at a glance
- [Getting Started](00_executive/GETTING_STARTED.md) - Onboarding guide

### Project Status & Licensing

- [Roadmap](../ROADMAP.md) - Current development focus and upcoming work (executive summary)
- [Detailed Roadmap](02_reference/roadmap/README.md) - **NEW**: Per-date detailed roadmap with component status and priorities
- [Known Issues & Limitations](02_reference/KNOWN_ISSUES.md) - **NEW**: Features that don't work yet or have limitations
- [Deterministic Training](00_executive/DETERMINISTIC_TRAINING.md) - SST config system and reproducibility guarantees
- [Audit Reports](00_executive/audits/README.md) - **NEW**: Quality assurance audits, documentation accuracy checks, and technical audits (public transparency)
- [Documentation Review Statement](../DOCUMENTATION_REVIEW.md) - **NEW**: Initial documentation review completed
- [Changelog](../CHANGELOG.md) - Recent technical and compliance changes (quick overview)
- [Changelog Index](02_reference/changelog/README.md) - Per-day detailed changelogs with file paths and config references
- [Licensing & Subscriptions](../LEGAL/SUBSCRIPTIONS.md) - AGPL vs commercial usage and subscription tiers
- [Legal Documentation Index](LEGAL_INDEX.md) - Complete legal and compliance documentation

### Who Should Read What

- **Execs / PMs / Stakeholders** → README, Quick Start, Architecture Overview, Roadmap, Changelog
- **Quants / Researchers** → Getting Started → Pipelines & Training tutorials → Intelligence Layer Overview
- **Infra / MLOps** → Setup tutorials, Configuration Reference, Systems Reference, Operations
- **Model Integration** → Model output interfaces and integration patterns
- **Legal / Compliance** → [Legal Documentation Index](LEGAL_INDEX.md)

## Tier A: Executive / High-Level

First-time users start here. See [00_executive/README.md](00_executive/README.md) for complete contents.

- [README](../README.md) - Project overview, licensing, contact
- [Quick Start](00_executive/QUICKSTART.md) - Get running in 5 minutes
- [Architecture Overview](00_executive/ARCHITECTURE_OVERVIEW.md) - System architecture
- [Getting Started](00_executive/GETTING_STARTED.md) - Onboarding guide
- [System Requirements](00_executive/SYSTEM_REQUIREMENTS.md) - Hardware and software requirements (verified stable up to 100 GB RAM)
- [Audit Reports](00_executive/audits/README.md) - Quality assurance audits, documentation checks, and technical audits (public transparency)

## Tier B: Tutorials / Walkthroughs

Step-by-step guides for common tasks. See [01_tutorials/README.md](01_tutorials/README.md) for complete contents.

### Setup
- [Installation](01_tutorials/setup/INSTALLATION.md) - System installation
- [Environment Setup](01_tutorials/setup/ENVIRONMENT_SETUP.md) - Python environment
- [GPU Setup](01_tutorials/setup/GPU_SETUP.md) - **UPDATED**: GPU configuration for target ranking, feature selection, and model training

### Pipelines
- [First Pipeline Run](01_tutorials/pipelines/FIRST_PIPELINE_RUN.md) - Run your first pipeline

### Development & Architecture
- [Refactoring & Wrappers](01_tutorials/REFACTORING_AND_WRAPPERS.md) - Understanding the modular structure and backward compatibility wrappers
- [Data Processing Walkthrough](01_tutorials/pipelines/DATA_PROCESSING_WALKTHROUGH.md) - Data pipeline guide
- [Feature Engineering Tutorial](01_tutorials/pipelines/FEATURE_ENGINEERING_TUTORIAL.md) - Feature creation

### Training
- [Intelligent Training Tutorial](01_tutorials/training/INTELLIGENT_TRAINING_TUTORIAL.md) - Automated target ranking, feature selection, training plan generation, and training (includes timestamped outputs and backup system)
- [Auto Target Ranking](01_tutorials/training/AUTO_TARGET_RANKING.md) - **NEW**: How to auto-discover, rank, and select top targets from your dataset
- [Training Routing Quick Start](02_reference/training_routing/QUICK_START.md) - **NEW**: Quick start for training routing system and 2-stage training
- [Training Routing End-to-End Flow](02_reference/training_routing/END_TO_END_FLOW.md) - **NEW**: Complete end-to-end pipeline documentation
- [Ranking and Selection Consistency](01_tutorials/training/RANKING_SELECTION_CONSISTENCY.md) - **NEW**: Unified pipeline behavior (interval handling, sklearn preprocessing, CatBoost configuration, Boruta gatekeeper)
- [Model Training Guide](01_tutorials/training/MODEL_TRAINING_GUIDE.md) - Manual training workflow (how to run it)
- [Walk-Forward Validation](01_tutorials/training/WALKFORWARD_VALIDATION.md) - Validation workflow
- [Feature Selection Tutorial](01_tutorials/training/FEATURE_SELECTION_TUTORIAL.md) - Manual feature selection
- [Experiments Operations](01_tutorials/training/EXPERIMENTS_OPERATIONS.md) - Step-by-step operations
- ⚠️ **Legacy**: [EXPERIMENTS Workflow](../LEGACY/EXPERIMENTS_WORKFLOW.md) - **DEPRECATED**: Use [Intelligent Training Pipeline](training/INTELLIGENT_TRAINING_TUTORIAL.md) instead
- ⚠️ **Legacy**: [EXPERIMENTS Quick Start](../LEGACY/EXPERIMENTS_QUICK_START.md) - **DEPRECATED**: Use [Intelligent Training Tutorial](training/INTELLIGENT_TRAINING_TUTORIAL.md) instead
- [Phase 1: Feature Engineering](01_tutorials/training/PHASE1_FEATURE_ENGINEERING.md) - Phase 1 documentation

### Configuration
- [Config Basics](01_tutorials/configuration/CONFIG_BASICS.md) - Configuration fundamentals
- [Experiment Config Guide](01_tutorials/configuration/EXPERIMENT_CONFIG_GUIDE.md) - **NEW**: Complete guide to experiment configuration files
- [Config Examples](01_tutorials/configuration/CONFIG_EXAMPLES.md) - Example configs
- [Advanced Config](01_tutorials/configuration/ADVANCED_CONFIG.md) - Advanced configuration
- [Config Audit](02_reference/configuration/CONFIG_AUDIT.md) - **NEW**: Config folder audit, hardcoded values tracking, and organization plan

## Tier C: Core Reference Docs

Complete technical reference for daily use. See [02_reference/README.md](02_reference/README.md) for complete contents.

### API Reference
- [Module Reference](02_reference/api/MODULE_REFERENCE.md) - Python API (includes `target_utils.py` and `sklearn_safe.py` utilities)
- [Intelligent Trainer API](02_reference/api/INTELLIGENT_TRAINER_API.md) - Intelligent training pipeline API reference
- [CLI Reference](02_reference/api/CLI_REFERENCE.md) - Command-line tools
- [Config Schema](02_reference/api/CONFIG_SCHEMA.md) - Configuration schema

### Project Reference
- [Changelog Index](02_reference/changelog/README.md) - Per-day detailed changelogs with file paths and config references

### Data Reference
- [Data Format Spec](02_reference/data/DATA_FORMAT_SPEC.md) - Data formats
- [Column Reference](02_reference/data/COLUMN_REFERENCE.md) - Column documentation
- [Data Sanity Rules](02_reference/data/DATA_SANITY_RULES.md) - Validation rules

### Models Reference
- [Model Catalog](02_reference/models/MODEL_CATALOG.md) - All available models
- [Model Config Reference](02_reference/models/MODEL_CONFIG_REFERENCE.md) - Model configurations
- [Training Parameters](02_reference/models/TRAINING_PARAMETERS.md) - Training settings

### Systems Reference
- [Pipeline Reference](02_reference/systems/PIPELINE_REFERENCE.md) - Data pipelines
- [Feature Importance Stability](03_technical/implementation/FEATURE_IMPORTANCE_STABILITY.md) - Feature importance stability tracking and analysis system
- [Feature Filtering Execution Order](03_technical/implementation/FEATURE_FILTERING_EXECUTION_ORDER.md) - Formalized feature filtering hierarchy
- [Feature Selection Execution Order](03_technical/implementation/FEATURE_SELECTION_EXECUTION_ORDER.md) - Formalized feature selection hierarchy
- [Feature Pruning Execution Order](03_technical/implementation/FEATURE_PRUNING_EXECUTION_ORDER.md) - Formalized feature pruning hierarchy
- [Data Loading and Preprocessing Execution Order](03_technical/implementation/DATA_LOADING_PREPROCESSING_EXECUTION_ORDER.md) - Formalized data pipeline hierarchy
- [Parallel Execution](03_technical/implementation/PARALLEL_EXECUTION.md) - Parallel execution infrastructure for target ranking and feature selection
- [Active Sanitization (Ghost Buster)](03_technical/implementation/ACTIVE_SANITIZATION.md) - **NEW**: Proactive feature quarantine system that prevents "ghost feature" discrepancies

### Decision-Making System ⚠️ EXPERIMENTAL (2025-12-12)
- [Decision Engine Guide](03_technical/implementation/decisioning/DECISION_ENGINE.md) - **NEW**: Complete guide to automated decision-making, policies, and requirements
- [Bayesian Policy Guide](03_technical/implementation/decisioning/BAYESIAN_POLICY.md) - **NEW**: Thompson sampling over discrete patch templates for adaptive config tuning
- [Verification Checklist](03_technical/implementation/decisioning/VERIFICATION_CHECKLIST.md) - **NEW**: How to verify decision application works correctly
- **Status**: Highly experimental, under active testing. See [TESTING_NOTICE.md](../TESTING_NOTICE.md) for details.
- [Training Routing System](02_reference/training_routing/README.md) - **NEW**: Config-driven routing decisions for cross-sectional vs symbol-specific training with automatic plan generation, 2-stage training (CPU→GPU), and one-command end-to-end pipeline

### API Reference
- [Module Reference](02_reference/api/MODULE_REFERENCE.md) - Python API for all modules
- [DATA_PROCESSING API](02_reference/api/DATA_PROCESSING_API.md) - Complete DATA_PROCESSING module API reference
- [Intelligent Trainer API](02_reference/api/INTELLIGENT_TRAINER_API.md) - Intelligent training pipeline API
- [CLI Reference](02_reference/api/CLI_REFERENCE.md) - Command-line interface reference

### Configuration Reference
- **[Modular Config System](02_reference/configuration/MODULAR_CONFIG_SYSTEM.md)** - Complete guide to modular configs, experiment configs, typed configs, migration (includes `logging_config.yaml`)
- [Configuration System Overview](02_reference/configuration/README.md) - Centralized configuration system overview (includes `logging_config.yaml` documentation)
- [Feature & Target Configs](02_reference/configuration/FEATURE_TARGET_CONFIGS.md) - Feature/target configuration guide
- [Training Pipeline Configs](02_reference/configuration/TRAINING_PIPELINE_CONFIGS.md) - System resources and training behavior
- [Training Routing Config](02_reference/training_routing/README.md) - **NEW**: Routing policy configuration (`routing_config.yaml`) for cross-sectional vs symbol-specific training decisions. Includes 2-stage training pipeline (CPU models first, then GPU models) and one-command end-to-end flow.
- [Safety & Leakage Configs](02_reference/configuration/SAFETY_LEAKAGE_CONFIGS.md) - Leakage detection and numerical stability
- [Model Configuration](02_reference/configuration/MODEL_CONFIGURATION.md) - Model hyperparameters and variants
- [Usage Examples](02_reference/configuration/USAGE_EXAMPLES.md) - Practical configuration examples (includes interval config and CatBoost examples)
- [Config Loader API](02_reference/configuration/CONFIG_LOADER_API.md) - Programmatic config loading (includes logging config utilities)
- [Config Cleaner API](02_reference/configuration/CONFIG_CLEANER_API.md) - **NEW**: Systematic parameter validation to prevent duplicate/unknown parameter errors
- [Config Overlays](02_reference/configuration/CONFIG_OVERLAYS.md) - Overlay system for environment-specific configs
- [Environment Variables](02_reference/configuration/ENVIRONMENT_VARIABLES.md) - Environment-based configuration
- [Config Migration](02_reference/configuration/migration/README.md) - Configuration migration and consolidation documentation

## Tier D: Deep Technical Appendices

Research notes, design rationale, advanced topics. See [03_technical/README.md](03_technical/README.md) for complete contents.

### Research
- [Intelligence Layer Overview](03_technical/research/INTELLIGENCE_LAYER.md) - Complete overview of intelligent training pipeline decision-making and automation
- [Leakage Analysis](03_technical/research/LEAKAGE_ANALYSIS.md) - Leakage research
- [Leakage Controls Evaluation](03_technical/architecture/LEAKAGE_CONTROLS_EVALUATION.md) - Leakage controls structural evaluation
- [SST Enforcement Design](../../TRAINING/utils/SST_ENFORCEMENT_DESIGN.md) - **NEW**: EnforcedFeatureSet contract eliminates split-brain across all training paths
- [Leakage Canary Test Guide](03_technical/testing/LEAKAGE_CANARY_TEST_GUIDE.md) - Pipeline integrity validation
- [Feature Importance Methodology](03_technical/research/FEATURE_IMPORTANCE_METHODOLOGY.md) - Feature importance research
- [Feature Importance Stability](03_technical/implementation/FEATURE_IMPORTANCE_STABILITY.md) - Feature importance stability tracking and analysis (see also Systems Reference)
- [Target Discovery](03_technical/research/TARGET_DISCOVERY.md) - Target research
- [Validation Methodology](03_technical/research/VALIDATION_METHODOLOGY.md) - Validation research

### Design
- [Architecture Deep Dive](03_technical/design/ARCHITECTURE_DEEP_DIVE.md) - System architecture
- [CLI vs Config Separation](03_technical/design/CLI_CONFIG_SEPARATION.md) - **NEW**: Policy for CLI/Config separation, SST compliance
- [Mathematical Foundations](03_technical/internal/MATHEMATICAL_FOUNDATIONS.md) - Math background (internal)
- [Optimization Engine](03_technical/internal/OPTIMIZATION_ENGINE.md) - Optimization design (internal)
- [C++ Integration](03_technical/internal/C++_INTEGRATION.md) - C++ components (internal)

### Benchmarks
- [Performance Metrics](03_technical/benchmarks/PERFORMANCE_METRICS.md) - Performance data
- [Model Comparisons](03_technical/benchmarks/MODEL_COMPARISONS.md) - Model benchmarks
- [Dataset Sizing](03_technical/benchmarks/DATASET_SIZING.md) - Dataset strategies

### Fixes
- [Known Issues](03_technical/fixes/KNOWN_ISSUES.md) - Current issues
- [Bug Fixes](03_technical/fixes/BUG_FIXES.md) - Fix history
- [Migration Notes](03_technical/fixes/MIGRATION_NOTES.md) - Migration guide
- [TensorFlow Executable Stack Fix](03_technical/fixes/TENSORFLOW_EXECUTABLE_STACK_FIX.md) - Fix for libtensorflow_cc.so executable stack error
- **Feature Selection and Config Fixes (2025-12-14)** – **NEW**:
  - [Feature Selection and Config Fixes Changelog](02_reference/changelog/2025-12-14-feature-selection-and-config-fixes.md) - Complete detailed changelog
  - **Status**: ✅ All fixes implemented and tested
  - **Fixes**: UnboundLocalError for np (11 model families), missing import, unpacking error, routing diagnostics, experiment config loading, target exclusion, lookback enforcement
- **Look-Ahead Bias Fixes (2025-12-14)** – **NEW**:
  - [Look-Ahead Bias Fix Plan](03_technical/fixes/LOOKAHEAD_BIAS_FIX_PLAN.md) - Complete analysis of 4 critical look-ahead bias issues and required fixes
  - [Safe Implementation Plan](03_technical/fixes/LOOKAHEAD_BIAS_SAFE_IMPLEMENTATION.md) - Feature flag-based implementation strategy with gradual rollout plan
  - **Status**: ✅ All fixes implemented (behind feature flags, default: OFF)
  - **Fixes**: Rolling windows exclude current bar, CV-based normalization, pct_change verification, feature renaming
- **Feature Selection Critical Fixes (2025-12-13)**:
  - [Implementation Verification](03_technical/fixes/2025-12-13-implementation-verification.md) - Complete verification of all 6 critical checks + 2 last-mile improvements
  - [Critical Fixes](03_technical/fixes/2025-12-13-critical-fixes.md) - Detailed root-cause analysis and fixes
  - [Telemetry Scoping Fix](03_technical/fixes/2025-12-13-telemetry-scoping-fix.md) - Telemetry scoping implementation
  - [Sharp Edges Verification](03_technical/fixes/2025-12-13-sharp-edges-verification.md) - Verification against user checklist

### Roadmaps
- [Alpha Enhancement Roadmap](03_technical/roadmaps/ALPHA_ENHANCEMENT_ROADMAP.md) - Enhancement plan
- [Future Work](03_technical/roadmaps/FUTURE_WORK.md) - Planned features

### Implementation
- [Feature Selection Implementation](03_technical/implementation/FEATURE_SELECTION_GUIDE.md) - Feature selection implementation details (see also [Ranking and Selection Consistency](01_tutorials/training/RANKING_SELECTION_CONSISTENCY.md) for unified pipeline behavior)
- [Training Optimization](03_technical/implementation/TRAINING_OPTIMIZATION_GUIDE.md) - Training optimization guide
- [SST Enforcement Design](../../TRAINING/utils/SST_ENFORCEMENT_DESIGN.md) - **NEW**: EnforcedFeatureSet contract, type boundary wiring, boundary assertions (provably split-brain free across all training paths)
- [SST Implementation Coverage](../../TRAINING/utils/SST_IMPLEMENTATION_COVERAGE.md) - **NEW**: Complete coverage matrix for SST enforcement across all training paths
- [Type Boundary Wiring](../../TRAINING/utils/TYPE_BOUNDARY_WIRING_COMPLETE.md) - **NEW**: Type boundary wiring implementation details
- [Boundary Assertions](../../TRAINING/utils/BOUNDARY_ASSERTIONS_COMPLETE.md) - **NEW**: Boundary assertion implementation details
- [Parallel Execution](03_technical/implementation/PARALLEL_EXECUTION.md) - Parallel execution infrastructure for target ranking and feature selection
- [Reproducibility Tracking](03_technical/implementation/REPRODUCIBILITY_TRACKING.md) - Automatic reproducibility verification across pipeline stages
- [Cohort-Aware Reproducibility](03_technical/implementation/COHORT_AWARE_REPRODUCIBILITY.md) - **NEW**: Cohort-aware reproducibility system with sample-adjusted drift detection
- [Trend Analyzer Verification](03_technical/implementation/TREND_ANALYZER_VERIFICATION.md) - **NEW**: Trend analysis system verification guide (target ranking, feature selection, cross-sectional ranking)
- [Reproducibility Structure](03_technical/implementation/REPRODUCIBILITY_STRUCTURE.md) - Complete directory structure guide with REPRODUCIBILITY organization
- [RESULTS Organization](03_technical/implementation/RESULTS_ORGANIZATION_OPTIONS.md) - **NEW**: RESULTS directory organization by sample size bins (current implementation)
- [Reproducibility API](03_technical/implementation/REPRODUCIBILITY_API.md) - **NEW**: API reference for reproducibility tracking
- [Reproducibility Error Handling](03_technical/implementation/REPRODUCIBILITY_ERROR_HANDLING.md) - **NEW**: Error classification and handling guide
- [Telemetry System](02_reference/changelog/2025-12-14-telemetry-system.md) - **NEW**: Sidecar-based telemetry with view isolation, hierarchical rollups (cohort → view → stage)
- [Reproducibility Improvements](03_technical/implementation/REPRODUCIBILITY_IMPROVEMENTS.md) - **NEW**: Summary of reproducibility improvements
- [Reproducibility Self-Test](03_technical/implementation/REPRODUCIBILITY_SELF_TEST.md) - **NEW**: Self-test checklist for validation
- [Cohort-Aware Implementation](03_technical/implementation/COHORT_AWARE_REPRODUCIBILITY_IMPLEMENTATION.md) - **NEW**: Detailed implementation notes
- [Safe Target Pattern](03_technical/implementation/SAFE_TARGET_PATTERN_IMPLEMENTATION.md) - Safe target pattern implementation
- [First Batch Specs](03_technical/implementation/FIRST_BATCH_SPECS_IMPLEMENTATION.md) - First batch specifications
- [Strategy Updates](03_technical/implementation/STRATEGY_UPDATES.md) - Training strategy updates
- ⚠️ **Legacy**: [Experiments Implementation](../LEGACY/EXPERIMENTS_IMPLEMENTATION.md) - **DEPRECATED**: See [Intelligent Training Tutorial](../01_tutorials/training/INTELLIGENT_TRAINING_TUTORIAL.md) instead
- [Pressure Test Plan](03_technical/implementation/PRESSURE_TEST_PLAN.md) - Testing plan
- [Performance Optimization](03_technical/implementation/PERFORMANCE_OPTIMIZATION.md) - Optimization work
- [Adding Proprietary Models](03_technical/implementation/ADDING_PROPRIETARY_MODELS.md) - Using BaseModelTrainer to add custom models
- [Decision Engine](03_technical/implementation/decisioning/DECISION_ENGINE.md) - ⚠️ **EXPERIMENTAL**: Automated decision-making system with configurable policies
- [Bayesian Policy](03_technical/implementation/decisioning/BAYESIAN_POLICY.md) - ⚠️ **EXPERIMENTAL**: Thompson sampling for adaptive config tuning
- [Decision Verification Checklist](03_technical/implementation/decisioning/VERIFICATION_CHECKLIST.md) - ⚠️ **EXPERIMENTAL**: How to verify decision application works correctly

### Refactoring
- **[Refactoring & Wrappers](01_tutorials/REFACTORING_AND_WRAPPERS.md)** - **NEW**: User-facing guide explaining wrapper mechanism, import patterns, and backward compatibility
- [Refactoring Summary](INTERNAL/REFACTORING_SUMMARY_INTERNAL.md) - Internal technical details of large file refactoring (2025-12-09)
- [Specialized Models Module](03_technical/refactoring/SPECIALIZED_MODELS.md) - Detailed documentation for `models/specialized/` module
- [Target Predictability Ranking Module](03_technical/refactoring/TARGET_PREDICTABILITY_RANKING.md) - Detailed documentation for `ranking/predictability/` module
- [Training Strategies Module](03_technical/refactoring/TRAINING_STRATEGIES.md) - Detailed documentation for `training_strategies/` module

### Legal & Compliance (Internal)

### Testing
- [Testing Plan](03_technical/testing/TESTING_PLAN.md) - Test strategy
- [Testing Summary](03_technical/testing/TESTING_SUMMARY.md) - Test results
- [Daily Testing](03_technical/testing/DAILY_TESTING.md) - Daily test procedures

### Operations
- ⚠️ **Legacy**: [Training Pipeline Debugging Status](../LEGACY/STATUS_DEBUGGING.md) - **OUTDATED**: Last updated 2025-12-09, issues resolved. See [CHANGELOG.md](../CHANGELOG.md) for current status
- [Journald Logging](03_technical/operations/JOURNALD_LOGGING.md) - Logging setup
- [Restore from Logs](03_technical/operations/RESTORE_FROM_LOGS.md) - Recovery procedures
- [Avoid Long Runs](03_technical/operations/AVOID_LONG_RUNS.md) - Performance tips
- [Systemd Deployment](03_technical/operations/SYSTEMD_DEPLOYMENT.md) - Deployment guide

## Additional Documentation

### System Specifications
- See [Ranking and Selection Consistency](01_tutorials/training/RANKING_SELECTION_CONSISTENCY.md) for unified pipeline behavior
- See [Feature Selection Implementation](03_technical/implementation/FEATURE_SELECTION_GUIDE.md) for implementation details

### System Documentation
- Trading integration modules have been removed from the core repository. The system focuses on ML research infrastructure and model training.

## Documentation Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the documentation structure and maintenance policy.
