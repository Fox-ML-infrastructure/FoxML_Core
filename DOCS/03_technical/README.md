# Technical Documentation

Deep technical appendices, research notes, design rationale, and advanced topics.

## Purpose

This directory contains detailed technical documentation for developers, researchers, and system architects. These documents cover implementation details, design decisions, research findings, and advanced topics.

## Contents

### Research
- **[Intelligence Layer Overview](research/INTELLIGENCE_LAYER.md)** - Intelligent training pipeline decision-making
- **[Feature Importance Methodology](research/FEATURE_IMPORTANCE_METHODOLOGY.md)** - Feature importance research
- **[Target Discovery](research/TARGET_DISCOVERY.md)** - Target research and discovery
- **[Validation Methodology](research/VALIDATION_METHODOLOGY.md)** - Validation research

### Architecture
- **[Leakage Controls Evaluation](architecture/LEAKAGE_CONTROLS_EVALUATION.md)** - Leakage controls structural evaluation and fixes
- See [architecture/](architecture/) for complete list

### Implementation
- **[Feature Selection Guide](implementation/FEATURE_SELECTION_GUIDE.md)** - Feature selection implementation
- **[Reproducibility Tracking](implementation/REPRODUCIBILITY_TRACKING.md)** - Reproducibility system
- **[Telemetry System](../02_reference/changelog/2025-12-14-telemetry-system.md)** - **NEW**: Sidecar-based telemetry with view isolation, hierarchical rollups
- **[Active Sanitization](implementation/ACTIVE_SANITIZATION.md)** - Proactive feature quarantine
- **[Parallel Execution](implementation/PARALLEL_EXECUTION.md)** - Parallel execution infrastructure
- **[Decision Engine](implementation/decisioning/DECISION_ENGINE.md)** - ⚠️ EXPERIMENTAL: Automated decision-making
- **[Bayesian Policy](implementation/decisioning/BAYESIAN_POLICY.md)** - ⚠️ EXPERIMENTAL: Adaptive config tuning
- **[Training Utils Documentation](implementation/training_utils/)** - Implementation details for training utilities (SST enforcement, duration system, lookback handling, etc.)
- See [implementation/](implementation/) for complete list

### Design
- **[Architecture Deep Dive](design/ARCHITECTURE_DEEP_DIVE.md)** - System architecture details
- **[CLI vs Config Separation](design/CLI_CONFIG_SEPARATION.md)** - CLI/Config separation policy
- See [design/](design/) for complete list

### Benchmarks
- **[Performance Metrics](benchmarks/PERFORMANCE_METRICS.md)** - Performance data
- **[Model Comparisons](benchmarks/MODEL_COMPARISONS.md)** - Model benchmarks
- **[Dataset Sizing](benchmarks/DATASET_SIZING.md)** - Dataset strategies

### Internal Documentation
- **[Data Overlap Analysis](internal/DATA_OVERLAP_ANALYSIS.md)** - Analysis of data overlap between target_rankings and REPRODUCIBILITY
- **[Reorganization Plan](internal/REORGANIZATION_PLAN.md)** - Plan for reorganizing results directory structure
- **[Reorganization Issues Check](internal/REORGANIZATION_ISSUES_CHECK.md)** - Verification of reorganization changes
- See [internal/](internal/) for complete list

### Fixes
- **[Known Issues](fixes/KNOWN_ISSUES.md)** - Current issues and limitations
- **[Migration Notes](fixes/MIGRATION_NOTES.md)** - Migration guide
- **Feature Selection and Config Fixes (2025-12-14)** – **NEW**:
  - **[Feature Selection and Config Fixes Changelog](../02_reference/changelog/2025-12-14-feature-selection-and-config-fixes.md)** - Complete detailed changelog
  - **Status**: ✅ All fixes implemented and tested
  - **Fixes**: UnboundLocalError for np (11 model families), missing import, unpacking error, routing diagnostics, experiment config loading, target exclusion, lookback enforcement
- **Look-Ahead Bias Fixes (2025-12-14)** – **NEW**:
  - **Status**: ✅ All fixes implemented (behind feature flags, default: OFF)
- **[SST Enforcement Design](fixes/2025-12-13-sst-enforcement-design.md)** - EnforcedFeatureSet contract, type boundary wiring, boundary assertions
- **[Leakage Controls Fixes](fixes/2025-12-13-leakage-validation-fix.md)** - Leakage validation and fingerprint tracking fixes
- **[Fingerprint Tracking](fixes/2025-12-13-lookback-fingerprint-tracking.md)** - Lookback fingerprint tracking system
- **[Fingerprint Improvements](fixes/2025-12-13-fingerprint-improvements.md)** - Set-invariant fingerprints and dataclass improvements
- See [fixes/](fixes/) for complete list

### Testing
- **[Testing Plan](testing/TESTING_PLAN.md)** - Test strategy
- **[Testing Summary](testing/TESTING_SUMMARY.md)** - Test results
- **[Daily Testing](testing/DAILY_TESTING.md)** - Daily test procedures
- **[Leakage Canary Test Guide](testing/LEAKAGE_CANARY_TEST_GUIDE.md)** - Pipeline integrity validation using canary targets

### Operations
- **[Journald Logging](operations/JOURNALD_LOGGING.md)** - Logging setup
- **[Systemd Deployment](operations/SYSTEMD_DEPLOYMENT.md)** - Deployment guide
- **[Restore from Logs](operations/RESTORE_FROM_LOGS.md)** - Recovery procedures
- See [operations/](operations/) for complete list

### Refactoring
- **[Specialized Models Module](refactoring/SPECIALIZED_MODELS.md)** - `models/specialized/` documentation
- **[Target Predictability Ranking](refactoring/TARGET_PREDICTABILITY_RANKING.md)** - `ranking/predictability/` documentation
- **[Training Strategies](refactoring/TRAINING_STRATEGIES.md)** - `training_strategies/` documentation

### Roadmaps
- **[Alpha Enhancement Roadmap](roadmaps/ALPHA_ENHANCEMENT_ROADMAP.md)** - Enhancement plan
- **[Future Work](roadmaps/FUTURE_WORK.md)** - Planned features

### Internal
- **[Mathematical Foundations](internal/MATHEMATICAL_FOUNDATIONS.md)** - Math background
- **[C++ Integration](internal/C++_INTEGRATION.md)** - C++ components
- **[SST Compliance](internal/SST_COMPLIANCE_CHECKLIST.md)** - SST compliance checklist
- See [internal/](internal/) for complete list

## Who Should Read This

- **Developers** - Implementation, Design, Refactoring
- **Researchers** - Research, Benchmarks
- **System Architects** - Design, Architecture Deep Dive
- **QA/Testing** - Testing, Fixes

## Related Documentation

- [Reference Documentation](../02_reference/) - API and configuration reference
- [Tutorials](../01_tutorials/) - Step-by-step guides
- [Executive Documentation](../00_executive/) - High-level overviews

