# Changelog

All notable changes to FoxML Core will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

**For more detailed information:**
- [Detailed Changelog](DOCS/02_reference/CHANGELOG_DETAILED.md) - Comprehensive change details with file paths and config references

---

## [Unreleased]

### Highlights

- **Systematic Config Parameter Validation System** (2025-12-11) — **NEW**: Created shared `TRAINING/utils/config_cleaner.py` utility to systematically prevent parameter passing errors across all model constructors. Uses `inspect.signature()` to validate parameters against actual estimator signatures, automatically removing duplicate and unknown parameters. Prevents entire class of "got multiple values for keyword argument" and "unexpected keyword argument" errors. Integrated into all model instantiation paths (multi-model feature selection, task types, cross-sectional ranking). Makes codebase resilient to future config drift from `inject_defaults`. Maintains SST (Single Source of Truth) while ensuring only valid parameters reach constructors.
- **Reproducibility Tracking Module** (2025-12-11) — **NEW**: Extracted reproducibility tracking into reusable `TRAINING/utils/reproducibility_tracker.py` module. Integrated into both target ranking and feature selection pipelines. Provides automatic comparison of run results with tolerance-based verification. Comprehensive documentation added in `DOCS/03_technical/implementation/REPRODUCIBILITY_TRACKING.md`. Generic design allows easy integration into any pipeline stage. Shows ✅ for reproducible runs (within 0.1% tolerance) and ⚠️ for differences.
- **Model Config Parameter Sanitization Fix** (2025-12-11) — **FIXED**: Resolved critical TypeError and ValueError errors affecting 7 model families (RandomForest, MLPRegressor, Lasso, CatBoost, XGBoost, LightGBM) when global config defaults were injected. All models now sanitize configs before instantiation, removing incompatible parameters (`random_seed` → `random_state` for sklearn, `n_jobs` → `thread_count` for CatBoost, early stopping params for XGBoost/LightGBM). Determinism preserved with explicit per-symbol/target seed setting.
- **Feature Importance Stability Tracking System** (2025-12-10) — **NEW**: Comprehensive system for tracking and analyzing feature importance stability across pipeline runs. Automatically captures snapshots from all integration points (target ranking, feature selection, quick pruning). Config-driven automation with stability metrics (top-K overlap, Kendall tau, selection frequency). Includes CLI tool for manual analysis and comprehensive documentation.
- **Auto-Fixer Backup Fix** (2025-12-10) — Fixed critical bug where auto-fixer was not creating backups when no leaks were detected. Backups are now created whenever auto-fix mode is triggered, preserving state history for debugging. Added comprehensive observability logging to auto-fixer initialization and detection.
- **Reproducibility Settings Centralization** (2025-12-10) — Centralized all reproducibility-critical settings (`random_state`, `shuffle`, validation splits) to Single Source of Truth. Removed 30+ hardcoded `random_state: 42` values across configs. All models now use `pipeline.determinism.base_seed` for consistent reproducibility.
- **Auto-Fixer Training Accuracy Fix** (2025-12-10) — Fixed critical bug where training accuracy was calculated but not stored in `model_metrics`, preventing auto-fixer from triggering on 100% training accuracy. Auto-fixer now correctly detects and creates backups when leakage is detected.
- **Silent Failures Fixed** (2025-12-10) — Added warnings for all silent config loading failures. Fixed YAML `None` return handling. Defaults injection now logs warnings when `defaults.yaml` is missing/broken. Random state fallback now logs warnings.
- **SST Enforcement & Complete Config Centralization** (2025-12-10) — All hardcoded configuration values across the entire TRAINING pipeline moved to YAML files. Automated SST enforcement test prevents hardcoded hyperparameters. Same config → same results across all pipeline stages.
- **Full Determinism** (2025-12-10) — All `random_state` values use centralized determinism system (`BASE_SEED`). Training strategies, feature selection, data splits, and model initializations are fully deterministic and reproducible.
- **Complete F821 Error Elimination** (2025-12-10) — Fixed all 194 undefined name errors across TRAINING and CONFIG directories. All files now pass Ruff F821 checks.
- **Pipeline Robustness** (2025-12-10) — Fixed critical syntax errors, variable initialization issues, and missing imports. Full end-to-end testing currently underway.
- **Large File Refactoring** (2025-12-09) — Split 3 monolithic files into modular components while maintaining 100% backward compatibility
- **Intelligent Training Framework** — Phase 1 completed with target ranking, feature selection, and model training pipelines unified
- **Leakage Safety Suite** — Production-grade auto-fixer with backup system, schema/registry validation
- **Modular Configuration** — Centralized YAML-based config system with typed schemas and validation

---

### Stability Guarantees

- **Training results reproducible** across hardware (deterministic seeds, config-driven hyperparameters)
- **Complete config centralization** — All pipeline parameters load from YAML files (single source of truth)
- **SST enforcement** — Automated test prevents hardcoded hyperparameters
- **Config schema backward compatible** (existing configs continue to work)
- **Modular architecture** (self-contained TRAINING module, zero external script dependencies)

### Known Issues & Limitations

- **Trading execution modules** removed from core repository; system focuses on ML research infrastructure
- **Feature engineering** requires human review and validation
- **End-to-end testing in progress** (2025-12-10) — Full pipeline validation underway after SST and Determinism fixes

---

### Added

- **CLI/Config separation policy** (2025-12-11) — New policy document defining CLI vs Config separation. CLI should only provide inputs, config overrides, and operational flags. All configuration values come from config files (SST compliant). See [CLI vs Config Separation](DOCS/03_technical/design/CLI_CONFIG_SEPARATION.md).
- **Intelligent training config section** (2025-12-11) — Added `intelligent_training` section to `pipeline_config.yaml` with all settings (auto_targets, top_n_targets, top_m_features, strategy, data limits, etc.). All settings now configurable via YAML instead of CLI.
- **Config cleaner utility** (2025-12-11) — New `TRAINING/utils/config_cleaner.py` module providing systematic parameter validation using `inspect.signature()` to prevent parameter passing errors. Integrated into all model instantiation paths. See [Detailed Changelog](DOCS/02_reference/CHANGELOG_DETAILED.md) for full details.
- **Reproducibility tracking module** (2025-12-11) — Reusable `ReproducibilityTracker` class for automatic reproducibility verification across pipeline stages. Compares runs with tolerance-based verification and flags reproducible vs different runs. See [Detailed Changelog](DOCS/02_reference/CHANGELOG_DETAILED.md) for full details.
- **Feature Importance Stability Tracking System** (2025-12-10) — Comprehensive system for tracking and analyzing feature importance stability across pipeline runs with automatic snapshot capture and config-driven automation.
- **Observability improvements** (2025-12-10) — Enhanced logging for auto-fixer, config loading, and defaults injection.
- **Config centralization** (2025-12-10) — All pipeline parameters now load from YAML files with single source of truth. SST enforcement system prevents hardcoded hyperparameters.
- **Target confidence & routing system** — Automatic quality assessment with configurable thresholds
- **Cross-sectional feature ranking** — Optional panel model for universe-level feature importance
- **Modular configuration system** — Typed schemas, experiment configs, validation
- **Leakage Safety Suite** — Production-grade backup system, automated detection and auto-fix

### Fixed

- **CLI/Config separation** (2025-12-11) — Enforced SST compliance by moving CLI settings to config files. Removed 15+ config-related CLI arguments from `intelligent_trainer.py`. CLI now only provides: required inputs (data-dir, symbols), config overrides (experiment-config), and operational flags (resume, force-refresh). All settings load from config files. See [CLI vs Config Separation](DOCS/03_technical/design/CLI_CONFIG_SEPARATION.md) for policy.
- **Silent error visibility** (2025-12-11) — Added comprehensive logging to all previously silent failure paths: ImportError fallbacks, importance extraction validation (None/type/length checks), bare except clauses, config loading failures, and empty results aggregation. All failures now have appropriate logging levels. See [Detailed Changelog](DOCS/02_reference/CHANGELOG_DETAILED.md) for full details.
- **Parameter passing errors** (2025-12-11) — Systematic fix using shared config cleaner utility prevents "got multiple values" and "unexpected keyword argument" errors across all model families. All model constructors now validate parameters against estimator signatures. See [Detailed Changelog](DOCS/02_reference/CHANGELOG_DETAILED.md) for full details.
- **Model config sanitization** (2025-12-11) — Fixed TypeError/ValueError errors when global defaults were injected. All models now sanitize configs before instantiation, removing incompatible parameters while preserving determinism.
- **Auto-fixer issues** (2025-12-10) — Fixed backup creation bug and training accuracy detection bug
- **Config loading** (2025-12-10) — Added warnings for all silent config loading failures, fixed YAML None handling
- **SST implementation** (2025-12-10) — Replaced all hardcoded values across entire TRAINING pipeline
- **Code quality** (2025-12-10) — Fixed all 194 F821 errors, missing imports, syntax errors

### Changed

- **Reproducibility settings** (2025-12-10) — Removed 30+ hardcoded `random_state: 42` values. All now use centralized determinism system.
- **Config cleanup** (2025-12-10) — Removed ~35+ duplicate default values. All now auto-injected from `defaults.yaml`.
- **Internal documentation** (2025-12-10) — Moved all internal docs to `INTERNAL_DOCS/` (never tracked). Cleaned up `CONFIG/` directory by removing internal audit/verification docs.
- **Config loading patterns** (2025-12-10) — All function parameters now load from config instead of hardcoded defaults
- **Logging system** — Replaced hardcoded flags with structured YAML configuration

---

### Security

- Enhanced compliance documentation for production use
- License enforcement procedures documented
- Copyright notice requirements standardized

### Documentation

- Documentation restructured into 4-tier hierarchy
- 55+ new documentation files created, 50+ existing files rewritten
- Comprehensive cross-linking and navigation improvements

---

## Versioning

Releases follow [Semantic Versioning](https://semver.org/):
- **MAJOR** version for incompatible API changes
- **MINOR** version for functionality added in a backwards-compatible manner
- **PATCH** version for backwards-compatible bug fixes

## Categories

- **Added** – New features
- **Changed** – Changes in existing functionality
- **Fixed** – Bug fixes
- **Security** – Security improvements
- **Documentation** – Documentation changes
