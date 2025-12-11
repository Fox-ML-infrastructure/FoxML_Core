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

- **Config cleaner utility** (2025-12-11) — New `TRAINING/utils/config_cleaner.py` module providing systematic parameter validation:
  - `clean_config_for_estimator()` function uses `inspect.signature()` to validate parameters against actual estimator constructors
  - Automatically removes duplicate parameters (if also passed explicitly via `extra_kwargs`)
  - Automatically removes unknown parameters (not in estimator's `__init__` signature)
  - Logs what was stripped for visibility (DEBUG level, can be raised to WARNING for auditing)
  - Handles edge cases: None configs, non-dict configs, None extra_kwargs
  - Maintains SST (Single Source of Truth) - values still come from config/defaults, but only valid keys are passed
  - Prevents entire class of parameter passing errors that would occur with config drift
  - Integrated into: `multi_model_feature_selection.py`, `task_types.py`, `cross_sectional_feature_ranker.py`
  - Makes codebase resilient to future changes in `inject_defaults` or model library updates
- **Reproducibility tracking module** (2025-12-11) — Reusable `ReproducibilityTracker` class for automatic reproducibility verification:
  - Generic module (`TRAINING/utils/reproducibility_tracker.py`) usable across all pipeline stages
  - Integrated into target ranking and feature selection pipelines
  - Compares current run to previous runs with configurable tolerances (0.1% for scores, 1% for importance)
  - Stores run history in JSON format (keeps last 10 runs per item by default)
  - Logs differences with percentage changes and absolute deltas
  - Flags reproducible runs (✅) vs different runs (⚠️)
  - Comprehensive documentation: `DOCS/03_technical/implementation/REPRODUCIBILITY_TRACKING.md`
  - Includes API reference, integration examples, best practices, and troubleshooting guide
- **Reproducibility comparison logging** (2025-12-11) — Automatic comparison of target ranking results to previous runs:
  - Stores run summaries in `reproducibility_log.json` (keeps last 10 runs per target)
  - Compares current vs previous: mean score, std score, importance, composite score
  - Logs differences with percentage changes and absolute deltas
  - Flags reproducible runs (✅) vs different runs (⚠️) with configurable tolerance (0.1% for scores, 1% for importance)
  - Helps verify deterministic behavior and catch non-reproducibility issues early
  - Integrated into `rank_target_predictability.py` - logs after each target evaluation summary
- **Feature Importance Stability Tracking System** (2025-12-10) — New `TRAINING/stability/feature_importance/` module with:
  - Automatic snapshot capture at all integration points (target ranking, feature selection, quick pruning)
  - Config-driven automation (`safety.feature_importance.auto_analyze_stability`)
  - Stability metrics: top-K overlap (Jaccard similarity), Kendall tau (rank correlation), selection frequency
  - CLI tool: `scripts/analyze_importance_stability.py` for manual analysis
  - Comprehensive documentation: `TRAINING/stability/FEATURE_IMPORTANCE_STABILITY.md`
  - Non-invasive hooks that don't break pipeline on failures
  - Standardized JSON snapshot format for easy analysis
- **Observability logging** (2025-12-10) — Added comprehensive initialization and operation logging to auto-fixer, config loading, and defaults injection. Now logs: auto-fixer initialization (paths, settings), detection results with confidence scores, defaults injection details, and config loading status.
- **Reproducibility settings centralization** (2025-12-10) — Created `CONFIG/defaults.yaml` with centralized `random_state`, `shuffle`, `validation_split` settings. All models inherit from Single Source of Truth.
- **Internal documentation organization** (2025-12-10) — Unified all internal docs into `INTERNAL_DOCS/` directory (never tracked). Moved audit trails, refactoring notes, and internal verification docs.
- **Config defaults system** (2025-12-10) — New `inject_defaults()` function automatically applies common defaults (dropout, activation, patience, n_jobs, etc.) to model configs unless explicitly overridden.
- **Complete SST config system** (2025-12-10) — All 52+ model files use centralized config for hyperparameters, test splits, and random seeds
- **Config centralization** (2025-12-10) — New config sections: `leakage_sentinels.*`, `auto_fixer.*`, `feature_pruning.*`
- **Determinism integration** (2025-12-10) — All training strategies and model initializations use `BASE_SEED`
- **Target confidence & routing system** — Automatic quality assessment with configurable thresholds and operational buckets
- **Cross-sectional feature ranking** — Optional panel model for universe-level feature importance
- **Modular configuration system** — Typed schemas, experiment configs, validation
- **Structured logging configuration** — Per-module verbosity controls via YAML
- **Leakage Safety Suite** — Production-grade backup system, automated detection and auto-fix

### Fixed

- **Silent error visibility improvements** (2025-12-11) — Enhanced observability by adding logging to all previously silent failure paths:
  - **ImportError fallback in target validation**: Added DEBUG logging when `validate_target` module is unavailable and fallback validation is used
  - **Importance extraction validation**: Added comprehensive validation and logging for importance extraction failures:
    - Validates importance is not None before use
    - Validates importance is correct type (pd.Series) and converts if needed
    - Validates importance length matches feature count and pads/truncates with warnings
    - Wraps entire extraction in try/except with ERROR-level logging on failures
  - **Bare except clauses**: Replaced all bare `except:` clauses with `except Exception as e:` and added DEBUG-level logging:
    - Stability selection bootstrap loop now logs when iterations fail
    - RFE score computation now logs when scoring fails
  - **Reproducibility tracker visibility**: Fixed issue where reproducibility comparison logs were not visible in main script output by ensuring logger propagation and fallback to main logger
  - **Config loading failures**: Added DEBUG-level logging to all config loading exception handlers (SHAP config, RFE config, Boruta config, max_samples, etc.) that previously silently fell back to defaults
  - **Empty results aggregation**: Added WARNING-level log when all model families fail and empty results are returned
  - All silent failures now have appropriate logging levels (DEBUG for expected fallbacks, WARNING for unexpected conditions, ERROR for actual failures)
- **Systematic parameter passing error prevention** (2025-12-11) — Implemented systematic fix for parameter passing errors using shared config cleaner utility:
  - **All model families in multi_model_feature_selection.py**: LightGBM, XGBoost, RandomForest, MLPRegressor, CatBoost, Lasso now use `clean_config_for_estimator()` helper
  - **All model constructors in task_types.py**: Fixed with proper closure handling (explicit copies in lambda closures to prevent reference issues)
  - **Cross-sectional feature ranker**: LightGBM and XGBoost now use config cleaner
  - **inject_defaults hardening**: Now handles None configs gracefully, prevents "argument of type 'NoneType' is not iterable" errors
  - Prevents "got multiple values for keyword argument" errors (e.g., `random_seed` passed both in config and explicitly)
  - Prevents "unexpected keyword argument" errors (e.g., `num_threads` for RandomForest, `n_jobs` for MLPRegressor/Lasso)
  - Lambda closure fix: All lambda functions in `task_types.py` now capture explicit copies (`config_final`, `extra_final`) to prevent reference issues
  - Future-proof: Any new parameters added by `inject_defaults` will be automatically filtered if they're duplicates or unknown
- **Model config parameter sanitization** (2025-12-11) — Fixed critical TypeError and ValueError errors when global config defaults (`random_seed`, `n_jobs`, `early_stopping_rounds`) were injected into model constructors. All model families now sanitize configs before instantiation:
  - **sklearn models** (RandomForest, MLPRegressor, Lasso): Remove `random_seed` (use `random_state` instead)
  - **CatBoost**: Remove `n_jobs` (uses `thread_count` instead)
  - **XGBoost/LightGBM**: Remove all early stopping params (`early_stopping_rounds`, `callbacks`, `eval_set`, `eval_metric`) in feature selection mode (requires `eval_set` which isn't available)
  - Determinism preserved: All models explicitly set `random_state`/`random_seed` using deterministic `model_seed` per symbol/target combination
  - Uses `.copy()` and `.pop()` for explicit parameter sanitization to prevent incompatible parameters from reaching model constructors
- **Auto-fixer backup creation** (2025-12-10) — Fixed critical bug where backups were not created when auto-fix mode triggered but no leaks were detected. Backups are now created whenever auto-fix mode is enabled, preserving state history for debugging.
- **Auto-fixer training accuracy detection** (2025-12-10) — Fixed critical bug where 100% training accuracy was logged but not stored in `model_metrics`, preventing auto-fixer from triggering. Now stores `training_accuracy` and `training_r2` in `model_metrics` for proper leakage detection.
- **Silent config loading failures** (2025-12-10) — Added warnings when `defaults.yaml` is missing/broken, when `pipeline_config.yaml` can't be loaded, and when YAML files return `None`. All silent failures now log warnings.
- **Empty directory cleanup** (2025-12-10) — Removed sloppy empty directories (`CONFIG/data/`, `CONFIG/leakage/`, `CONFIG/system/`). Added `.gitkeep` to `CONFIG/backups/` for structure preservation.
- **Complete SST implementation** (2025-12-10) — Replaced ALL hardcoded values across entire TRAINING pipeline (52+ model files)
- **Config loading robustness** (2025-12-10) — Fixed syntax errors and variable initialization issues in config loading patterns
- **Complete F821 error elimination** (2025-12-10) — Fixed all 194 undefined name errors (missing imports, logger initialization, circular imports)
- **Missing imports** (2025-12-10) — Fixed `NameError: name 'pl' is not defined` and other import issues
- **Feature list validation** (2025-12-10) — Added robust validation for `selected_features` parameter
- **Import and syntax fixes** (2025-12-10) — Fixed `__future__` import placement, removed circular imports
- **Large file refactoring** (2025-12-09) — Split 3 monolithic files into modular components
- **Model family status tracking** — Comprehensive debugging for multi-model feature selection
- **Interval detection robustness** — Fixed timestamp gap filtering to ignore outliers
- **Feature selection pipeline** — Boruta errors, CatBoost loss function, sklearn NaN/dtype handling

### Changed

- **Reproducibility settings** (2025-12-10) — Removed 30+ hardcoded `random_state: 42` values from configs. All now use centralized `defaults.randomness.random_state` → `pipeline.determinism.base_seed`. Centralized `shuffle: true` setting for train/test splits.
- **Config cleanup** (2025-12-10) — Removed ~35+ duplicate default values from individual config files (dropout, activation, patience, aggregation settings, output settings). All now auto-injected from `defaults.yaml`.
- **Internal documentation** (2025-12-10) — Moved all internal docs to `INTERNAL_DOCS/` (never tracked). Cleaned up `CONFIG/` directory by removing internal audit/verification docs.
- **Config loading patterns** (2025-12-10) — All function parameters with hardcoded defaults now use `Optional[Type] = None` and load from config
- **Determinism system** (2025-12-10) — All `random_state=42` hardcoded values replaced with `BASE_SEED`
- **Logging system refactored** — Replaced hardcoded flags with structured YAML configuration
- **Leakage filtering** — Supports ranking mode with permissive rules, strict rules for training

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
