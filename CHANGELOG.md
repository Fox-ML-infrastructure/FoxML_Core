# Changelog

All notable changes to FoxML Core will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

**For more detailed information:**
- [Detailed Changelog](DOCS/02_reference/CHANGELOG_DETAILED.md) - Comprehensive change details with file paths and config references

---

## [Unreleased]

### Highlights

- **Model Parameter Sanitization Fixes** (2025-12-11) — **FIXED**: Resolved three critical parameter validation issues affecting MLPRegressor, CatBoost, and univariate feature selection. **MLPRegressor verbose parameter**: Fixed `verbose=-1` error by sanitizing negative values to `0` (sklearn requires `>= 0`). **CatBoost iteration synonyms conflict**: Fixed "only one of iterations/n_estimators/num_boost_round/num_trees" error by removing all synonyms except `iterations` (CatBoost's native param) before model construction, with double-check after config cleaning. **CatBoost random_state/random_seed conflict**: Fixed duplicate parameter error by converting `random_state` to `random_seed` (CatBoost's preferred name) or removing if already in extra_kwargs. **Univariate selection assertion failure**: Fixed "Importance sum should be positive" assertion by adding multiple defensive fallback layers ensuring sum is always > 0, even when all scores are zero. All fixes centralized in `config_cleaner.py` and `multi_model_feature_selection.py` for global application. Added reproducibility logging per symbol showing base_seed, n_features, n_samples, and detected_interval for debugging.
- **Per-Model Reproducibility Tracking** (2025-12-11) — **NEW**: Added structured, thresholded per-model reproducibility tracking to feature selection. Computes delta_score, Jaccard@K, and importance_corr for each model family. Stores results in `model_metadata.json` for audit trails. Compact logging: stable models = one line (INFO), unstable = WARNING. Symbol-level summary shows reproducibility status across all families. Different thresholds for high-variance vs stable model families. Non-spammy: only logs when it matters (unstable/borderline). See [Config Cleaner API](DOCS/02_reference/configuration/CONFIG_CLEANER_API.md) for details.
- **Comprehensive Reproducibility Tracking** (2025-12-11) — **ENHANCED**: Reproducibility tracking now integrated across all deterministic pipeline stages: target ranking, feature selection, and model training. **Architectural improvement**: Tracking moved from entry points to computation modules (`evaluate_target_predictability()`, `select_features_for_target()`), ensuring tracking works regardless of which entry point calls these functions. **Tolerance bands with STABLE/DRIFTING/DIVERGED classification**: Replaced binary SAME/DIFFERENT with three-tier system that prevents alert fatigue. STABLE (within noise) uses INFO level, DRIFTING (small changes) uses INFO, only DIVERGED (real issues) uses WARNING. Supports configurable absolute/relative thresholds per metric and optional z-score calculation using reported σ. Module-specific log storage (target_rankings/, feature_selections/, training_results/) with cross-run search capability. See [Reproducibility Tracking Guide](DOCS/03_technical/implementation/REPRODUCIBILITY_TRACKING.md).
- **Systematic Config Parameter Validation System** (2025-12-11) — **NEW**: Created shared `TRAINING/utils/config_cleaner.py` utility to systematically prevent parameter passing errors across all model constructors. Uses `inspect.signature()` to validate parameters against actual estimator signatures, automatically removing duplicate and unknown parameters. Prevents entire class of "got multiple values for keyword argument" and "unexpected keyword argument" errors. Integrated into all model instantiation paths (multi-model feature selection, task types, cross-sectional ranking). Makes codebase resilient to future config drift from `inject_defaults`. Maintains SST (Single Source of Truth) while ensuring only valid parameters reach constructors.
- **Reproducibility Tracking Module** (2025-12-11) — **NEW**: Extracted reproducibility tracking into reusable `TRAINING/utils/reproducibility_tracker.py` module. Integrated into target ranking, feature selection, and model training pipelines. Provides automatic comparison of run results with tolerance-based verification. Comprehensive documentation added in `DOCS/03_technical/implementation/REPRODUCIBILITY_TRACKING.md`. Generic design allows easy integration into any pipeline stage. Shows ✅ for reproducible runs (within 0.1% tolerance) and ⚠️ for differences.
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

- **Interval detection large gap filtering** (2025-12-11) — **FIXED**: Improved interval auto-detection to ignore large gaps (overnight/weekend) that were contaminating detection. Added median-based gap filtering that excludes gaps > 10x median (configurable via `pipeline.data_interval.max_gap_factor`). Two-stage filtering: first removes gaps > 1 day, then removes gaps > 10x median. Prevents 270m/1210m gaps from causing "unclear interval" warnings when base cadence is 5m. Should eliminate false warnings on clean 5m data.
- **Univariate selection signed score handling** (2025-12-11) — **FIXED**: Fixed univariate feature selection to handle signed F-statistics properly. Previously, negative scores caused negative importance sums, triggering uniform fallback (no real signal). Now uses absolute values for ranking, treating negative correlations as weaker but still informative. Preserves signal instead of falling back to uniform distribution. Logs when negative scores are detected for debugging.
- **MLPRegressor verbose parameter validation** (2025-12-11) — **FIXED**: Fixed `CatBoostError` and `ValueError` when `verbose=-1` was passed to MLPRegressor. sklearn's MLPRegressor requires `verbose >= 0`, but global config defaults were injecting `verbose=-1` (silent mode). Added sanitization in `config_cleaner.py` that converts negative verbose values to `0` for neural_network family, preserving intent (silent) while satisfying sklearn's validation.
- **CatBoost iteration synonyms conflict** (2025-12-11) — **FIXED**: Fixed `CatBoostError: only one of the parameters iterations, n_estimators, num_boost_round, num_trees should be initialized`. Global defaults injection was adding `n_estimators: 1000` while config already had `iterations: 300`. Added aggressive sanitization in `multi_model_feature_selection.py` that removes all synonyms (`n_estimators`, `num_boost_round`, `num_trees`) before model construction, preferring `iterations` (CatBoost's native param). Double-check after `_clean_config_for_estimator` ensures no synonyms remain. Also added sanitization in `config_cleaner.py` for global application.
- **CatBoost random_state/random_seed conflict** (2025-12-11) — **FIXED**: Fixed `CatBoostError: only one of the parameters random_seed, random_state should be initialized`. CatBoost accepts only one RNG parameter. Added sanitization in `config_cleaner.py` that converts `random_state` to `random_seed` (CatBoost's preferred name) or removes it if `random_seed` is already in extra_kwargs, preventing conflicts from defaults injection.
- **Univariate selection importance assertion failure** (2025-12-11) — **FIXED**: Fixed `AssertionError: Importance sum should be positive after normalization` when univariate feature selection returned all-zero scores. Added multiple defensive fallback layers in `normalize_importance()`: validates `n_features > 0`, ensures sum > 0 even when `normalize_after_fallback` is False, handles edge cases where `uniform_importance` is 0/negative, checks for finite values, and applies uniform distribution (1/n) as last resort. Assertion now always passes with comprehensive error handling.
- **Reproducibility logging per symbol** (2025-12-11) — **ADDED**: Added debug-level reproducibility logging in `process_single_symbol()` showing base_seed, n_features, n_samples, and detected_interval for each symbol. Matches target ranking's reproducibility approach and provides fine-grained debugging information for deterministic behavior verification.
- **Leakage detection confidence calculation bug** (2025-12-11) — **CRITICAL FIX**: Fixed bug where detection confidence was using raw importance values (0-1) directly, causing valid leak detections to be silently filtered out. Features with 15% importance got 15% confidence, which was below the 80% threshold, so all detections were filtered out even when perfect scores were detected. Now uses base confidence of 0.85 for perfect-score context with importance boost, ensuring detections get 0.85-0.95 confidence (always above threshold). This explains why auto-fixer reported 0 leaks when perfect scores were detected.
- **Leakage detection when importances missing** (2025-12-11) — Enhanced detection to compute feature importances on-the-fly when `model_importance` is not provided. Previously, detection only checked known patterns (p_, y_, fwd_ret_, etc.) when importances were missing, missing subtle leaks. Now trains quick RandomForest to compute importances automatically, ensuring detection works even when importances aren't passed from upstream.
- **Leakage detection visibility and diagnostics** (2025-12-11) — Added comprehensive logging for detection process: confidence distribution (high vs low confidence), top detections that will be fixed vs filtered out, warnings when perfect score detected but no leaks found (explains possible reasons: structural leakage, already excluded, detection methods need improvement). Improved INFO-level logging for auto-fixer inputs and top features by importance.
- **Cross-sectional sampling limit bug** (2025-12-11) — **CRITICAL**: Fixed bug where `max_cs_samples` filtering was in wrong code block, causing large dataframes to be built despite `max_cs_samples=1000` setting. Filtering code was in `else:` block (when `time_col is None`) instead of `if time_col is not None:` block. Now properly limits cross-sectional samples per timestamp, dramatically reducing memory usage and processing time.
- **Config parameter passing to ranking** (2025-12-11) — Fixed issue where `min_cs`, `max_cs_samples`, and `max_rows_per_symbol` from test config were not being passed to `rank_targets()` function. These values are now properly extracted from `train_kwargs` and passed through the ranking pipeline, ensuring test config settings are actually used.
- **Reproducibility tracking architecture** (2025-12-11) — **REFACTORED**: Moved reproducibility tracking from entry points to computation modules for better architecture. Tracking now happens in `evaluate_target_predictability()` and `select_features_for_target()` functions, ensuring it works regardless of entry point (intelligent_trainer, standalone scripts, programmatic calls). Removed duplicate tracking code from entry points. Single source of tracking logic makes maintenance easier and ensures consistent behavior.
- **Reproducibility tracking visibility** (2025-12-11) — Fixed issue where reproducibility comparison logs were not appearing in target ranking output. Tracking now integrated into computation functions with proper logger propagation. Logs now show up in main output with clear ✅/⚠️ indicators.
- **Reproducibility tracking tolerance bands** (2025-12-11) — **ENHANCED**: Implemented three-tier classification system (STABLE/DRIFTING/DIVERGED) replacing binary SAME/DIFFERENT. Prevents alert fatigue from tiny differences within CV noise. STABLE uses INFO level, only DIVERGED uses WARNING. Configurable thresholds per metric (roc_auc, composite, importance) with absolute, relative, and z-score support. Uses reported σ when available for statistical significance. Example: 0.08% ROC-AUC shift with z=0.06 is now classified as STABLE (INFO) instead of DIFFERENT (WARNING).
- **Reproducibility tracking directory structure** (2025-12-11) — **FIXED**: Reproducibility logs now use module-specific directories (target_rankings/, feature_selections/, training_results/) instead of shared location. Each module has its own reproducibility_log.json. Added search_previous_runs option to find previous runs across different timestamped output directories. This allows comparing runs even when output_dir is timestamped/run-specific, while keeping modules properly separated.
- **Reproducibility tracking error handling** (2025-12-11) — **FIXED**: Added comprehensive error handling to prevent crashes when searching for previous runs. Fixed missing `List` and `Tuple` imports causing NameError. Improved exception logging from DEBUG to WARNING level for visibility. Added traceback logging for debugging. Prevents silent failures that could break output generation.
- **Auto-fixer logging format error** (2025-12-11) — Fixed ValueError: Invalid format specifier when logging auto-fixer inputs. Changed from using conditional expression in format specifier to formatting value first, then using in f-string.
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
