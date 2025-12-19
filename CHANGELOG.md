# Changelog

All notable changes to FoxML Core will be documented in this file.

> **Note**: This project is under active development. See [NOTICE.md](NOTICE.md) for more information.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

**For detailed technical changes:**
- [Changelog Index](DOCS/02_reference/changelog/README.md) – Per-day detailed changelogs with file paths, config keys, and implementation notes.

---

## [Unreleased]

### Recent Highlights

#### 2025-12-19 Updates (Target Evaluation Config Fixes)
- **Config Precedence Fix**: Fixed issue where `max_targets_to_evaluate` from experiment config was not properly overriding test config values. Experiment config now correctly takes priority over test config, with debug logging showing the precedence chain.
- **Target Whitelist Support**: Added `targets_to_evaluate` whitelist field that works with `auto_targets: true`. Users can now specify a specific list of targets to evaluate while still using auto-discovery, providing fine-grained control over target evaluation.
- **Enhanced Debug Logging**: Added comprehensive debug logging for config loading, showing where each config value comes from and the full precedence chain. Config trace now includes `intelligent_training` section overrides.
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-19-target-evaluation-config-fixes.md)

#### 2025-12-18 Updates (TRAINING Folder Reorganization)
- **TRAINING Folder Reorganization**: Comprehensive reorganization of `TRAINING/` folder structure to improve modularity and reduce duplication. Consolidated small directories (`features/`, `datasets/`, `memory/`, `live/`) into `data/` and `common/`. Merged overlapping directories (`strategies/` into `training_strategies/`, data processing modules into `data/`). Reorganized entry points (`unified_training_interface.py`, `target_router.py` into `orchestration/`). Moved output directories to `RESULTS/` and archived `EXPERIMENTS/`. Fixed config loader import warnings (changed to debug level). All changes maintain backward compatibility via re-export wrappers. 100% of key imports passing.
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-18-training-folder-reorganization.md)

#### 2025-12-18 Updates (Code Modularization & Refactoring)
- **Large File Modularization**: Split 7 large files (2,000-6,800 lines) into smaller, maintainable modules. Created 23 new utility/module files including `model_evaluation/`, `reproducibility/`, `diff_telemetry/`, `multi_model_feature_selection/`, `intelligent_trainer/`, and `leakage_detection/` subdirectories. Total: 103 files changed, ~2,000+ lines extracted.
- **Common Utilities Centralization**: Created 6 new centralized utility modules (`file_utils`, `cache_manager`, `config_hashing`, `process_cleanup`, `path_setup`, `family_constants`) to eliminate duplication across the codebase.
- **Utils Folder Reorganization**: Reorganized `TRAINING/utils/` into domain-specific subdirectories (`ranking/utils/`, `orchestration/utils/`, `common/utils/`). Maintained full backward compatibility via re-exports.
- **Import Error Fixes**: Fixed all import errors from refactoring, including missing imports, circular dependencies, and module export issues.
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-18-code-modularization.md)

#### 2025-12-17 Updates (Training Pipeline Audit Fixes)
- **Training Pipeline Contract Fixes**: Fixed 12 critical contract breaks across family IDs, routing, plan consumption, feature schema, counting/tracking, diff telemetry, and reproducibility verification. Key fixes: family normalization (XGBoost → xgboost), reproducibility tracking string/Enum handling, preflight filtering for all routes, routing plan DISABLED status respect, symbol-specific route includes all eligible symbols, feature pipeline threshold fix (allowed→present not requested→present), schema mismatch diagnostics with close matches, diff telemetry severity logic (fixed "minor" severity when no changes), and output digests for reproducibility verification (metrics_sha256, artifacts_manifest_sha256, predictions_sha256).
- **Feature Pipeline Improvements**: Fixed feature drop threshold to use allowed→present denominator (prevents false positives when registry intentionally prunes). Added close-match diagnostics for missing allowed features to help diagnose name mismatches.
- **Routing/Plan Integration**: Training now respects routing plan's CS: DISABLED status. Training plan with 0 jobs now hard-fails (error, not warning). Added validation logging for routing decision count mismatches.
- **Diff Telemetry Severity Fix**: Fixed severity logic to be purely derived from report. Added `severity_reason` field. Fixed bug where `total_changes=0` incorrectly returned "minor" severity. Now correctly returns "none" when no changes, "critical" when not comparable, with clear reasons.
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-17-training-pipeline-audit-fixes.md)

#### 2025-12-17 Updates (Licensing & Reproducibility)
- **Licensing Model Reverted**: Returned to **AGPL v3 + Commercial** dual licensing model. The brief "source-available, tax-exempt universities only" model was a mistake and has been reverted. All documentation updated to reflect standard dual-license model. Sorry for any confusion during the transition.
- **Documentation Restructure**: README.md restructured to be builder-first with minimal licensing notice. All commercial details moved to `LEGAL/LICENSING.md` for better audience separation (builders vs. buyers).
- **FEATURE_SELECTION Reproducibility Enhancement**: 
  - Integrated hyperparameters, train_seed, and library versions tracking into FEATURE_SELECTION stage
  - FEATURE_SELECTION now has same reproducibility requirements as TRAINING stage
  - Hyperparameters extracted from `model_families_config` (primary model family, usually LightGBM)
  - `train_seed` tracked from config (`pipeline.determinism.base_seed`) (note: config key name preserved for backward compatibility)
  - Library versions collected via `collect_environment_info()` and stored in `environment.library_versions`
  - All three factors (hyperparameters, train_seed, library_versions) now included in comparison group for FEATURE_SELECTION
  - Runs with different hyperparameters, train_seed, or library versions are no longer considered comparable
  - Fixed `lib_sig` UnboundLocalError by initializing to None for all stages
  - Fixed `fcntl` import errors in both `reproducibility_tracker.py` and `diff_telemetry.py`
  - Updated `log_run` API to accept `additional_data_override` parameter for passing hyperparameters
  - Legacy API path (`log_comparison`) also extracts and passes hyperparameters via `additional_data['training']`
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-17-feature-selection-reproducibility.md)

#### 2025-12-17 Updates (continued)
- **Audit-Grade Metadata Enhancement**: Added comprehensive environment tracking (python_version, platform, hostname, cuda_version, dependencies_lock_hash), data source details (source, dataset_id, bar_size, timezone, market_calendar), evaluation details (target_definition, n_features, feature_family_counts), and comparable_key for run comparison. Fixed CV/embargo inconsistency - embargo now explicitly set to 0.0 when CV is enabled (not "not_applicable"). Added splitter_impl and enabled flags to cv_details for clarity.
- **Research-Grade Metrics Enhancement**: Added per-fold distributional stats (fold_scores, min_score, max_score, median_score) for detecting fold collapse and bimodality. Added composite_score definition and versioning for reproducible comparison. Enhanced leakage reporting with structured object (status, checks_run, violations) instead of simple flag. Fixed naming redundancy (removed features_final, kept n_features_post_prune).

#### 2025-12-17 Updates
- **Fixed Field Name Mismatch in Diff Telemetry**: Corrected field name alignment between required field validation and metadata construction. Changed `date_start`/`date_end` → `date_range_start`/`date_range_end`, `target` → `target_name`, and `N_effective` → `n_effective`. This ensures `metadata.json` and `metrics.json` are properly written to cohort directories. Improved error visibility by changing diff telemetry exception handler from DEBUG to WARNING level.
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-17-field-name-fix.md)

#### 2025-12-16 Updates
- **Diff Telemetry Integration**: Integrated diff telemetry into metadata.json and metrics.json outputs. Full audit trail in metadata (fingerprints, sources, excluded factors), lightweight queryable fields in metrics (flags, counts, summaries). All stages (TARGET_RANKING, FEATURE_SELECTION, TRAINING) now share the same telemetry contract. Backwards compatible with stable shapes for all edge cases.
- **Diff Telemetry Digest Hardening**: Implemented fail-fast assertion for JSON-primitive-only types (removed `default=str` fallback) and full SHA256 hash (64 hex chars, 256 bits) for maximum integrity. Normalization bugs now fail immediately rather than being silently hidden.
- **Metadata/Metrics Output Per Target/Feature**: Fixed metadata.json and metrics.json to be written per target (CROSS_SECTIONAL and SYMBOL_SPECIFIC views) and per feature selection (per target, per view, per symbol). Made diff telemetry completely optional (non-fatal) so core metadata is always written even if diff telemetry fails. Added fallback for legacy mode to still write to cohort directories. Diff telemetry now works with multiple runs - finds previous comparable runs across all runs matching the comparison group key.
- **Feature Selection Structure**: Organized feature selection outputs to match target ranking layout (feature_importances/, metadata/, artifacts/). Eliminated scattered files and nested REPRODUCIBILITY directories.
- **Canonical Family ID System**: Migrated all model family registries to snake_case canonical IDs. All registries (`MODMAP`, `TRAINER_MODULE_MAP`, `POLICY`, `FAMILY_CAPS`) now use consistent snake_case keys (e.g., `"lightgbm"`, `"xgboost"`, `"meta_learning"`). Added startup validation to prevent key drift.
- **Feature Audit System**: Added comprehensive feature drop tracking with per-feature drop reasons. Generates CSV reports showing why features were dropped at each stage (registry filter, Polars select, pandas coercion, NaN drop, non-numeric drop).
- **Training Pipeline Fixes**: Fixed family name canonicalization, banner suppression in child processes, reproducibility tracking string/Enum handling, and model saving packaging bugs.
→ [Detailed Changelogs](DOCS/02_reference/changelog/2025-12-16-diff-telemetry-integration.md) | [Training Pipeline Fixes](DOCS/02_reference/changelog/2025-12-16-training-pipeline-fixes.md) | [Feature Selection Structure](DOCS/02_reference/changelog/2025-12-16-feature-selection-structure.md)

#### 2025-12-15 Updates
- **Metrics System Rename**: Renamed telemetry to metrics throughout codebase for better branding. All metrics stored locally - no user data collection, no external transmission.
- **Seed Tracking Fix**: Fixed missing seed field in metadata.json for full reproducibility tracking.
- **Feature Selection Improvements**: Output structure refactor, model family normalization, experiment config documentation.
- **CatBoost GPU Fixes**: Critical fixes for GPU mode compatibility and feature importance output.
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-15-consolidated.md)

#### 2025-12-14 Updates
- **IP Assignment Signed**: All IP legally assigned to Fox ML Infrastructure LLC. ✅ Legally effective.
- **Execution Modules**: ALPACA_trading and IBKR_trading modules added with compliance framework. ⚠️ ALPACA has minor issues, IBKR untested.
- **Enhanced Drift Tracking**: Fingerprints, drift tiers (OK/WARN/ALERT), critical metrics, sanity checks, Parquet files.
→ [Detailed Changelogs](DOCS/02_reference/changelog/README.md#december)

#### 2025-12-14 Updates (continued)
- **Metrics System**: Sidecar-based metrics with view isolation, hierarchical rollups, local-only storage. [Renamed from Telemetry on 2025-12-15]
- **Feature Selection Fixes**: Critical bug fixes for feature selection pipeline, experiment config loading, target exclusion.
- **Look-Ahead Bias Fixes**: Comprehensive data leakage fixes behind feature flags (default: OFF).
→ [Detailed Changelogs](DOCS/02_reference/changelog/README.md#december)

#### 2025-12-13 Updates
- **SST Enforcement Design**: Provably split-brain free system with EnforcedFeatureSet contract.
- **Single Source of Truth**: Eliminated split-brain in lookback computation.
- **Leakage Controls**: Unified leakage budget calculator, fingerprint tracking system.
- **Feature Selection Unification**: Shared ranking harness, comprehensive hardening.
- **Duration System**: Generalized duration parsing with interval-aware strictness.
- **Config Consolidation**: Major CONFIG directory restructure, config trace logging.
→ [Detailed Changelogs](DOCS/02_reference/changelog/README.md#december)

#### 2025-12-12 Updates
- **GPU Acceleration**: GPU support for XGBoost, CatBoost, and LightGBM.
- **Experiment Configuration**: Reusable experiment configs with auto target discovery.
- **Active Sanitization**: Proactive feature quarantine system.
- **License Banner**: Professional startup banner with licensing information.
- **Critical Bug Fixes**: Mutual Information SST compliance, XGBoost 3.1+ GPU compatibility.
→ [Detailed Changelogs](DOCS/02_reference/changelog/README.md#december)

---

### Stability Guarantees

- **Training results reproducible** across runs and hardware (centralized seeds + config-only hyperparameters)
- **Complete config centralization** – Pipeline behavior is controlled by YAML (Single Source of Truth)
- **SST enforcement** – Automated test prevents accidental reintroduction of hardcoded hyperparameters
- **Config schema backward compatible** – Existing configs continue to work with deprecation warnings where applicable
- **Modular architecture** – TRAINING module is self-contained with zero external script dependencies

### Known Issues & Limitations

- **Trading / execution modules** are out of scope for the core repo; FoxML Core focuses on ML research & infra
- **Feature engineering** still requires human review and domain validation
- **Full end-to-end test suite** is being expanded following SST + reproducibility changes
- **LOSO CV splitter**: LOSO view currently uses combined data; dedicated CV splitter is a future enhancement
- **Placebo test per symbol**: Symbol-specific strong targets should be validated with placebo tests - future enhancement

**For complete list:** See [Known Issues & Limitations](DOCS/02_reference/KNOWN_ISSUES.md)

---

### Added

**Recent additions (2025-12-12 - 2025-12-17):**
- Audit-Grade Metadata Fields - Environment info (python_version, platform, hostname, cuda_version, dependencies_hash), data source details, evaluation details, comparable_key for run comparison (2025-12-17)
- Research-Grade Metrics - Per-fold distributional stats (fold_scores, min/max/median), composite score definition/versioning, structured leakage reporting (2025-12-17)
- Feature Audit System - Per-feature drop tracking with CSV reports showing drop reasons (2025-12-16)
- Canonical Family ID System - Unified snake_case family IDs across all registries with startup validation (2025-12-16)
- Registry Validation - Startup assertions to prevent non-canonical keys and collisions (2025-12-16)
- Target Pattern Exclusion - Per-experiment control to exclude specific target types (2025-12-14)
- Look-Ahead Bias Fixes - Feature flag-based fixes for data leakage (2025-12-14)
- Shared Ranking Harness - Unified evaluation contract for target ranking and feature selection
- Feature Selection Unification - Complete parity with target ranking
- Experiment Configuration System - Reusable experiment configs with auto target discovery
- Active Sanitization System - Proactive feature quarantine before training
- Dual-View Target Ranking System - Multiple evaluation views with automatic routing
- Config-Based Pipeline Interface - Minimal command-line usage
- Sample Size Binning System - Automatic binning of runs by sample size
- Trend Analysis System - Automated trend tracking across runs
- License & Commercial Use Banner - Professional startup banner
- Training Routing & Planning System - Config-driven routing decisions
- Reproducibility Tracking System - End-to-end tracking with STABLE/DRIFTING/DIVERGED classification

**For complete details:** See [Changelog Index](DOCS/02_reference/changelog/README.md)

---

### Fixed

**Recent fixes (2025-12-12 - 2025-12-17):**
- **Training Pipeline Contract Fixes** (2025-12-17): Fixed 12 critical contract breaks: family normalization (XGBoost → xgboost alias), reproducibility tracking string/Enum crashes, LightGBM save hook `_pkg_ver` referenced-before-assignment, preflight filtering for all routes (SYMBOL_SPECIFIC now uses validated_families), router default for swing targets (binary classification), training plan 0 jobs hard error, routing plan CS: DISABLED respect, routing decision count mismatch detection, symbol-specific route includes all eligible symbols, feature pipeline threshold fix (allowed→present) and schema mismatch diagnostics, diff telemetry severity logic fix (severity now purely derived from report with `severity_reason` field), and output digests for reproducibility verification (metrics_sha256, artifacts_manifest_sha256, predictions_sha256 - enables comparison of outputs across reruns).
- **CV/Embargo Metadata Inconsistency** (2025-12-17): Fixed inconsistent embargo_minutes handling - when CV is enabled (cv_method set), embargo is now explicitly set to 0.0 (disabled) rather than marked as "not_applicable". Only marks as "not_applicable" when CV is actually disabled. Added embargo_enabled flag for clarity.
- **Metrics Naming Redundancy** (2025-12-17): Removed redundant features_final field, kept n_features_post_prune (more descriptive). Clarified n_features_pre vs n_features_post_prune semantics.
- **Training Pipeline Plumbing Fixes** (2025-12-16): Fixed family name canonicalization mismatches, banner suppression in child processes, reproducibility tracking string/Enum handling, model saving packaging bugs (`_pkg_ver`, `joblib` imports), feature selector vs trainer confusion
- **Feature Selection and Config Fixes** (2025-12-14): Fixed UnboundLocalError for np (11 model families), missing import, unpacking error, routing diagnostics, experiment config loading, target exclusion, lookback enforcement
- **Look-Ahead Bias Fixes** (2025-12-14): Rolling windows exclude current bar, CV-based normalization, feature renaming, symbol-specific logging, feature selection bug (task_type collision)
- **Leakage Controls Structural Fixes** (2025-12-13): Unified lookback calculator, calendar feature classification, separate purge/embargo validation, fingerprint tracking, leakage canary test
- **Feature Selection Critical Fixes** (2025-12-13): Shared harness unpack crashes, CatBoost dtype mis-typing, RFE/linear model failures, stability cross-model mixing, telemetry scoping
- **Single Source of Truth** (2025-12-13): Eliminated split-brain in lookback computation, POST_PRUNE invariant check, _Xd pattern inference, readline library conflict
- Documentation Link Fixes, Resolved Config System, Reproducibility Tracker Fixes, Critical Horizon Unit Bug, XGBoost 3.1+ GPU Compatibility, CatBoost GPU Verification, Process Deadlock Fix, Config Path Consolidation

**For complete details:** See [Changelog Index](DOCS/02_reference/changelog/README.md)

---

### Changed

- **Reproducibility & Defaults** (2025-12-10): Removed hardcoded `random_state` and similar defaults (note: internal module names preserved for backward compatibility)
- **Logging** (2025-12-10): Replaced scattered logging flags with structured YAML-driven configuration
- **Documentation** (2025-12-09+): Restructured into 4-tier hierarchy with improved cross-linking

---

### Security

- Enhanced compliance and production-use documentation
- Documented license enforcement procedures and copyright notice requirements

---

## Versioning

Releases follow [Semantic Versioning](https://semver.org/):

- **MAJOR** – Incompatible API changes
- **MINOR** – Backwards-compatible functionality
- **PATCH** – Backwards-compatible bug fixes

## Categories

- **Added** – New features  
- **Changed** – Changes in existing functionality  
- **Fixed** – Bug fixes  
- **Security** – Security / compliance improvements  
- **Documentation** – Documentation changes
