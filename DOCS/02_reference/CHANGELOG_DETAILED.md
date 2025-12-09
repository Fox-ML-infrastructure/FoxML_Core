# Detailed Changelog

This document provides comprehensive change details for FoxML Core. For a quick overview, see the [root changelog](../../CHANGELOG.md). For internal release notes with full technical details, see [`DOCS/internal/DETAILED_RELEASE_NOTES.md`](../internal/DETAILED_RELEASE_NOTES.md).

---

## [Unreleased]

### Status

Phase 1 functioning properly - Ranking and selection pipelines unified with consistent behavior. Boruta refactored as statistical gatekeeper. All documentation updated and cross-linked.

**Note**: Phase 1 of the pipeline (intelligent training framework) is functioning properly. Ranking and selection pipelines now have unified behavior (interval handling, sklearn preprocessing, CatBoost configuration). Boruta has been refactored from "just another importance scorer" to a statistical gatekeeper that modifies consensus scores via bonuses/penalties. All documentation has been updated with cross-links and references to new config files and utilities. Backward functionality remains fully operational. All existing training workflows continue to function as before, and legacy config locations are still supported with deprecation warnings.

---

### Highlights

- Phase 1 intelligent training framework completed and functioning properly
- Ranking & selection pipelines unified (interval handling, preprocessing, CatBoost)
- Boruta refactored into a statistical gatekeeper with base vs final consensus tracking
- New modular configuration + structured logging system
- Leakage Safety Suite hardened (auto-fixer + backup system + schema/registry)
- Documentation and legal reorganized into a 4-tier, enterprise-ready docs hierarchy
- Commercial license & pricing updated for enterprise quant infrastructure

---

### Added

#### Intelligent Training & Ranking

**Unified Ranking and Selection Pipelines**

- **Unified interval handling** — `explicit_interval` parameter now wired through entire ranking pipeline (orchestrator → rank_targets → evaluate_target_predictability → train_and_evaluate_models). All interval detection respects `data.bar_interval` from experiment config, eliminating spurious auto-detection warnings. Fixed "Nonem" logging issue in interval detection fallback.
- **Shared sklearn preprocessing** — All sklearn-based models in ranking now use `make_sklearn_dense_X()` helper (same as feature selection) for consistent NaN/dtype/inf handling. Applied to Lasso, Mutual Information, Univariate Selection, Boruta, and Stability Selection.
- **Unified CatBoost builder** — CatBoost in ranking now uses same target type detection and loss function selection as feature selection. Auto-detects classification vs regression and sets appropriate `loss_function` (`Logloss`/`MultiClass`/`RMSE`) with YAML override support.
- **Shared target utilities** — New `TRAINING/utils/target_utils.py` module with reusable helpers (`is_classification_target()`, `is_binary_classification_target()`, `is_multiclass_target()`) used consistently across ranking and selection.

See [`docs/01_tutorials/training/RANKING_SELECTION_CONSISTENCY.md`](../01_tutorials/training/RANKING_SELECTION_CONSISTENCY.md) for complete details.

**Boruta Statistical Gatekeeper**

- **Boruta as gatekeeper, not scorer** — Refactored Boruta from "just another importance scorer" to a statistical gatekeeper that modifies consensus scores via bonuses/penalties. Boruta is now excluded from base consensus calculation and only applied as a modifier, eliminating double-counting.
- **Base vs final consensus separation** — Feature selection now tracks both `consensus_score_base` (model families only) and `consensus_score` (with Boruta gatekeeper effect). Added `boruta_gate_effect` column showing pure Boruta impact (final - base) for debugging and analysis.
- **Boruta implementation improvements**:
  - Switched from `RandomForest` to `ExtraTreesClassifier/Regressor` for more random, stability-oriented importance testing
  - Configurable hyperparams: `n_estimators: 500` (vs RF's 200), `max_depth: 6` (vs RF's 15), `perc: 95` (more conservative)
  - Configurable `class_weight`, `n_jobs`, `verbose` via YAML
  - Fixed `X_clean` error by using `X_dense` and `y` from `make_sklearn_dense_X()`
- **Magnitude sanity checks** — Added configurable magnitude ratio warning (`boruta_magnitude_warning_threshold: 0.5`) that warns if Boruta bonuses/penalties exceed 50% of base consensus range.
- **Ranking impact metric** — Calculates and logs how many features changed in top-K set when comparing base vs final consensus.
- **Debug output** — New `feature_importance_with_boruta_debug.csv` file with explicit columns for Boruta gatekeeper analysis.
- **Config migration** — All Boruta hyperparams and gatekeeper settings moved to `CONFIG/feature_selection/multi_model.yaml` (no hardcoded values).

**GPU & Training Infrastructure**

- **LightGBM GPU support** in target ranking with automatic detection and usage (CUDA/OpenCL), GPU verification diagnostics, and fallback to CPU if GPU unavailable
- **TRAINING module self-contained** — Moved all utility dependencies from `SCRIPTS/` to `TRAINING/utils/`. TRAINING module now has zero dependencies on `SCRIPTS/` folder.
- Base trainer scaffolding for 2D and 3D models (`base_2d_trainer.py`, `base_3d_trainer.py`)

#### Configuration & Logging

**Modular Configuration System**

- **Typed configuration schemas** (`CONFIG/config_schemas.py`):
  - `ExperimentConfig` - Experiment-level configuration (data, targets, overrides)
  - `FeatureSelectionConfig` - Feature selection module configuration
  - `TargetRankingConfig` - Target ranking module configuration
  - `TrainingConfig` - Training module configuration
  - All configs validated on load (required fields, value ranges, type checking)
- **Configuration builder** (`CONFIG/config_builder.py`):
  - `load_experiment_config()` - Load experiment configs from YAML
  - `build_feature_selection_config()` - Build typed configs by merging experiment + module configs
  - `build_target_ranking_config()` - Build typed configs for target ranking
  - `build_training_config()` - Build typed configs for training
  - Automatic fallback to legacy config locations with deprecation warnings
- **New config directory structure**:
  - `CONFIG/experiments/` - Experiment-level configs (what are we running?)
  - `CONFIG/feature_selection/` - Feature selection module configs
  - `CONFIG/target_ranking/` - Target ranking module configs
  - `CONFIG/training/` - Training module configs
  - Prevents config "crossing" between pipeline components
- **Experiment configs** (preferred way):
  - Single YAML file defines data, targets, and module overrides
  - Use via `--experiment-config` CLI argument
  - Example: `python TRAINING/train.py --experiment-config my_experiment`
- **Backward compatibility**:
  - All legacy config locations still supported
  - Deprecation warnings guide migration to new locations
  - Old code continues to work without changes
- **CLI improvements**:
  - `--experiment-config` argument for using experiment configs
  - `--max-targets-to-evaluate` option for faster E2E testing (limits evaluation, not just return count)
  - `--data-dir` and `--symbols` now optional when experiment config provided
- **Config validation**:
  - Required fields validated on load
  - Value ranges checked (e.g., `cv_folds >= 2`, `max_samples_per_symbol >= 1`)
  - Type checking (paths converted to `Path` objects)
  - Clear error messages for invalid configs

See [`docs/02_reference/configuration/MODULAR_CONFIG_SYSTEM.md`](configuration/MODULAR_CONFIG_SYSTEM.md) for complete documentation.

**Structured Logging Configuration**

- **Logging configuration schema** (`CONFIG/config_schemas.py`):
  - `LoggingConfig` - Global logging configuration with module and backend controls
  - `ModuleLoggingConfig` - Per-module verbosity controls (gpu_detail, cv_detail, edu_hints, detail)
  - `BackendLoggingConfig` - Backend library verbosity (native_verbosity, show_sparse_warnings)
- **Logging configuration YAML** (`CONFIG/logging_config.yaml`):
  - Global logging level control
  - Per-module verbosity flags (rank_target_predictability, feature_selection, etc.)
  - Backend verbosity controls (LightGBM, XGBoost, TensorFlow)
  - Profile support (default, debug_run, quiet)
- **Logging config utilities** (`CONFIG/logging_config_utils.py`):
  - `LoggingConfigManager` singleton for centralized config management
  - `get_module_logging_config()` - Get module-specific logging config
  - `get_backend_logging_config()` - Get backend-specific logging config
  - Profile support for switching between quiet/verbose modes
- **Integration**:
  - `rank_target_predictability.py` uses config for GPU detail, CV detail, and educational hints
  - `lightgbm_trainer.py` uses backend config for verbose parameter
  - No hardcoded logging flags scattered throughout codebase
  - Easy to switch between quiet production runs and verbose debug runs via config

**Centralized Training Configs**

- Centralized configuration system with 9 training config YAML files:
  - Pipeline config
  - GPU config
  - Memory config
  - Preprocessing config
  - Threading config
  - Safety config
  - Callbacks config
  - Optimizer config
  - System config
- Config loader with nested access and family-specific overrides

#### Leakage Safety Suite

**Production-Grade Backup System for Auto-Fixer**

- Per-target timestamped backup structure: `CONFIG/backups/{target}/{timestamp}/files + manifest.json`
- Automatic retention policy: Keeps last N backups per target (configurable, default: 20)
- High-resolution timestamps: Uses microseconds to avoid collisions in concurrent scenarios
- Manifest files with full provenance: Includes backup_version, source, target_name, timestamp, git_commit, file paths
- Atomic restore operations: Writes to temp file first, then atomic rename (prevents partial writes)
- Enhanced error handling: Lists available timestamps on unknown timestamp, validates manifest structure
- Comprehensive observability: Logs backup creation, pruning, and restore operations with full context
- Config-driven settings: `max_backups_per_target` configurable via `system_config.yaml` (default: 20, 0 = no limit)
- Restoration helpers: `list_backups()` and `restore_backup()` static methods for backup management
- Backward compatible: Legacy flat structure still supported (with warning) when no target_name provided
- Git commit tracking: Captures git commit hash in manifest for debugging and provenance

**Automated Leakage Detection and Auto-Fix System**

- `LeakageAutoFixer` class for automatic detection and remediation of data leakage
- Integration with leakage sentinels (shifted-target, symbol-holdout, randomized-time tests)
- Automatic config file updates (`excluded_features.yaml`, `feature_registry.yaml`)
- Auto-fixer triggers automatically when perfect scores (≥0.99) are detected during target ranking
- **Checks against pre-excluded features**: Filters out already-excluded features before detection to avoid redundant work
- **Configurable auto-fixer thresholds** in `safety_config.yaml`:
  - CV score threshold (default: 0.99)
  - Training accuracy threshold (default: 0.999)
  - Training R² threshold (default: 0.999)
  - Perfect correlation threshold (default: 0.999)
  - Minimum confidence for auto-fix (default: 0.8)
  - Maximum features to fix per run (default: 20) - prevents overly aggressive fixes
  - Enable/disable auto-fixer flag
- **Auto-rerun after leakage fixes**:
  - Automatic rerun of target evaluation after auto-fixer modifies configs
  - Configurable via `safety_config.yaml` (`auto_rerun` section):
    - `enabled`: Enable/disable auto-rerun (default: `true`)
    - `max_reruns`: Maximum reruns per target (default: `3`)
    - `rerun_on_perfect_train_acc`: Rerun on perfect training accuracy (default: `true`)
    - `rerun_on_high_auc_only`: Rerun on high AUC alone (default: `false`)
  - Stops automatically when no leakage detected or no config changes
  - Tracks attempt count and final status (`OK`, `SUSPICIOUS_STRONG`, `LEAKAGE_UNRESOLVED`, etc.)
- **Pre-training leak scan**:
  - Detects near-copy features before model training (catches obvious leaks early)
  - Binary classification: detects features matching target with ≥99.9% accuracy
  - Regression: detects features with ≥99.9% correlation with target
  - Automatically removes leaky features before model training
  - Configurable thresholds in `safety_config.yaml` (min_match, min_corr)
- **Feature/Target Schema** (`CONFIG/feature_target_schema.yaml`):
  - Explicit schema for classifying columns (metadata, targets, features)
  - Feature families with mode-specific rules (ranking vs. training)
  - Ranking mode: more permissive (allows basic OHLCV/TA features)
  - Training mode: strict rules (enforces all leakage filters)
- **Configurable leakage detection thresholds**:
  - All hardcoded thresholds moved to `CONFIG/training_config/safety_config.yaml`
  - Pre-scan thresholds (min_match, min_corr, min_valid_pairs)
  - Ranking feature requirements (min_features_required, min_features_for_model)
  - Warning thresholds (classification, regression with forward_return/barrier variants)
  - Model alert thresholds (suspicious_score)
- **Feature registry system** (`CONFIG/feature_registry.yaml`):
  - Structural rules based on temporal metadata (`lag_bars`, `allowed_horizons`, `source`)
  - Automatic filtering based on target horizon to prevent leakage
  - Support for short-horizon targets (added horizon=2 for 10-minute targets)
- **Leakage sentinels** (`TRAINING/common/leakage_sentinels.py`):
  - Shifted target test – detects features encoding future information
  - Symbol holdout test – detects symbol-specific leakage
  - Randomized time test – detects temporal information leakage
- **Feature importance diff detector** (`TRAINING/common/importance_diff_detector.py`):
  - Compares feature importances between full vs. safe feature sets
  - Identifies suspicious features with high importance in full model but low in safe model

See [`docs/03_technical/research/LEAKAGE_ANALYSIS.md`](../03_technical/research/LEAKAGE_ANALYSIS.md) and [`docs/02_reference/configuration/SAFETY_LEAKAGE_CONFIGS.md`](configuration/SAFETY_LEAKAGE_CONFIGS.md) for complete documentation.

#### Documentation & Legal

**Documentation Restructure**

- **4-tier documentation hierarchy** implemented:
  - Tier A: Executive / High-Level
  - Tier B: Tutorials / Walkthroughs
  - Tier C: Core Reference Docs
  - Tier D: Technical Deep Dives
- **Documentation centralized** in `docs/` folder:
  - Moved all CONFIG documentation to `docs/02_reference/configuration/`
  - Moved all TRAINING documentation to `docs/` folder
  - Code directories now contain only code and minimal README pointers
- **Comprehensive legal documentation index** (`docs/LEGAL_INDEX.md`):
  - Complete index of all legal, licensing, compliance, and enterprise documentation
  - Organized by category: Licensing, Terms & Policies, Enterprise & Compliance, Security, Legal Agreements, Consulting Services
- **Cross-linking and navigation** improved throughout all documentation
- 55+ new documentation files created, 50+ existing files rewritten and standardized

**Roadmap Restructure**

- Added "What Works Today" section highlighting immediate production-ready capabilities
- Renamed Phase 3.5 to Phase 4 (multi-GPU & NVLink exploration)
- Reorganized development priorities into near-term focus and longer-term/R&D categories
- Removed date-specific targets in favor of general development guidelines
- Added branding clarification (FoxML Core vs Fox ML Infrastructure)
- Refined wording throughout for external/enterprise consumption

See [`ROADMAP.md`](../../ROADMAP.md) for complete roadmap.

**Compliance Documentation Suite**

- `LICENSE_ENFORCEMENT.md` – License enforcement procedures and compliance requirements
- `COMMERCIAL_USE.md` – Quick reference guide for commercial use
- `COMPLIANCE_FAQ.md` – Frequently asked compliance questions
- `PRODUCTION_USE_NOTIFICATION.md` – Production use notification form
- `COPYRIGHT_NOTICE.md` – Copyright notice requirements

See [`docs/LEGAL_INDEX.md`](../LEGAL_INDEX.md) for complete legal documentation index.

#### Commercial

- **Commercial license pricing** updated to enterprise quant infrastructure standards:
  - 1–10 employees: $150,000/year
  - 11–50 employees: $350,000/year
  - 51–250 employees: $750,000/year
  - 251–1000 employees: $1,500,000–$2,500,000/year
  - 1000+ employees: $5,000,000–$12,000,000+ /year (custom enterprise quote)
- **Optional enterprise add-ons**:
  - Dedicated Support SLA: $5,000–$20,000/month
  - Integration & On-Prem Setup: $100,000–$500,000 one-time
  - Onboarding: $25,000–$75,000 one-time
  - Private Slack / Direct Founder Access: $30,000–$120,000/year
  - Additional User Seats: $500–$2,000 per seat/year
  - Adaptive Intelligence Layer (Tier 5 only): +$2,000,000–$5,000,000/year
- Enhanced copyright headers across codebase (2025-2026 Fox ML Infrastructure LLC)

See [`COMMERCIAL_LICENSE.md`](../../COMMERCIAL_LICENSE.md) for complete pricing details.

---

### Changed

**Logging System Refactored**

- Replaced hardcoded logging flags with structured configuration system
- `rank_target_predictability.py` now uses config-driven logging (GPU detail, CV detail, educational hints)
- `lightgbm_trainer.py` uses backend config for verbose parameter instead of hardcoded `-1`
- All logging verbosity controlled via `CONFIG/logging_config.yaml` without code changes
- Supports profiles (default, debug_run, quiet) for easy switching between modes

**Leakage Safety Suite Improvements**

- **Leakage filtering now supports ranking mode**:
  - `filter_features_for_target()` accepts `for_ranking` parameter
  - Ranking mode: permissive rules, allows basic OHLCV/TA features even if in always_exclude
  - Training mode: strict rules (default, backward compatible)
  - Ensures ranking has sufficient features to evaluate target predictability
- **Random Forest training accuracy no longer triggers critical leakage**:
  - High training accuracy (≥99.9%) now logged as warning, not error
  - Tree models can overfit to 100% training accuracy without leakage
  - Real leakage defense: schema filters + pre-training scan + time-purged CV
  - Prevents false positives from overfitting detection

**Configuration System**

- All model trainers updated to use centralized configs (preprocessing, callbacks, optimizers, safety guards)
- Pipeline, threading, memory, GPU, and system settings integrated into centralized config system

**Legal**

- Updated company address in Terms of Service (STE B 212 W. Troy St., Dothan, AL 36303)

---

### Fixed

**Feature Selection Pipeline Fixes**

- **Boruta `X_clean` error** — Fixed `NameError: name 'X_clean' is not defined` in Boruta feature selection. Now correctly uses `X_dense` and `y` from `make_sklearn_dense_X()` sanitization.
- **Interval detection "Nonem" warning** — Fixed logging issue where interval detection fallback showed "Using default: Nonem" instead of actual default value. Now properly passes default parameter through call chain.
- **Boruta double-counting** — Fixed issue where Boruta was contributing to both base consensus (as a model family) and gatekeeper modifier. Now excluded from base consensus, only applied as gatekeeper modifier.
- **Boruta feature count mismatch** — Fixed `ValueError: X has N features, but ExtraTreesClassifier is expecting M features` error. Boruta now uses `train_score = math.nan` (not `0.0`) to indicate "not applicable" since it's a selector, not a predictor. Added NaN handling in logging and checkpoint serialization.
- **Config hardcoded values** — Moved all hardcoded config values (RFE estimator params, Boruta hyperparams, Stability Selection params, Boruta gatekeeper thresholds) to YAML config files for full configurability.

**Logging Configuration**

- Fixed method name mismatch in `logging_config_utils.py` (`get_backend_logging_config` now correctly calls `get_backend_config`)
- Fixed logger initialization order in `rank_target_predictability.py` (config import before logger usage)

**Ranking Pipeline**

- Fixed `_perfect_correlation_models` NameError in target ranking
- Fixed insufficient features handling (now properly filters targets with <2 features)
- Fixed early exit logic when leakage detected (removed false positive triggers)
- Improved error messages when no targets selected after ranking
- **Fixed progress logging denominator** - Now correctly shows `[1/23]` instead of `[1/63]` when using `--max-targets-to-evaluate`
- **Interval detection warnings in ranking** — Fixed spurious interval auto-detection warnings by wiring `explicit_interval` through entire ranking call chain. All `detect_interval_from_dataframe()` calls now respect `data.bar_interval` from experiment config.
- **CatBoost loss function for classification** — Fixed CatBoost using `RMSE` loss for binary classification targets. Now auto-detects target type and sets `loss_function` appropriately (`Logloss` for binary, `MultiClass` for multiclass, `RMSE` for regression).
- **Sklearn NaN/dtype handling in ranking** — Replaced ad-hoc `SimpleImputer` usage with shared `make_sklearn_dense_X()` helper for all sklearn-based models. Ensures consistent preprocessing across ranking and feature selection pipelines.
- **Shared target type detection** — Created `TRAINING/utils/target_utils.py` with reusable helpers used consistently across ranking and feature selection.

**Path Resolution & Imports**

- **Fixed inconsistent repo root calculations** - `feature_selector.py` and `target_ranker.py` now use `parents[2]` consistently
- **Auto-fixer import path** — fixed `parents[3]` to `parents[2]` in `leakage_auto_fixer.py` for correct repo root detection
- **Leakage filtering path resolution** — fixed config path lookup in `leakage_filtering.py` when moved to `TRAINING/utils/`
- **Path resolution in moved files** — corrected `parents[2]` vs `parents[3]` for repo root detection
- **Import paths after module migration** — all `scripts.utils.*` imports updated to `TRAINING.utils.*`

**Auto-Fixer**

- **Auto-fixer training accuracy detection** — now passes actual training accuracy (from `model_metrics`) instead of CV scores to auto-fixer
- **Auto-fixer pattern-based fallback** — added fallback detection when `model_importance` is missing
- **Auto-fixer pre-excluded feature check** — now filters out already-excluded features before detection to prevent redundant exclusions
- **Hardcoded safety net in leakage filtering** — added fallback patterns to exclude known leaky features even when config fails to load

**GPU & Model Issues**

- VAE serialization issues — custom Keras layers now properly imported before deserialization
- Sequential models 3D preprocessing issues — input shape handling corrected
- XGBoost source-build stability — persistent build directory and non-editable install
- Readline symbol lookup errors — environment variable fixes
- TensorFlow GPU initialization — CUDA library path resolution
- Type conversion issues in callback configs (min_lr, factor, patience, etc.)
- LSTM timeout issues — dynamic batch and epoch scaling implemented
- Transformer OOM errors — reduced batch size and attention heads, dynamic scaling
- CNN1D, LSTM, Transformer input shape mismatches — 3D to 2D reshape fixes
- **LightGBM GPU verbose parameter** — moved `verbose` from `fit()` to model constructor (LightGBM API requirement)

**Documentation**

- Comprehensive documentation updates — Updated all documentation with cross-links and new content
- Added `RANKING_SELECTION_CONSISTENCY.md` guide explaining unified pipeline behavior
- Updated all configuration docs to include `logging_config.yaml` references
- Added comprehensive cross-links throughout all documentation files
- Updated main `INDEX.md` with new files and references
- Fixed broken references to non-existent files
- Created `CROSS_REFERENCES.md` tracking document
- All API references now include utility modules (`target_utils.py`, `sklearn_safe.py`)
- All training tutorials now reference unified pipeline behavior
- Configuration examples updated with interval config and CatBoost examples

---

### Security

- Enhanced compliance documentation for production use
- License enforcement procedures documented
- Copyright notice requirements standardized

---

### Documentation

- Roadmap restructured for external consumption (see [`ROADMAP.md`](../../ROADMAP.md))
- Modular configuration system documentation (see [`docs/02_reference/configuration/MODULAR_CONFIG_SYSTEM.md`](configuration/MODULAR_CONFIG_SYSTEM.md))
- Documentation cleanup and consolidation — Integrated old folders, fixed cross-references, reorganized structure
- Comprehensive cross-linking and navigation improvements

---

## Future Work

### Adaptive Intelligence Layer (Planned)

The current intelligence layer provides automated target ranking, feature selection, leakage detection, and auto-fixing. Future enhancements will include:

- **Adaptive learning over time**: System learns from historical leakage patterns and feature performance
- **Dynamic threshold adjustment**: Automatically tunes detection thresholds based on observed patterns
- **Predictive leakage prevention**: Proactively flags potential leakage before training begins
- **Multi-target optimization**: Optimizes feature selection across multiple targets simultaneously

Adaptive intelligence layer design is documented in planning materials.

---

## Related Documentation

- [Root Changelog](../../CHANGELOG.md) - Quick overview of changes
- [Internal Release Notes](../internal/DETAILED_RELEASE_NOTES.md) - Full technical details
- [Roadmap](../../ROADMAP.md) - Development priorities and upcoming work
- [Documentation Index](../INDEX.md) - Complete documentation navigation

