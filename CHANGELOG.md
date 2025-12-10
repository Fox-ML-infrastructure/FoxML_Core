# Changelog

All notable changes to FoxML Core will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

**For more detailed information:**
- [Detailed Changelog](DOCS/02_reference/CHANGELOG_DETAILED.md) - Comprehensive change details with file paths and config references

---

## [Unreleased]

### Highlights

- **SST Enforcement System & Complete Hardcoded Value Elimination** (2025-12-10) — Implemented automated SST (Single Source of Truth) enforcement test that scans all training code for hardcoded hyperparameters. Fixed all remaining hardcoded values: top 10% importance patterns now configurable, batch_size uses training profiles (default/debug/throughput_optimized), all seeds properly marked with FALLBACK_DEFAULT_OK, diagnostic models marked as DESIGN_CONSTANT_OK. Created internal SST documentation (`DOCS/03_technical/internal/SST_DETERMINISM_GUARANTEES.md`, `SST_COMPLIANCE_CHECKLIST.md`) and public-facing deterministic training guide (`DOCS/00_executive/DETERMINISTIC_TRAINING.md`). Test passes: no unmarked hardcoded hyperparameters remain.
- **Complete Single Source of Truth (SST) config centralization** (2025-12-10) — **ALL** hardcoded configuration values across the entire TRAINING pipeline moved to YAML files. Every model trainer in `model_fun/` and `models/` now loads hyperparameters, test splits, and random seeds from centralized config. Feature pruning, leakage detection, training strategies, data preprocessing, and all 52+ model files use the same config system. Ensures complete reproducibility: same config → same results across all pipeline stages.
- **Determinism system integration** (2025-12-10) — All `random_state` values now use the centralized determinism system (`BASE_SEED`) instead of hardcoded values. Training strategies, feature selection, data splits, and all model initializations are now fully deterministic and reproducible.
- **Pipeline robustness fixes** (2025-12-10) — Fixed critical syntax errors and variable initialization issues in config loading patterns. All `if _CONFIG_AVAILABLE:` blocks now have proper `else:` clauses to ensure variables are always initialized, preventing "referenced before assignment" errors. Fixed missing polars imports in data preparation and data loader modules. Added feature list validation to prevent type errors at STEP 2 → STEP 3 transition. Full end-to-end testing currently underway.
- **Large file refactoring completed** (2025-12-09) — Split 3 large monolithic files (4.5k, 3.4k, 2.5k lines) into modular components while maintaining 100% backward compatibility
- **Model family status tracking** — Added comprehensive debugging for multi-model feature selection to identify which families succeed/fail and why
- **Interval detection robustness** — Fixed timestamp gap filtering to ignore outliers (weekends, data gaps) before computing median
- **Legal compliance enhanced** — Added IP assignment agreement, regulatory disclaimers, and explicit "No Financial Advice" sections
- Phase 1 intelligent training framework completed and functioning properly
- Ranking & selection pipelines unified (interval handling, preprocessing, CatBoost)
- Boruta refactored into a statistical gatekeeper with base vs final consensus tracking
- **Target confidence & routing system** — Automatic quality assessment with configurable thresholds and operational buckets (core/candidate/experimental)
- New modular configuration + structured logging system
- Leakage Safety Suite hardened (auto-fixer + backup system + schema/registry)
- Documentation and legal reorganized into a 4-tier, enterprise-ready docs hierarchy
- Commercial license & pricing updated for enterprise quant infrastructure

---

### Stability Guarantees

- **Training results reproducible** across hardware (deterministic seeds, config-driven hyperparameters)
- **Complete config centralization** (2025-12-10) — All pipeline parameters load from YAML files (single source of truth). No hardcoded thresholds, limits, or hyperparameters in core pipeline code. Automated test enforces SST compliance.
- **Full determinism** (2025-12-10) — All random seeds use centralized determinism system. Same config → same results across all pipeline stages.
- **SST enforcement** (2025-12-10) — Automated test (`TRAINING/tests/test_no_hardcoded_hparams.py`) prevents hardcoded hyperparameters from being introduced. Only explicitly marked fallbacks (`FALLBACK_DEFAULT_OK`) and design constants (`DESIGN_CONSTANT_OK`) are allowed.
- **Config schema backward compatible** (existing configs continue to work)
- **Auto-fixer non-destructive by design** (atomic backups, manifest tracking, restore capabilities)
- **Leakage detection thresholds configurable** (no hardcoded magic numbers)
- **Modular architecture** (self-contained TRAINING module, zero external script dependencies)

### Known Issues & Limitations

- **Trading execution modules** have been removed from the core repository; the system focuses on ML research infrastructure
- **Feature engineering** still requires human review and validation (initial feature set was for testing)
- **Adaptive intelligence layer** in early phase (leakage detection and auto-fixer are production-ready, but adaptive learning over time is planned)
- **Ranking pipeline** may occasionally log false-positive leakage warnings for tree models (RF overfitting detection is conservative by design)
- **Later phases of the experiments workflow** (core models and sequential models) require implementation beyond Phase 1
- **End-to-end testing in progress** (2025-12-10) — Full pipeline validation currently underway after SST and Determinism fixes. All syntax, config loading, and import issues have been resolved. Comprehensive testing across all model families and targets is ongoing.

---

### Added

- **Complete Single Source of Truth config system** (2025-12-10) — All 52+ model files now use centralized config:
  - **Base trainer helpers**: `_get_test_split_params()`, `_get_random_state()`, `_get_learning_rate()` methods for consistent config access
  - **Model hyperparameters**: All `n_estimators`, `max_depth`, `learning_rate`, `alpha`, `C` values load from `models.{family}.{param}` config paths
  - **Train/test splits**: All use `preprocessing.validation.test_size` from config
  - **Random seeds**: All use `BASE_SEED` from determinism system (with config fallback)
  - **Neural network optimizers**: All Adam/optimizer learning rates load from `optimizer.learning_rate` config
- **Config centralization** (2025-12-10) — New config sections in `safety_config.yaml`:
  - `leakage_sentinels.*` — All leakage detection thresholds (shifted target, symbol holdout, randomized time)
  - `auto_fixer.*` — Auto-fixer settings (perfect score threshold, min confidence, backup limits, test size)
- **Feature pruning config** (2025-12-10) — All feature pruning parameters now configurable via `preprocessing_config.yaml`:
  - `feature_pruning.cumulative_threshold` — Minimum cumulative importance threshold
  - `feature_pruning.min_features` — Minimum features to keep
  - `feature_pruning.n_estimators` — Number of trees for quick pruning
  - `feature_pruning.max_depth` — Tree depth for pruning model
  - `feature_pruning.learning_rate` — Learning rate for pruning model
- **Determinism integration** (2025-12-10) — All training strategies, data preprocessing, and model initializations now use `BASE_SEED` from determinism system instead of hardcoded `random_state=42`

### Fixed

- **Complete Single Source of Truth implementation** (2025-12-10) — Replaced ALL hardcoded values across entire TRAINING pipeline:
  - **Model trainers** (`model_fun/` - 34 files): All `test_size`, `random_state`, `n_estimators`, `max_depth`, `learning_rate`, `alpha` now load from config
  - **Specialized models** (`models/` - 18 files): All hardcoded hyperparameters and splits now use config
  - **Base trainer helpers**: Added `_get_random_state()`, `_get_learning_rate()` methods for consistent config access
  - **Strategies**: RandomForest fallback `n_estimators` now loads from config
  - All model initializations (XGBoost, LightGBM, Neural Networks, Ridge, SGD, etc.) use config-driven hyperparameters
  - All train/test splits use `preprocessing.validation.test_size` from config
  - All random seeds use `BASE_SEED` from determinism system
- **Config loading pattern robustness** (2025-12-10) — Fixed critical syntax errors and variable initialization issues:
  - Fixed `SyntaxError` in `data_loading.py` where `if _CONFIG_AVAILABLE:` was incorrectly placed in function parameter list
  - Fixed `SyntaxError` in `leakage_detection.py` where config loading was incorrectly embedded in function call parameters
  - Fixed `UnboundLocalError` in `model_evaluation.py` where `MIN_FEATURES_FOR_MODEL` and `MIN_FEATURES_AFTER_LEAK_REMOVAL` could be undefined when `_CONFIG_AVAILABLE` was `False`
  - All `if _CONFIG_AVAILABLE:` blocks now have proper `else:` clauses ensuring variables are always initialized before use
  - Comprehensive audit confirmed no similar patterns exist elsewhere in the pipeline
- **Missing import fixes** (2025-12-10) — Fixed `NameError: name 'pl' is not defined` errors at STEP 2 → STEP 3 transition:
  - Added `import polars as pl` to `data_preparation.py` and `data_loader.py`
  - Added missing type imports (`Dict`, `List`, `Tuple`, `Path`) to `data_loader.py`
  - Added `USE_POLARS` environment variable handling to `data_loader.py`
- **Feature list validation** (2025-12-10) — Added robust validation for `selected_features` parameter in training pipeline:
  - Validates feature lists are proper list/tuple types before use
  - Handles empty feature lists gracefully (falls back to auto-discovery)
  - Prevents `TypeError` when calling `len()` on invalid types
  - Improves robustness at STEP 2 (Feature Selection) → STEP 3 (Model Training) transition

#### Intelligent Training & Ranking

- **Unified ranking and selection pipelines** — Consistent interval handling, sklearn preprocessing, and CatBoost configuration across both pipelines. Shared target utilities and preprocessing helpers ensure identical behavior. See [`DOCS/01_tutorials/training/RANKING_SELECTION_CONSISTENCY.md`](DOCS/01_tutorials/training/RANKING_SELECTION_CONSISTENCY.md).
- **Boruta statistical gatekeeper** — Refactored from importance scorer to gatekeeper that modifies consensus scores via bonuses/penalties. Tracks base vs final consensus with explicit gate effect. ExtraTrees-based implementation with configurable thresholds. See [`CONFIG/feature_selection/multi_model.yaml`](CONFIG/feature_selection/multi_model.yaml).
- **Target confidence & routing system** — Automatic quality assessment for each target with configurable thresholds. Computes Boruta coverage, model coverage, score strength, and agreement ratio. Routes targets into operational buckets (core/candidate/experimental) based on confidence + score_tier. All thresholds and routing rules configurable via YAML. See [`CONFIG/feature_selection/multi_model.yaml`](CONFIG/feature_selection/multi_model.yaml) `confidence` section and [`DOCS/01_tutorials/training/INTELLIGENT_TRAINING_TUTORIAL.md`](DOCS/01_tutorials/training/INTELLIGENT_TRAINING_TUTORIAL.md#target-confidence-and-routing).
- **Cross-sectional feature ranking** — Optional panel model for universe-level feature importance. Trains single model across all symbols simultaneously to identify CORE (strong both), SYMBOL_SPECIFIC, CS_SPECIFIC, and WEAK features. Config-controlled via `aggregation.cross_sectional_ranking` in `multi_model.yaml`. Only runs if `len(symbols) >= min_symbols` (default: 5). See [`DOCS/02_reference/configuration/FEATURE_TARGET_CONFIGS.md`](DOCS/02_reference/configuration/FEATURE_TARGET_CONFIGS.md#cross-sectional-ranking-panel-model).
- **LightGBM GPU support** in target ranking with automatic detection and CPU fallback
- **TRAINING module self-contained** — All utilities migrated from `SCRIPTS/` to `TRAINING/utils/`, zero external dependencies

#### Configuration & Logging

- **Modular configuration system** with typed schemas, experiment configs, and validation. Single YAML file defines data, targets, and module overrides. Backward compatible with legacy configs. See [`DOCS/02_reference/configuration/MODULAR_CONFIG_SYSTEM.md`](DOCS/02_reference/configuration/MODULAR_CONFIG_SYSTEM.md).
- **Structured logging configuration** — Per-module and backend verbosity controls via YAML. Profile support (default, debug_run, quiet). No hardcoded logging flags. See [`CONFIG/logging_config.yaml`](CONFIG/logging_config.yaml).
- **Centralized training configs** — 9 YAML files for pipeline, GPU, memory, preprocessing, threading, safety, callbacks, optimizer, and system settings

#### Leakage Safety Suite

- **Production-grade backup system for auto-fixer** — Per-target, timestamped backups with manifests and git commit provenance. Atomic restore operations with retention policy and detailed error handling. Config-driven settings documented in safety config. Full behavior documented in [`DOCS/03_technical/research/LEAKAGE_ANALYSIS.md`](DOCS/03_technical/research/LEAKAGE_ANALYSIS.md).
- **Automated leakage detection and auto-fix** — Automatic detection and remediation with configurable thresholds. Pre-training leak scan, auto-rerun after fixes, and integration with leakage sentinels. See [`DOCS/02_reference/configuration/SAFETY_LEAKAGE_CONFIGS.md`](DOCS/02_reference/configuration/SAFETY_LEAKAGE_CONFIGS.md).
- **Feature/target schema system** — Explicit schema for classifying columns with mode-specific rules (ranking vs training). Feature registry with temporal metadata filtering. See [`CONFIG/feature_target_schema.yaml`](CONFIG/feature_target_schema.yaml) and [`CONFIG/feature_registry.yaml`](CONFIG/feature_registry.yaml).

#### GPU & Training Infra

- Base trainer scaffolding for 2D and 3D models
- Sequential models 3D preprocessing fixes
- XGBoost source-build stability improvements
- TensorFlow GPU initialization fixes

#### Docs & Legal

- **Documentation restructured** — 4-tier hierarchy with centralized docs in `DOCS/`. Code directories contain only code and minimal README pointers. See [`DOCS/INDEX.md`](DOCS/INDEX.md).
- **Roadmap restructured** — Added "What Works Today" section, reorganized priorities, refined wording for external consumption. See [`ROADMAP.md`](ROADMAP.md).
- **Legal documentation suite** — Compliance docs, license enforcement procedures, commercial use guides. See [`DOCS/LEGAL_INDEX.md`](DOCS/LEGAL_INDEX.md).
- **Legal documentation updates** — Enhanced decision matrix, FAQ, and subscription documentation for clarity and completeness. See [`LEGAL/DECISION_MATRIX.md`](LEGAL/DECISION_MATRIX.md), [`LEGAL/FAQ.md`](LEGAL/FAQ.md), [`LEGAL/SUBSCRIPTIONS.md`](LEGAL/SUBSCRIPTIONS.md).
- 55+ new documentation files created, 50+ existing files rewritten and standardized

#### Commercial

- **Commercial license pricing** updated to enterprise quant infrastructure standards (see [`COMMERCIAL_LICENSE.md`](COMMERCIAL_LICENSE.md) for full pricing tiers)
- Optional enterprise add-ons (dedicated support, integration, onboarding, private access)
- Enhanced copyright headers across codebase

---

### Changed

- **Config loading patterns** (2025-12-10) — All function parameters with hardcoded defaults now use `Optional[Type] = None` and load from config when `None`:
  - Feature pruning: `cumulative_threshold`, `min_features`, `n_estimators`, `random_state`
  - Leakage auto-fixer: `min_confidence`, `max_backups_per_target`, `symbol_holdout_test_size`
  - Leakage sentinels: All threshold parameters
  - Training strategies: `test_size`, `random_state` for all model creation
  - Data preprocessing: `test_size`, `random_state` for train/test splits
  - Unified training interface: `seed`, `test_size` load from config
- **Determinism system** (2025-12-10) — All `random_state=42` hardcoded values replaced with `BASE_SEED` from determinism system:
  - Training strategies (`single_task.py`, `cascade.py`)
  - Feature pruning utilities
  - Data preprocessing splits
  - Model creation in strategies

- **Logging system refactored** — Replaced hardcoded flags with structured configuration. All verbosity controlled via YAML without code changes. See [`CONFIG/logging_config.yaml`](CONFIG/logging_config.yaml).
- **Leakage filtering supports ranking mode** — Permissive rules for ranking, strict rules for training. Prevents false positives from overfitting detection.
- All model trainers updated to use centralized configs (preprocessing, callbacks, optimizers, safety guards)
- Pipeline, threading, memory, GPU, and system settings integrated into centralized config system

---

### Fixed

- **Large file refactoring** (2025-12-09) — Split 3 monolithic files into modular components:
  - `models/specialized_models.py`: 4,518 → 82 lines (split into `models/specialized/`)
  - `ranking/rank_target_predictability.py`: 3,454 → 56 lines (split into `ranking/predictability/`)
  - `train_with_strategies.py`: 2,523 → 66 lines (split into `training_strategies/`)
  - All imports remain 100% backward compatible. See [`DOCS/03_technical/refactoring/`](DOCS/03_technical/refactoring/) for details.
- **Missing imports in refactored modules** — Fixed `NameError` and `ImportError` issues:
  - Added missing `time`, `json`, `datetime`, `pandas`, `numpy`, `logging` imports
  - Fixed `time.time()` → `_t.time()` usage (time imported as `_t`)
  - Fixed `feat_cols` to use actual column names after filtering
  - Corrected path resolution (`parents[1]` → `parents[2]`) for repo root
- **Model family status tracking** — Added comprehensive debugging for multi-model feature selection:
  - Tracks success/failure per family per symbol with detailed error info
  - Logs clear summaries showing which families succeeded/failed
  - Persists status to `model_family_status.json` for post-run analysis
  - Logs excluded families during aggregation with error types
  - See [`DOCS/03_technical/debugging/MODEL_FAMILY_STATUS_TRACKING.md`](DOCS/03_technical/debugging/MODEL_FAMILY_STATUS_TRACKING.md)
- **Interval detection robustness** — Fixed timestamp gap filtering:
  - Now filters out insane gaps (> 1 day) before computing median
  - Prevents outliers (weekends, data gaps, bad rows) from contaminating detection
  - Warning only fires if ALL gaps are insane, not just one outlier
  - See [`TRAINING/utils/data_interval.py`](TRAINING/utils/data_interval.py)
- **Feature selection pipeline** — Boruta `X_clean` error, double-counting, feature count mismatches. Interval detection warnings, CatBoost loss function for classification, sklearn NaN/dtype handling. See [`DOCS/02_reference/CHANGELOG_DETAILED.md`](DOCS/02_reference/CHANGELOG_DETAILED.md) for detailed notes.
- **Interval detection** — Fixed negative delta warnings from unsorted timestamps or wraparound. Now uses `abs()` on time deltas before unit detection and conversion. Prevents spurious warnings like "Timestamp delta -789300000000000.0 doesn't map to reasonable interval". See [`TRAINING/utils/data_interval.py`](TRAINING/utils/data_interval.py) and [`TRAINING/ranking/rank_target_predictability.py`](TRAINING/ranking/rank_target_predictability.py).
- **Ranking cache JSON serialization** — Fixed `TypeError: Object of type Timestamp is not JSON serializable` when saving ranking cache. Added `_json_default()` serializer to handle pandas Timestamp, numpy types, and datetime objects. See [`TRAINING/orchestration/intelligent_trainer.py`](TRAINING/orchestration/intelligent_trainer.py).
- **Feature selection interval warnings** — Fixed interval auto-detection warnings in feature selection path by wiring `explicit_interval` from `experiment_config.data.bar_interval` through the feature selection call chain. Eliminates spurious "Timestamp delta doesn't map to reasonable interval" warnings when `data.bar_interval` is configured. See [`TRAINING/orchestration/intelligent_trainer.py`](TRAINING/orchestration/intelligent_trainer.py).
- **Path resolution** — Fixed inconsistent repo root calculations across moved files
- **Auto-fixer** — Import paths, training accuracy detection, pre-excluded feature checks
- **GPU and model issues** — VAE serialization, sequential models 3D preprocessing, XGBoost stability, TensorFlow GPU initialization, LSTM timeouts, Transformer OOM errors
- **Progress logging** — Fixed denominator when using `--max-targets-to-evaluate`

---

### Security

- Enhanced compliance documentation for production use
- License enforcement procedures documented
- Copyright notice requirements standardized

---

### Documentation

- Modular configuration system documentation (see [`DOCS/02_reference/configuration/MODULAR_CONFIG_SYSTEM.md`](DOCS/02_reference/configuration/MODULAR_CONFIG_SYSTEM.md))
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

## Versioning

Releases follow [Semantic Versioning](https://semver.org/):
- **MAJOR** version for incompatible API changes
- **MINOR** version for functionality added in a backwards-compatible manner
- **PATCH** version for backwards-compatible bug fixes

---

## Categories

- **Added** – New features
- **Changed** – Changes in existing functionality
- **Commercial** – Business/pricing/licensing changes
- **Deprecated** – Soon-to-be removed features
- **Removed** – Removed features
- **Fixed** – Bug fixes
- **Security** – Security improvements
- **Documentation** – Documentation changes
