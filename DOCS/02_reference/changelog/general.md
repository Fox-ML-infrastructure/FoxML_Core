# Changelog — General

**Intelligent Training Framework, Leakage Safety Suite, Configuration System, Documentation**

For a quick overview, see the [root changelog](../../../CHANGELOG.md).  
For other dates, see the [changelog index](README.md).

**Note (2025-12-09)**: Large monolithic files have been refactored into modular components for better maintainability. References to `specialized_models.py`, `rank_target_predictability.py`, and `train_with_strategies.py` now refer to thin backward-compatibility wrappers that import from the new modular structure. **All existing imports continue to work unchanged** - this is an internal refactoring with no user-facing API changes. For user-facing documentation, see [Refactoring & Wrappers Guide](../../01_tutorials/REFACTORING_AND_WRAPPERS.md).

---

## Added

### Intelligent Training & Ranking

**Unified Ranking and Selection Pipelines**

- **Unified interval handling** — `explicit_interval` parameter now wired through entire ranking pipeline (orchestrator → rank_targets → evaluate_target_predictability → train_and_evaluate_models). All interval detection respects `data.bar_interval` from experiment config, eliminating spurious auto-detection warnings. Fixed "Nonem" logging issue in interval detection fallback.
- **Interval detection negative delta fix** — Fixed warnings from negative timestamp deltas (unsorted timestamps or wraparound). Now uses `abs()` on time deltas before unit detection and conversion in `TRAINING/utils/data_interval.py` and `TRAINING/ranking/rank_target_predictability.py`. Prevents spurious warnings like "Timestamp delta -789300000000000.0 doesn't map to reasonable interval".
- **Shared sklearn preprocessing** — All sklearn-based models in ranking now use `make_sklearn_dense_X()` helper (same as feature selection) for consistent NaN/dtype/inf handling. Applied to Lasso, Mutual Information, Univariate Selection, Boruta, and Stability Selection.
- **Unified CatBoost builder** — CatBoost in ranking now uses same target type detection and loss function selection as feature selection. Auto-detects classification vs regression and sets appropriate `loss_function` (`Logloss`/`MultiClass`/`RMSE`) with YAML override support.
- **Shared target utilities** — New `TRAINING/utils/target_utils.py` module with reusable helpers (`is_classification_target()`, `is_binary_classification_target()`, `is_multiclass_target()`) used consistently across ranking and selection.

See [`DOCS/01_tutorials/training/RANKING_SELECTION_CONSISTENCY.md`](../../01_tutorials/training/RANKING_SELECTION_CONSISTENCY.md) for complete details.

**Interval Detection Fix**

- **Negative delta handling** — Fixed warnings from negative timestamp deltas (unsorted timestamps or wraparound). Now uses `abs()` on time deltas before unit detection and conversion in `TRAINING/utils/data_interval.py` and `TRAINING/ranking/rank_target_predictability.py`. Prevents spurious warnings like "Timestamp delta -789300000000000.0 doesn't map to reasonable interval". Interval detection is fundamentally about magnitude of the typical step, not direction.

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

**Target Confidence & Routing System**

- **Automatic target quality assessment** — New `compute_target_confidence()` function in `TRAINING/ranking/multi_model_feature_selection.py` computes per-target metrics:
  - Boruta coverage (confirmed/tentative/rejected counts, with `boruta_used` guard to prevent false positives when Boruta is disabled)
  - Model coverage (successful vs available models)
  - Score strength (mean/max scores, plus mean_strong_score for tree ensembles + CatBoost + NN)
  - Agreement ratio (fraction of top-K features appearing in ≥2 models, computed per-target across all symbols)
  - Score tier (orthogonal metric: HIGH/MEDIUM/LOW signal strength based on mean_strong_score and max_score thresholds)
- **Confidence bucketing** — Targets classified into HIGH/MEDIUM/LOW confidence based on configurable thresholds in `CONFIG/feature_selection/multi_model.yaml`:
  - HIGH: All of boruta_confirmed ≥ 5, agreement_ratio ≥ 0.4, mean_score ≥ 0.05, model_coverage ≥ 0.7
  - MEDIUM: Any of boruta_confirmed ≥ 1, agreement_ratio ≥ 0.25, mean_score ≥ 0.02
  - LOW: Fallback with specific reasons (boruta_zero_confirmed, low_model_agreement, low_model_scores, low_model_coverage, multiple_weak_signals)
- **Operational routing** — New `TRAINING/orchestration/target_routing.py` module with `classify_target_from_confidence()` routes targets into buckets:
  - **core**: Production-ready (HIGH confidence, `allowed_in_production: true`)
  - **candidate**: Worth trying (MEDIUM confidence with decent scores, `allowed_in_production: false`)
  - **experimental**: Fragile signal (LOW confidence, especially boruta_zero_confirmed, `allowed_in_production: false`)
- **Configurable thresholds** — All confidence thresholds, score tier thresholds, and routing rules configurable via `CONFIG/feature_selection/multi_model.yaml` `confidence` section. Backward compatible with sensible defaults matching previous hardcoded values.
- **Run-level summaries** — Automatically generates `target_confidence_summary.json` (list of all targets) and `target_confidence_summary.csv` (human-readable table with all metrics + routing decisions) for easy inspection.
- **Integration** — Wired into `intelligent_trainer.py` to automatically compute and log confidence/routing decisions per target after feature selection. Creates run-level summary after training completes. See `TRAINING/orchestration/target_routing.py` and [Intelligent Training Tutorial](../../01_tutorials/training/INTELLIGENT_TRAINING_TUTORIAL.md).
- **Output artifacts**:
  - Per-target: `target_confidence.json`, `target_routing.json`
  - Run-level: `target_confidence_summary.json`, `target_confidence_summary.csv`

**GPU & Training Infrastructure**

- **LightGBM GPU support** in target ranking with automatic detection and usage (CUDA/OpenCL), GPU verification diagnostics, and fallback to CPU if GPU unavailable
- **TRAINING module self-contained** — Moved all utility dependencies from `SCRIPTS/` to `TRAINING/utils/`. TRAINING module now has zero dependencies on `SCRIPTS/` folder.
- Base trainer scaffolding for 2D and 3D models (`base_2d_trainer.py`, `base_3d_trainer.py`)

### Configuration & Logging

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

See [`DOCS/02_reference/configuration/MODULAR_CONFIG_SYSTEM.md`](../configuration/MODULAR_CONFIG_SYSTEM.md) for complete documentation.

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

### Leakage Safety Suite

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

See [`DOCS/02_reference/configuration/SAFETY_LEAKAGE_CONFIGS.md`](../configuration/SAFETY_LEAKAGE_CONFIGS.md) for complete documentation.

### Documentation & Legal

**Documentation Restructure**

- **4-tier documentation hierarchy** implemented:
  - Tier A: Executive / High-Level
  - Tier B: Tutorials / Walkthroughs
  - Tier C: Core Reference Docs
  - Tier D: Technical Deep Dives
- **Documentation centralized** in `DOCS/` folder:
  - Moved all CONFIG documentation to `DOCS/02_reference/configuration/`
  - Moved all TRAINING documentation to `DOCS/` folder
  - Code directories now contain only code and minimal README pointers
- **Comprehensive legal documentation index** (`DOCS/LEGAL_INDEX.md`):
  - Complete index of all legal, licensing, compliance, and enterprise documentation
  - Organized by category: Licensing, Terms & Policies, Enterprise & Compliance, Security, Legal Agreements, Consulting Services
- **Legal documentation updates**:
  - Enhanced decision matrix (`LEGAL/DECISION_MATRIX.md`) for clarity on licensing decisions and use case classification
  - Updated FAQ (`LEGAL/FAQ.md`) with comprehensive answers to common questions about commercial licensing, AGPL usage, and subscription tiers
  - Refined subscription documentation (`LEGAL/SUBSCRIPTIONS.md`) for better clarity on business use, academic use, and license requirements
- **Target confidence & routing documentation**:
  - Added section to [Feature & Target Configs](../configuration/FEATURE_TARGET_CONFIGS.md) documenting confidence thresholds and routing rules
  - Updated [Intelligent Training Tutorial](../../01_tutorials/training/INTELLIGENT_TRAINING_TUTORIAL.md) with confidence/routing section and updated output structure
  - Updated `CONFIG/feature_selection/README.md` with confidence/routing information
- **Cross-linking and navigation** improved throughout all documentation
- 55+ new documentation files created, 50+ existing files rewritten and standardized

**Roadmap Restructure**

- Added "What Works Today" section highlighting immediate production-ready capabilities
- Renamed Phase 3.5 to Phase 4 (multi-GPU & NVLink exploration)
- Reorganized development priorities into near-term focus and longer-term/R&D categories
- Removed date-specific targets in favor of general development guidelines
- Added branding clarification (FoxML Core vs Fox ML Infrastructure)
- Refined wording throughout for external/enterprise consumption

See [`ROADMAP.md`](../../02_reference/roadmap/ROADMAP.md) for complete roadmap.

**Compliance Documentation Suite**

- `LICENSE_ENFORCEMENT.md` – License enforcement procedures and compliance requirements
- `COMMERCIAL_USE.md` – Quick reference guide for commercial use
- `COMPLIANCE_FAQ.md` – Frequently asked compliance questions
- `PRODUCTION_USE_NOTIFICATION.md` – Production use notification form
- `COPYRIGHT_NOTICE.md` – Copyright notice requirements

See [`DOCS/LEGAL_INDEX.md`](../../LEGAL_INDEX.md) for complete legal documentation index.

### Commercial

- **Commercial license pricing** restructured for better market fit and accessibility:
  - **Paid pilots (non-negotiable entry point):** Pilot (30 days): $15,000–$30,000; Pilot+ (60–90 days): $35,000–$90,000 (50–100% credit toward first-year license)
  - **Team/Desk-based tiers** (based on using team/desk size, not total company headcount):
    - Team (1–5 users, 1 env): $75,000/year
    - Desk (6–20 users, up to 2 env): $150,000–$300,000/year
    - Division (21–75 users, up to 3 env): $350,000–$750,000/year
    - Enterprise (76–250 users, multi-env): $800,000–$2,000,000/year
    - >250 users / multi-region / regulated bank: Custom $2,000,000+/year
  - **Alternative: Platform fee + per-seat model** (platform: $50k–$200k/year, per-seat: $5k–$15k/user/year, per-env: $25k–$100k/env/year)
  - **Support add-ons** (priced as 15–30% of license, not separate giant contracts):
    - Standard Support (included): Email/issues, best-effort, no SLA
    - Business Support: +$25,000/year
    - Enterprise Support: +$75,000–$150,000/year
    - Premium / 24×5 Support: +$200,000–$400,000/year (only if we can staff it)
  - **Services** (founder-led, realistic pricing):
    - Onboarding / Integration: $25,000–$150,000 one-time
    - On-Prem / High-Security Deployment: $150,000–$500,000+ one-time (custom)
  - **Policy:** Licensing scoped to business unit/desk, not parent company total headcount
- Enhanced copyright headers across codebase (2025-2026 Fox ML Infrastructure LLC)

See [`COMMERCIAL_LICENSE.md`](../../../../COMMERCIAL_LICENSE.md) and [`LEGAL/SUBSCRIPTIONS.md`](../../../../LEGAL/SUBSCRIPTIONS.md) for complete pricing details.

---

## Changed

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

## Security

- Enhanced compliance documentation for production use
- License enforcement procedures documented
- Copyright notice requirements standardized

---

## Documentation

- Roadmap restructured for external consumption (see [`ROADMAP.md`](../../02_reference/roadmap/ROADMAP.md))
- Modular configuration system documentation (see [`DOCS/02_reference/configuration/MODULAR_CONFIG_SYSTEM.md`](../configuration/MODULAR_CONFIG_SYSTEM.md))
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

## Related

- [Changelog Index](README.md)
- [Root Changelog](../../../CHANGELOG.md)
- [2025-12-11](2025-12-11.md)
- [2025-12-10](2025-12-10.md)
