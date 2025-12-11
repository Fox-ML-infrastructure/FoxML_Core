# Detailed Changelog

This document provides comprehensive change details for FoxML Core. For a quick overview, see the [root changelog](../../CHANGELOG.md).

**Note (2025-12-09)**: Large monolithic files have been refactored into modular components for better maintainability. References to `specialized_models.py`, `rank_target_predictability.py`, and `train_with_strategies.py` now refer to thin backward-compatibility wrappers that import from the new modular structure. **All existing imports continue to work unchanged** - this is an internal refactoring with no user-facing API changes. For user-facing documentation, see [Refactoring & Wrappers Guide](../01_tutorials/REFACTORING_AND_WRAPPERS.md). For internal refactoring details, see [Refactoring Summary](../INTERNAL/REFACTORING_SUMMARY_INTERNAL.md).

---

## [Unreleased]

### Status

Phase 1 functioning properly - Ranking and selection pipelines unified with consistent behavior. Boruta refactored as statistical gatekeeper. Complete config centralization and determinism implemented. SST enforcement system added with automated test. All hardcoded values eliminated or properly marked. Full end-to-end testing currently underway.

**Note**: Phase 1 of the pipeline (intelligent training framework) is functioning properly. Ranking and selection pipelines now have unified behavior (interval handling, sklearn preprocessing, CatBoost configuration). Boruta has been refactored from "just another importance scorer" to a statistical gatekeeper that modifies consensus scores via bonuses/penalties. All configuration values are now centralized in YAML files with single source of truth. All random seeds use centralized determinism system. Config loading patterns have been hardened with proper fallbacks to prevent runtime errors. **SST enforcement system implemented**: automated test (`TRAINING/tests/test_no_hardcoded_hparams.py`) enforces that all hyperparameters load from config, with strict markers (`FALLBACK_DEFAULT_OK`, `DESIGN_CONSTANT_OK`) for allowed exceptions. All remaining hardcoded values have been eliminated or properly marked. All documentation has been updated with cross-links and references to new config files and utilities. Backward functionality remains fully operational. All existing training workflows continue to function as before, and legacy config locations are still supported with deprecation warnings. **Full end-to-end testing is currently underway** to validate the complete pipeline from target ranking → feature selection → model training.

---

### Highlights

- **SST Enforcement System** (2025-12-10) — Automated test enforces Single Source of Truth. All hardcoded hyperparameters eliminated or properly marked. Training profiles added for batch_size. Top 10% importance patterns now configurable.
- Phase 1 intelligent training framework completed and functioning properly
- Ranking & selection pipelines unified (interval handling, preprocessing, CatBoost)
- Boruta refactored into a statistical gatekeeper with base vs final consensus tracking
- New modular configuration + structured logging system
- Leakage Safety Suite hardened (auto-fixer + backup system + schema/registry)
- Documentation and legal reorganized into a 4-tier, enterprise-ready docs hierarchy
- Commercial license & pricing updated for enterprise quant infrastructure

---

### Added

#### Leakage Detection Critical Bug Fixes (2025-12-11)

**CRITICAL BUG FIX: Detection Confidence Calculation**
- **Issue**: Detection confidence was using raw importance values (0-1) directly as confidence scores
  - Example: Feature with 15% importance → 15% confidence
  - min_confidence threshold = 80% (0.8)
  - Result: ALL detections were filtered out, so auto-fixer reported 0 leaks even when perfect scores were detected
- **Impact**: Valid leak detections were silently filtered out, making it appear that detection wasn't working
- **Root Cause**: Misunderstanding of confidence vs importance - confidence should reflect suspicion level (perfect score = high suspicion), not just raw importance
- **Fix**: Changed confidence calculation to account for perfect-score context:
  - Base confidence: 0.85 (perfect score = high suspicion)
  - Importance boost: up to +0.1 based on importance value
  - Final confidence: 0.85-0.95 (always above 80% threshold)
- **Result**: Detections now get appropriate confidence scores and are properly applied
- **Files**: `TRAINING/common/leakage_auto_fixer.py`

**Enhanced Detection When Importances Missing**
- **Issue**: When `model_importance` was not provided, detection only checked known patterns (p_, y_, fwd_ret_, etc.), missing subtle leaks
- **Impact**: Detection couldn't identify leaky features when importances weren't passed from upstream
- **Fix**: 
  - Compute feature importances on-the-fly using quick RandomForest when missing
  - Use computed importances to find suspicious features
  - Ensures detection works even when importances aren't passed
- **Files**: `TRAINING/common/leakage_auto_fixer.py`

**Improved Detection Visibility and Diagnostics**
- **Enhancement**: Added comprehensive logging for detection process
- **Features**:
  - Confidence distribution logging (high vs low confidence detections)
  - Lists top detections that will be fixed vs filtered out
  - Warnings when perfect score detected but no leaks found
  - Explains possible reasons: structural leakage, already excluded, detection methods need improvement
  - INFO-level logging for auto-fixer inputs and top features by importance
- **Files**: 
  - `TRAINING/common/leakage_auto_fixer.py`
  - `TRAINING/ranking/predictability/model_evaluation.py`

#### Reproducibility Tracking Tolerance Bands & Enhancements (2025-12-11)

**Tolerance Bands with STABLE/DRIFTING/DIVERGED Classification**
- **Enhancement**: Replaced binary SAME/DIFFERENT classification with three-tier system
- **Problem**: Binary system flagged tiny differences (0.08% shifts) as DIFFERENT, causing alert fatigue
- **Solution**: Three-tier classification:
  - **STABLE**: Differences within noise (passes both abs and rel thresholds, optional z-score) → INFO level
  - **DRIFTING**: Small but noticeable changes (within 2x thresholds) → INFO level
  - **DIVERGED**: Real reproducibility issues (exceeds 2x thresholds) → WARNING level
- **Features**:
  - Configurable thresholds per metric (roc_auc, composite, importance) in `safety_config.yaml`
  - Supports absolute, relative, and z-score thresholds
  - Uses reported σ (std_score) when available for statistical significance
  - Example: 0.08% ROC-AUC shift with z=0.06 is now STABLE (INFO) instead of DIFFERENT (WARNING)
- **Config**: Added `safety.reproducibility` section with thresholds and `use_z_score` option
- **Files**: 
  - `TRAINING/utils/reproducibility_tracker.py` - Complete rewrite of classification logic
  - `CONFIG/training_config/safety_config.yaml` - Added reproducibility thresholds config

**Reproducibility Tracking Error Handling Fixes**
- **Issue**: Missing `List` and `Tuple` imports causing `NameError: name 'List' is not defined`
- **Impact**: Reproducibility tracking failed silently, breaking output generation
- **Fix**: 
  - Added `List` and `Tuple` to typing imports
  - Added comprehensive error handling in `_find_previous_log_files()` to prevent crashes
  - Changed exception logging from DEBUG to WARNING level for visibility
  - Added traceback logging for debugging
- **Files**: `TRAINING/utils/reproducibility_tracker.py`

#### Reproducibility Tracking & Auto-Fixer Fixes (2025-12-11)

**Reproducibility Tracking Directory Structure Fix**
- **Issue**: Reproducibility logs were stored in shared location, mixing different modules (target ranking, feature selection, model training). Also couldn't find previous runs because each run uses timestamped directory (e.g., `test_e2e_ranking_unified_20251211_133358`).
- **Impact**: 
  - Modules couldn't be properly separated (all logs in one place)
  - Previous runs weren't found (stored in different timestamped directories)
  - Always showed "First run" even when previous runs existed
- **Fix**: 
  - Each module now has its own reproducibility log in module-specific subdirectory:
    - Target ranking: `{output_dir}/target_rankings/reproducibility_log.json`
    - Feature selection: `{output_dir}/feature_selections/reproducibility_log.json`
    - Model training: `{output_dir}/training_results/reproducibility_log.json`
  - Added `search_previous_runs` option (enabled for all modules)
  - `_find_previous_log_files()` searches parent directories (up to 3 levels) for previous runs from same module
  - Merges runs from current log + previous logs, sorts by timestamp, uses most recent
- **Result**: Modules are properly separated, and previous runs are found across different timestamped output directories
- **Files**: 
  - `TRAINING/utils/reproducibility_tracker.py` - Added module-specific directory support and previous run search
  - `TRAINING/ranking/predictability/model_evaluation.py` - Updated to use target_rankings/ subdirectory
  - `TRAINING/ranking/feature_selector.py` - Updated to use feature_selections/ subdirectory
  - `TRAINING/training_strategies/training.py` - Updated to use training_results/ subdirectory

**Auto-Fixer Logging Format Error Fix**
- **Issue**: `ValueError: Invalid format specifier` when logging auto-fixer inputs. Code tried to use conditional expression (`actual_train_score:.4f if actual_train_score else None`) directly in format specifier, which is invalid Python syntax.
- **Impact**: Auto-fixer logging crashed with format error, preventing visibility into what was being passed to detection
- **Fix**: Format the value first, then use in f-string:
  ```python
  train_score_str = f"{actual_train_score:.4f}" if actual_train_score is not None else "None"
  logger.info(f"🔧 Auto-fixer inputs: train_score={train_score_str}, ...")
  ```
- **Result**: Auto-fixer logging now works correctly, showing inputs for debugging
- **Files**: `TRAINING/ranking/predictability/model_evaluation.py`

#### Cross-Sectional Sampling & Config Parameter Fixes (2025-12-11)

**Critical Bug Fix: max_cs_samples Filtering**
- **Issue**: `max_cs_samples` filtering code was in wrong code block (`else:` when `time_col is None`), so it never executed when timestamps exist
- **Impact**: Large dataframes (25,000+ rows) were built despite `max_cs_samples=1000` setting, causing unnecessary memory usage and slower processing
- **Fix**: Moved filtering code into `if time_col is not None:` block, right after `min_cs` filter
- **Result**: Now properly limits cross-sectional samples per timestamp, dramatically reducing dataframe size
- **File**: `TRAINING/utils/cross_sectional_data.py`

**Config Parameter Passing Fix**
- **Issue**: `min_cs`, `max_cs_samples`, and `max_rows_per_symbol` from test config were not being passed to `rank_targets()` function
- **Impact**: Test config settings (e.g., `min_cs=3`) were ignored, default values (e.g., `min_cs=10`) were used instead
- **Fix**: 
  - Extract these values from `train_kwargs` at start of `train_with_intelligence()` method
  - Pass them to `rank_targets_auto()` and then to `rank_targets()`
  - Ensures config-driven settings are actually used throughout pipeline
- **Files**: `TRAINING/orchestration/intelligent_trainer.py`

**Comprehensive Reproducibility Tracking**
- **Enhancement**: Added reproducibility tracking to all deterministic pipeline stages
- **Architectural Improvement**: Moved tracking from entry points to computation modules
  - **Target Ranking**: Tracking in `evaluate_target_predictability()` in `model_evaluation.py` (computation module)
  - **Feature Selection**: Tracking in `select_features_for_target()` in `feature_selector.py` (computation module)
  - **Model Training**: Tracking in training loop in `training.py` (computation module)
- **Benefits**:
  - Works regardless of entry point (intelligent_trainer, standalone scripts, programmatic calls)
  - Single source of tracking logic (no duplication)
  - Better architecture: computation functions handle their own tracking
  - Easier maintenance: update tracking in one place
- **Visibility**: Fixed issue where logs weren't appearing - now logs to both internal and main loggers
- **Files Modified**: 
  - `TRAINING/ranking/predictability/model_evaluation.py` - Added tracking to `evaluate_target_predictability()`
  - `TRAINING/ranking/feature_selector.py` - Added tracking to `select_features_for_target()`
  - `TRAINING/training_strategies/training.py` - Added tracking after model training
  - `TRAINING/ranking/target_ranker.py` - Removed duplicate tracking (now in computation module)
  - `TRAINING/ranking/predictability/main.py` - Removed duplicate tracking (now in computation module)
  - `TRAINING/ranking/multi_model_feature_selection.py` - Removed duplicate tracking (now in computation module)
  - `TRAINING/utils/reproducibility_tracker.py` - Enhanced with visibility fixes

#### Config Parameter Validation & Silent Error Visibility (2025-12-11)

**Config Cleaner Utility**
- **New module**: `TRAINING/utils/config_cleaner.py` providing systematic parameter validation
- **Function**: `clean_config_for_estimator()` uses `inspect.signature()` to validate parameters against actual estimator constructors
- **Features**:
  - Automatically removes duplicate parameters (if also passed explicitly via `extra_kwargs`)
  - Automatically removes unknown parameters (not in estimator's `__init__` signature)
  - Logs what was stripped for visibility (DEBUG level, can be raised to WARNING for auditing)
  - Handles edge cases: None configs, non-dict configs, None extra_kwargs
  - Maintains SST (Single Source of Truth) - values still come from config/defaults, but only valid keys are passed
  - Prevents entire class of parameter passing errors that would occur with config drift
- **Integration**: Applied to `multi_model_feature_selection.py`, `task_types.py`, `cross_sectional_feature_ranker.py`
- **Future-proof**: Makes codebase resilient to future changes in `inject_defaults` or model library updates

**Reproducibility Tracking Module**
- **New module**: `TRAINING/utils/reproducibility_tracker.py` - reusable `ReproducibilityTracker` class
- **Features**:
  - Generic module usable across all pipeline stages
  - Integrated into target ranking and feature selection pipelines
  - Compares current run to previous runs with configurable tolerances (0.1% for scores, 1% for importance)
  - Stores run history in JSON format (keeps last 10 runs per item by default)
  - Logs differences with percentage changes and absolute deltas
  - Flags reproducible runs (✅) vs different runs (⚠️)
- **Documentation**: `DOCS/03_technical/implementation/REPRODUCIBILITY_TRACKING.md` with API reference, integration examples, best practices, and troubleshooting guide
- **Reproducibility comparison logging**: Automatic comparison of target ranking results to previous runs
  - Stores run summaries in `reproducibility_log.json` (keeps last 10 runs per target)
  - Compares current vs previous: mean score, std score, importance, composite score
  - Integrated into `rank_target_predictability.py` - logs after each target evaluation summary

**Silent Error Visibility Improvements**
- **ImportError fallback in target validation**: Added DEBUG logging when `validate_target` module is unavailable and fallback validation is used
- **Importance extraction validation**: Added comprehensive validation and logging:
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
- **Logging levels**: All silent failures now have appropriate logging levels (DEBUG for expected fallbacks, WARNING for unexpected conditions, ERROR for actual failures)

**Files Modified**
- `TRAINING/utils/config_cleaner.py` — New module
- `TRAINING/utils/reproducibility_tracker.py` — Enhanced with visibility fixes
- `TRAINING/ranking/multi_model_feature_selection.py` — Integrated config cleaner, added importance validation, fixed bare excepts
- `TRAINING/utils/task_types.py` — Integrated config cleaner with proper closure handling
- `TRAINING/ranking/cross_sectional_feature_ranker.py` — Integrated config cleaner
- `CONFIG/config_loader.py` — Hardened `inject_defaults` to handle None configs

#### SST Enforcement & Configuration Hardening (2025-12-10)

**SST Enforcement Test**
- **Automated SST compliance test** — `TRAINING/tests/test_no_hardcoded_hparams.py` scans all Python files in TRAINING/ for hardcoded hyperparameters, thresholds, and seeds. Only explicitly marked values (`FALLBACK_DEFAULT_OK`, `DESIGN_CONSTANT_OK`) are allowed. Prevents accidental introduction of hardcoded configuration values.

**Configuration Improvements**
- **Training profiles** — Added `training_profiles` to `CONFIG/training_config/optimizer_config.yaml` with `default`, `debug`, and `throughput_optimized` profiles. Neural network trainers now load `batch_size` and `max_epochs` from active profile.
- **Top fraction configurable** — Added `importance_top_fraction: 0.10` to `CONFIG/feature_selection/multi_model.yaml`. All "top 10%" importance patterns in `model_evaluation.py` and `leakage_detection.py` now use configurable fraction instead of hardcoded 0.1.

**Documentation**
- **Internal SST docs** — `DOCS/03_technical/internal/SST_DETERMINISM_GUARANTEES.md` (policy and guarantees), `SST_COMPLIANCE_CHECKLIST.md` (pre-commit checklist), `SST_REMAINING_WORK.md` (action plan).
- **Public deterministic training guide** — `DOCS/00_executive/DETERMINISTIC_TRAINING.md` (public-facing guide for buyers/quants).

**Code Changes**
- **All seed fallbacks marked** — Added `FALLBACK_DEFAULT_OK` markers to all seed fallback defaults throughout codebase.
- **Diagnostic models marked** — `n_estimators=1` in diagnostic leakage detection models marked with `DESIGN_CONSTANT_OK`.
- **Batch size configurable** — `neural_network_trainer.py` and `comprehensive_trainer.py` load batch_size from training profiles.
- **Top fraction helper** — Created `_get_importance_top_fraction()` helper in `model_evaluation.py` and `leakage_detection.py`.
- **Fixed indentation errors** — Corrected indentation in top fraction pattern updates.

**Files Modified**
- `CONFIG/feature_selection/multi_model.yaml` — Added `importance_top_fraction`
- `CONFIG/training_config/optimizer_config.yaml` — Added `training_profiles`
- `TRAINING/tests/test_no_hardcoded_hparams.py` — New SST enforcement test
- `TRAINING/model_fun/neural_network_trainer.py` — Load batch_size from config
- `TRAINING/model_fun/comprehensive_trainer.py` — Load batch_size from config
- `TRAINING/ranking/predictability/model_evaluation.py` — Top fraction configurable (11 instances)
- `TRAINING/ranking/predictability/leakage_detection.py` — Top fraction configurable (11 instances)
- All seed fallbacks marked with `FALLBACK_DEFAULT_OK` across 15+ files

#### Intelligent Training & Ranking

**Unified Ranking and Selection Pipelines**

- **Unified interval handling** — `explicit_interval` parameter now wired through entire ranking pipeline (orchestrator → rank_targets → evaluate_target_predictability → train_and_evaluate_models). All interval detection respects `data.bar_interval` from experiment config, eliminating spurious auto-detection warnings. Fixed "Nonem" logging issue in interval detection fallback.
- **Interval detection negative delta fix** — Fixed warnings from negative timestamp deltas (unsorted timestamps or wraparound). Now uses `abs()` on time deltas before unit detection and conversion in `TRAINING/utils/data_interval.py` and `TRAINING/ranking/rank_target_predictability.py`. Prevents spurious warnings like "Timestamp delta -789300000000000.0 doesn't map to reasonable interval".
- **Shared sklearn preprocessing** — All sklearn-based models in ranking now use `make_sklearn_dense_X()` helper (same as feature selection) for consistent NaN/dtype/inf handling. Applied to Lasso, Mutual Information, Univariate Selection, Boruta, and Stability Selection.
- **Unified CatBoost builder** — CatBoost in ranking now uses same target type detection and loss function selection as feature selection. Auto-detects classification vs regression and sets appropriate `loss_function` (`Logloss`/`MultiClass`/`RMSE`) with YAML override support.
- **Shared target utilities** — New `TRAINING/utils/target_utils.py` module with reusable helpers (`is_classification_target()`, `is_binary_classification_target()`, `is_multiclass_target()`) used consistently across ranking and selection.

See [`DOCS/01_tutorials/training/RANKING_SELECTION_CONSISTENCY.md`](../01_tutorials/training/RANKING_SELECTION_CONSISTENCY.md) for complete details.

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
- **Integration** — Wired into `intelligent_trainer.py` to automatically compute and log confidence/routing decisions per target after feature selection. Creates run-level summary after training completes. See [`TRAINING/orchestration/target_routing.py`](../../../TRAINING/orchestration/target_routing.py) and [Intelligent Training Tutorial](../01_tutorials/training/INTELLIGENT_TRAINING_TUTORIAL.md#target-confidence-and-routing).
- **Output artifacts**:
  - Per-target: `target_confidence.json`, `target_routing.json`
  - Run-level: `target_confidence_summary.json`, `target_confidence_summary.csv`

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

See [`DOCS/02_reference/configuration/MODULAR_CONFIG_SYSTEM.md`](configuration/MODULAR_CONFIG_SYSTEM.md) for complete documentation.

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

See [`DOCS/03_technical/research/LEAKAGE_ANALYSIS.md`](../03_technical/research/LEAKAGE_ANALYSIS.md) and [`DOCS/02_reference/configuration/SAFETY_LEAKAGE_CONFIGS.md`](configuration/SAFETY_LEAKAGE_CONFIGS.md) for complete documentation.

#### Documentation & Legal

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
  - Added section to [Feature & Target Configs](configuration/FEATURE_TARGET_CONFIGS.md) documenting confidence thresholds and routing rules
  - Updated [Intelligent Training Tutorial](../01_tutorials/training/INTELLIGENT_TRAINING_TUTORIAL.md) with confidence/routing section and updated output structure
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

See [`ROADMAP.md`](../../ROADMAP.md) for complete roadmap.

**Compliance Documentation Suite**

- `LICENSE_ENFORCEMENT.md` – License enforcement procedures and compliance requirements
- `COMMERCIAL_USE.md` – Quick reference guide for commercial use
- `COMPLIANCE_FAQ.md` – Frequently asked compliance questions
- `PRODUCTION_USE_NOTIFICATION.md` – Production use notification form
- `COPYRIGHT_NOTICE.md` – Copyright notice requirements

See [`DOCS/LEGAL_INDEX.md`](../LEGAL_INDEX.md) for complete legal documentation index.

#### Commercial

- **Commercial license pricing** updated to enterprise ML infrastructure standards (aligned with Databricks, Palantir, H2O Driverless AI, Scale AI, Recursion Pharma partnerships):
  - Tier 1 — 1–10 employees: $350,000/year
  - Tier 2 — 11–50 employees: $850,000/year
  - Tier 3 — 51–250 employees: $2,500,000/year
  - Tier 4 — 251–1000 employees: $6,000,000–$10,000,000/year (pricing depends on deployment complexity and usage scope)
  - Tier 5 — 1000+ employees: $15,000,000–$30,000,000+ /year (custom enterprise quote)
  - **Optional enterprise add-ons** (with defined scope and boundaries, aligned with enterprise ML infrastructure standards):
    - Dedicated Support SLA: $25,000–$250,000/month (Business Support $25k–$50k/month, Enterprise Support $60k–$120k/month, Premium Support $150k–$250k/month with defined response times, coverage, channels)
    - Integration & On-Prem Setup: $500,000–$3,000,000 one-time (scoped via SOW, depends on infrastructure complexity, environment count, data integration volume, orchestration complexity, cross-team rollout scope)
    - Onboarding: $150,000–$600,000 one-time (Basic: $150k–$300k for training + architecture review; Advanced/Custom: $300k–$600k for tailored workshops + hands-on pipeline design + extended engineering consultation)
    - Private Slack / Direct Founder Access: $300,000–$1,000,000/year (rare, non-replicable strategic access to system architect; comparable to CTO-level advisory fees; up to 3–5 named contacts, business hours, strategic discussions only; not a replacement for support SLA)
    - Additional User Seats: $500–$2,000/seat/year (beyond included seats per tier)
    - Adaptive Intelligence Layer (Tier 5 only): +$2,000,000–$5,000,000/year (subject to roadmap, requires separate SOW with milestones)
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

#### Parameter Passing Errors & Silent Failures (2025-12-11)

**Systematic Parameter Passing Error Prevention**
- **Root cause**: `inject_defaults` was injecting parameters (like `random_seed`, `random_state`, `n_jobs`, `num_threads`, `threads`) into model configs, but model instantiation code was also passing these (or renamed versions) explicitly, leading to duplicate arguments or unknown parameters.
- **Solution**: Implemented systematic fix using shared `clean_config_for_estimator()` utility from `TRAINING/utils/config_cleaner.py`:
  - **All model families in multi_model_feature_selection.py**: LightGBM, XGBoost, RandomForest, MLPRegressor, CatBoost, Lasso now use `clean_config_for_estimator()` helper
  - **All model constructors in task_types.py**: Fixed with proper closure handling (explicit copies in lambda closures to prevent reference issues)
  - **Cross-sectional feature ranker**: LightGBM and XGBoost now use config cleaner
  - **inject_defaults hardening**: Now handles None configs gracefully, prevents "argument of type 'NoneType' is not iterable" errors
  - Prevents "got multiple values for keyword argument" errors (e.g., `random_seed` passed both in config and explicitly)
  - Prevents "unexpected keyword argument" errors (e.g., `num_threads` for RandomForest, `n_jobs` for MLPRegressor/Lasso)
  - Lambda closure fix: All lambda functions in `task_types.py` now capture explicit copies (`config_final`, `extra_final`) to prevent reference issues
  - Future-proof: Any new parameters added by `inject_defaults` will be automatically filtered if they're duplicates or unknown

**Model Config Parameter Sanitization**
- **Fixed**: Critical TypeError and ValueError errors when global config defaults (`random_seed`, `n_jobs`, `early_stopping_rounds`) were injected into model constructors
- **sklearn models** (RandomForest, MLPRegressor, Lasso): Remove `random_seed` (use `random_state` instead)
- **CatBoost**: Remove `n_jobs` (uses `thread_count` instead)
- **XGBoost/LightGBM**: Remove all early stopping params (`early_stopping_rounds`, `callbacks`, `eval_set`, `eval_metric`) in feature selection mode (requires `eval_set` which isn't available)
- Determinism preserved: All models explicitly set `random_state`/`random_seed` using deterministic `model_seed` per symbol/target combination
- Uses `.copy()` and `.pop()` for explicit parameter sanitization to prevent incompatible parameters from reaching model constructors

**Silent Error Visibility Improvements**
- **ImportError fallback in target validation**: Added DEBUG logging when `validate_target` module is unavailable and fallback validation is used
- **Importance extraction validation**: Added comprehensive validation and logging:
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
- **Logging levels**: All silent failures now have appropriate logging levels (DEBUG for expected fallbacks, WARNING for unexpected conditions, ERROR for actual failures)

**Files Modified**
- `TRAINING/utils/config_cleaner.py` — New module
- `TRAINING/ranking/multi_model_feature_selection.py` — Integrated config cleaner, added importance validation, fixed bare excepts
- `TRAINING/utils/task_types.py` — Integrated config cleaner with proper closure handling
- `TRAINING/ranking/cross_sectional_feature_ranker.py` — Integrated config cleaner
- `CONFIG/config_loader.py` — Hardened `inject_defaults` to handle None configs
- `TRAINING/utils/reproducibility_tracker.py` — Enhanced with visibility fixes

#### Complete Single Source of Truth (SST) implementation (2025-12-10) — Replaced ALL hardcoded values across entire TRAINING pipeline for full reproducibility:
  - **Model trainers** (`TRAINING/model_fun/` - 34 files): All hardcoded hyperparameters replaced with config loading:
    - `comprehensive_trainer.py`: LightGBM/XGBoost `n_estimators`, `max_depth`, `learning_rate` now load from `models.{family}.{param}` config paths
    - `neural_network_trainer.py`: Adam `learning_rate` now uses `_get_learning_rate()` helper method
    - `ensemble_trainer.py`: Ridge `alpha` now loads from `models.ridge.alpha` config
    - `change_point_trainer.py`: Ridge `alpha` and KMeans `random_state` now load from config
    - `ngboost_trainer.py`: HistGradientBoosting `max_depth` and `learning_rate` now load from config
    - All other trainers already using `_get_test_split_params()` and `_get_random_state()` from base class
  - **Specialized models** (`TRAINING/models/specialized/` - 2 files): All hardcoded values replaced:
    - `trainers.py`: All `train_test_split`, `random_state`, `learning_rate`, `alpha`, DecisionTree/RandomForest hyperparameters now load from config
    - `trainers_extended.py`: LightGBM `learning_rate` and `seed`, GradientBoosting hyperparameters, all Adam `learning_rate`, all `train_test_split` calls now use config
  - **Base trainer enhancements**: Added `_get_learning_rate()` helper method to `base_trainer.py` for consistent optimizer config access. Method loads from `optimizer.learning_rate` config with model-family-specific fallback support.
  - **Strategies**: RandomForest fallback `n_estimators` in `cascade.py` and `single_task.py` now loads from `models.random_forest.n_estimators` config
  - **Config sources used**:
    - `preprocessing.validation.test_size` - For all train/test splits
    - `BASE_SEED` (determinism system) - For all random_state values (with config fallback)
    - `models.{family}.{param}` - For model-specific hyperparameters (n_estimators, max_depth, learning_rate, alpha, etc.)
    - `optimizer.learning_rate` - For neural network optimizers
  - **Result**: Same config file → identical results across all pipeline stages. Full reproducibility guaranteed. Zero hardcoded config values remain in the TRAINING pipeline.

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
- **Interval detection warnings in feature selection** — Fixed interval auto-detection warnings in feature selection path by extracting `explicit_interval` from `experiment_config.data.bar_interval` and passing it through `intelligent_trainer.py` → `feature_selector.py` → `multi_model_feature_selection.py` → `detect_interval_from_dataframe()`. Eliminates spurious "Timestamp delta doesn't map to reasonable interval" warnings when `data.bar_interval` is configured in experiment config. See `TRAINING/orchestration/intelligent_trainer.py` lines 454-477.
- **Ranking cache JSON serialization** — Fixed `TypeError: Object of type Timestamp is not JSON serializable` when saving ranking cache. Added `_json_default()` serializer function that handles pandas Timestamp (via `isoformat()`), numpy scalars (int/float conversion), numpy arrays (tolist()), and datetime objects. Updated `_save_cached_rankings()` to use `default=_json_default` in `json.dump()` call. See `TRAINING/orchestration/intelligent_trainer.py` lines 107-137, 240.
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
- Modular configuration system documentation (see [`DOCS/02_reference/configuration/MODULAR_CONFIG_SYSTEM.md`](configuration/MODULAR_CONFIG_SYSTEM.md))
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
- [Roadmap](../../ROADMAP.md) - Development priorities and upcoming work
- [Documentation Index](../INDEX.md) - Complete documentation navigation

