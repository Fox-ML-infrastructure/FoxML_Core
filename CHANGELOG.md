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

#### 2025-12-22 (Performance Audit System for Multiplicative Work Detection)
- **Feature**: Added comprehensive performance audit system to detect "accidental multiplicative work"
- **Instrumentation**: Tracks call counts and timing for heavy functions (CatBoost importance, build_panel, train_model, etc.)
- **Automatic Reports**: Generates audit report at end of training run showing multipliers, nested loops, and cache opportunities
- **Impact**: Proactively identifies performance bottlenecks where expensive operations are called multiple times unnecessarily
- **Files Changed**: `performance_audit.py` (NEW), `intelligent_trainer.py`, `multi_model_feature_selection.py`, `shared_ranking_harness.py`, `model_evaluation.py`, `leakage_detection.py`
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-22-performance-audit-system.md)

#### 2025-12-22 (Training Results Organization and Pipeline Integrity Fixes)
- **Bug Fix**: Fixed nested `training_results/training_results/` folder structure - models now save to simple `training_results/<family>/` structure
- **Bug Fix**: Filtered feature selectors (lasso, mutual_information, univariate_selection, etc.) before training execution to prevent training errors
- **Bug Fix**: Fixed family name normalization in isolation_runner (NeuralNetwork → neural_network) before TRAINER_MODULE_MAP lookup
- **Bug Fix**: Fixed reproducibility tracking Path/string handling to prevent `'str' object has no attribute 'name'` errors
- **Enhancement**: Made training plan 0 jobs explicit (downgraded ERROR to WARNING with clear disabled state message)
- **Enhancement**: Added fingerprint validation for routing decisions to prevent stale data reuse from previous runs
- **Enhancement**: Added routing decisions target matching validation (set equality check)
- **Enhancement**: Moved feature registry filtering upstream into feature selection (strict mode, same as training)
- **Enhancement**: Fixed horizon→bars logic to use trading days calendar (390 minutes per trading session, not 1440)
- **Enhancement**: Added registry filtering metadata to feature selection output (selected_features_total, selected_features_registry_allowed)
- **Enhancement**: Added config documentation clarifying which families are selectors vs trainers
- **Impact**: Prevents feature count collapse (selecting 100 features where 92 are forbidden), eliminates training errors from invalid families, ensures consistent folder structure
- **Files Changed**: `intelligent_trainer.py`, `training.py`, `isolation_runner.py`, `training_plan_consumer.py`, `target_routing.py`, `multi_model_feature_selection.py`, `sst_contract.py`, `feature_selector.py`, `multi_model.yaml`
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-22-training-results-organization-fixes.md)

#### 2025-12-22 (CatBoost CV Efficiency with Early Stopping in Feature Selection)
- **Performance Improvement**: Implemented efficient CV with early stopping per fold for CatBoost in feature selection
- **Feature Enhancement**: Added fold-level stability analysis (mean importance, variance tracking) for rigorous feature selection
- **Impact**: Training time reduced from 3 hours to <30 minutes (6-18x speedup) while maintaining CV rigor
- **Reverted**: Previous CV skip approach - CV is now kept for stability diagnostics and accuracy
- **Files Changed**: `multi_model_feature_selection.py`
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-22-catboost-cv-efficiency-with-early-stopping.md)

#### 2025-12-21 (CatBoost Formatting Error and CV Skip Fixes)
- **Bug Fix**: Fixed CatBoost `train_val_gap` format specifier error causing `ValueError: Invalid format specifier` when logging scores
- **Performance Fix**: Always skip CV for CatBoost in feature selection to prevent 3-hour training times (CV doesn't use early stopping per fold)
- **Impact**: Training time reduced from 3 hours to <5 minutes for single symbol (36x speedup)
- **Backward Compatible**: No change for users with `cv_n_jobs <= 1` (they already skip CV)
- **Files Changed**: `multi_model_feature_selection.py`
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-21-catboost-formatting-and-cv-skip-fixes.md)

#### 2025-12-21 (CatBoost Logging and n_features Extraction Fixes)
- **Bug Fix**: Fixed CatBoost logging ValueError when `val_score` is not available (conditionally format value before using in f-string)
- **Bug Fix**: Fixed n_features extraction for FEATURE_SELECTION to check nested `evaluation` dict where it's actually stored in `full_metadata`
- **Root Cause**: `_build_resolved_context()` only checked flat paths but `n_features` is stored in `resolved_metadata['evaluation']['n_features']`
- **Files Changed**: `multi_model_feature_selection.py`, `diff_telemetry.py`
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-21-catboost-logging-and-n-features-extraction-fixes.md)

#### 2025-12-21 (Training Plan Model Families and Feature Summary Fixes)
- **Bug Fix**: Fixed training plan to use correct trainer families from experiment config (automatically filters out feature selectors like random_forest, catboost, lasso)
- **Enhancement**: Added global feature summary (`globals/selected_features_summary.json`) with actual feature lists per target per view for auditing
- **Bug Fix**: Fixed REPRODUCIBILITY directory creation to only occur within run directories, not at RESULTS root level
- **Enhancement**: Added comprehensive documentation for feature storage locations and flow from phase 2 to phase 3
- **Enhancement**: Enhanced logging to show families parameter flow and feature selector filtering
- **Files Changed**: `training_plan_generator.py`, `intelligent_trainer.py`, `diff_telemetry.py`
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-21-training-plan-model-families-and-feature-summary-fixes.md)

#### 2025-12-21 (Feature Selection Routing and Training View Tracking Fixes)
- **Bug Fix**: Fixed path resolution warning that was walking to root directory
- **Enhancement**: Added view tracking (CROSS_SECTIONAL/SYMBOL_SPECIFIC) to feature selection routing metadata
- **Bug Fix**: Added route/view information to training reproducibility tracking for proper output separation
- **Bug Fix**: Fixed BOTH route to use symbol-specific features for symbol-specific model training (was using CS features incorrectly)
- **Enhancement**: Added view information to per-target routing_decision.json files
- **Files Changed**: `feature_selection_reporting.py`, `target_routing.py`, `intelligent_trainer.py`, `training.py`
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-21-feature-selection-routing-and-training-view-tracking.md)

#### 2025-12-21 (CatBoost Verbosity and Feature Selection Reproducibility Fixes)
- **Bug Fix**: Fixed CatBoost verbosity parameter conflict causing training failures (removed conflicting `logging_level` parameter)
- **Bug Fix**: Added missing `n_features` to feature selection reproducibility tracking (fixes diff telemetry validation warnings)
- **Files Changed**: `multi_model_feature_selection.py`, `feature_selector.py`
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-21-catboost-verbosity-and-reproducibility-fixes.md)

#### 2025-12-21 (CatBoost Performance Diagnostics and Comprehensive Fixes)
- **Performance Fix**: Reduced iterations cap from 2000 to 300 (matching target ranking), added comprehensive timing logs and diagnostics
- **Diagnostics**: Added performance timing (CV, fit, importance), diagnostic logging (iterations, scores, gaps), pre-training checks, enhanced overfitting detection
- **Analysis**: Created comparison document identifying differences between feature selection and target ranking stages
- **Files Changed**: `multi_model_feature_selection.py`, `docs/analysis/catboost_feature_selection_vs_target_ranking_comparison.md`
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-21-catboost-performance-diagnostics.md)

#### 2025-12-21 (CatBoost Early Stopping Fix for Feature Selection)
- **Performance Fix**: Added early stopping to CatBoost final fit in feature selection, reducing training time from ~3 hours to <30 minutes
- **Files Changed**: `multi_model_feature_selection.py`, `multi_model.yaml`
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-21-catboost-early-stopping-fix.md)

#### 2025-12-21 (Run Comparison Fixes for Target-First Structure)
- **Bug Fix**: Fixed diff telemetry and trend analyzer to properly find and compare runs across target-first structure
- **Files Changed**: `diff_telemetry.py`, `trend_analyzer.py`, `reproducibility_tracker.py`
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-21-run-comparison-fixes.md)

#### 2025-12-20 (Threading, Feature Pruning, and Path Resolution Fixes)
- **Performance**: Added threading for CatBoost/Elastic Net in feature selection (2-4x speedup)
- **Bug Fix**: Added `ret_zscore_*` to exclusion patterns (fixes data leakage)
- **Bug Fix**: Fixed path resolution errors causing permission denied
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-20-threading-feature-pruning-path-fixes.md)

#### 2025-12-20 (Untrack DATA_PROCESSING Folder)
- **Repository Cleanup**: Untracked `DATA_PROCESSING/` folder from git, updated paths to `RESULTS/`
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-20-untrack-data-processing-folder.md)

#### 2025-12-20 (CatBoost Fail-Fast for 100% Training Accuracy)
- **Performance**: Added fail-fast for CatBoost when training accuracy >= 99.9% (saves 40+ minutes)
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-20-catboost-fail-fast-for-overfitting.md)

#### 2025-12-20 (Elastic Net Graceful Failure Handling)
- **Performance**: Fixed Elastic Net to fail-fast when all coefficients are zero (saves 30+ minutes)
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-20-elastic-net-graceful-failure-handling.md)

#### 2025-12-20 (Path Resolution Fix)
- **Bug Fix**: Fixed path resolution stopping at `RESULTS/` instead of finding run directory
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-20-path-resolution-fix.md)

#### 2025-12-20 (Feature Selection Output Organization and Elastic Net Fail-Fast)
- **Bug Fix**: Fixed feature selection outputs using target-first structure, added Elastic Net fail-fast
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-20-feature-selection-output-organization-and-elastic-net-fail-fast.md)

#### 2025-12-20 (Unified Threading Utilities)
- **Refactoring**: Centralized threading utilities for all model families in feature selection and target ranking
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-20-threading-feature-pruning-path-fixes.md)

#### 2025-12-20 (Incremental Decision File Saving)
- **Feature**: Routing decisions saved immediately after each target completes (crash resilience)

#### 2025-12-20 (Snapshot Index Symbol Key Fix & SST Metrics Architecture)
- **Bug Fix**: Fixed snapshot index key format to include symbol, implemented SST metrics architecture

#### 2025-12-20 (Legacy REPRODUCIBILITY Directory Cleanup)
- **Refactoring**: Removed legacy directory creation, new runs use target-first structure only

#### 2025-12-19 (Target-First Directory Structure Migration)
- **Architecture**: Migrated all output artifacts to target-first structure (`targets/<target>/`)
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-19-target-first-structure-migration.md)

#### 2025-12-19 (Feature Selection Error Fixes)
- **Bug Fix**: Fixed `NameError` in feature selection, improved error messaging
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-19-feature-selection-error-fixes.md)

#### 2025-12-19 (Target Evaluation Config Fixes)
- **Bug Fix**: Fixed config precedence for `max_targets_to_evaluate`, added target whitelist support
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-19-target-evaluation-config-fixes.md)

#### 2025-12-18 (TRAINING Folder Reorganization)
- **Folder Reorganization**: Consolidated small directories into `data/` and `common/`, merged overlapping directories, reorganized entry points into `orchestration/`. All changes maintain backward compatibility via re-export wrappers.
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-18-training-folder-reorganization.md)

#### 2025-12-18 (Code Modularization)
- **Large File Modularization**: Split 7 large files (2,000-6,800 lines) into smaller modules. Created 23 new utility/module files. Total: 103 files changed, ~2,000+ lines extracted.
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-18-code-modularization.md)

#### 2025-12-17 (Training Pipeline Audit Fixes)
- **Contract Fixes**: Fixed 12 critical contract breaks (family normalization, reproducibility tracking, routing, feature pipeline, diff telemetry, output digests).
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-17-training-pipeline-audit-fixes.md)

#### 2025-12-17 (Licensing & Reproducibility)
- **Licensing Model**: Reverted to AGPL v3 + Commercial dual licensing model.
- **FEATURE_SELECTION Reproducibility**: Integrated hyperparameters, train_seed, and library versions tracking.
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-17-feature-selection-reproducibility.md)

#### Older Updates
For detailed changelogs from 2025-12-16 and earlier, see the [Changelog Index](DOCS/02_reference/changelog/README.md#december).

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

**Recent additions:**
- Target Whitelist Support - `targets_to_evaluate` field for fine-grained target selection (2025-12-19)
- Audit-Grade Metadata Fields - Environment info, data source details, evaluation details (2025-12-17)
- Research-Grade Metrics - Per-fold distributional stats, composite score versioning (2025-12-17)
- Feature Audit System - Per-feature drop tracking with CSV reports (2025-12-16)
- Canonical Family ID System - Unified snake_case family IDs with startup validation (2025-12-16)
- Diff Telemetry Integration - Full audit trail in metadata, lightweight queryable fields in metrics (2025-12-16)
- Experiment Configuration System - Reusable experiment configs with auto target discovery
- Dual-View Target Ranking System - Multiple evaluation views with automatic routing
- Training Routing & Planning System - Config-driven routing decisions
- Reproducibility Tracking System - End-to-end tracking with STABLE/DRIFTING/DIVERGED classification

**For complete details:** See [Changelog Index](DOCS/02_reference/changelog/README.md)

---

### Fixed

**Recent fixes:**
- **Config Precedence** (2025-12-19): Fixed `max_targets_to_evaluate` from experiment config not overriding test config
- **Training Pipeline Contract Fixes** (2025-12-17): Fixed 12 critical contract breaks (family normalization, reproducibility tracking, routing, feature pipeline, diff telemetry, output digests)
- **CV/Embargo Metadata** (2025-12-17): Fixed inconsistent embargo_minutes handling when CV is enabled
- **Field Name Mismatch** (2025-12-17): Fixed diff telemetry field name alignment (date_start/end → date_range_start/end, etc.)
- **Training Pipeline Plumbing** (2025-12-16): Fixed family canonicalization, banner suppression, reproducibility tracking, model saving bugs
- **Feature Selection Fixes** (2025-12-14): Fixed UnboundLocalError, missing imports, experiment config loading, target exclusion
- **Look-Ahead Bias** (2025-12-14): Rolling windows exclude current bar, CV-based normalization fixes
- **Single Source of Truth** (2025-12-13): Eliminated split-brain in lookback computation

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
