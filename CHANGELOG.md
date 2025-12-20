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

#### 2025-12-20 (Untrack DATA_PROCESSING Folder)
- **Repository Cleanup**: Untracked `DATA_PROCESSING/` folder from git (22 files) - folder remains on disk but is now ignored
- **Dependency Updates**: Updated default output paths in `multi_model_feature_selection.py` and `CONFIG/ranking/features/config.yaml` from `DATA_PROCESSING/` to `RESULTS/`
- **Documentation Cleanup**: Removed 3 DATA_PROCESSING-specific documentation files and updated `DOCS/INDEX.md`
- **No Core Impact**: Verified TRAINING pipeline is completely independent - no Python imports, runtime dependencies, or data dependencies on DATA_PROCESSING
- **Files Changed**: `.gitignore`, `multi_model_feature_selection.py`, `config.yaml`, `DOCS/INDEX.md` (4 modified, 25 deleted)
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-20-untrack-data-processing-folder.md)

#### 2025-12-20 (CatBoost Fail-Fast for 100% Training Accuracy)
- **CatBoost Overfitting Detection**: Added fail-fast mechanism for CatBoost when training accuracy reaches 100% (or >= 99.9% threshold)
- **Time Savings**: Prevents wasting 40+ minutes on expensive feature importance computation when model is overfitting/memorizing
- **Early Detection**: Checks training accuracy immediately after fit, before expensive operations start
- **Graceful Failure**: Skips expensive operations and continues with other models instead of blocking
- **Files Changed**: `multi_model_feature_selection.py`
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-20-catboost-fail-fast-for-overfitting.md)

#### 2025-12-20 (Elastic Net Graceful Failure Handling - Prevent Full Fit)
- **Elastic Net Error Handling**: Fixed Elastic Net to gracefully handle "all coefficients zero" failures and prevent expensive full fit operations from running
- **Quick Pre-Check**: Quick pre-check now sets a flag to skip expensive operations (cross_val_score, pipeline.fit) when failure is detected early
- **CROSS_SECTIONAL Fix**: Fixed issue where CROSS_SECTIONAL view would waste time running expensive operations even after detecting failure (SYMBOL_SPECIFIC already worked due to wrapper)
- **Time Savings**: Prevents wasting 30+ minutes on operations that will fail - now fails fast in ~1-2 minutes
- **Files Changed**: `model_evaluation.py`
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-20-elastic-net-graceful-failure-handling.md)

#### 2025-12-20 (Path Resolution Fix - Stop at Run Directory, Not RESULTS Root)
- **Path Resolution Bug Fix**: Fixed path resolution logic that incorrectly stopped at `RESULTS/` directory instead of continuing to find the actual run directory
- **Root Cause**: Bug was always present but only surfaced after removing legacy root-level writes - previously legacy writes masked the issue
- **Fix**: Changed path resolution to only stop when it finds a run directory (has `targets/`, `globals/`, or `cache/` subdirectories), not at `RESULTS/` itself
- **Impact**: No more `RESULTS/targets/` created outside run directories - all files now correctly go to `RESULTS/runs/{run}/targets/<target>/`
- **Files Changed**: `multi_model_feature_selection.py`, `model_evaluation.py`, `feature_selection_reporting.py`, `target_routing.py`
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-20-path-resolution-fix.md)

#### 2025-12-20 (Feature Selection Output Organization and Elastic Net Fail-Fast)
- **Output Organization**: Fixed feature selection outputs being overwritten at run root - now uses target-first structure (`targets/<target>/reproducibility/`) exclusively
- **Decision Routing Updates**: Updated `load_target_confidence()` and `save_target_routing_metadata()` to use target-first structure with separate `globals/feature_selection_routing.json` (doesn't modify `routing_decisions.json`)
- **Aggregation Function**: Added `_aggregate_feature_selection_summaries()` to collect per-target summaries into `globals/` after all targets complete
- **Elastic Net Fail-Fast**: Added quick pre-check (max_iter=50) and reduced max_iter cap (500) to fail in ~1-2 minutes instead of 30+ minutes when over-regularized
- **Syntax Error Fix**: Fixed missing `except` block in `feature_selection_reporting.py` causing import failures
- **Reduced Log Noise**: Changed reproducibility tracker warning to debug level for expected fallback scenarios
- **Files Changed**: `feature_selection_reporting.py`, `multi_model_feature_selection.py`, `target_routing.py`, `intelligent_trainer.py`, `reproducibility_tracker.py`, `model_evaluation.py`
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-20-feature-selection-output-organization-and-elastic-net-fail-fast.md)

#### 2025-12-20 (Unified Threading Utilities for Feature Selection and Target Ranking)
- **Threading Utilities Integration**: Refactored all model training in feature selection and target ranking to use centralized threading utilities from `TRAINING/common/threads.py`
- **GPU-Aware Thread Management**: All models now automatically limit CPU threads (OMP=1, MKL=1) when GPU is enabled, preventing CPU bottlenecks during GPU training
- **Smart Thread Allocation**: Models use `plan_for_family()` to determine optimal OMP vs MKL thread allocation based on model family type (tree models use OMP, linear models use MKL)
- **Consistent Thread Control**: All model families (LightGBM, XGBoost, CatBoost, Random Forest, Neural Network, Lasso, Ridge, Elastic Net, RFE, Boruta, Stability Selection, HistGradientBoosting) now use `thread_guard()` context manager for safe thread limiting
- **Feature Selection**: Updated `TRAINING/ranking/multi_model_feature_selection.py` - all 9 model families now use threading utilities
- **Target Ranking**: Updated `TRAINING/ranking/predictability/model_evaluation.py` - all 11 model families now use threading utilities
- **Benefits**: Prevents thread conflicts, optimizes resource usage, ensures GPU training doesn't get CPU-bound, maintains consistency across all training phases
- **Files Changed**: `multi_model_feature_selection.py` (LightGBM, XGBoost, CatBoost, Random Forest, Neural Network, Lasso, RFE, Boruta, Stability Selection), `model_evaluation.py` (LightGBM, XGBoost, CatBoost, Random Forest, Neural Network, Lasso, Ridge, Elastic Net, RFE, Stability Selection, HistGradientBoosting)

#### 2025-12-20 (Incremental Decision File Saving)
- **Incremental Decision Saving**: Routing decision files are now saved immediately after each target completes evaluation, not just at the end
- **Crash Resilience**: If the process crashes, decisions for completed targets are already saved
- **Per-Target Decisions**: Each target's decision is written to `targets/<target>/decision/routing_decision.json` as soon as that target finishes
- **Global Summary Still Updated**: `globals/routing_decisions.json` is still written at the end with the complete summary
- **Files Changed**: `target_routing.py` (added `_compute_single_target_routing_decision()` and `_save_single_target_decision()`), `target_ranker.py` (integrated incremental saving in both parallel and sequential paths)

#### 2025-12-20 (Snapshot Index Symbol Key Fix & SST Metrics Architecture)
- **Fixed Snapshot Index Overwrites**: Updated snapshot index key format from `run_id:stage:target:view` to `run_id:stage:target:view:symbol` to prevent overwrites when processing multiple symbols for the same target in SYMBOL_SPECIFIC view
- **Backward Compatibility**: All snapshot index readers handle old formats (legacy, previous, and current) automatically
- **SST Metrics Architecture**: Implemented Single Source of Truth for metrics:
  - **Canonical Location**: `targets/<target>/reproducibility/<view>/cohort=<id>/metrics.parquet` (immutable, audit-grade)
  - **Debug Export**: `metrics.json` in same location (derived from parquet)
  - **Reference Pointers**: `targets/<target>/metrics/view=.../latest_ref.json` points to canonical location
  - **No Duplication**: Full metrics payloads only in reproducibility/cohort directories, metrics/ contains only references
  - **Reading Logic**: All readers check canonical location first, then reference pointers, then legacy locations
- **Files Changed**: `diff_telemetry.py` (snapshot key format), `metrics.py` (SST architecture), `metrics_aggregator.py`, `diff_telemetry.py` (reading logic)

#### 2025-12-20 (Legacy REPRODUCIBILITY Directory Cleanup)
- **Removed Legacy Directory Creation**: Removed all code that creates the legacy `REPRODUCIBILITY/` directory structure
- **Preserved Backward Compatibility**: Fallback reading logic still supports reading from existing legacy directories
- **Target-First Only**: New runs now exclusively use the target-first structure (`targets/<target>/reproducibility/`)
- **Files Changed**: `shared_ranking_harness.py`, `model_evaluation.py` - removed legacy directory creation while preserving reading fallback

#### 2025-12-19 (Target-First Directory Structure Migration)
- **Target-First Organization**: Migrated all output artifacts to target-first structure (`targets/<target>/`) for better organization and decision-making
- **Global Summaries**: Added `globals/` directory for run-level summaries (routing decisions, target rankings, confidence summaries, stats)
- **Per-Target Metadata**: Added `targets/<target>/metadata.json` aggregating all target information
- **Run Manifest**: Added `manifest.json` at run root with experiment config and target index
- **Legacy Support**: All reading logic checks target-first structure first, then falls back to legacy for backward compatibility
- **No Legacy Writes**: Removed all writes to legacy `REPRODUCIBILITY/` structure - new runs only create target-first structure
- **Complete Migration**: All artifacts (metadata, metrics, diffs, snapshots, decisions, models, feature selection) now use target-first structure
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-19-target-first-structure-migration.md)

#### 2025-12-19 (Feature Selection Error Fixes)
- **cohort_metadata undefined error fix**: Fixed `NameError` in feature selection when cohort metadata extraction fails. Added safe initialization and guards in fallback paths.
- **Improved error messaging**: Updated insufficient data span error message to clarify that fallback to per-symbol processing is expected for long-horizon targets.
- **Reduced log noise**: Changed log level from WARNING to INFO for expected fallback scenarios.
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-19-feature-selection-error-fixes.md)

#### 2025-12-19 (Target Evaluation Config Fixes)
- **Config Precedence Fix**: Fixed `max_targets_to_evaluate` from experiment config not overriding test config. Experiment config now correctly takes priority.
- **Target Whitelist Support**: Added `targets_to_evaluate` whitelist that works with `auto_targets: true` for fine-grained control.
- **Enhanced Debug Logging**: Added config precedence chain logging and config trace for `intelligent_training` overrides.
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
