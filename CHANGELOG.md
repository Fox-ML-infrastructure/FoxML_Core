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
