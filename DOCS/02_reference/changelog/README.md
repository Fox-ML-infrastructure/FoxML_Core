# Changelog Index

This directory contains detailed per-day changelogs for FoxML Core. For the lightweight root changelog, see [CHANGELOG.md](../../../CHANGELOG.md).

## 2025

### December

- **2025-12-23 (Comprehensive Model Timing Metrics)** — Added comprehensive timing metrics (start-time and elapsed-time logging) for all 12 model families in target ranking and feature selection. Provides visibility into execution sequence, individual model performance, and overall pipeline timing to help identify bottlenecks. All models now log start time (🚀) and elapsed time (⏱️) with percentage breakdown in overall summary.
  → [View](2025-12-23-comprehensive-model-timing-metrics.md)

- **2025-12-22 (CatBoost CV Efficiency with Early Stopping in Feature Selection)** — Implemented efficient CV with early stopping per fold for CatBoost in feature selection, replacing previous CV skip approach. Maintains CV rigor for fold-level stability analysis (mean importance, variance tracking) while reducing training time from 3 hours to <30 minutes (6-18x speedup). Enables identifying features with persistent signal vs. noisy features. Reverted previous CV skip to maintain best practices for time-series feature selection.
  → [View](2025-12-22-catboost-cv-efficiency-with-early-stopping.md)

- **2025-12-22 (Boruta Performance Optimizations)** — Implemented quality-preserving optimizations for Boruta feature selection to address performance bottlenecks. Added time budget enforcement (10 min default), conditional execution (skip for >200 features or >20k samples), adaptive max_iter based on dataset size, subsampling for large datasets, and caching integration. All parameters SST-compliant (loaded from config). Reduces Boruta feature selection time from hours to minutes while maintaining model quality.
  → [View](2025-12-22-boruta-performance-optimizations.md)

- **2025-12-22 (CatBoost Formatting TypeError Fix)** — Fixed `TypeError: unsupported format string passed to NoneType.__format__` when `cv_mean` or `val_score` is `None` in CatBoost overfitting check logging. Pre-format values before using in f-string to prevent format specifier errors. Prevents runtime errors in CatBoost logging, training pipeline completes successfully regardless of CV or validation score availability.
  → [View](2025-12-22-catboost-formatting-typeerror-fix.md)

- **2025-12-22 (Trend Analyzer Operator Precedence Fix)** — Fixed operator precedence bug in trend analyzer path detection that prevented correct identification of runs in comparison groups. Added explicit parentheses to ensure `d.is_dir()` is evaluated before checking subdirectories. Enables proper run detection in comparison groups, trend analyzer correctly identifies all runs with `targets/`, `globals/`, or `REPRODUCIBILITY/` subdirectories.
  → [View](2025-12-22-trend-analyzer-operator-precedence-fix.md)

- **2025-12-21 (CatBoost Formatting Error and CV Skip Fixes)** — Fixed CatBoost `train_val_gap` format specifier error causing `ValueError: Invalid format specifier`. Always skip CV for CatBoost in feature selection to prevent 3-hour training times (CV doesn't use early stopping per fold, runs full 300 iterations per fold). Training time reduced from 3 hours to <5 minutes for single symbol (36x speedup). Backward compatible: no change for users with `cv_n_jobs <= 1`. **NOTE**: This approach was later reverted in favor of efficient CV with early stopping (see 2025-12-22 entry).
  → [View](2025-12-21-catboost-formatting-and-cv-skip-fixes.md)

- **2025-12-21 (CatBoost Logging and n_features Extraction Fixes)** — Fixed CatBoost logging ValueError when `val_score` is not available (conditionally format value before using in f-string). Fixed n_features extraction for FEATURE_SELECTION to check nested `evaluation` dict where it's actually stored in `full_metadata`. Root cause: `_build_resolved_context()` only checked flat paths but `n_features` is stored in `resolved_metadata['evaluation']['n_features']`.
  → [View](2025-12-21-catboost-logging-and-n-features-extraction-fixes.md)

- **2025-12-21 (Training Plan Model Families and Feature Summary Fixes)** — Fixed training plan to use correct trainer families from experiment config (automatically filters out feature selectors). Added global feature summary with actual feature lists per target per view for auditing. Fixed REPRODUCIBILITY directory creation to only occur within run directories. Added comprehensive documentation for feature storage locations and flow.
  → [View](2025-12-21-training-plan-model-families-and-feature-summary-fixes.md)

- **2025-12-21 (Feature Selection Routing and Training View Tracking Fixes)** — Fixed path resolution warning walking to root directory. Added view tracking (CROSS_SECTIONAL/SYMBOL_SPECIFIC) to feature selection routing metadata. Added route/view information to training reproducibility tracking for proper output separation. Fixed BOTH route to use symbol-specific features for symbol-specific model training (was incorrectly using CS features). Added view information to per-target routing_decision.json files.
  → [View](2025-12-21-feature-selection-routing-and-training-view-tracking.md)

- **2025-12-21 (CatBoost Verbosity and Feature Selection Reproducibility Fixes)** — Fixed CatBoost verbosity parameter conflict causing training failures (removed conflicting `logging_level` parameter). Added missing `n_features` to feature selection reproducibility tracking (fixes diff telemetry validation warnings).
  → [View](2025-12-21-catboost-verbosity-and-reproducibility-fixes.md)

- **2025-12-21 (CatBoost Performance Diagnostics and Comprehensive Fixes)** — Reduced iterations cap from 2000 to 300 (matching target ranking), added comprehensive performance timing logs, diagnostic logging (iterations, scores, gaps), pre-training data quality checks, and enhanced overfitting detection. Created comparison document identifying differences between feature selection and target ranking stages.
  → [View](2025-12-21-catboost-performance-diagnostics.md)

- **2025-12-21 (CatBoost Early Stopping Fix for Feature Selection)** — Fixed CatBoost training taking 3 hours by adding early stopping to final fit. Added train/val split and eval_set support to enable early stopping, reducing training time from ~3 hours to <30 minutes.
  → [View](2025-12-21-catboost-early-stopping-fix.md)

- **2025-12-21 (Run Comparison Fixes for Target-First Structure)** — Fixed diff telemetry and trend analyzer to properly find and compare runs across target-first structure.
  → [View](2025-12-21-run-comparison-fixes.md)

- **2025-12-20 (Threading, Feature Pruning, and Path Resolution Fixes)** — Added threading/parallelization to feature selection (CatBoost/Elastic Net), excluded `ret_zscore_*` targets from features to prevent leakage, and fixed path resolution errors causing permission denied. Feature selection now matches target ranking performance.
  → [View](2025-12-20-threading-feature-pruning-path-fixes.md)

- **2025-12-20 (Untrack DATA_PROCESSING Folder)** — Untracked `DATA_PROCESSING/` folder from git (22 files), updated default output paths to use `RESULTS/` instead, removed DATA_PROCESSING-specific documentation. Verified TRAINING pipeline is completely independent - no core functionality affected.
  → [View](2025-12-20-untrack-data-processing-folder.md)

- **2025-12-20 (CatBoost Fail-Fast for 100% Training Accuracy)** — Added fail-fast mechanism for CatBoost when training accuracy reaches 100% (>= 99.9% threshold), preventing 40+ minutes wasted on expensive feature importance computation when model is overfitting.
  → [View](2025-12-20-catboost-fail-fast-for-overfitting.md)

- **2025-12-20 (Elastic Net Graceful Failure Handling)** — Fixed Elastic Net to gracefully handle "all coefficients zero" failures and prevent expensive full fit operations from running. Quick pre-check now sets a flag to skip expensive operations when failure is detected early.
  → [View](2025-12-20-elastic-net-graceful-failure-handling.md)

- **2025-12-20 (Path Resolution Fix)** — Fixed path resolution logic that incorrectly stopped at `RESULTS/` directory instead of continuing to find the actual run directory. Changed to only stop when it finds a run directory (has `targets/`, `globals/`, or `cache/` subdirectories).
  → [View](2025-12-20-path-resolution-fix.md)

- **2025-12-20 (Feature Selection Output Organization)** — Fixed feature selection outputs being overwritten at run root - now uses target-first structure exclusively. Added Elastic Net fail-fast mechanism and fixed syntax error in feature_selection_reporting.py.
  → [View](2025-12-20-feature-selection-output-organization-and-elastic-net-fail-fast.md)

- **2025-12-19 (Target Evaluation Config Fixes)** — Fixed config precedence issue where `max_targets_to_evaluate` from experiment config was not properly overriding test config values. Added `targets_to_evaluate` whitelist support that works with `auto_targets: true`, allowing users to specify a specific list of targets to evaluate while still using auto-discovery. Enhanced debug logging shows config precedence chain and config trace now includes `intelligent_training` section overrides.
  → [View](2025-12-19-target-evaluation-config-fixes.md)

- **2025-12-18 (TRAINING Folder Reorganization)** — Comprehensive reorganization of `TRAINING/` folder structure: consolidated small directories (`features/`, `datasets/`, `memory/`, `live/`) into `data/` and `common/`, merged overlapping directories (`strategies/` into `training_strategies/`, data processing modules into `data/`), reorganized entry points into `orchestration/`, moved output directories to `RESULTS/`, fixed config loader import warnings. All changes maintain backward compatibility via re-export wrappers. 100% of key imports passing.
  → [View](2025-12-18-training-folder-reorganization.md)

- **2025-12-18 (Code Modularization)** — Major code refactoring: Split 7 large files (2,000-6,800 lines) into modular components, created 23 new utility/module files, reorganized utils folder into domain-specific subdirectories, centralized common utilities (file_utils, cache_manager, config_hashing, etc.), fixed all import errors, maintained full backward compatibility. Total: 103 files changed, ~2,000+ lines extracted.
  → [View](2025-12-18-code-modularization.md)

- **2025-12-17 (Metric Deltas in Diff Artifacts)** — Fixed empty `metric_deltas` issue. Implemented 3-tier reporting (summary, structured deltas, full metrics), z-score noise detection, impact classification, and proper separation of nondeterminism from regression. All numeric metrics now captured and deltas always computed.
  → [View](2025-12-17-metric-deltas-in-diff-artifacts.md)

- **2025-12-17 (Training Pipeline Audit Fixes)** — Fixed 10 critical contract breaks across family IDs, routing, plan consumption, feature schema, and counting/tracking. Key fixes: family normalization, reproducibility tracking, preflight filtering, routing plan respect, symbol-specific route, feature pipeline threshold and diagnostics.
  → [View](2025-12-17-training-pipeline-audit-fixes.md)

- **2025-12-16 (Feature Selection Structure)** — Organized feature selection outputs to match target ranking layout. Eliminated scattered files and nested REPRODUCIBILITY directories.
  → [View](2025-12-16-feature-selection-structure.md)

- **2025-12-15 (Consolidated)** — Metrics system rename, seed tracking fixes, feature selection improvements, CatBoost GPU fixes, privacy documentation updates.  
  → [View](2025-12-15-consolidated.md)
  
- **2025-12-15 (CatBoost GPU Fixes)** — Fixed CatBoost GPU mode requiring Pool objects, sklearn clone compatibility, and missing feature importance output. CatBoost GPU training now works correctly and feature importances are saved to results directory.  
  → [View](2025-12-15-catboost-gpu-fixes.md)
  
- **2025-12-15 (Metrics Rename)** — Renamed telemetry to metrics throughout codebase. All metrics stored locally - no user data collection.  
  → [View](2025-12-15-metrics-rename.md)

- **2025-12-14 (IP Assignment Agreement Signed)** — IP Assignment Agreement signed, legally assigning all IP from individual to Fox ML Infrastructure LLC. ✅ Legally effective.  
  → [View](2025-12-14-ip-assignment-signed.md)

- **2025-12-14 (Execution Modules Added)** — Trading modules added with compliance framework, documentation organization, copyright headers  
  → [View](2025-12-14-execution-modules.md)

- **2025-12-14 (Enhanced Drift Tracking)** — Fingerprints (git commit, config hash, data fingerprint), drift tiers (OK/WARN/ALERT), critical metrics tracking, sanity checks, Parquet files for queryable data  
  → [View](2025-12-14-drift-tracking-enhancements.md)

- **2025-12-14 (Telemetry System)** — Sidecar-based telemetry with view isolation, hierarchical rollups (cohort → view → stage), baseline key format for drift comparison, config-driven behavior, Parquet files  
  → [View](2025-12-14-telemetry-system.md)

- **2025-12-14 (Feature Selection and Config Fixes)** — Fixed UnboundLocalError for np (11 model families), missing import, unpacking error, routing diagnostics, experiment config loading, target exclusion, lookback enforcement  
  → [View](2025-12-14-feature-selection-and-config-fixes.md)

- **2025-12-14 (Look-Ahead Bias Fixes)** — Rolling windows exclude current bar, CV-based normalization, pct_change verification, feature renaming, symbol-specific logging, feature selection bug fix  
  → [View](2025-12-14-lookahead-bias-fixes.md)

- **2025-12-13 (SST Enforcement Design)** — EnforcedFeatureSet contract, type boundary wiring, boundary assertions, no rediscovery rule, full coverage across all training paths  
  → [View](2025-12-13-sst-enforcement-design.md)

- **2025-12-13 (Single Source of Truth)** — Eliminated split-brain in lookback computation, POST_PRUNE invariant check, _Xd pattern inference, readline library conflict fix  
  → [View](2025-12-13-single-source-of-truth.md)

- **2025-12-13 (Fingerprint Tracking)** — Fingerprint Tracking System, LookbackResult Dataclass, Explicit Stage Logging, Leakage Canary Test  
  → [View](2025-12-13-fingerprint-tracking.md)

- **2025-12-13 (Feature Selection Unification)** — Shared Ranking Harness, Comprehensive Hardening, Same Output Structure, Config-Driven Setup  
  → [View](2025-12-13-feature-selection-unification.md)

- **2025-12-13 (Duration System)** — Generalized Duration Parsing System, Lookback Detection Precedence Fix, Documentation Review, Non-Auditable Status Markers  
  → [View](2025-12-13-duration-system.md)

- **2025-12-13** — Config Path Consolidation, Config Trace System, Max Samples Fix, Output Directory Binning Fix  
  → [View](2025-12-13.md)

- **2025-12-12** — Trend Analysis System Extension (Feature Selection), Cohort-Aware Reproducibility System, RESULTS Directory Organization, Integrated Backups, Enhanced Metadata  
  → [View](2025-12-12.md)

- **2025-12-11** — Training Routing System, Reproducibility Tracking, Leakage Fixes, Interval Detection, Param Sanitization, Cross-Sectional Stability  
  → [View](2025-12-11.md)

- **2025-12-10** — SST Enforcement, Determinism System, Config Centralization  
  → [View](2025-12-10.md)

- **General** — Intelligent Training Framework, Leakage Safety Suite, Configuration System, Documentation  
  → [View](general.md)

---

## Documentation Audits

Quality assurance audits and accuracy checks (publicly available for transparency):

- **[Documentation Accuracy Check](../../00_executive/audits/DOCS_ACCURACY_CHECK.md)** - Accuracy audit results and fixes (2025-12-13)
- **[Unverified Claims Analysis](../../00_executive/audits/DOCS_UNVERIFIED_CLAIMS.md)** - Claims without verified test coverage (2025-12-13)
- **[Marketing Language Removal](../../00_executive/audits/MARKETING_LANGUAGE_REMOVED.md)** - Marketing terms removed for accuracy (2025-12-13)
- **[Dishonest Statements Fixed](../../00_executive/audits/DISHONEST_STATEMENTS_FIXED.md)** - Final pass fixing contradictions and overselling (2025-12-13)

---

## Navigation

- [Root Changelog](../../../CHANGELOG.md) - Executive summary
- [Documentation Index](../../INDEX.md) - Complete documentation navigation
