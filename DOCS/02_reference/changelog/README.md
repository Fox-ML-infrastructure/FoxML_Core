# Changelog Index

This directory contains detailed per-day changelogs for FoxML Core. For the lightweight root changelog, see [CHANGELOG.md](../../../CHANGELOG.md).

## 2026

### January

- **2026-01-09 (Symbol Parameter Propagation Fix - All Stages)** — Applied comprehensive fixes to FEATURE_SELECTION and TRAINING stages: added view auto-detection in feature_selector.py, added symbol validation in save_feature_importances_for_reproducibility(), added validation warning in get_training_snapshot_dir(), and verified ArtifactPaths.model_dir() symbol parameter. All three stages now have consistent symbol parameter propagation and validation. FEATURE_SELECTION and TRAINING stages will correctly route symbol-specific data to SYMBOL_SPECIFIC/symbol=.../ directories. Maintains SST principles with consistent patterns across all stages.
  → [View](2026-01-09-sst-enum-migration.md)

- **2026-01-09 (Symbol Parameter Propagation Fix - Complete Path Construction)** — Fixed multiple issues where symbol parameter was not properly propagated: 1) "SYMBOL_SPECIFIC view requires symbol" error in save_feature_importances(), 2) Universe directories created under SYMBOL_SPECIFIC/ instead of SYMBOL_SPECIFIC/symbol=.../, 3) Missing feature importance snapshots per symbol. Fixed by passing symbol to resolve_write_scope(), adding fallback logic for symbol_for_importances, ensuring symbol is derived for all ensure_scoped_artifact_dir() calls, and adding validation in get_scoped_artifact_dir(). All artifact directories now correctly route to SYMBOL_SPECIFIC/symbol=.../universe=.../ paths. Maintains SST principles.
  → [View](2026-01-09-sst-enum-migration.md)

- **2026-01-09 (Symbol-Specific Routing Fix - View Propagation and Auto-Detection)** — Fixed critical bug where single-symbol runs were incorrectly routed to CROSS_SECTIONAL directories instead of SYMBOL_SPECIFIC/symbol=.../ directories. Root cause: auto-detected view was overridden by run context. Fixed by ensuring requested_view_from_context uses auto-detected SYMBOL_SPECIFIC view instead of loading from context, and added safety net auto-detection for view_for_importances. Fixes propagate through entire pipeline: requested_view → resolved_data_config → view_for_writes → all path construction. All artifacts (feature importances, metrics, snapshots) now route correctly. Maintains SST principles and backward compatibility.
  → [View](2026-01-09-sst-enum-migration.md)

- **2026-01-09 (Comprehensive JSON/Parquet Serialization Fixes - SST Solution)** — Fixed critical JSON/Parquet serialization failures by creating centralized SST helpers (`sanitize_for_serialization()`, `safe_json_dump()`, `safe_dataframe_from_dict()`) that handle enum objects and pandas Timestamps. Replaced all direct `json.dump()` calls with `safe_json_dump()` across 11 files (136 instances). Fixed metrics duplication in metrics.json by extracting nested metrics dict before writing. All JSON and Parquet files now write successfully. Maintains SST principles with centralized helpers.
  → [View](2026-01-09-sst-enum-migration.md)

- **2026-01-09 (Root Cause Fixes - NoneType Errors and Path Construction)** — Fixed root cause of persistent NoneType.replace() error by passing additional_data to extract_run_id() call (enables multi-source extraction). Fixed critical bug where symbol-specific data was written to CROSS_SECTIONAL directories - symbol check now happens FIRST before view determination, forcing SYMBOL_SPECIFIC view when symbol is set. Fixed path construction in 6 locations (main save, drift.json, metadata lookup, metrics rollup, snapshot). All fixes maintain SST principles and preserve hash verification data. Verified all files compile and symbol-specific data routes correctly.
  → [View](2026-01-09-sst-enum-migration.md)

- **2026-01-09 (NoneType Replace Error Fixes - All Stages)** — Fixed persistent `'NoneType' object has no attribute 'replace'` error across all three stages: multi-source `run_id` extraction in reproducibility_tracker.py with NameError handling, defensive checks for audit_result.get('run_id') in cross_sectional_feature_ranker.py, defensive checks for timestamp in diff_telemetry.py, and defensive checks for _run_name in intelligent_trainer.py (3 instances). All fixes include fallback to datetime.now().isoformat() and preserve all hash verification data. Verified all files compile and all stages are protected.
  → [View](2026-01-09-sst-enum-migration.md)

- **2026-01-09 (Comprehensive File Write Fixes - Enum Normalization and NoneType Error Resolution)** — Fixed missing JSON/parquet/CSV files in globals and output locations by normalizing enum values to strings in all snapshot creation and JSON write operations. Fixed persistent NoneType.replace() error in reproducibility tracking with defensive checks. All fixes maintain overwrite protection and backward compatibility. Verified all three stages (TARGET_RANKING, FEATURE_SELECTION, TRAINING) now output files correctly.
  → [View](2026-01-09-sst-enum-migration.md)

- **2026-01-09 (SST Import Shadowing Fixes)** — Fixed all `UnboundLocalError` issues from SST refactoring: removed `Stage` from local import in `model_evaluation.py` (line 8129), added global `Stage` import in `shared_ranking_harness.py`, removed redundant local `Path` imports (9 instances across 3 files). Verified all path construction and JSON serialization work correctly with enum values. All critical modules now import without errors.
  → [View](2026-01-09-sst-enum-migration.md)

- **2026-01-09 (Comprehensive SST Path Construction Fixes - All Stages)** — Fixed enum-to-string conversion issues in path construction across all three stages: updated target_first_paths.py (get_target_reproducibility_dir, find_cohort_dir_by_id, target_repro_dir), output_layout.py (__init__, repro_dir), and training_strategies/reproducibility/io.py (get_training_snapshot_dir) to normalize Stage/View enums to strings before path construction. All path construction functions now explicitly convert enum values to strings using .value property. Fixes missing metric artifacts and JSON files across TARGET_RANKING, FEATURE_SELECTION, and TRAINING stages. Maintains backward compatibility with string inputs. Verified all three stages work correctly with enum inputs.
  → [View](2026-01-09-sst-enum-migration.md)

- **2026-01-09 (Metric Output JSON Serialization Fixes)** — Fixed broken metric outputs where Stage/View enum objects were written directly to JSON: updated all metric output functions in metrics.py (write_cohort_metrics, _write_metrics, _write_drift, generate_view_rollup, generate_stage_rollup) to normalize enum inputs to strings before JSON serialization, updated reproducibility_tracker.py to normalize ctx.stage/ctx.view before passing to write_cohort_metrics(), fixed NoneType.replace() error by adding explicit None checks. All functions now accept Union[str, Stage] and Union[str, View] for SST compatibility. Fixes broken metric outputs in CROSS_SECTIONAL view for target ranking and all other stages. Maintains backward compatibility with string inputs. Verified all JSON outputs contain string values.
  → [View](2026-01-09-sst-enum-migration.md)

- **2026-01-09 (Additional SST Improvements)** — Completed additional SST improvements: migrated remaining string literal comparisons to enum comparisons (10 files, 20+ instances), standardized config hashing to use canonical_json/sha256 helpers (4 files), verified all changes maintain JSON output format and metric tracking. All enum comparisons use `View.from_string()` / `Stage.from_string()` for backward compatibility. All test suites pass. Rollback point created at tag `sst-import-fixes-complete`.
  → [View](2026-01-09-sst-enum-migration.md)

- **2026-01-09 (SST Enum Migration and WriteScope Adoption)** — Complete migration to SST architecture: View/Stage enum adoption (29 files), WriteScope function migration (4 functions), unified helper functions (cohort ID, config hashing, scope resolution, universe signatures). Fixed syntax errors (indentation issues in model_evaluation.py, cross_sectional_feature_ranker.py, multi_model_feature_selection.py). All changes maintain backward compatibility with existing JSON files, snapshots, and metrics.
  → [View](2026-01-09-sst-enum-migration.md)

- **2026-01-09 (SST Compliance Fixes - Complete Migration)** — Fixed SST (Single Source of Truth) inconsistencies: added `normalize_target_name()` helper and replaced **ALL** remaining instances (39+ total) of manual target normalization across **ALL** files (20 files), replaced **ALL** remaining custom path resolution loops (30+ total) with `run_root()` helper across **ALL** files (19 files), added `view` and `symbol` parameters to `compute_cross_sectional_stability()` to use SST-resolved values, fixed hardcoded `universe_sig="ALL"` in metrics aggregator (now extracts from cohort metadata), removed internal document references from public changelogs. **Complete migration** - the codebase now uses SST helpers consistently throughout. All changes verified to maintain functionality and determinism. All changes maintain backward compatibility.
  → [View](2026-01-09-sst-consistency-fixes.md)

- **2026-01-08 (FEATURE_SELECTION Reproducibility Fixes)** — Fixed critical reproducibility issues in FEATURE_SELECTION stage: CatBoost missing from results (removed filter excluding failed models, handle empty dicts), training snapshot validation (verify files exist after creation), duplicate cohort directories (normalize cs_config structure for consistent hashing, consolidated CS panel metrics into same cohort directory), missing universe_sig in metadata (fixed duplicate assignment), missing snapshot/diff files (added validation after finalize_run), duplicate universe scopes (removed hardcoded "ALL" default, added fallback from run_identity), missing per-model snapshots (improved error logging), missing deterministic_config_fingerprint (fixed path resolution). Created comprehensive documentation explaining which snapshots exist and which one to use (`multi_model_aggregated` = source of truth, `cross_sectional_panel` = optional stability). Fixed null pointer bugs in cohort consolidation (proper null checks for `cohort_dir` and `audit_result`). All fixes maintain backward compatibility.
  → [View](2026-01-08-feature-selection-reproducibility-fixes.md)

- **2026-01-08 (File Overwrite and Plan Creation Fixes)** — Fixed critical bugs causing data loss in `globals/` directory files and missing routing/training plan creation. Fixed `run_context.json` stage history loss, `run_hash.json` creation issues (previous run lookup, error logging), and routing/training plan creation (error visibility, save verification, manifest update). All fixes maintain backward compatibility.
  → [View](2026-01-08-file-overwrite-and-plan-creation-fixes.md)

- **2026-01-08 (Commercial License Clarity and Support Documentation)** — Clarified commercial license requirements in README - now explicitly states "required for proprietary/closed deployments or to avoid AGPL obligations (especially SaaS/network use)". Added root-level SUPPORT.md for easier discovery without navigating LEGAL folder. Makes commercial license trigger crystal clear to legal teams and improves discoverability of support information.
  → [View](2026-01-08-commercial-license-clarity-and-support.md)

- **2026-01-08 (README Improvements for Credibility)** — Comprehensive README improvements: reorganized structure for better funnel, qualified claims (bitwise determinism scope, data transmission), added quick start snippet, concepts glossary, licensing TL;DR, and "why this exists" positioning. Resolved internal contradictions (production vs active development). Moved OSRS joke to maintain credibility. Makes README read like "infra, not a notebook" with clear positioning.
  → [View](2026-01-08-readme-improvements-for-credibility.md)

- **2026-01-08 (Comprehensive Config Dump and Hardware Docs)** — Added automatic copying of all CONFIG files to `globals/configs/` when runs are created, preserving directory structure. Enables easy run recreation without needing original CONFIG folder access. Updated README with detailed CPU/GPU recommendations for optimal performance and stability. All changes maintain backward compatibility.
  → [View](2026-01-08-comprehensive-config-dump-and-hardware-docs.md)

- **2026-01-08 (File Locking for JSON Writes)** — Added file locking around all JSON writes (metadata.json, snapshot.json, metrics.json, diff files) to prevent race conditions when multiple processes write to the same files concurrently. Created `_write_atomic_json_with_lock()` helper that sanitizes data (converts Timestamps to ISO strings) and acquires exclusive file lock before writing. Applied across all pipeline stages (TARGET_RANKING, FEATURE_SELECTION, TRAINING). Prevents data corruption, Timestamp serialization errors, and race conditions. Uses same locking pattern as existing index files. All changes maintain backward compatibility.
  → [View](2026-01-08-file-locking-for-json-writes.md)

- **2026-01-08 (Metrics SHA256 Structure Fix)** — Fixed misleading "metrics_sha256 cannot be computed" error logs. Root cause: metrics were spread as top-level keys in `run_data` instead of nested under `'metrics'` key. Fixed by adding `"metrics": metrics` to `run_data` in 3 locations, with fallback logic to reconstruct from top-level keys for backward compatibility. Metrics digest now computed correctly without error logs. All changes maintain backward compatibility.
  → [View](2026-01-08-metrics-sha256-structure-fix.md)

- **2026-01-08 (Task-Aware Routing Fix)** — Fixed critical bug where routing used fixed [0,1] thresholds on `auc` field, breaking regression targets (R² can be negative). Implemented unified `skill01` score that normalizes both regression IC and classification AUC-excess to [0,1] range. Fixed IC extraction bug (now extracts IC from model_metrics instead of using R²). Enhanced suspicious detection to be task-aware (uses tstat for stability check). Added routing and training plan hashes to manifest for fast change detection. All changes are backward compatible.
  → [View](2026-01-08-task-aware-routing-fix.md)

- **2026-01-08 (Comprehensive Config Dump and Documentation Updates)** — Added automatic copying of all CONFIG files to `globals/configs/` when runs are created, preserving directory structure. Enables easy run recreation without needing original CONFIG folder access. Updated README and tutorials to reflect 3-stage pipeline capabilities, dual-view support, and hardware requirements. Added detailed CPU/GPU recommendations for optimal performance and stability.
  → [View](2026-01-08-comprehensive-config-dump-and-documentation-updates.md)

- **2026-01-08 (Dual Ranking and Filtering Mismatch Fix)** — Fixed critical filtering mismatch between TARGET_RANKING and FEATURE_SELECTION that caused false positives. Removed "unknown but safe" features from ranking mode (now uses safe_family + registry only). Added dual ranking: screen evaluation (safe+registry) and strict evaluation (registry-only) with mismatch telemetry. Updated promotion logic to filter by `strict_viability_flag`. Prevents targets from ranking high using features unavailable in training. All new fields are optional and backward compatible. Improves pipeline trustworthiness without breaking existing functionality.
  → [View](2026-01-08-dual-ranking-filtering-mismatch-fix.md)

- **2026-01-08 (Cross-Stage Issue Fixes)** — Fixed similar issues across FEATURE_SELECTION and TRAINING stages: Path import cleanup (removed redundant try-block imports, fixed root cause in schema.py), type casting for numeric config values (prevents type errors), universe signature extraction (prefers run_identity.dataset_signature over batch subsets), and config name audit (verified all get_cfg() calls use correct canonical paths). Fixed "name 'Path' is not defined" error affecting TARGET_RANKING and FEATURE_SELECTION stages. Ensures consistency and type safety across all pipeline stages.
  → [View](2026-01-08-cross-stage-issue-fixes.md)

- **2026-01-08 (Manifest and Determinism Fixes)** — Fixed manifest.json schema consistency (always includes run_metadata and target_index fields) and deterministic fingerprint computation (excludes git.dirty field). Added update_manifest_with_run_hash() to ensure manifest completeness at end of run. Deterministic fingerprints are now truly stable across runs with identical settings.
  → [View](2026-01-08-manifest-and-determinism-fixes.md)

- **2026-01-08 (Config Cleanup and Symlink Removal)** — Removed all symlinks from CONFIG directory and updated all code to use canonical paths directly. Removed 23 symlinks (6 root-level, 17 in training_config/, legacy directories). Updated config loader to use canonical paths only (no fallback logic). Updated all code references and documentation. Verified run hash and config tracking unchanged (fingerprints based on content, not paths). All configurable settings now fully accessible via config files.
  → [View](2026-01-08-config-cleanup-and-symlink-removal.md)

- **2026-01-08 (Metrics Cleanup and Run Hash)** — Metrics JSON restructuring for smaller, non-redundant, semantically unambiguous output. Added full run hash with change detection. Updated delta computation for grouped metrics structure. All stages now output clean, grouped metrics with task-gating.
  → [View](2026-01-08-metrics-cleanup-and-run-hash.md)

- **2026-01-04 (Reproducibility File Output Fixes)** — Fixed critical bugs preventing reproducibility files from being written to cohort directories. Fixed path detection to handle target-first structure (`reproducibility/...` vs `REPRODUCIBILITY/...`). Fixed `snapshot.json`, `baseline.json`, and diff files (`diff_prev.json`, `metric_deltas.json`, `diff_baseline.json`) not being written. Fixed previous snapshot lookup to search in target-first structure instead of legacy `REPRODUCIBILITY/...`. Added error handling and logging for all writes. All files now correctly written to target-first structure for both CROSS_SECTIONAL and SYMBOL_SPECIFIC views.
  → [View](2026-01-04-reproducibility-file-output-fixes.md)

- **2026-01-04 (GPU/CPU Determinism Config Fix)** — Fixed critical disconnect between reproducibility config settings and actual GPU/CPU device selection. Replaced 4 hardcoded `set_global_determinism()` calls with config-aware `init_determinism_from_config()`. Added strict mode checks to GPU detection in target ranking (LightGBM, XGBoost, CatBoost), feature selection (LightGBM), and training phase (XGBoost, PyTorch, TensorFlow). Fixed `CUDA_VISIBLE_DEVICES` to hide GPUs in strict mode. Fixed `UnboundLocalError` from redundant `import os`. `REPRO_MODE=strict` now properly forces CPU across all phases for true deterministic runs.
  → [View](2026-01-04-gpu-cpu-determinism-config-fix.md)

- **2026-01-04 (RunIdentity Wiring Fixes and Path Organization)** — Fixed critical bugs preventing `run_identity` signatures from appearing in TARGET_RANKING snapshots: (1) Added `run_identity` and `prediction_fingerprint` parameters to `_save_to_cohort()` function. (2) Fixed `log_comparison()` to use locally-computed `partial_identity` instead of null parameter. (3) Added `train_seed` fallback chain and `hparams_signature` computation for TARGET_RANKING. (4) Simplified SYMBOL_SPECIFIC paths - removed redundant `universe=` prefix to match cohort path pattern. TARGET_RANKING snapshots now contain populated signatures for determinism verification.
  → [View](2026-01-04-run-identity-wiring-and-path-organization.md)

- **2026-01-03 (Deterministic Run Identity System)** — Implemented comprehensive deterministic run identity system for reproducibility tracking. Added `RunIdentity` SST dataclass with two-phase construction (partial → final), strict/replicate key separation, registry-based feature fingerprinting with per-feature metadata hashing, and configurable enforcement modes (strict/relaxed/legacy). Hash-based snapshot storage keyed by identity. Feature fingerprinting now includes registry metadata and explicit provenance markers. Partial identities cannot be saved. Stability analysis refuses invalid groups in non-legacy modes.
  → [View](2026-01-03-deterministic-run-identity.md)

## 2025

### December

- **2025-12-23 (Dominance Quarantine and Leakage Safety Enhancements)** — Implemented comprehensive dominance quarantine system for feature-level leakage detection and recovery. Features with dominant importance are detected, confirmed via rerun with suspects removed, and quarantined if score drops significantly. Only blocks target/view if leakage persists after quarantine. Added hard-exclusion of forward-looking features (`time_in_profit_*`) for forward-return targets. Added config-driven small-panel leniency (downgrades BLOCKED to SUSPECT when n_symbols < 10). Fixed `detect_leakage()` import conflict causing TypeError crash. All changes maintain SST compliance and backward compatibility.
  → [View](2025-12-23-dominance-quarantine-and-leakage-safety.md)

- **2025-12-23 (Mode Selection and Pipeline Safety Fixes)** — Fixed 4 critical red flags identified in training logs that connect to "no symbol metrics / 0 jobs / stale routing" issues. Key fixes: (1) Mode selection logic fixed - small panels (<10 symbols) now select SYMBOL_SPECIFIC instead of CROSS_SECTIONAL, preventing missing symbol metrics. (2) Unknown lookback invariant enforced - hard assertion that no inf lookbacks remain after gatekeeper quarantine, prevents RuntimeError. (3) Purge inflation protection - estimates effective samples after purge increase, warns when <30% remaining, fails early when <minimum threshold (configurable). (4) Dev mode job guarantee - generates fallback jobs when router produces 0 jobs in dev_mode, ensuring E2E tests always have jobs. All fixes maintain per-target calculation (purge depends on features selected for each target).
  → [View](2025-12-23-mode-selection-and-pipeline-safety-fixes.md)

- **2025-12-23 (Training Pipeline Integrity and Canonical Layout Migration)** — Fixed 7 critical integrity issues and removed competing `training_results/` hierarchy. Standardized on target-first canonical layout: `run_root/targets/<target>/models/...` as SST. Key fixes: (1) Routing fingerprint mismatch now fails-fast in prod or auto-regenerates in dev (no silent continue). (2) Feature registry bypass fixed - 0 allowed features = error unless dev_mode (metadata stamped). (3) Training families bug fixed - uses `training.model_families` from config with assertion. (4) Routing 0-jobs fixed - metrics aggregation fallback + auto-dev thresholds for small datasets. (5) Stale routing decisions fixed - enforce single known path, removed legacy fallbacks. (6) Removed `training_results/` entirely - all models in canonical location via ArtifactPaths builder. (7) Router pattern added for `*_oc_same_day` targets. Created ArtifactPaths (SST) and optional mirror generation for browsing.
  → [View](2025-12-23-training-pipeline-integrity-and-canonical-layout.md)

- **2025-12-23 (Training Pipeline Organization and Config Fixes)** — Comprehensive refactoring to fix blocking correctness bugs, data integrity issues, and structural cleanup with centralized path SST. Key fixes: (1) Quarantined unknown lookback features before budget call, preventing RuntimeError. (2) Fixed reproducibility files and feature importances to use view/symbol subdirectories, eliminating overwrites. (3) Split model families config - training.model_families for training, feature_selection.model_families for feature selection. (4) Removed legacy METRICS/ creation, reorganized globals/ into subfolders (routing/, training/, summaries/). All changes maintain backward compatibility for reading.
  → [View](2025-12-23-training-pipeline-organization-and-config-fixes.md)

- **2025-12-23 (Boruta Timeout and CatBoost Pickle Error Fixes)** — Fixed two critical errors in feature selection: (1) Improved Boruta timeout error handling to detect timeout errors even when wrapped as ValueError, preventing confusing error messages and pipeline crashes. (2) Fixed CatBoost pickle error by moving importance worker function to module level, enabling multiprocessing for importance extraction. Both fixes improve pipeline stability and error clarity.
  → [View](2025-12-23-boruta-catboost-error-handling-fixes.md)

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
