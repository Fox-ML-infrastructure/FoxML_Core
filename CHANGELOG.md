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

#### 2026-01-06 (Feature Selection Telemetry and Stage Labeling Fixes)
- **Critical Fix**: Fixed fs_snapshot stage mislabeling - snapshots from TARGET_RANKING were incorrectly labeled as "FEATURE_SELECTION".
  - `schema.py`: Added `stage` parameter to `FeatureSelectionSnapshot.from_importance_snapshot()`
  - `schema.py`: Fixed `get_index_key()` to use `self.stage` instead of hardcoded "FEATURE_SELECTION"
  - `hooks.py`: Added `stage` parameter to `save_snapshot_hook()` and `save_snapshot_from_series_hook()`
  - `io.py`: Added `stage` parameter to `create_fs_snapshot_from_importance()`
  - `reporting.py`: Updated TARGET_RANKING caller to pass `stage="TARGET_RANKING"`
- **Bug Fix**: Fixed all 8 model families now get snapshots during feature selection (was only XGBoost).
  - `multi_model_feature_selection.py`: Set `allow_legacy=True` in `save_snapshot_from_series_hook()` calls
  - Added `create_stage_identity` fallback when `run_identity` is None
- **Bug Fix**: Fixed `predictions_sha256` always null in `snapshot_index.json` for TARGET_RANKING.
  - `model_evaluation.py`: Aggregated prediction fingerprints from `model_metrics` and passed to `tracker.log_comparison()`
- **Bug Fix**: Fixed TRAINING stage missing `run_identity` in `log_comparison()` calls.
  - `intelligent_trainer.py`: Added `training_identity = create_stage_identity(...)` before training
  - `training.py`: Added `run_identity` and `experiment_config` parameters, with `create_stage_identity` fallback
- **Bug Fix**: Fixed TensorFlow models failing with "Random ops require a seed to be set when determinism is enabled".
  - `isolation_runner.py`: Added `tf.random.set_seed()` in `_bootstrap_family_runtime()` for TensorFlow families
- **Bug Fix**: Fixed excessive LightGBM warning spam ("No further splits with positive gain").
  - `quantile_lightgbm_trainer.py`: Changed `verbosity: 0` to `verbosity: -1` in `_quantile_params`
- **Bug Fix**: Fixed feature selection telemetry warnings for missing `universe_sig`, `min_cs`, `max_cs_samples`, `train_seed`.
  - `cross_sectional_feature_ranker.py`: Propagated `universe_sig` and added `seed`, `min_cs`, `max_cs_samples` to `RunContext`
  - `cohort_metadata_extractor.py`: Ensured `cs_config` always includes `min_cs` and `max_cs_samples` keys
- **Impact**: All three pipeline stages (TARGET_RANKING, FEATURE_SELECTION, TRAINING) now have complete determinism tracking with correctly labeled snapshots, prediction hashes, and run identities.
- **Files Changed**: `schema.py`, `hooks.py`, `io.py`, `reporting.py`, `multi_model_feature_selection.py`, `model_evaluation.py`, `intelligent_trainer.py`, `training.py`, `isolation_runner.py`, `quantile_lightgbm_trainer.py`, `cross_sectional_feature_ranker.py`, `cohort_metadata_extractor.py`

#### 2026-01-05 (Seed Injection and License Cleanup)
- **Enhancement**: Added automatic seed injection to all model configs for complete tracking.
  - `data_loading.py`: `get_model_config()` now auto-injects seed from SST (global.seed or pipeline.determinism.base_seed)
  - Normalizes seed key per model family: `seed` (LightGBM, XGBoost), `random_state` (sklearn), `random_seed` (CatBoost)
  - Skips deterministic models (Lasso, Ridge) and models that handle seed explicitly (Boruta, Stability Selection)
- **Bug Fix**: Fixed stability selection loop variable (`_` → `i`) causing `UnboundLocalError`.
- **Bug Fix**: Fixed Boruta config cleanup to remove all seed keys (`seed`, `random_state`, `random_seed`).
- **Enhancement**: `train_seed` now populated for all stages (including TARGET_RANKING) for traceability.
  - SST fallback ensures train_seed is always set from config, even when not required for comparison
- **Cleanup**: Removed `HUMANITARIAN_LICENSE.md` and references.
  - Simplified license structure to AGPL-3.0 + Commercial License only
  - Removed contract_tests from git tracking (now in .gitignore)
- **Impact**: Seeds are now fully traceable in model configs and comparison groups. All runs have train_seed populated.
- **Files Changed**: `data_loading.py`, `model_evaluation.py`, `diff_telemetry.py`, `README.md`, `HUMANITARIAN_LICENSE.md` (deleted)

#### 2026-01-05 (Feature Loading Determinism Fix)
- **Critical Fix**: Fixed non-deterministic feature ordering causing different feature counts between runs.
  - `leakage_filtering.py`: Fixed `list(set(...))` → `sorted(set(...))` for deterministic ordering
  - `leakage_filtering.py`: Added `sorted()` to final return to ensure consistent feature order
  - `data_loading.py`: Added `sorted()` to feature_names and reordered DataFrame columns to match
- **Critical Fix**: Added `feature_signature` to TARGET_RANKING required fields.
  - `diff_telemetry/types.py`: Added `feature_signature` to `REQUIRED_FIELDS_BY_STAGE_BASE["TARGET_RANKING"]`
  - `diff_telemetry.py`: Updated `_build_comparison_group_from_context()` to extract `feature_signature` for TARGET_RANKING
  - Added fallback to compute `feature_signature` from `ctx.feature_names` if not available
- **Fix**: Added `train_seed` and `universe_sig` fallbacks for FEATURE_SELECTION tracking.
  - `intelligent_trainer.py`: Added fallback to config `base_seed` (default 42) for `train_seed`
  - `intelligent_trainer.py`: Added `universe_sig` computation from `symbols_to_use`
  - Eliminates `ComparisonGroup missing required fields: [train_seed]` warnings
- **Fix**: Fixed `artifacts_manifest_sha256` not computed for CROSS_SECTIONAL TARGET_RANKING.
  - `diff_telemetry.py`: Normalized artifact lookup to handle different directory structures:
    - SYMBOL_SPECIFIC: `symbol=.../feature_importances/`
    - CROSS_SECTIONAL: `universe=.../feature_importances/`
  - Now correctly finds artifacts in both views
- **Impact**: Runs with identical config/data now produce identical feature sets with identical ordering. Runs with different feature sets are correctly marked as incomparable. FEATURE_SELECTION comparison tracking now works. Artifact manifests computed for both views.
- **Files Changed**: `leakage_filtering.py`, `data_loading.py`, `diff_telemetry.py`, `diff_telemetry/types.py`, `intelligent_trainer.py`

#### 2026-01-04 (Reproducibility File Output Fixes)
- **Critical Fix**: Fixed `snapshot.json` not being written to cohort directories.
  - `diff_telemetry.py`: Fixed path detection to handle target-first structure (`reproducibility/...` vs `REPRODUCIBILITY/...`)
  - Added error handling and logging for snapshot writes
  - Snapshot extraction now uses snapshot object fields instead of path parsing
- **Critical Fix**: Fixed `baseline.json` not being written to cohort directories.
  - `diff_telemetry.py`: `_save_baseline_to_cohort()` now allows lowercase `reproducibility` paths (was blocking on uppercase `REPRODUCIBILITY`)
  - Added error handling and logging
- **Critical Fix**: Fixed diff files (`diff_prev.json`, `metric_deltas.json`, `diff_baseline.json`) not being written.
  - `diff_telemetry.py`: `save_diff()` now correctly handles target-first paths
  - Fixed path parsing to extract stage/view/target from snapshot object when available
  - Ensured files are written even when path parsing fails but directory is already correct
  - Added error handling and logging for all diff file writes
- **Critical Fix**: Fixed path reconstruction for finding previous snapshots.
  - `diff_telemetry.py`: `finalize_run()` now searches in target-first structure (`targets/<target>/reproducibility/...`) instead of legacy `REPRODUCIBILITY/...`
  - Handles both CROSS_SECTIONAL and SYMBOL_SPECIFIC views correctly
  - Falls back to legacy structure if target-first not found
- **Impact**: All reproducibility files (`snapshot.json`, `baseline.json`, `diff_prev.json`, `metric_deltas.json`, `diff_baseline.json`) are now correctly written to cohort directories for both CROSS_SECTIONAL and SYMBOL_SPECIFIC views. Previous snapshot lookup now works correctly in target-first structure.
- **Files Changed**: `diff_telemetry.py`

#### 2026-01-04 (GPU/CPU Determinism Config Fix)
- **Critical Fix**: Fixed config settings being ignored - replaced 4 hardcoded `set_global_determinism()` calls with config-aware `init_determinism_from_config()`.
  - All entry points now read from `reproducibility.yaml` and respect `REPRO_MODE` environment variable
  - `TRAINING/common/determinism.py`: Added `init_determinism_from_config()` function
  - `ranking/predictability/main.py`, `multi_model_feature_selection.py`, `intelligent_trainer.py`, `training_strategies/utils.py`: Replaced hardcoded calls
- **Critical Fix**: GPU device selection now respects strict mode - added `is_strict_mode()` checks to all GPU detection blocks.
  - `model_evaluation.py`: LightGBM, XGBoost, CatBoost now force CPU in strict mode
  - `multi_model_feature_selection.py`: LightGBM now forces CPU in strict mode
  - `xgboost_trainer.py`, `seq_torch_base.py`, `neural_network_trainer.py`: Training phase now forces CPU in strict mode
- **Critical Fix**: Fixed `CUDA_VISIBLE_DEVICES` to hide GPUs in strict mode (`CUDA_VISIBLE_DEVICES="-1"`).
- **Bug Fix**: Fixed `UnboundLocalError` in `set_global_determinism()` from redundant `import os` shadowing module-level import.
- **Impact**: `REPRO_MODE=strict` now properly forces CPU across all phases (target ranking, feature selection, training) for true deterministic runs. Config settings in `reproducibility.yaml` are now respected.
- **Files Changed**: `determinism.py`, `model_evaluation.py`, `multi_model_feature_selection.py`, `intelligent_trainer.py`, `training_strategies/utils.py`, `xgboost_trainer.py`, `seq_torch_base.py`, `neural_network_trainer.py`

#### 2026-01-04 (RunIdentity Wiring Fixes and Path Organization)
- **Critical Fix**: Fixed `NameError: name 'run_identity' is not defined` in `_save_to_cohort()`.
  - `reproducibility_tracker.py`: Added `run_identity` and `prediction_fingerprint` parameters to `_save_to_cohort()` function signature
  - Updated all 4 call sites to pass the new parameters through to `finalize_run()`
- **Critical Fix**: Fixed `run_identity=run_identity` using null parameter instead of computed identity.
  - `model_evaluation.py`: Changed `log_comparison(..., run_identity=run_identity)` to use `partial_identity` (computed at line 6291 with actual data)
  - This was the root cause of null signatures in TARGET_RANKING snapshots
- **Feature**: Populated `train_seed` and `hparams_signature` for TARGET_RANKING stage.
  - `model_evaluation.py`: Added fallback chain for `train_seed`: experiment_config.seed → config loader → default 42
  - `model_evaluation.py`: Compute `hparams_signature` from evaluation model families for reproducibility tracking
- **Critical Fix**: Simplified SYMBOL_SPECIFIC path organization - removed redundant `universe=` prefix.
  - `output_layout.py`: `repro_dir()` now returns `SYMBOL_SPECIFIC/symbol={sym}/` instead of `SYMBOL_SPECIFIC/universe={sig}/symbol={sym}/`
  - This matches the cohort path pattern used by `reproducibility_tracker.py`
  - CROSS_SECTIONAL unchanged (still uses `universe=` to identify symbol set)
- **Impact**: TARGET_RANKING snapshots now contain populated `dataset_signature`, `task_signature`, `routing_signature`, `train_seed`, `hparams_signature`. SYMBOL_SPECIFIC directory structure is now consistent between cohorts and feature importances.
- **Files Changed**: `reproducibility_tracker.py`, `model_evaluation.py`, `output_layout.py`

#### 2026-12-30 (Prediction Hashing for Determinism Verification and Drift Detection)
- **Feature**: Implemented prediction hashing system for determinism verification and live drift detection.
  - `prediction_hashing.py` (NEW): Utility for computing stable, reproducible prediction fingerprints
  - `PredictionFingerprint` dataclass: Contains strict hash, live hash, row_ids_hash, classes_hash, kind, dtype, shape
  - `prediction_fingerprint()`: Computes both strict (bitwise) and live (quantized) hashes
  - `canonicalize_preds()`: Standardizes predictions (dtype, NaN handling, optional quantization)
  - `compare_prediction_fingerprints()`: Compares two fingerprints with row identity validation
- **Feature**: Prediction hash integration into snapshot schema:
  - `schema.py`: Added `prediction_hash`, `prediction_hash_live`, `prediction_row_ids_hash`, `prediction_classes_hash`, `prediction_kind` fields
  - `hooks.py`: `save_snapshot_hook()` accepts `prediction_fingerprint` parameter
  - `analysis.py`: `compute_stability_metrics()` tracks `pred_hash_match_rate` and flags strict mode mismatches
- **Feature**: Live-safe hashing scheme:
  - Row identity binding (hashes include row_ids to ensure order independence)
  - Classification-specific handling (class order, probabilities vs labels)
  - Quantized hash for drift tolerance (default 1e-6)
  - NaN canonicalization for cross-platform stability
- **Impact**: Prediction hashes can now be tracked as metrics alongside feature importance, enabling:
  - Strict determinism verification (bitwise identical predictions = identical hashes)
  - Live drift detection (quantized hashes tolerate float jitter)
  - Audit trail (tamper-evident prediction records)
- **Files Created**: `TRAINING/common/utils/prediction_hashing.py`
- **Files Changed**: `schema.py`, `hooks.py`, `analysis.py`

#### 2026-01-03 (Diff Telemetry - RunIdentity Wiring and Path Fix)
- **Feature**: Wired `run_identity` SST object through entire snapshot creation chain for authoritative signatures.
  - `diff_telemetry.py`: Added `run_identity` and `prediction_fingerprint` to `normalize_snapshot()`, `finalize_run()`, `_build_comparison_group_from_context()`, `_compute_predictions_digest()`
  - `reproducibility_tracker.py`: Added `run_identity` and `prediction_fingerprint` to `log_comparison()`, passed through to `finalize_run()` at both call sites
  - `target_ranker.py`: Added `run_identity` to `evaluate_target_predictability()` and `rank_targets()`, wired through 6 internal call sites
  - `model_evaluation.py`: Added `run_identity` to `evaluate_target_predictability()` function signature
  - `feature_selector.py`: Updated `log_comparison()` call to pass `run_identity`
- **Critical Fix**: Fixed CROSS_SECTIONAL vs SYMBOL_SPECIFIC path organization bug in multiple locations.
  - `reproducibility_tracker.py`: If symbol is set OR cohort_id starts with `sy_`, force `view = "SYMBOL_SPECIFIC"` (3 locations)
  - `diff_telemetry.py`: `save_snapshot()` and `save_diff()` now prefer snapshot's view/symbol over path parsing
  - Prevents symbol-specific cohorts (e.g., BAC) being written to CROSS_SECTIONAL directory
- **Impact**: Snapshots now contain populated `feature_signature`, `hparams_signature`, `dataset_signature`, `predictions_sha256` from SST. Comparison groups correctly scope by view/symbol, ensuring BAC only compares to BAC.
- **Files Changed**: `diff_telemetry.py`, `reproducibility_tracker.py`, `target_ranker.py`, `model_evaluation.py`, `feature_selector.py`

#### 2026-01-03 (Determinism SST - Production-Ready Reproducibility)
- **Feature**: Implemented production-grade Single Source of Truth (SST) for determinism enforcement.
  - `repro_bootstrap.py` (NEW): Pre-import bootstrap that sets thread env vars BEFORE numpy/torch are imported
  - `bin/run_deterministic.sh` (NEW): Launcher script that sets `PYTHONHASHSEED` before Python starts
  - `reproducibility.yaml` (NEW): SST config for strict/best_effort modes with subsettings
- **Feature**: Strict mode for bitwise reproducible runs (financial audit compliance):
  - Forces CPU-only execution (`device_type: cpu`)
  - Single-threaded (`n_jobs=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`)
  - Hard-fails if `PYTHONHASHSEED` not set before Python starts
  - Hard-fails if numeric libraries imported before bootstrap
- **Feature**: `create_estimator()` single choke point for all model instantiation:
  - Applies determinism params automatically (random_state, n_jobs, device_type)
  - Uses correct sklearn API keys per library (n_jobs vs num_threads)
  - `normalize_seed()` prevents seed=0 and overflow issues
  - `resolve_seed()` uses SHA256 for stable, versioned seed derivation
- **Feature**: `stable_sort()` for deterministic ordering at key boundaries (features, targets, folds, symbols).
- **Feature**: Import-time strict assertion in `determinism.py` catches bootstrap-before-import violations.
- **Testing**: Two-process strict determinism test (`test_determinism_strict.py`) with bitwise prediction hash comparison.
- **Config**: `determinism_test.yaml` experiment config for verifying reproducibility.
- **Impact**: Runs with `bin/run_deterministic.sh` produce bitwise identical results across separate process invocations. Verified with prediction hash comparison.
- **Files Created**: `repro_bootstrap.py`, `bin/run_deterministic.sh`, `reproducibility.yaml`, `determinism_test.yaml`, `test_determinism_strict.py`
- **Files Changed**: `determinism.py` (major additions), `intelligent_trainer.py` (bootstrap import)

#### 2026-01-03 (Deterministic Run Identity System)
- **Critical Fix**: Added `set_global_determinism()` to `intelligent_trainer.py` - **this was the root cause of non-reproducible runs**.
  - Same config + same data now produces identical fingerprints every run
  - Sets `PYTHONHASHSEED`, numpy/random seeds, and ML library seeds at module load time
  - Must be called before any ML library imports
- **Feature**: Implemented comprehensive deterministic run identity system for reproducibility tracking.
  - `fingerprinting.py`: Added `RunIdentity` SST dataclass with two-phase construction (partial → final)
  - `fingerprinting.py`: Strict/replicate key separation (strict includes seed, replicate excludes for cross-seed stability)
  - `fingerprinting.py`: Registry-based feature fingerprinting with per-feature metadata hashing
  - `fingerprinting.py`: Finance-safe timestamp canonicalization (`canonicalize_timestamp()` with epoch inference)
  - `config_hashing.py`: Centralized canonicalization (`canonical_json`, `sha256_full`, `sha256_short`)
  - `identity_config.yaml`: Production-grade configurable enforcement (strict/relaxed/legacy modes)
- **Feature**: Finance-safe dataset identity includes:
  - `symbols_digest`: Hash of sorted symbols (not raw list)
  - `start_ts_utc`, `end_ts_utc`: Canonicalized UTC timestamps from actual data
  - Row-shaping filters: `max_rows_per_symbol`, `max_samples_per_symbol`, `interval`, `min_cs_samples`
  - `sampling_method`: Algorithm version tracking
  - Strict mode raises on empty timestamps; relaxed mode marks `timestamp_canon_failed`
- **Feature**: Hash-based snapshot storage keyed by identity (`replicate/<replicate_key>/<strict_key>.json`).
- **Feature**: Configurable identity enforcement via `CONFIG/identity_config.yaml`:
  - `identity.mode`: strict (fail on missing), relaxed (log and continue), legacy (backward compat)
  - `stability.filter_mode`: replicate (cross-seed), strict (same seed), legacy
  - `feature_identity.mode`: registry_resolved (per-feature metadata), names_only (deprecated)
- **Enhancement**: Feature fingerprinting now includes registry metadata (lag_bars, source, allowed_horizons, version) and explicit provenance markers (`registry_explicit`, `registry_mixed`, `registry_inferred`, `names_only_degraded`).
- **Safety**: Partial identities cannot be saved (enforced via `is_final` flag and `finalize()` validation).
- **Safety**: Stability analysis refuses invalid/mismatched signature groups in non-legacy modes.
- **Impact**: Runs with identical configurations produce identical identity keys and fingerprints. Stability analysis only compares truly equivalent runs. No silent degradation in strict mode.
- **Files Changed**: `intelligent_trainer.py` (determinism), `fingerprinting.py`, `config_hashing.py`, `hooks.py`, `feature_selector.py`, `model_evaluation.py`, `cross_sectional_feature_ranker.py`, `multi_model_feature_selection.py`, `identity_config.yaml` (NEW)

#### 2026-01-02 (Horizon-Aware Routing and Telemetry Comparison Fixes)
- **Feature**: Implemented horizon-aware routing thresholds for regression targets.
  - `routing.yaml`: Regression `min_score` and `strong_score` now support horizon tiers: `short` (<60min), `medium` (60min-4h), `long` (4h-1d), `very_long` (>1d)
  - `training_router.py`: Added `_resolve_horizon_minutes()` and `_get_horizon_tier()` helper functions
  - `training_router.py`: Modified `_get_score_threshold()` to accept optional `horizon_minutes` and select appropriate tier
  - Long-horizon targets (e.g., `fwd_ret_5d`) no longer blocked due to inherently lower R² scores
- **Critical Fix**: Fixed telemetry comparing different symbols/universes incorrectly (e.g., AAPL metrics compared to AVGO).
  - `diff_telemetry/types.py`: Added `symbol` and `universe_sig` fields to `ComparisonGroup`
  - `diff_telemetry/types.py`: Updated `to_key()` to include both fields in comparison key
  - `diff_telemetry.py`: Added `symbol` and `universe_sig` parameters to `_build_comparison_group_from_context()`
  - `diff_telemetry.py`: Extracted `universe_sig` from `additional_data` with `cs_config` fallback
  - `diff_telemetry.py`: Added explicit symbol check in `_check_comparability()` for SYMBOL_SPECIFIC view
  - Now AAPL only compares to previous AAPL runs, not AVGO; CS runs only compare to same universe
- **Impact**: Long-horizon regression targets route correctly. Telemetry diffs are accurate (no more misleading "auc: +99%" changes from cross-symbol comparisons). Stability warnings "Found snapshots from 2 different symbols/universes" should stop appearing.
- **Files Changed**: `routing.yaml`, `training_router.py`, `diff_telemetry/types.py`, `diff_telemetry.py`

#### 2026-01-02 (universe_sig Propagation Fixes and High-AUC Investigation)
- **Fix**: Improved `universe_sig` propagation in feature selection pipeline to eliminate `Missing universe_sig` warnings.
  - `feature_selector.py`: Added early initialization of `universe_sig_for_writes`, `view_for_writes`, `symbol_for_writes` before shared harness block
  - `feature_selector.py`: Added `resolve_write_scope()` call in SYMBOL_SPECIFIC branch to set SST scope variables
  - `feature_selector.py`: Added `universe_sig` computation in fallback path when shared harness fails
  - `shared_ranking_harness.py`: Added missing `requested_view` and `output_dir` parameters to CROSS_SECTIONAL call
  - `model_evaluation.py`: Fixed log message to show effective view (`view_for_writes`) instead of requested view, preventing confusing "CROSS_SECTIONAL (symbol=X)" messages when SS->CS promotion is blocked
- **Investigation**: Analyzed high AUC warnings (>0.90 ROC-AUC across 5 model families) for MFE targets.
  - Confirmed feature exclusion patterns are working correctly (232 columns excluded, 287 safe)
  - Verified `fwd_ret_*`, `mfe_*`, `mdd_*`, `max_return_*` columns are properly rejected
  - Root cause: Small sample overfitting (1997 rows/symbol with 210+ features) + likely class imbalance
  - Existing safeguard "Small panel detected, downgrading severity" is appropriate behavior
- **Impact**: Cleaner logs, correct view display, better scope tracking for reproducibility.
- **Files Changed**: `feature_selector.py`, `shared_ranking_harness.py`, `model_evaluation.py`

#### 2026-01-02 (Critical Bug Fixes: seed/random_state and Pipeline Stability)
- **Critical Fix**: Fixed 25+ instances of `seed=` being passed to sklearn APIs that expect `random_state=`, causing `TypeError: got an unexpected keyword argument 'seed'` across trainers and feature selection.
  - `train_test_split()` in 11 trainer files
  - `HistGradientBoostingRegressor`, `RandomForestRegressor` in `ensemble_trainer.py`, `ngboost_trainer.py`
  - `MLPRegressor`, `Lasso`, `ElasticNet`, `Ridge` in `multi_model_feature_selection.py`
  - `permutation_importance()` in `importance_extractors.py`
  - `LGBMRegressor/LGBMClassifier` in `feature_pruning.py`
- **Critical Fix**: Removed invalid `seed=` parameter from `Ridge` calls (Ridge is deterministic, no random component).
- **Critical Fix**: Fixed `UnboundLocalError: local variable 'cohort_metadata' referenced before assignment` in `reproducibility_tracker.py` when NON_COHORT mode triggered.
- **Task-Type Aware Routing**: Implemented separate routing thresholds for classification (AUC-based) vs regression (R²-based) targets.
  - `routing.yaml`: `min_score` and `strong_score` now accept `{classification: X, regression: Y}` format
  - `training_router.py`: Added `_get_score_threshold()` helper, updated eligibility evaluation
  - `metrics_aggregator.py`: Now propagates `task_type` and `metric_name` in routing candidates
- **Routing Fix**: Added `UNKNOWN` to `stability_allowlist` to allow first runs (no stability history).
- **Impact**: Training pipeline no longer crashes with TypeError on seed parameters. Regression targets with negative R² scores now route correctly. First-run targets no longer blocked by missing stability.
- **Files Changed**: `ensemble_trainer.py`, `ngboost_trainer.py`, `lightgbm_trainer.py`, `xgboost_trainer.py`, `mlp_trainer.py`, `lstm_trainer.py`, `cnn1d_trainer.py`, `transformer_trainer.py`, `vae_trainer.py`, `meta_learning_trainer.py`, `ftrl_proximal_trainer.py`, `reward_based_trainer.py`, `gmm_regime_trainer.py`, `gan_trainer.py`, `multi_task_trainer.py`, `change_point_trainer.py`, `quantile_lightgbm_trainer.py`, `base_trainer.py`, `data_preprocessor.py`, `multi_model_feature_selection.py`, `importance_extractors.py`, `feature_pruning.py`, `reproducibility_tracker.py`, `training_router.py`, `metrics_aggregator.py`, `routing.yaml`

#### 2025-12-31 (universe_sig Propagation and Scope Tracking Fixes)
- **Critical Fix**: Fixed `Missing universe_sig` warnings across FEATURE_SELECTION and TARGET_RANKING stages.
  - `feature_selector.py`: Added `universe_sig_for_ctx` extraction and propagation to `RunContext` constructors
  - `model_evaluation.py`: Fixed `universe_sig=view` bug (was passing view string as universe_sig), added proper extraction
  - `reproducibility_tracker.py`: NON_COHORT fallback now preserves `universe_sig`, `symbols`, `min_cs`, `max_cs_samples` in `cs_config`
  - `cross_sectional_feature_ranker.py`: Added `horizon_minutes` extraction to prevent COHORT_AWARE downgrade
- **Bug Fix**: Fixed `universe_sig=view` typo in `model_evaluation.py` line 1357 (was passing "CROSS_SECTIONAL" as universe_sig).
- **Bug Fix**: Fixed `Missing required fields for COHORT_AWARE mode: ['horizon_minutes']` by extracting horizon from target column name.
- **Impact**: Scope tracking now works correctly, no more legacy path fallbacks, diff telemetry receives complete metadata.
- **Files Changed**: `feature_selector.py`, `model_evaluation.py`, `reproducibility_tracker.py`, `cross_sectional_feature_ranker.py`

#### 2025-12-30 (SST Variable Naming Unification and Vocabulary Lock)
- **Refactoring**: Implemented Single Source of Truth (SST) for variable naming across the modeling pipeline.
  - Unified `view`/`mode`/`scope` → canonical `view` with `View` enum
  - Unified `n_samples`/`N_effective`/`n_rows` → canonical `n_effective`
  - Unified `date_start`/`start_date`/`date_range.start` → canonical accessors
  - Unified `universe_id`/`universe_sig`/`symbol_hash` → canonical `universe_sig`
  - Unified `target_name`/`target_col`/`target` → canonical `target_column`
  - Unified `random_state`/`seed` → canonical `seed` (internal), `random_state` (sklearn APIs)
- **Duplicate Function Consolidation**: Merged 6 duplicate fingerprint/hash functions into centralized implementations.
  - `_compute_feature_fingerprint` → single implementation in `fingerprinting.py`
  - `get_git_commit` → single implementation in `git_utils.py`
  - `compute_config_hash`, `compute_data_fingerprint` → centralized in `fingerprinting.py`
- **Fallback Chain Elimination**: Removed 50+ instances of redundant `.get()` fallback chains like `config.get('view') or config.get('mode') or config.get('scope')`.
- **Data Contracts**: Updated `RunContext`, `WriteScope`, `View` enum with SST accessor methods.
- **Impact**: Consistent naming across codebase, reduced cognitive load, eliminated vocabulary drift bugs.
- **Files Changed**: `run_context.py`, `scope_resolution.py`, `reproducibility_tracker.py`, `intelligent_trainer.py`, `feature_selector.py`, `model_evaluation.py`, `cross_sectional_data.py`, `training_router.py`, `metrics_aggregator.py`, and 40+ other files

#### 2025-12-29 (Routing Pipeline Metrics Lookup and Alias Validation)
- **Critical Fix**: Fixed routing pipeline producing 0 jobs due to metrics path mismatch. `_load_symbol_metrics()` was looking for `multi_model_metadata.json` directly in `symbol=X/` but metrics were written inside `cohort=Y/metrics.json`. Now descends into cohort directories with deterministic selection (mtime, name tuple for tie-breaking).
- **Critical Fix**: Removed incorrect `mlp` → `neural_network` alias from `FAMILY_ALIASES`. `mlp` and `neural_network` are distinct trainers with separate implementations (`MLPTrainer` vs `NeuralNetworkTrainer`). `normalize_family("mlp")` now correctly returns `"mlp"`.
- **New Feature**: Added `validate_sst_contract()` for comprehensive alias validation at startup:
  - No alias key may shadow a canonical key (strict policy)
  - Alias target must exist in canonical registry
  - Cross-source conflict detection (same key, different targets)
  - Normalization collision detection across alias keys
  - Chain and cycle detection with path tracking
- **Safety Enhancement**: Family invariant check now validates against `training.model_families` (SST) instead of `feature_selection.model_families`. Uses `normalize_family()` on both sides for apples-to-apples comparison.
- **Safety Enhancement**: Fail-fast on 0 jobs - `ValueError("FATAL: Training plan has 0 jobs...")` raised with routing diagnostics. Exception explicitly re-raised past outer `try/except` handlers.
- **Bug Fix**: `target_repro_dir()` now uses `path_mode` variable derived from SST `resolved_mode` instead of mutating caller's `view` parameter. Warn-once pattern for view mismatches.
- **Bug Fix**: Moved `SPECIAL_CASES` to module scope in `sst_contract.py` so both `normalize_family()` and `validate_sst_contract()` reference the same source of truth. Removed canonical self-map `"xgboost": "xgboost"`.
- **Config**: Added optional `routing.allow_mode_fallback` flag (default: false) to allow CS metrics fallback when symbol metrics missing.
- **Impact**: Training pipeline no longer fails with 0 jobs on valid data. Alias conflicts detected at startup. Future alias mistakes (shadow, chain, cycle, missing target) fail fast.
- **Files Changed**: `metrics_aggregator.py`, `intelligent_trainer.py`, `routing_integration.py`, `target_first_paths.py`, `sst_contract.py`, `registry_validation.py`

#### 2025-12-27 (Humanitarian & Public Benefit License Exception)
- **Licensing**: Added Humanitarian & Public Benefit License Exception as a third licensing option alongside AGPL-3.0 and Commercial License
- **New Document**: Created `HUMANITARIAN_LICENSE.md` with explicit eligibility criteria, conditions, and FAQ
- **Eligible Organizations**: Registered non-profits, accredited academic/research institutions, government/public-sector research bodies, NGOs/charities, companies conducting non-commercial humanitarian work (with organizational isolation requirement)
- **Anti-Abuse Measures**: Includes indirect revenue clause blocking lead generation, investor signaling, efficiency gains for commercial operations, and similar loopholes
- **Transition Clause**: Organizations transitioning to commercial use must obtain a commercial license
- **Termination Clause**: Upon termination, continued use constitutes unlicensed commercial use
- **Documentation Updates**: Updated README.md, COMMERCIAL_LICENSE.md, LEGAL/LICENSING.md, LEGAL/QUICK_REFERENCE.md, LEGAL/ACADEMIC.md to reference the new exception
- **Impact**: Enables genuine humanitarian use (climate research, public health NGOs, disaster response analytics) while blocking VC-backed startups, hedge funds, and internal corporate research teams
- **Files Changed**: `HUMANITARIAN_LICENSE.md` (NEW), `README.md`, `COMMERCIAL_LICENSE.md`, `LEGAL/LICENSING.md`, `LEGAL/QUICK_REFERENCE.md`, `LEGAL/ACADEMIC.md`

#### 2025-12-23 (Scope Violation Firewall and Output Layout)
- **Critical Fix**: Implemented scope violation firewall to prevent symbol-specific cohorts (`cohort=sy_*`) from being incorrectly saved under `CROSS_SECTIONAL/` paths. This was causing feature importance and artifacts to be saved at the wrong scope, breaking reproducibility.
- **New Feature**: Created `OutputLayout` dataclass as Single Source of Truth (SST) for all output paths with hard invariants:
  - `view` must be `CROSS_SECTIONAL` or `SYMBOL_SPECIFIC`
  - `SYMBOL_SPECIFIC` requires `symbol`, `CROSS_SECTIONAL` cannot have `symbol`
  - `universe_sig` (hash of sorted symbols) required for all scopes
- **Validation**: Added `validate_cohort_id()` method that catches `cs_` vs `sy_` prefix mismatches at write time with explicit `startswith()` checks
- **Firewall**: Updated `_save_to_cohort()` to validate every write via OutputLayout. Falls back to legacy paths with warning if metadata incomplete. Telemetry trap logs `SCOPE VIOLATION RISK` for debugging bad callers.
- **Migration**: All `_compute_cohort_id()` callers updated to pass required `view` parameter (6 call sites)
- **Path Extensions**: Extended `target_first_paths.py` and `artifact_paths.py` functions to accept optional `universe_sig` parameter for cross-run reproducibility
- **Config**: Added `safety.output_layout.strict_scope_partitioning` flag (default: false, set true after validation)
- **Impact**: Prevents silent data corruption from scope violations, ensures correct view+universe partitioning, enables cross-run reproducibility
- **Files Changed**: `output_layout.py` (NEW), `reproducibility_tracker.py`, `target_first_paths.py`, `artifact_paths.py`, `intelligent_trainer.py`, `safety.yaml`, `test_scope_violation_firewall.py` (NEW)
→ [Audit Trail](docs/audit/2024-12-23/21-15-07_scope_violation_firewall_pr1/)

#### 2025-12-23 (Feature Registry, Symbol Discovery, and Mode Resolution Fixes)
- **Feature Registry Fix**: Fixed 114 features with incorrect `lag_bars: 0` and `AUTO-REJECTED` status. Correct lookback values now set using proper minute-to-bar (`ceil(min/5)`) and day-to-bar (`ceil(day*1440/5)`) conversions. Calendar features (`_hour`, `day_of_week`, etc.) correctly unrejected with `lag_bars=0`.
- **New Feature**: Auto-discover symbols from `data_dir` when `symbols: []` (empty). Optional `symbol_batch_size` limits selection with deterministic seeded sampling. Symbols flow through entire pipeline (ranking → feature selection → routing → training).
- **Bug Fix**: Universe-scoped mode resolution - `resolved_mode` is now keyed by universe signature (blake2s hash of sorted symbols) instead of being globally immutable. Different symbol universes correctly resolve to different modes (e.g., 25 symbols → CROSS_SECTIONAL, 1 symbol → SINGLE_SYMBOL_TS).
- **Bug Fix**: Fixed string accumulation in `mode_reason` logging - stores `original_reason` separately and references it directly on reuse (no recursive nesting).
- **Bug Fix**: Fixed multiple `IndentationError` and `SyntaxError` issues across `intelligent_trainer.py`, `training.py`, `leakage_budget.py`, `leakage_detection.py`.
- **Bug Fix**: Fixed `IntelligentTrainer` import issue - package `__init__.py` now re-exports class from sibling module file.
- **Impact**: Training no longer logs `lag_bars=0 → fallback to inference` for known features. Symbol discovery enables flexible experiment configs. Mode resolution correctly handles per-symbol evaluation loops.
- **Files Changed**: `feature_registry.yaml`, `config_schemas.py`, `config_builder.py`, `symbol_discovery.py` (NEW), `run_context.py`, `cross_sectional_data.py`, `intelligent_trainer.py`, `training.py`, `leakage_budget.py`, `leakage_detection.py`, `fix_feature_registry.py` (NEW)

#### 2025-12-23 (Dominance Quarantine and Leakage Safety Enhancements)
- **New Feature**: Implemented dominance quarantine system (auto-suspect → confirm → quarantine workflow) for feature-level leakage detection and recovery. Features with dominant importance (30%+ share, 3× next feature) are detected, confirmed via rerun with suspects removed, and quarantined if score drops significantly. Only blocks target/view if leakage persists after quarantine.
- **Safety Enhancement**: Hard-exclude `time_in_profit_*` and similar forward-looking profit/PnL features for forward-return targets (prevents label-proxy leakage before dominance detection triggers).
- **Safety Enhancement**: Config-driven small-panel leniency - downgrades leakage BLOCKED to SUSPECT when `n_symbols < 10`, allowing dominance quarantine to attempt recovery before blocking (critical for E2E tests).
- **Bug Fix**: Fixed `detect_leakage()` import conflict causing `TypeError: unexpected keyword argument 'X'` crash. Removed conflicting import, added error handling so leakage detection failures don't crash target evaluation.
- **Impact**: Prevents permanent feature drops on first trigger, allows recovery via rerun, makes pipeline resilient to leakage detection errors, better small-panel support.
- **Files Changed**: `safety.yaml`, `dominance_quarantine.py` (NEW), `model_evaluation.py`, `target_conditional_exclusions.py`, `leakage_detection.py`, `metrics_aggregator.py`, `shared_ranking_harness.py`, `multi_model_feature_selection.py`, `training.py`, `model_evaluation/__init__.py`, test files
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-23-dominance-quarantine-and-leakage-safety.md)

#### 2025-12-23 (Mode Selection and Pipeline Safety Fixes)
- **Critical Fixes**: Fixed 4 red flags from training logs: (1) Mode selection - small panels (<10 symbols) now select SYMBOL_SPECIFIC instead of CROSS_SECTIONAL, preventing missing symbol metrics → 0 jobs. (2) Unknown lookback invariant - hard assertion that no inf lookbacks remain after gatekeeper quarantine, prevents RuntimeError in compute_budget. (3) Purge inflation protection - estimates effective samples after purge increase, warns when <30% remaining, fails early when <minimum threshold (configurable via `training_config.routing.min_effective_samples_after_purge`). (4) Dev mode job guarantee - generates fallback jobs when router produces 0 jobs in dev_mode, ensuring E2E tests always have jobs.
- **Impact**: Prevents "no symbol metrics", "0 jobs", and RuntimeError from unknown lookback features. Purge calculation remains per-target and depends on features selected for each target.
- **Files Changed**: `cross_sectional_data.py`, `shared_ranking_harness.py`, `resolved_config.py`, `training_plan_generator.py`
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-23-mode-selection-and-pipeline-safety-fixes.md)

#### 2025-12-23 (Training Pipeline Integrity and Canonical Layout Migration)
- **Critical Fixes**: Fixed 7 integrity issues: routing fingerprint mismatch (fail-fast), feature registry bypass (0-features = error), training families source bug (uses config), routing 0-jobs (metrics fallback + auto-dev thresholds), stale routing decisions (single path), output layout inconsistencies (removed training_results/), router pattern miss (*_oc_same_day)
- **Structural Refactoring**: Removed competing `training_results/` hierarchy, standardized on target-first canonical layout: `run_root/targets/<target>/models/...` as SST
- **New ArtifactPaths Builder**: Created single source of truth for all artifact paths - all model saves/loads use `ArtifactPaths.model_dir()`
- **Optional Mirrors**: Added optional family-first browsing via symlinks or manifest (config-driven, disabled by default)
- **Impact**: Eliminates structural ambiguity, ensures reproducibility, enforces config-centered control, prevents stale decision loads
- **Files Changed**: `artifact_paths.py` (new), `artifact_mirror.py` (new), `training.py`, `intelligent_trainer.py`, `target_routing.py`, `leakage_filtering.py`, `metrics_aggregator.py`, `training_router.py`, `target_router.py`, `target_first_paths.py`
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-23-training-pipeline-integrity-and-canonical-layout.md)

#### 2025-12-23 (Training Pipeline Organization and Config Fixes)
- **Refactoring**: Comprehensive refactoring to fix blocking correctness bugs, data integrity issues, and structural cleanup with centralized path SST
- **Correctness**: Quarantined unknown lookback features before budget call, preventing RuntimeError from features with `inf` lookback
- **Data Integrity**: Fixed reproducibility files and feature importances to use view/symbol subdirectories, eliminating overwrites across views/symbols
- **Config Correctness**: Split model families config - `training.model_families` for training, `feature_selection.model_families` for feature selection
- **Structure**: Removed legacy METRICS/ creation, reorganized globals/ into subfolders (routing/, training/, summaries/)
- **Impact**: Prevents data corruption, fixes config routing, improves organization, maintains backward compatibility
- **Files Changed**: `target_first_paths.py`, `shared_ranking_harness.py`, `feature_selection_reporting.py`, `multi_model_feature_selection.py`, `feature_selector.py`, `intelligent_trainer.py`, `training_plan_consumer.py`, `routing_integration.py`, `metrics_aggregator.py`, `training_router.py`, `training_plan_generator.py`
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-23-training-pipeline-organization-and-config-fixes.md)

#### 2025-12-23 (Boruta Timeout and CatBoost Pickle Error Fixes)
- **Bug Fix**: Improved Boruta timeout error handling to detect timeout errors even when wrapped as ValueError
- **Bug Fix**: Fixed CatBoost pickle error by moving importance worker function to module level for multiprocessing
- **Impact**: Prevents pipeline crashes, improves error clarity, enables CatBoost importance extraction to complete successfully
- **Files Changed**: `model_evaluation.py`, `multi_model_feature_selection.py`
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-23-boruta-catboost-error-handling-fixes.md)

#### 2025-12-23 (Comprehensive Model Timing Metrics)
- **Enhancement**: Added comprehensive timing metrics (start-time and elapsed-time logging) for all 12 model families in target ranking and feature selection
- **New Models**: Added timing for XGBoost, Random Forest, Lasso, Elastic Net, Ridge, Neural Network, Mutual Information, Stability Selection, Histogram Gradient Boosting
- **Existing Models**: Added start logging for LightGBM, CatBoost, and Boruta (already had elapsed timing)
- **Impact**: Enables easy diagnosis of performance bottlenecks by showing execution sequence, individual model times, and overall percentage breakdown
- **Files Changed**: `model_evaluation.py`
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-23-comprehensive-model-timing-metrics.md)

#### 2025-12-22 (License Banner Pricing Structure Update)
- **Enhancement**: Restructured license banner with more realistic and approachable pricing ladder
- **New Tiers**: Added Commercial Evaluation ($1k–$5k) and Commercial License tiers ($10k–$25k small team, $25k–$60k team)
- **Clarification**: Split License vs Support (Enterprise tier includes SLA, response times, on-call)
- **Fix**: Corrected AGPL wording - "make source available to users" (can be internal, not necessarily public)
- **Pricing**: Reduced Pilot from $35k to $10k–$20k, Enterprise starts at $120k+/year with SLA
- **Impact**: More approachable entry point for organizations, clearer value proposition at each tier
- **Files Changed**: `license_banner.py`
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-22-license-banner-pricing-update.md)

#### 2025-12-22 (Performance Audit System for Multiplicative Work Detection)
- **Feature**: Added comprehensive performance audit system to detect "accidental multiplicative work"
- **Instrumentation**: Tracks call counts and timing for heavy functions (CatBoost importance, build_panel, train_model, etc.)
- **Automatic Reports**: Generates audit report at end of training run showing multipliers, nested loops, and cache opportunities
- **Impact**: Proactively identifies performance bottlenecks where expensive operations are called multiple times unnecessarily
- **Files Changed**: `performance_audit.py` (NEW), `intelligent_trainer.py`, `multi_model_feature_selection.py`, `shared_ranking_harness.py`, `model_evaluation.py`, `leakage_detection.py`
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-22-performance-audit-system.md)

#### 2025-12-22 (Trend Analyzer Indentation Fix)
- **Bug Fix**: Fixed critical indentation errors in `trend_analyzer.py` that prevented the module from loading
- **Bug Fix**: Corrected `if targets_dir.exists():` indentation inside `for current_run_dir in runs_to_process:` loop (line 204)
- **Bug Fix**: Fixed `try:` block indentation inside `for cohort_dir in view_dir.iterdir():` loop (line 301)
- **Bug Fix**: Fixed cascading indentation issues in nested blocks (SYMBOL_SPECIFIC and CROSS_SECTIONAL views)
- **Impact**: Trend analyzer now loads correctly and can analyze trends across runs, generate summaries, and process both view types
- **Files Changed**: `trend_analyzer.py`
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-22-trend-analyzer-indentation-fix.md)

#### 2025-12-22 (CatBoost Overfitting Detection and Legacy Structure Removal)
- **Performance Fix**: Fixed CatBoost feature selection taking 6+ hours by implementing config-driven overfitting detection and process-based timeout
- **Feature**: Added shared overfitting detection helper with policy-based gating (train_acc threshold, train/CV gap, train/val gap, feature count cap)
- **Feature**: Added process-based timeout (30 minutes) for expensive PredictionValuesChange importance computation
- **Feature**: Added comprehensive timing diagnostics with `timed()` context manager (CV, fit, importance stages)
- **Enhancement**: Deterministic fallback importance (gain/split/none) when skipping expensive PVC - preserves comparability
- **Bug Fix**: Fixed YAML config structure - moved `feature_importance` to correct location under `leakage_detection`
- **Code Cleanup**: Removed legacy `RESULTS/REPRODUCIBILITY/FEATURE_SELECTION/...` structure creation
- **Bug Fix**: Fixed path resolution to not stop at `RESULTS/` directory - now finds actual run directory correctly
- **Impact**: Prevents 6-hour hangs, ensures all writes go to target-first structure only, comprehensive timing visibility
- **Files Changed**: `safety.yaml`, `overfitting_detection.py` (NEW), `multi_model_feature_selection.py`, `feature_selection_reporting.py`, `feature_selector.py`, `io.py`, `hooks.py`
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-22-catboost-overfitting-detection-and-legacy-structure-removal.md)

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

#### 2025-12-22 (Boruta Performance Optimizations)
- **Performance Enhancement**: Implemented quality-preserving optimizations for Boruta feature selection to address performance bottlenecks
- **Optimizations**: Time budget enforcement (10 min default), conditional execution (skip for >200 features or >20k samples), adaptive max_iter based on dataset size, subsampling for large datasets, caching integration
- **Impact**: Reduces Boruta feature selection time from hours to minutes while maintaining model quality
- **SST Compliance**: All parameters loaded from config, no hardcoded defaults
- **Files Changed**: `multi_model_feature_selection.py`, `model_evaluation.py`, `multi_model.yaml`
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-22-boruta-performance-optimizations.md)

#### 2025-12-22 (CatBoost Formatting TypeError Fix)
- **Bug Fix**: Fixed `TypeError: unsupported format string passed to NoneType.__format__` when `cv_mean` or `val_score` is `None` in CatBoost overfitting check logging
- **Solution**: Pre-format values before using in f-string to prevent format specifier errors
- **Impact**: Prevents runtime errors in CatBoost logging, training pipeline completes successfully regardless of CV or validation score availability
- **Files Changed**: `multi_model_feature_selection.py`
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-22-catboost-formatting-typeerror-fix.md)

#### 2025-12-22 (Trend Analyzer Operator Precedence Fix)
- **Bug Fix**: Fixed operator precedence bug in trend analyzer path detection that prevented correct identification of runs in comparison groups
- **Solution**: Added explicit parentheses to ensure `d.is_dir()` is evaluated before checking subdirectories
- **Impact**: Enables proper run detection in comparison groups, trend analyzer correctly identifies all runs with `targets/`, `globals/`, or `REPRODUCIBILITY/` subdirectories
- **Files Changed**: `trend_analyzer.py`
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-22-trend-analyzer-operator-precedence-fix.md)

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
