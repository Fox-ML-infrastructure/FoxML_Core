# Changelog

All notable changes to FoxML Core will be documented in this file.

> **Note**: This project is under active development. See [NOTICE.md](NOTICE.md) for more information.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

**For detailed technical changes:**
- [Changelog Index](DOCS/02_reference/changelog/README.md) – Per-day detailed changelogs with file paths, config keys, and implementation notes.

---

## [Unreleased]

### Recent Highlights (Last 7 Days)

#### 2026-01-08
**Comprehensive Config Dump** - Added automatic copying of all CONFIG files to `globals/configs/` when runs are created, preserving directory structure. Enables easy run recreation without needing original CONFIG folder access.

**Documentation Enhancements** - Updated README and tutorials to reflect 3-stage pipeline capabilities, dual-view support, and hardware requirements. Added detailed CPU/GPU recommendations for optimal performance and stability.

**File Locking for JSON Writes** - Added file locking around all JSON writes to prevent race conditions. Applied across all pipeline stages.

**Metrics SHA256 Structure Fix** - Fixed metrics digest computation by ensuring metrics are nested under `'metrics'` key in `run_data`.

**Task-Aware Routing Fix** - Fixed routing to use unified `skill01` score normalizing both regression IC and classification AUC-excess to [0,1] range.

**Dual Ranking and Filtering Mismatch Fix** - Fixed filtering mismatch between TARGET_RANKING and FEATURE_SELECTION, added dual ranking with mismatch telemetry.

**Cross-Stage Consistency Fixes** - Fixed Path imports, type casting, and config loading across all stages.

**Manifest and Determinism Fixes** - Fixed manifest schema consistency and deterministic fingerprint computation.

**Config Cleanup** - Removed all symlinks from CONFIG directory, code now uses canonical paths directly.

**Metrics Restructuring** - Grouped metrics structure for cleaner, non-redundant output with task-gating.

**Full Run Hash** - Deterministic run identifier with change detection and aggregation.

**Phase 3.1 Composite Score Fixes** - SE-based stability and skill-gating for family-correct, comparable scoring.

**Snapshot Contract Unification** - P0/P1 correctness fixes for TARGET_RANKING and FEATURE_SELECTION.

> **For detailed technical changes:** See [Changelog Index](DOCS/02_reference/changelog/README.md) for per-day detailed changelogs with file paths, config keys, and implementation notes.

#### 2026-01-07
**Expanded Model Families for TARGET_RANKING/FEATURE_SELECTION** - Full task-type coverage.
- **NEW**: `logistic_regression` family - Standalone classification baseline (binary/multiclass)
- **NEW**: `ftrl_proximal` family - Online learning approximation for binary classification
- **NEW**: `ngboost` family - Probabilistic gradient boosting with uncertainty estimation
- **ENABLED**: `ridge` and `elastic_net` families (were disabled, now enabled for regression)
- All new families:
  - Use `stable_seed_from()` for deterministic seeds (SST pattern)
  - Integrate with existing fingerprinting (prediction, feature, hparams)
  - Automatically populate `fs_snapshot_index.json` via wrapper code
  - Support task-type filtering via `FAMILY_CAPS.supported_tasks`
- **NEW**: Feature selection families added to `FAMILY_CAPS`: `rfe`, `boruta`, `stability_selection`

**Seed Parameter Normalization** - Fixes `unexpected keyword argument 'seed'` errors.
- **FIX**: Convert `seed` → `random_state` for sklearn models in TARGET_RANKING (`model_evaluation.py`)
  - Lasso, Ridge, Elastic Net, Random Forest, Neural Network
- **FIX**: Convert `seed` → `random_state` for sklearn models in leakage detection (`leakage_detection.py`)
  - Lasso, Random Forest, Neural Network
- **FIX**: Add `BASE_SEED` initialization to both files via `init_determinism_from_config()`
- FEATURE_SELECTION already correct (uses `_clean_config_for_estimator` which strips `seed`)

**Task-Type Model Filtering** - Prevents incompatible families from polluting aggregations.
- **NEW**: `supported_tasks` field in `FAMILY_CAPS` for constrained families
  - `elastic_net`, `ridge`, `lasso`: regression only
  - `logistic_regression`: binary, multiclass only
  - `ngboost`: regression, binary only
  - `quantile_lightgbm`: regression only
- **NEW**: `is_family_compatible()` helper in `utils.py` (SST single source of truth)
- **FIX**: Filter applied in all 3 stages before training:
  - Stage 1 (TARGET_RANKING): `model_evaluation.py`
  - Stage 2 (FEATURE_SELECTION): `multi_model_feature_selection.py`
  - Stage 3 (TRAINING): `training.py`
- Tree families (lightgbm, xgboost, catboost) have no restriction - all tasks allowed

**Task-Aware Metrics Schema** - No more `pos_rate: 0.0` on regression targets.
- **NEW**: `CONFIG/ranking/metrics_schema.yaml` with task-specific metric definitions
- **NEW**: `compute_target_stats()` in `metrics_schema.py` (cached schema loader)
- **FIX**: Regression targets emit `y_mean`, `y_std`, `y_min`, `y_max`, `y_finite_pct`
- **FIX**: Binary classification emits `pos_rate` (with configurable `pos_label`)
- **FIX**: Multiclass emits `class_balance` dict, `n_classes` (no `pos_rate`)
- Replaced 2 unconditional `pos_rate` writes in `model_evaluation.py`

**Canonical Metric Naming** - Unambiguous metric names across all stages.
- **NEW**: Naming scheme `<metric_base>__<view>__<aggregation>` (e.g., `spearman_ic__cs__mean`)
- **NEW**: `canonical_names` section in `metrics_schema.yaml` with task+view mappings
- **NEW**: `get_canonical_metric_name(task_type, view)` helper in `metrics_schema.py`
- **NEW**: `get_canonical_metric_names_for_output()` for snapshot metrics population
- **FIX**: `TargetPredictabilityScore` now includes `view` field and `primary_metric_name` property
- **FIX**: All stages emit canonical names alongside deprecated `auc` field for backward compat
  - Regression: `spearman_ic__cs__mean`, `r2__sym__mean`
  - Binary: `roc_auc__cs__mean`, `roc_auc__sym__mean`
  - Multiclass: `accuracy__cs__mean`, `accuracy__sym__mean`
- **DEPRECATED**: `auc` field preserved for backward compatibility (will be removed in v2.0)

**Classification Target Metrics Serialization Fix** - Fixes empty `outputs` for classification targets.
- **FIX**: `class_balance` dict keys now use strings instead of integers
  - PyArrow/Parquet doesn't support integer dict keys, causing silent serialization failures
  - Affected: `compute_target_stats()` in `metrics_schema.py` for binary/multiclass classification
- **FIX**: `_write_metrics()` now writes JSON first, then Parquet
  - JSON is more resilient; ensures metrics.json exists even if Parquet fails
- **NEW**: `_prepare_for_parquet()` helper recursively stringifies nested dict keys
- **FIX**: Shadowed `view` variable bug in `reproducibility_tracker.py` `_save_to_cohort()`
  - Was setting `view = None` then checking `if view:` (always False)
  - Now uses `metrics_view` to avoid shadowing the function parameter

**FEATURE_SELECTION Stability Analysis and Diff Telemetry Fixes**
- **FIX**: `io.py` now skips `manifest.json` when loading snapshots from `replicate/` directories
  - `manifest.json` has different schema (no top-level `run_id`), causing KeyError during stability analysis
  - Added to skip list alongside `fs_snapshot.json` in both `load_snapshots` functions
- **FIX**: `feature_selector.py` now populates `library_versions` in `additional_data`
  - Required for diff telemetry `ComparisonGroup` validation (FEATURE_SELECTION stage)
  - Collects Python version, lightgbm, sklearn, numpy, pandas versions
  - Fixes `ComparisonGroup missing required fields: ['hyperparameters_signature', 'library_versions_signature']` warning
- **FIX**: `get_snapshot_base_dir()` now accepts `ensure_exists` parameter (default True)
  - When False, returns path without creating directories (for read operations)
  - Prevents empty `reproducibility/CROSS_SECTIONAL/feature_importance_snapshots/` directories
  - `metrics_aggregator.py` now passes `ensure_exists=False` when searching for snapshots

**Sample Limit Consistency Across Stages** - Consistent data for TR/FS/TRAINING.
- **FIX**: `cross_sectional_feature_ranker.py` now respects `max_rows_per_symbol`
  - Was loading ALL data (188k samples) instead of config limit (2k per symbol)
- **FIX**: `compute_cross_sectional_importance()` accepts `max_rows_per_symbol` parameter
- **FIX**: `feature_selector.py` passes `max_samples_per_symbol` to CS ranker
- All stages now use consistent `.tail(N)` sampling for reproducibility

**TRAINING Stage Full Parity Tracking** - Complete audit trail for Stage 3.
- **NEW**: `TrainingSnapshot` schema in `TRAINING/training_strategies/reproducibility/schema.py`
  - Model artifact hash (`model_artifact_sha256`) for tamper detection
  - Prediction fingerprint (`predictions_sha256`) for determinism verification
  - Full comparison_group parity with TR/FS stages
- **NEW**: `training_snapshot_index.json` global index for all training runs
- **NEW**: `create_and_save_training_snapshot()` SST-compliant entry point
- **FIX**: Training snapshots created for both CROSS_SECTIONAL and SYMBOL_SPECIFIC models
- End-to-end chain: TR snapshot → FS snapshot → Training snapshot

**FS Snapshot Full Parity with TARGET_RANKING** - Complete audit trail for FEATURE_SELECTION stage.
- **FIX**: Seed derivation now uses `base_seed` (42) directly instead of deriving from `universe_sig`
  - Ensures TR/FS/TRAINING stages have consistent seeds for determinism verification
- **NEW**: `FeatureSelectionSnapshot` now includes full parity fields:
  - `snapshot_seq`: Sequence number for this run
  - `metrics_sha256`: Hash of outputs.metrics for drift detection
  - `artifacts_manifest_sha256`: Hash of output artifacts for tampering detection
  - `fingerprint_sources`: Documentation of what each fingerprint means
  - Full `comparison_group` with `n_effective`, `hyperparameters_signature`, `feature_registry_hash`, `comparable_key`
- **NEW**: Hooks (`save_snapshot_hook`, `save_snapshot_from_series_hook`) accept full parity fields
- **NEW**: `create_fs_snapshot_from_importance` accepts and passes through all parity fields

**OutputLayout & Path Functions Stage Support** - Complete stage-scoped path coverage.
- **NEW**: `OutputLayout` now accepts `stage` parameter and includes `stage=` in `repro_dir()` paths
- **NEW**: `target_repro_dir()` and `target_repro_file_path()` accept `stage` parameter
- **FIX**: All 12 `OutputLayout` callers now pass explicit stage (TARGET_RANKING/FEATURE_SELECTION)
- **FIX**: Dominance quarantine paths use stage-aware paths
- **FIX**: `artifacts_manifest_sha256` now computes correctly (artifacts in expected stage-scoped paths)
- **FIX**: `analyze_all_stability_hook` now uses `iter_stage_dirs()` for proper stage-aware scanning
- **FIX**: Stability metrics now keyed by stage (`TARGET_RANKING/target/method` vs `FEATURE_SELECTION/target/method`)
- **FIX**: `save_snapshot_hook` now passes `stage` to `get_snapshot_base_dir()` (was ignored)
- **FIX**: `feature_selector.py` callers now pass `stage="FEATURE_SELECTION"` explicitly

#### 2026-01-06 (Updated)
**SST Stage Factory & Identity Passthrough** - Stage-aware reproducibility tracking.
- **NEW**: SST stage factory in `run_context.py`: `save_stage_transition()`, `get_current_stage()`, `resolve_stage()`
- **NEW**: Stage-aware reproducibility paths: `stage=TARGET_RANKING/`, `stage=FEATURE_SELECTION/`
- **NEW**: Path scanning helpers for dual-structure support: `iter_stage_dirs()`, `find_cohort_dirs()`, `parse_reproducibility_path()`
- **FIX**: Identity passthrough to `log_run()` in `reproducibility_tracker.py`
- **FIX**: FEATURE_SELECTION identity finalization now logs at WARNING level (was silent DEBUG)
- **FIX**: Partial identity signatures used as fallback when finalization fails
- **FIX**: `fs_snapshot_index.json` fingerprints now populated from FEATURE_SELECTION stage data
- **FIX**: `cross_sectional_panel` snapshots now use partial fallback (was silently failing)
- **FIX**: `multi_model_feature_selection.py` per-family snapshots now use partial fallback
- [Full details →](DOCS/02_reference/changelog/2026-01-06-sst-stage-factory-identity-passthrough.md)

**Comprehensive Determinism Tracking** - Complete end-to-end tracking chain.
- All 8 model families get snapshots (was only XGBoost)
- Training stage now tracks prediction fingerprints
- Feature selection tracks input vs output signatures (`feature_signature_input` / `feature_signature_output`)
- Stage dependencies explicit in snapshots (`selected_targets`, `selected_features`)
- Seeds derived from identity for true determinism
- **FIX**: `allow_legacy=True` now respected for partial RunIdentity (was being ignored)
- **FIX**: Defensive model_metrics handling to ensure fingerprints reach aggregation
- **FIX**: Per-model RunIdentity in TARGET_RANKING prevents replicate folder overwrites
- **FIX**: `predictions_sha256` now populated via `log_run` API path (was only in fallback path)
- [Full details →](DOCS/02_reference/changelog/2026-01-06-determinism-tracking-comprehensive.md)

**View-Scoped Artifact Paths** - Proper separation by view/symbol.
- Artifacts scoped: `targets/<target>/reproducibility/<VIEW>/[symbol=<symbol>/]<artifact_type>/`
- CROSS_SECTIONAL vs SYMBOL_SPECIFIC no longer collide
- Backwards compatible with unscoped paths

**Snapshot Output Fixes** - Critical stage case mismatch resolved.
- Fixed FEATURE_SELECTION snapshots not being written (case mismatch)
- Human-readable manifests for hash-based directories
- Per-model prediction hashes in TARGET_RANKING

#### 2026-01-05
**Determinism and Seed Fixes** - Feature ordering and seed injection.
- Fixed non-deterministic feature ordering (`list(set(...))` → `sorted(set(...))`)
- Automatic seed injection to all model configs
- `feature_signature` added to TARGET_RANKING required fields
- [Full details →](DOCS/02_reference/changelog/2026-01-05-determinism-and-seed-fixes.md)

#### 2026-01-04
**Reproducibility File Output Fixes** - All files now written correctly.
- Fixed `snapshot.json`, `baseline.json`, diff files not being written
- Path reconstruction for target-first structure
- [Full details →](DOCS/02_reference/changelog/2026-01-04-reproducibility-file-output-fixes.md)

**GPU/CPU Determinism Config Fix** - Config settings now respected.
- Replaced hardcoded `set_global_determinism()` with config-aware `init_determinism_from_config()`
- GPU detection respects strict mode
- [Full details →](DOCS/02_reference/changelog/2026-01-04-gpu-cpu-determinism-config-fix.md)

#### 2026-01-03
**Determinism SST** - Production-grade reproducibility.
- `RunIdentity` SST with two-phase construction
- Strict/replicate key separation
- `bin/run_deterministic.sh` launcher
- [Full details →](DOCS/02_reference/changelog/2026-01-03-deterministic-run-identity.md)

---

### Older Updates

See the [Changelog Index](DOCS/02_reference/changelog/README.md) for detailed changelogs organized by date:

- **2026-01-02**: Horizon-aware routing, telemetry comparison fixes
- **2025-12-30**: Prediction hashing for determinism verification
- **2025-12-23**: Dominance quarantine, leakage safety, model timing
- **2025-12-22**: CatBoost/Boruta optimizations, performance audit
- **2025-12-21**: CatBoost fixes, feature selection routing
- **2025-12-20**: Threading utilities, target-first structure
- **2025-12-19**: Target-first migration, config fixes
- **2025-12-18**: TRAINING folder reorganization
- **2025-12-17**: Training pipeline audit, licensing
- **2025-12-16**: Diff telemetry integration
- **2025-12-15**: CatBoost GPU fixes, metrics rename
- **2025-12-14**: Drift tracking, lookahead bias fixes
- **2025-12-13**: SST enforcement, fingerprint tracking
- **2025-12-10–12**: Initial infrastructure setup

---

## Version History

### v0.1.0 (In Development)
- Initial release of FoxML Core
- Multi-model feature selection pipeline
- Target ranking with predictability scoring
- Comprehensive reproducibility tracking system
- Deterministic training with strict mode support
