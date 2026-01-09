# Changelog

All notable changes to FoxML Core will be documented in this file.

## 2026-01-09

### Comprehensive SST Path Construction Fixes - All Stages
- **Fixed Path Construction with Enum Values**: Resolved enum-to-string conversion issues in path construction across all three stages
  - `target_first_paths.py`: Fixed `get_target_reproducibility_dir()`, `find_cohort_dir_by_id()`, and `target_repro_dir()` to normalize Stage/View enums to strings before path construction
  - `output_layout.py`: Fixed `__init__()` and `repro_dir()` to normalize View and Stage enums to strings
  - `training_strategies/reproducibility/io.py`: Fixed `get_training_snapshot_dir()` to normalize View and Stage enums to strings
  - All path construction functions now explicitly convert enum values to strings using `.value` property before building paths
  - Fixes missing metric artifacts and JSON files across TARGET_RANKING, FEATURE_SELECTION, and TRAINING stages
  - Maintains backward compatibility with string inputs
  - Verified all three stages work correctly with enum inputs

### Metric Output JSON Serialization Fixes
- **Fixed Broken Metric Outputs**: Resolved JSON serialization issues where Stage/View enum objects were written directly to JSON
  - `metrics.py`: Updated `write_cohort_metrics()`, `_write_metrics()`, `_write_drift()`, `generate_view_rollup()`, and `generate_stage_rollup()` to normalize enum inputs to strings before JSON serialization
  - `reproducibility_tracker.py`: Updated `generate_metrics_rollups()` and fixed `ctx.stage`/`ctx.view` normalization before passing to `write_cohort_metrics()`
  - All metric output functions now accept `Union[str, Stage]` and `Union[str, View]` for SST compatibility
  - All enum values are explicitly converted to strings using `.value` property before being added to JSON dictionaries
  - Fixes broken metric outputs in CROSS_SECTIONAL view for target ranking and all other stages
  - Maintains backward compatibility with string inputs
  - Verified all JSON outputs contain string values (not enum objects)

### SST Implementation Progress Summary - Complete
- **Phase 1 Complete**: View and Stage enum migrations (29 files total)
- **Phase 2 Complete**: WriteScope function migration (4 functions updated)
- **Phase 3 Complete**: Helper function audits and standardization
  - Phase 3.1: Scope resolution migration - Complete
  - Phase 3.2: RunIdentity factory audit - Complete
  - Phase 3.3: Cohort ID unification - Complete
  - Phase 3.4: Config hash audit - Complete
  - Phase 3.5: Universe signature audit - Complete
- **All SST candidates implemented**: Complete migration to SST-centric architecture with consistent enums, WriteScope objects, and unified helpers

## 2026-01-09

### Syntax and Import Fixes
- **Fixed Syntax Errors**: Resolved all syntax and indentation errors in TRAINING pipeline
  - `model_evaluation.py`: Fixed indentation error in try block (line 6011-6026)
  - `cross_sectional_feature_ranker.py`: Fixed indentation error after try statement (line 703)
  - `multi_model_feature_selection.py`: Fixed orphaned else block and try block indentation (lines 4325-4327, 5453)
  - `intelligent_trainer.py`: Fixed `UnboundLocalError` for `Path` - removed redundant local import that shadowed global import (line 3970)
  - All Python files now compile without syntax errors
  - All critical modules import successfully

### SST Import Shadowing Fixes
- **Fixed UnboundLocalError Issues**: Resolved all import shadowing issues from SST refactoring
  - `model_evaluation.py:8129`: Removed `Stage` from local import (already imported globally at line 41) - fixes `UnboundLocalError: local variable 'Stage' referenced before assignment` at line 5390
  - `shared_ranking_harness.py:286`: Added global `Stage` import and removed redundant local import - prevents `UnboundLocalError` in `create_run_context` method
  - Removed redundant local `Path` imports in `diff_telemetry.py` (3 instances), `target_routing.py`, and `training.py` (5 instances) where `Path` is already imported globally
  - Verified all path construction functions correctly convert enum values to strings using `str(enum)` which returns `.value`
  - Verified all JSON serialization correctly handles enum values (enums inherit from `str` so serialize correctly, and `_sanitize_for_json` explicitly converts enums to `.value` for safety)
  - All critical modules now import without `UnboundLocalError` issues
  - All path construction and file writes verified to work correctly with enum values

### Additional SST Improvements
- **String Literal to Enum Migration**: Replaced hardcoded string comparisons with enum comparisons
  - `metrics.py`: 7+ instances - replaced `view == "SYMBOL_SPECIFIC"` with `view_enum == View.SYMBOL_SPECIFIC`
  - `trend_analyzer.py`: 4+ instances - replaced `stage == "TARGET_RANKING"` with `stage_enum == Stage.TARGET_RANKING`
  - `target_routing.py`: 3+ instances - replaced string lists and comparisons with View enum values
  - `cache_manager.py`, `artifact_mirror.py`, `leakage_detection/reporting.py`, `model_evaluation/reporting.py`, `reproducibility/io.py`, `manifest.py`, `hooks.py`, `metrics_schema.py`: All string comparisons migrated to enum comparisons
  - All enum comparisons use `View.from_string()` or `Stage.from_string()` for normalization, ensuring backward compatibility
  - All path construction uses `view_enum.value` to ensure string output for filesystem paths
  - JSON output format unchanged - enum values serialize as strings via `.value` property

- **Config Hashing Standardization**: Replaced manual hashlib calls with canonical_json/sha256 helpers
  - `fingerprinting.py`: Universe signature computation now uses `canonical_json()` + `sha256_short()`
  - `cohort_metadata_extractor.py`: Data fingerprinting now uses `sha256_short()` helper
  - `reproducibility/utils.py`: Comparison key hashing now uses `sha256_short()` helper
  - `diff_telemetry/types.py`: Hash computation now uses `canonical_json()` + `sha256_short()` helpers
  - All changes maintain same hash output format (backward compatible)
  - Binary file hashing (lock files, binary data) kept as-is (appropriate use of hashlib)

- **Verification**: All changes verified to maintain JSON output format and metric tracking
  - Enum comparisons work correctly with both string and enum inputs
  - Enum values serialize as strings in JSON (via `.value` property)
  - Path construction produces identical paths (enum `.value` matches original strings)
  - All test suites pass: imports, enum access, path construction, JSON serialization

### SST Implementation Complete - All Phases Verified
- **Comprehensive Verification**: All SST migration phases completed and verified
  - **Enum Migration**: 29 files migrated to use View and Stage enums (only 2 stage strings remain in comments, 23 view strings in appropriate contexts)
  - **WriteScope Migration**: 4 functions now accept WriteScope objects with backward compatibility
  - **Helper Unification**: All scope resolution, cohort ID, config hashing, and universe signature computations use unified SST helpers
  - **Backward Compatibility**: All changes maintain full backward compatibility with existing JSON files, snapshots, and metrics
  - **No Breaking Changes**: All file paths, JSON serialization, and existing data formats remain unchanged

### SST Config Hash Audit (Phase 3.4) - Complete
- **Config Hash Standardization**: Updated manual config hash computations to use shared helpers from `config_hashing.py`
  - `reproducibility_tracker.py`: Replaced manual `json.dumps()` + `hashlib.sha256()` with `canonical_json()` + `sha256_short()`
  - `diff_telemetry.py`: Replaced manual `hashlib.sha256()` calls with `sha256_short()` helper
  - All config hashing now uses consistent logic: `canonical_json()` for normalization, `sha256_full()` or `sha256_short()` for hashing
  - Hash lengths standardized: 8 chars for short hashes, 16 chars for medium, 64 chars for full identity keys

### SST Cohort ID Unification (Phase 3.3) - Complete
- **Cohort ID Generation Unification**: Created unified `compute_cohort_id()` helper in `cohort_id.py`
  - Extracted duplicate logic from `ReproducibilityTracker._compute_cohort_id()` and `compute_cohort_id_from_metadata()`
  - Both implementations now delegate to unified helper (SST-compliant)
  - Uses View enum for consistent view handling
  - Uses `extract_universe_sig()` helper for SST-compliant universe signature access
  - All cohort ID generation now uses single source of truth

### SST RunIdentity Factory Audit (Phase 3.2) - Complete
- **RunIdentity Construction Audit**: Verified all RunIdentity constructions follow SST patterns
  - Factory `create_stage_identity()` is used for creating new identities from scratch (9 instances)
  - Manual constructions are for legitimate use cases: updating existing identities, finalizing partial identities, or copying with modifications
  - All new identity creation uses factory pattern (SST-compliant)
  - Remaining manual constructions are for identity updates/copies (correct pattern)

### SST Universe Signature Audit (Phase 3.5) - Complete
- **Universe Signature Consistency**: Verified all universe signature computations use `compute_universe_signature()` helper
  - All 22 instances across 10 files verified to use `compute_universe_signature()` from `run_context.py`
  - Fallback manual computation in `fingerprinting.py` is defensive and acceptable (only used if helper unavailable)
  - One instance in `model_evaluation.py` is for `symbols_digest` (metadata), not universe signature - correctly different
  - All universe signature computations now consistent and SST-compliant

### SST Scope Resolution Migration (Phase 3.1) - Complete
- **Manual Scope Resolution Replacement**: Replaced manual `resolved_data_config.get('view')` and `resolved_data_config.get('universe_sig')` patterns with `resolve_write_scope()` helper
  - `feature_selector.py`: Consolidated manual universe_sig and view extraction into single `resolve_write_scope()` call
  - `model_evaluation.py`: Replaced manual view/universe_sig extraction with `resolve_write_scope()` for canonical scope resolution
  - All scope resolution now uses SST helper, ensuring consistent scope handling across codebase
  - Remaining `.get()` calls are for telemetry/metadata purposes (not scope resolution)

### SST WriteScope Migration (Phase 2.2) - Complete
- **WriteScope Function Migration**: Migrating functions to accept WriteScope objects for SST consistency
  - `get_scoped_artifact_dir()` and `ensure_scoped_artifact_dir()` in `target_first_paths.py` now accept `scope: WriteScope` parameter
  - `model_output_dir()` in `target_first_paths.py` now accepts `scope: WriteScope` parameter
  - `build_cohort_metadata()` in `cohort_metadata.py` now accepts `scope: WriteScope` parameter
  - All functions maintain backward compatibility with loose (view, symbol, universe_sig, stage) parameters
  - When `scope` is provided, extracts view, symbol, universe_sig, and stage from WriteScope
  - All call sites remain compatible (using deprecated parameters for now)

### SST WriteScope Migration (Phase 2.1) - Identification
- **WriteScope Adoption Planning**: Identified functions accepting loose (view, symbol, universe_sig) tuples for WriteScope migration
  - Functions identified for migration:
    - `get_scoped_artifact_dir()` and `ensure_scoped_artifact_dir()` in `target_first_paths.py` - accept view, symbol, universe_sig, stage
    - `model_output_dir()` in `target_first_paths.py` - accepts view, symbol, universe_sig
    - `build_cohort_metadata()` in `cohort_metadata.py` - accepts view, universe_sig, symbol
  - Migration strategy: Start with internal functions (lowest call sites), work outward to public APIs
  - Maintain backward compatibility with wrapper functions where needed

### SST Stage Enum Migration (Phase 1.2) - Complete
- **Stage Enum Adoption**: Migrated 17 files to use `Stage` enum from `scope_resolution.py` instead of hardcoded string literals
  - Function signatures updated to accept `Union[str, Stage]` for backward compatibility
  - All comparisons use enum instances: `stage_enum == Stage.TARGET_RANKING` instead of `stage == "TARGET_RANKING"`
  - All JSON serialization uses `str(stage_enum)` which returns `.value` via `__str__`
  - All path construction uses `str(stage_enum)` for consistent string conversion
- **Backward Compatibility Guaranteed**:
  - `Stage.from_string()` normalizes strings from JSON/metadata to enum instances (handles "MODEL_TRAINING" → "TRAINING" alias)
  - Existing JSON files (metadata.json, snapshot.json, metrics.json) continue to work unchanged
  - File paths remain identical (enum `__str__` returns `.value`)
- **Files Updated**:
  - `reproducibility_tracker.py` - Complete Stage enum migration with proper normalization and comparisons
  - `target_first_paths.py` - Already accepts `Union[str, Stage]` for stage parameters
  - `predictability/main.py`, `model_evaluation/reporting.py`, `leakage_detection.py` - TARGET_RANKING stage enum
  - `multi_model_feature_selection.py`, `feature_selector.py` - FEATURE_SELECTION stage enum
  - `target_ranker.py`, `dominance_quarantine.py` - TARGET_RANKING stage enum
  - `training.py`, `reproducibility/io.py`, `reproducibility/schema.py` - TRAINING stage enum
  - `intelligent_trainer.py` - TARGET_RANKING and FEATURE_SELECTION stage enum
- **No Breaking Changes**: All existing snapshots, metrics, and JSON files remain fully compatible

### SST View Enum Migration (Phase 1.1)
- **View Enum Adoption**: Migrated 17 files to use `View` enum from `scope_resolution.py` instead of hardcoded string literals
  - All function signatures now accept `Union[str, View]` for backward compatibility
  - All comparisons use enum instances: `view_enum == View.CROSS_SECTIONAL` instead of `view == "CROSS_SECTIONAL"`
  - All JSON serialization uses `view_enum.value` to ensure string output
  - All path construction uses `str(view_enum)` which returns `.value` via `__str__`
- **Backward Compatibility Guaranteed**:
  - `_sanitize_for_json()` explicitly converts Enum types to `.value` for JSON serialization
  - `View.from_string()` normalizes strings from JSON/metadata to enum instances
  - Existing JSON files (metadata.json, snapshot.json, metrics.json) continue to work unchanged
  - File paths remain identical (enum `__str__` returns `.value`)
- **Files Updated**:
  - `feature_selection_reporting.py`, `multi_model_feature_selection.py`, `metrics_aggregator.py`
  - `training.py`, `artifact_paths.py`, `cohort_metadata.py`, `cross_sectional_data.py`
  - `shared_ranking_harness.py`, `output_layout.py`, `diff_telemetry.py`, `target_first_paths.py`
  - `intelligent_trainer.py`, `target_ranker.py`, `feature_selector.py`, `cross_sectional_feature_ranker.py`
  - `model_evaluation.py`, `reproducibility_tracker.py`
- **No Breaking Changes**: All existing snapshots, metrics, and JSON files remain fully compatible

### SST Compliance Fixes
- **Target Name Normalization**: Added `normalize_target_name()` helper function and replaced **ALL** remaining instances (39+ total) of manual target normalization across **ALL** files - ensures consistent filesystem-safe target names across all path construction
- **Path Resolution Consistency**: Replaced **ALL** remaining custom path resolution loops (30+ total) with `run_root()` helper across **ALL** files for consistent run root directory resolution
- **Cross-Sectional Stability SST Parameters**: Added `view` and `symbol` parameters to `compute_cross_sectional_stability()` to use SST-resolved values instead of hardcoded defaults - ensures consistency with main feature selection stage
- **Universe Signature Fix**: Fixed hardcoded `universe_sig="ALL"` in `metrics_aggregator.py` - now extracts from cohort metadata with proper fallback chain
- **Internal Document Cleanup**: Removed all references to internal documentation from public-facing changelogs
- **SST Audit**: Verified RunIdentity construction patterns and TRAINING stage SST usage - all verified as correct
- **Determinism Verification**: All helper replacements verified to produce identical output as manual code, ensuring no non-determinism introduced
- **Complete Migration**: **ALL** remaining SST helper opportunities have been migrated - the codebase now uses SST helpers consistently throughout
- See [detailed changelog](DOCS/02_reference/changelog/2026-01-09-sst-consistency-fixes.md) for full details

## 2026-01-08

### FEATURE_SELECTION Cohort Consolidation
- **Consolidated Duplicate Cohort Directories**: Fixed duplicate cohort directories in FEATURE_SELECTION stage - cross-sectional panel now writes `metrics_cs_panel.json` to the same cohort directory as main feature selection (`metrics.json`), eliminating duplicate directories and making output structure cleaner
- **Cohort ID Passing**: Main feature selection now passes `cohort_id` to cross-sectional panel computation to ensure both use the same cohort directory
- **Bug Fixes**: Fixed null pointer errors when `cohort_dir` or `audit_result` are `None` - added proper null checks before accessing attributes
- **Backward Compatibility**: If `cohort_id` is not provided, CS panel falls back to creating its own cohort (legacy behavior)
- See [detailed changelog](DOCS/02_reference/changelog/2026-01-08-feature-selection-reproducibility-fixes.md) for full details

### FEATURE_SELECTION Reproducibility Fixes
- **CatBoost Missing from Results:** Fixed CatBoost disappearing from results - removed `importance.sum() > 0` filter that excluded failed models, handle empty dicts by creating zero importance Series, ensure failed models appear in aggregation with zero consensus score
- **Training Snapshot Validation:** Added validation that snapshot files actually exist after creation, improved error logging with full traceback at warning level
- **Duplicate Cohort Directories:** Fixed inconsistent `cs_config` structure causing different `config_hash` values - normalize to always include all keys (even if None) for consistent hashing
- **Missing universe_sig:** Fixed duplicate assignment overwriting `universe_sig` in metadata
- **Missing snapshot/diff files:** Added validation after `finalize_run()` to verify required files are created
- **Duplicate universe scopes:** Removed hardcoded `universe_sig="ALL"` default, use SST universe signature consistently, added fallback to extract from `run_identity.dataset_signature`
- **Missing per-model snapshots:** Improved error logging from debug to warning level for per-model snapshot failures
- **Missing deterministic_config_fingerprint:** Fixed path resolution to walk up directory tree to find run root
- **Documentation:** Created comprehensive guide explaining which snapshots exist (`multi_model_aggregated` = source of truth, `cross_sectional_panel` = optional stability analysis) and which one to use
- See [detailed changelog](DOCS/02_reference/changelog/2026-01-08-feature-selection-reproducibility-fixes.md) for full details

### File Overwrite and Plan Creation Fixes
- **run_context.json:** Fixed stage history loss - now preserves `current_stage` and `stage_history` when `save_run_context()` is called after `save_stage_transition()`
- **run_hash.json:** Fixed creation issues - improved error logging, fixed previous run lookup to search parent directories, added validation for missing snapshot indices
- **Routing/Training Plans:** Fixed plan creation - improved error logging (visible warnings instead of debug), added plan save verification, fixed manifest update to occur after plans are created
- See [detailed changelog](DOCS/02_reference/changelog/2026-01-08-file-overwrite-and-plan-creation-fixes.md) for full details

### Commercial License Clarity and Support Documentation
- **README:** Clarified commercial license requirements - now explicitly states "required for proprietary/closed deployments or to avoid AGPL obligations (especially SaaS/network use)"
- **SUPPORT.md:** Added root-level support documentation for easier discovery
- See [detailed changelog](DOCS/02_reference/changelog/2026-01-08-commercial-license-clarity-and-support.md) for full details

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
