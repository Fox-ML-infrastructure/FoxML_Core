# 2026-01-06: Comprehensive Determinism Tracking

## Comprehensive Determinism Tracking

- **Enhancement**: All 8 model families now get feature importance snapshots (was only XGBoost).
  - `feature_selector.py`: Changed `allow_legacy=(final_identity is None)` to `allow_legacy=True`
- **Enhancement**: Training stage now computes and tracks prediction fingerprints.
  - `training.py`: Added prediction fingerprint computation for both SYMBOL_SPECIFIC and CROSS_SECTIONAL models
  - Uses `compute_prediction_fingerprint_for_model()` with aggregated hash across model components
- **Enhancement**: Feature selection now tracks input vs output feature signatures.
  - `fingerprinting.py`: Added `feature_signature_input` and `feature_signature_output` to `RunIdentity`
  - `schema.py`: Added `feature_fingerprint_input` and `feature_fingerprint_output` to `FeatureSelectionSnapshot`
  - Enables diffing "what went in" vs "what came out" of feature selection
- **Enhancement**: Stage dependencies now explicit in snapshots.
  - `feature_selector.py`: Added `selected_targets` to FS snapshot inputs
  - `training.py`: Added `selected_features` to Training snapshot additional_data
- **Enhancement**: Seeds now derived from identity for true determinism.
  - `fingerprinting.py`: `create_stage_identity()` derives `train_seed` from `base_seed + universe_sig`
  - Same universe + same config = same seed every time
- **Bug Fix**: Fixed `UnboundLocalError: hashlib` in `create_stage_identity()`.
  - Removed shadowing local import that caused error when main import succeeded
- **Impact**: Complete determinism tracking chain: TR → FS → Training with prediction fingerprints at all stages.
- **Files Changed**: `feature_selector.py`, `training.py`, `fingerprinting.py`, `schema.py`

## View-Scoped Artifact Paths

- **Enhancement**: Artifacts now scoped by view/symbol for proper separation.
  - `target_first_paths.py`: Added `get_scoped_artifact_dir()` and `ensure_scoped_artifact_dir()` helpers
  - Path pattern: `targets/<target>/reproducibility/<VIEW>/[symbol=<symbol>/]<artifact_type>/`
- **Updated Callers**:
  - `model_evaluation.py`: Feature exclusions and featureset artifacts now use view-scoped paths
  - `io.py`: `get_snapshot_base_dir()` now accepts `view` and `symbol` parameters
  - `hooks.py`: Updated `save_snapshot_hook()` to pass view/symbol to `get_snapshot_base_dir()`
  - `cross_sectional_feature_ranker.py`: Updated snapshot base dir calls for CROSS_SECTIONAL view
  - `metrics_aggregator.py`: Updated stability analysis to search both scoped and unscoped paths
- **Backwards Compatibility**: Reader functions search both new scoped paths and old unscoped paths.
- **Impact**: Artifacts for different views (CROSS_SECTIONAL vs SYMBOL_SPECIFIC) no longer collide.

## Snapshot Output Fixes and Human-Readable Manifests

- **Critical Fix**: Fixed stage case mismatch causing FEATURE_SELECTION snapshots to not be written.
  - `model_evaluation.py`: Changed `stage="target_ranking"` to `stage="TARGET_RANKING"` (uppercase)
  - `feature_selector.py`: Changed fallback from `"feature_selection"` to `"FEATURE_SELECTION"` (uppercase)
  - `reproducibility_tracker.py`: Normalized stage comparisons to use `.upper()` (lines 4024, 4026, 4109, 4142)
- **Enhancement**: Disabled fs_snapshot.json during TARGET_RANKING to reduce confusion.
  - `reporting.py`: Added `write_fs_snapshot=False` to `save_snapshot_hook()` call
  - TARGET_RANKING now only writes `snapshot.json` (complete) + hash-based snapshots
  - FEATURE_SELECTION continues to write `fs_snapshot.json` (human-readable)
- **Enhancement**: Added per-model prediction hashes to TARGET_RANKING for auditability.
  - `model_evaluation.py`: Added `prediction_hashes` dict to `additional_data` with per-model hashes
- **Enhancement**: Added human-readable manifests to feature_importance_snapshots.
  - `io.py`: Added `_update_directory_manifest()` for per-directory manifest.json
  - `io.py`: Added `update_global_importance_manifest()` for root manifest.json
  - Manifests map hash-based filenames to method/target/timestamp metadata
- **Impact**: FEATURE_SELECTION snapshots now written correctly. Hash-based directories navigable via manifest.json.
- **Files Changed**: `model_evaluation.py`, `feature_selector.py`, `reproducibility_tracker.py`, `reporting.py`, `io.py`

## Feature Selection Telemetry and Stage Labeling Fixes

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
  - `model_evaluation.py`: Added aggregation to NEW `log_run` API path (was only in fallback `except ImportError` path)
  - `reproducibility_tracker.py`: Added `prediction_fingerprint` parameter to `log_run()` and pass through to `log_comparison()`
- **Bug Fix**: Fixed replicate folder only containing xgboost (all models overwriting same file).
  - Root cause: All models shared same `strict_key` because single identity was created from first model's hparams
  - `model_evaluation.py`: Removed `break` statement, now builds `per_model_identities` dict with each model's unique identity
  - `reporting.py`: Updated `save_feature_importances()` to accept `Dict[str, RunIdentity]` and look up per-model identity
  - Each model now gets unique `strict_key`/`replicate_key` based on its actual `hparams_signature`
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
