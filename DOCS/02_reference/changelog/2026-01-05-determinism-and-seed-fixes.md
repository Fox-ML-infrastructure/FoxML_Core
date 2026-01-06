# 2026-01-05: Determinism and Seed Fixes

## Seed Injection and License Cleanup

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

## Feature Loading Determinism Fix

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
