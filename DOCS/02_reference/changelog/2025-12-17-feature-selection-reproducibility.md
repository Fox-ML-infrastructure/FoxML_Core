# Feature Selection Reproducibility Enhancement (2025-12-17)

## Summary

Enhanced FEATURE_SELECTION stage to track hyperparameters, train_seed, and library versions for full reproducibility. FEATURE_SELECTION now has the same reproducibility requirements as TRAINING stage.

## Changes

### Reproducibility Tracking

**Files Modified:**
- `TRAINING/utils/diff_telemetry.py`
- `TRAINING/utils/reproducibility_tracker.py`
- `TRAINING/ranking/feature_selector.py`

**Changes:**

1. **Hyperparameters Tracking**
   - Extract hyperparameters from `model_families_config` (primary model family, usually LightGBM)
   - Store in `metadata.json` under `training.hyperparameters`
   - Include in comparison group via `hyperparameters_signature` hash
   - Pass via `additional_data['training']` in legacy API
   - Pass via `additional_data_override['training']` in new `log_run` API

2. **Train Seed Tracking**
   - Extract `train_seed` from config (`pipeline.determinism.base_seed`)
   - Store in `metadata.json` under `training.train_seed`
   - Include in comparison group for FEATURE_SELECTION and TRAINING stages
   - Pass as both `seed` and `train_seed` in `additional_data`

3. **Library Versions Tracking**
   - Collect via `collect_environment_info()` in `reproducibility_tracker.py`
   - Store in `metadata.json` under `environment.library_versions`
   - Include in comparison group via `library_versions_signature` hash
   - Tracked libraries: pandas, numpy, scikit-learn, lightgbm, xgboost, tensorflow, ngboost

4. **Comparison Group Updates**
   - FEATURE_SELECTION now includes: `hyperparameters_signature`, `train_seed`, `library_versions_signature`
   - Runs with different hyperparameters, train_seed, or library versions are NOT comparable
   - Aligns FEATURE_SELECTION reproducibility requirements with TRAINING stage

### Bug Fixes

1. **Fixed `lib_sig` UnboundLocalError**
   - Initialized `lib_sig = None` at start of `_build_comparison_group_from_context`
   - Prevents error for TARGET_RANKING stage (where library versions are not applicable)

2. **Fixed `fcntl` Import Errors**
   - Added `import fcntl` to `TRAINING/utils/reproducibility_tracker.py`
   - Added `import fcntl` to `TRAINING/utils/diff_telemetry.py`
   - Required for file locking in `_write_atomic_json`

### API Changes

1. **`log_run` API Enhancement**
   - Added optional `additional_data_override` parameter
   - Allows passing hyperparameters and other metadata separately from RunContext
   - Merged into `additional_data` before processing

2. **Legacy API Support**
   - `log_comparison` API already extracts hyperparameters from `additional_data['training']`
   - Both APIs now support hyperparameters tracking

## Impact

**Before:**
- FEATURE_SELECTION runs with different hyperparameters were considered comparable
- Different train_seed values didn't affect comparability
- Library version differences were ignored

**After:**
- FEATURE_SELECTION runs are only comparable if they have:
  - Same hyperparameters (learning_rate, max_depth, n_estimators, etc.)
  - Same train_seed
  - Same library versions
- Full reproducibility tracking for FEATURE_SELECTION stage
- Consistent reproducibility requirements across FEATURE_SELECTION and TRAINING stages

## Documentation Updates

- Updated `DOCS/03_technical/telemetry/COMPARABILITY_REQUIREMENTS.md` to reflect FEATURE_SELECTION requirements
- Updated `DOCS/03_technical/implementation/REPRODUCIBILITY_TRACKING.md` with new requirements
- Updated `CHANGELOG.md` with detailed changes

## Testing

- Verified hyperparameters extraction from `model_families_config`
- Verified `train_seed` extraction from config
- Verified library versions collection
- Verified comparison group includes all three factors
- Fixed UnboundLocalError and import errors

## Related

- See [Comparability Requirements](../../03_technical/telemetry/COMPARABILITY_REQUIREMENTS.md) for complete comparability rules
- See [Reproducibility Tracking](../../03_technical/implementation/REPRODUCIBILITY_TRACKING.md) for API usage

