# 2025-12-22: Training Results Organization and Pipeline Integrity Fixes

## Summary

Fixed critical issues with training results organization, model family filtering, routing decisions validation, and feature registry filtering. These fixes prevent silent failures, data integrity issues, and ensure the pipeline is honest about what it's doing.

## Changes

### 1. Training Results Folder Structure Simplification

**Problem**: Models were being saved to nested `training_results/training_results/<family>/<target>/` structure, creating duplicate and confusing organization.

**Solution**: Simplified to `training_results/<family>/` structure (no nested training_results, no target subfolders for model files).

**Files Changed**:
- `TRAINING/training_strategies/execution/training.py`:
  - Line 1341: Changed model saving to use `output_dir / family /` directly
  - Removed nested `training_results/training_results/` path creation
  - Updated CS route, SYMBOL_SPECIFIC route, and BOTH route saving
  - Symbol-specific models save to `training_results/<family>/symbol=<symbol>/`

**Impact**: Cleaner folder structure, easier to find models by family type.

---

### 2. Feature Selector Filtering Before Training

**Problem**: Feature selectors (lasso, mutual_information, univariate_selection, random_forest, catboost) were being passed to training execution, causing errors like "Family 'lasso' not found in MODMAP".

**Solution**: Added filtering in `intelligent_trainer.py` to remove feature selectors before passing families to training execution.

**Files Changed**:
- `TRAINING/orchestration/intelligent_trainer.py`:
  - Lines 2112-2120: Added `FEATURE_SELECTORS` constant and filtering logic
  - Line 2195: Added final defensive filter before training
  - Filters out: `random_forest`, `catboost`, `lasso`, `mutual_information`, `univariate_selection`, `elastic_net`, `ridge`, `lasso_cv`

**Impact**: Prevents training errors from attempting to train feature selectors.

---

### 3. Family Name Normalization Fix

**Problem**: `NeuralNetwork` not found in TRAINER_MODULE_MAP (should be `neural_network`). Family names weren't normalized before lookup.

**Solution**: Added normalization in `isolation_runner.py` before TRAINER_MODULE_MAP lookup.

**Files Changed**:
- `TRAINING/common/isolation_runner.py`:
  - Lines 489-503: Added `normalize_family()` call to convert "NeuralNetwork" → "neural_network" before lookup
  - Uses `normalize_family()` from `sst_contract.py` for consistent normalization

**Impact**: Prevents "Family not found" errors from case mismatches.

---

### 4. Reproducibility Tracking Path Fix

**Problem**: `'str' object has no attribute 'name'` errors when `output_dir` was passed as string instead of Path.

**Solution**: Fixed Path/string conversion and ensured `tracker_input_adapter` handles Enum/string conversion.

**Files Changed**:
- `TRAINING/training_strategies/execution/training.py`:
  - Lines 1220-1228: Added Path conversion before operations
  - Lines 1307-1322: Uses `tracker_input_adapter` for Enum/string conversion

**Impact**: Eliminates reproducibility tracking errors.

---

### 5. Training Plan 0 Jobs Handling

**Problem**: Training plan with 0 jobs logged ERROR but silently fell back, making it unclear that plan was disabled.

**Solution**: Downgraded to WARNING with explicit message that plan is disabled.

**Files Changed**:
- `TRAINING/orchestration/training_plan_consumer.py`:
  - Lines 134-143: Changed ERROR to WARNING with clear message about plan being disabled

**Impact**: Makes disabled state explicit, not a silent fallback.

---

### 6. Routing Decisions Fingerprint Validation

**Problem**: Routing decisions from previous runs could be loaded, causing mismatches (e.g., 10 decisions vs 4 filtered targets).

**Solution**: Added fingerprint computation when saving routing decisions and validation when loading.

**Files Changed**:
- `TRAINING/ranking/target_routing.py`:
  - Lines 408-418: Added fingerprint computation (targets + symbols hash) when saving
  - Lines 450-516: Added fingerprint validation when loading (with `expected_targets` parameter)
  - Returns empty dict if fingerprint doesn't match
- `TRAINING/orchestration/intelligent_trainer.py`:
  - Lines 2257-2272: Pass `expected_targets` and `validate_fingerprint=True` to load function
  - Added set equality check to ensure routing decisions match filtered targets

**Impact**: Prevents loading stale routing decisions from previous runs.

---

### 7. Feature Registry Filtering Upstream

**Problem**: Feature selection ranked from ALL features, but training-time registry filtering removed many, causing "feature count collapse" (requested=100 → allowed=8).

**Solution**: Moved registry filtering upstream into feature selection step, using STRICT mode (same as training).

**Files Changed**:
- `TRAINING/ranking/multi_model_feature_selection.py`:
  - Lines 3143-3197: Changed `for_ranking=False` (strict mode) instead of permissive ranking mode
  - Re-validates selected_features from shared harness with strict registry filtering
  - Added registry_stats tracking for metadata
- `TRAINING/ranking/feature_selector.py`:
  - Lines 1520-1530: Added `registry_filtering` metadata to feature selection summary

**Impact**: Prevents selecting features that will be rejected at training time. Makes pipeline honest about registry constraints.

---

### 8. Horizon→Bars Trading Days Fix

**Problem**: Horizon calculation assumed 24/7 trading (5d = 7200 minutes = 1440 bars), but targets use trading days calendar.

**Solution**: Changed day multiplier from 1440 (24 hours) to 390 (6.5 hours = trading session).

**Files Changed**:
- `TRAINING/common/utils/sst_contract.py`:
  - Line 135: Changed `{'regex': r'(\d+)d', 'multiplier': 1440}` to `{'regex': r'(\d+)d', 'multiplier': 390}`
  - 1d = 390 minutes (trading session), 5d = 1950 minutes (5 trading sessions)

**Impact**: Correct horizon→bars conversion for trading days targets.

---

### 9. Config Documentation for Family Split

**Problem**: Config didn't clearly document which families are selectors vs trainers.

**Solution**: Added documentation in config explaining which families are for selection only.

**Files Changed**:
- `CONFIG/ranking/features/multi_model.yaml`:
  - Lines 13-24: Added documentation explaining selector vs trainer families
  - Notes that selectors are automatically filtered at runtime

**Impact**: Clearer config documentation, though runtime filtering already works.

---

## Testing

All fixes have been tested and pass linting. The changes are backward compatible:
- Existing configs continue to work (runtime filtering handles selectors)
- Folder structure changes are additive (old paths still work, new structure is primary)
- Fingerprint validation is opt-in (can disable with `validate_fingerprint=False`)

## Migration Notes

- **Folder Structure**: Old nested `training_results/training_results/` structure is no longer created. Models are in `training_results/<family>/`.
- **Routing Decisions**: Old routing decisions without fingerprints will still load, but new saves include fingerprints for validation.
- **Horizon Calculation**: Day-based horizons now use trading days (390 min/day) instead of calendar days (1440 min/day). This affects `fwd_ret_5d` and similar targets.

## Related Issues

- Feature count collapse (requested=100 → allowed=8) - FIXED
- Training errors from invalid families (lasso, mutual_information) - FIXED
- Nested training_results folder structure - FIXED
- Routing decisions mismatch (10 vs 4) - FIXED
- Training plan 0 jobs silent fallback - FIXED
- Horizon→bars wrong for trading days - FIXED

