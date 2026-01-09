# Cross-Stage Issue Fixes (2026-01-08)

## Summary

Fixed similar issues across FEATURE_SELECTION and TRAINING stages that mirror the issues already fixed in TARGET_RANKING. These fixes ensure consistency, type safety, and proper universe signature extraction across all pipeline stages.

---

## Path Import Cleanup (TRAINING stage)

**Problem**: Redundant `Path` imports inside try blocks in `schema.py` could cause scope issues if imports failed.

**Solution**:
- **UPDATED**: Added `from pathlib import Path` at module level in `schema.py`
- **REMOVED**: Redundant `from pathlib import Path` imports inside try blocks (lines 260, 386, 398)
- **UPDATED**: `save_training_snapshot()` in `io.py` - Added defensive `isinstance(output_dir, Path)` check before path operations

**Files Changed**:
- `TRAINING/training_strategies/reproducibility/schema.py`:
  - Added module-level `Path` import
  - Removed redundant try-block imports
- `TRAINING/training_strategies/reproducibility/io.py`:
  - Added defensive Path type checking in `save_training_snapshot()`

**Result**: Consistent Path import pattern across all stages, preventing potential scope issues.

---

## Type Casting for Config Values

**Problem**: Config values read from YAML may be strings, causing type errors when used in numeric comparisons.

**Solution**:
- **UPDATED**: Added explicit `float()` cast for `collapse_threshold` in `training.py`
- **UPDATED**: Added explicit `int()` casts for numeric config values:
  - `base_seed` values in `feature_selector.py` (3 locations)
  - `max_cs_samples` in `data_preparation.py`
  - `default_lookback` in `main.py` and `execution/main.py`
  - `mkl_threads` and `lookback` in `utils.py`

**Files Changed**:
- `TRAINING/training_strategies/execution/training.py`:
  - `collapse_threshold = float(get_cfg(...))` - Prevents type errors in numeric comparisons
- `TRAINING/ranking/feature_selector.py`:
  - `seed_value = int(get_cfg("pipeline.determinism.base_seed", ...))` (3 locations)
- `TRAINING/training_strategies/execution/data_preparation.py`:
  - `max_cs_samples = int(get_cfg(...))`
- `TRAINING/training_strategies/execution/main.py`:
  - `default_lookback = int(get_cfg(...))`
- `TRAINING/training_strategies/main.py`:
  - `default_lookback = int(get_cfg(...))`
- `TRAINING/training_strategies/utils.py`:
  - `MKL_THREADS_DEFAULT = int(get_cfg(...))`
  - `lookback = int(get_cfg(...))`

**Result**: All numeric config values are explicitly cast, preventing type mismatch errors similar to the `composite_score.py` fix.

---

## Universe Signature Extraction (FEATURE_SELECTION stage)

**Problem**: Fallback universe signature computation could use batch subset (`symbols_to_process`) instead of full run universe, causing incorrect scoping.

**Solution**:
- **UPDATED**: Modified fallback logic in `feature_selector.py` to prefer `run_identity.dataset_signature` (full run universe) over computing from `symbols_to_process`
- **PATTERN**: Matches the fix applied in TRAINING stage (preferring `run_identity.dataset_signature` over `mtf_data.keys()`)

**Files Changed**:
- `TRAINING/ranking/feature_selector.py` (lines 1367-1381):
  - First tries to extract from `run_identity.dataset_signature` if available
  - Only computes from `symbols_to_process` as last resort
  - Added logging to indicate source of universe_sig

**Result**: FEATURE_SELECTION stage now consistently uses full run universe signature, preventing incorrect scoping from batch subsets.

---

## Config Name Audit

**Problem**: Need to verify all `get_cfg()` calls use correct `config_name` values matching canonical paths.

**Solution**:
- **AUDITED**: All `get_cfg()` calls in FEATURE_SELECTION and TRAINING stages
- **VERIFIED**: All config names match mappings in `CONFIG/config_loader.py`:
  - `pipeline_config` → `pipeline/pipeline.yaml` ✓
  - `safety_config` → `pipeline/training/safety.yaml` ✓
  - `preprocessing_config` → `pipeline/training/preprocessing.yaml` ✓
  - `threading_config` → `pipeline/threading.yaml` ✓
  - `training_config` → `pipeline/training/intelligent.yaml` ✓
  - `multi_model_feature_selection` → `ranking/features/multi_model.yaml` ✓

**Files Audited**:
- `TRAINING/ranking/feature_selector.py` - 21 `get_cfg()` calls, all correct
- `TRAINING/training_strategies/execution/training.py` - 2 `get_cfg()` calls, all correct
- `TRAINING/training_strategies/strategies/*.py` - 5 `get_cfg()` calls, all correct
- `TRAINING/training_strategies/execution/data_preparation.py` - 1 `get_cfg()` call, correct
- `TRAINING/training_strategies/utils.py` - 2 `get_cfg()` calls, all correct

**Result**: All config loading paths are correct and use canonical paths (no symlinks).

---

## Related Changes

- Path import fixes in `TRAINING/stability/feature_importance/io.py` (TARGET_RANKING stage)
- Type casting fixes in `TRAINING/ranking/predictability/composite_score.py` (TARGET_RANKING stage)
- Universe signature fixes in `TRAINING/training_strategies/execution/training.py` (TRAINING stage)

---

**Path Import Root Cause Fix**
- **FIXED**: Added module-level `Path` import in `TRAINING/stability/feature_importance/schema.py`
- **REMOVED**: Redundant `Path` import inside `from_importance_snapshot()` method try block
- **ROOT CAUSE**: `FeatureSelectionSnapshot.from_importance_snapshot()` was importing `Path` inside a try block, causing "name 'Path' is not defined" errors when the import scope was incorrect
- **COVERAGE**: Fix applies to both TARGET_RANKING and FEATURE_SELECTION stages (shared module). TRAINING stage uses different schema file already fixed earlier.

**Files Changed**:
- `TRAINING/stability/feature_importance/schema.py`:
  - Added `from pathlib import Path` at module level (line 12)
  - Removed redundant `from pathlib import Path` inside `from_importance_snapshot()` method

**Impact**: 
- ✅ Consistent Path import pattern across all stages
- ✅ Type-safe config value handling prevents runtime errors
- ✅ Universe signature extraction uses full run universe across all stages
- ✅ All config loading paths verified and correct
- ✅ Cross-stage consistency for determinism and reproducibility tracking
- ✅ Fixed "name 'Path' is not defined" error for all model types in TARGET_RANKING and FEATURE_SELECTION stages
