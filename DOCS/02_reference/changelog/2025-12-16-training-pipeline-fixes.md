# Training Pipeline Fixes - 2025-12-16

## Overview

This changelog documents critical fixes to the training pipeline, including canonical family ID system, feature audit instrumentation, and various plumbing fixes.

## Changes

### Canonical Family ID System

**Problem**: Model family names were inconsistent across registries (TitleCase in some, snake_case in others), causing lookup failures like `KeyError: "Family 'CatBoost' not found in MODMAP or TRAINER_MODULE_MAP"`.

**Solution**: Migrated all registries to use snake_case canonical IDs consistently.

**Files Changed**:
- `TRAINING/training_strategies/family_runners.py` - Updated `MODMAP` keys to snake_case
- `TRAINING/common/isolation_runner.py` - Updated `TRAINER_MODULE_MAP` keys to snake_case
- `TRAINING/common/runtime_policy.py` - Updated `POLICY` keys to snake_case (both `CROSS_SECTIONAL_POLICIES` and `SEQUENTIAL_POLICIES`)
- `TRAINING/training_strategies/utils.py` - Updated `FAMILY_CAPS` keys to snake_case, improved `normalize_family_name()` function

**Key Changes**:
- All registry keys now use snake_case (e.g., `"lightgbm"`, `"xgboost"`, `"meta_learning"`)
- `normalize_family_name()` function converts any input to canonical snake_case
- All lookups normalize family names before accessing registries
- Added startup validation to prevent key drift

**New Files**:
- `TRAINING/utils/registry_validation.py` - Startup assertions to enforce canonical keys
- `tests/test_family_canonicalization.py` - Unit tests for canonicalization and registry invariants

### Feature Audit System

**Problem**: Feature collapse (e.g., 100 → 52 → 12 features) was happening silently without clear diagnostics.

**Solution**: Added comprehensive feature drop tracking with per-feature drop reasons.

**Files Changed**:
- `TRAINING/utils/feature_audit.py` - New module for feature drop tracking
- `TRAINING/training_strategies/data_preparation.py` - Integrated audit tracking at all drop points

**Features**:
- Tracks features at each stage: requested → registry_allowed → present_in_polars → kept_for_training → used_in_X
- Records drop reasons: `missing_in_df`, `excluded_by_registry`, `all_null`, `non_numeric`, `failed_coercion`
- Generates CSV reports: `feature_audit_{target}_summary.csv` and `feature_audit_{target}_drops.csv`
- Reports written to `output/artifacts/feature_audits/`

### Training Pipeline Fixes

**Fixed Issues**:

1. **Family Name Canonicalization**: Fixed mismatches between config family names and registry keys
   - Files: `TRAINING/training_strategies/training.py`, `TRAINING/training_strategies/family_runners.py`
   - All family names are now normalized before registry lookups

2. **Banner Suppression**: Fixed licensing banner printing in child processes
   - Files: `TRAINING/common/license_banner.py`, `TRAINING/common/threads.py`
   - Banner now suppressed via environment variables (`FOXML_SUPPRESS_BANNER`, `TRAINER_ISOLATION_CHILD`, `TRAINER_CHILD_FAMILY`)

3. **Reproducibility Tracking**: Fixed `'str' object has no attribute 'name'` error
   - File: `TRAINING/utils/reproducibility_tracker.py`
   - Added defensive handling for both Enum and string inputs

4. **Model Saving**: Fixed `_pkg_ver` and `joblib` referenced before assignment
   - File: `TRAINING/training_strategies/training.py`
   - Moved `_pkg_ver` definition and `joblib` import to proper scope

5. **Feature Selector vs Trainer**: Fixed confusion between feature selectors (`mutual_information`, `univariate_selection`) and actual trainers
   - Files: `TRAINING/training_strategies/training.py`, `TRAINING/training_strategies/family_runners.py`
   - Added preflight validation to skip selectors from training pipeline

### Testing

**New Tests**:
- `tests/test_family_canonicalization.py`:
  - `test_normalize_idempotent()` - Ensures canonicalization is idempotent
  - `test_registry_keys_canonical_and_unique()` - Validates all registry keys are canonical and collision-free
  - `test_registry_coverage()` - Ensures registries have consistent coverage
  - `test_lookup_accepts_variants()` - Verifies lookups accept various input formats

### Documentation Updates

**Updated Files**:
- `CHANGELOG.md` - Added 2025-12-16 updates section
- `DOCS/03_technical/implementation/ADDING_PROPRIETARY_MODELS.md` - Updated to reflect snake_case registry keys

## Migration Notes

**Breaking Changes**: None (backward compatible - normalization handles both TitleCase and snake_case inputs)

**Action Required**: 
- If you have custom model families, ensure registry keys are snake_case
- Run `pytest tests/test_family_canonicalization.py` to validate your registries

## Technical Details

### Canonicalization Function

The `normalize_family_name()` function converts any input to snake_case:
- `"LightGBM"` → `"lightgbm"`
- `"XGBoost"` → `"xgboost"`
- `"MetaLearning"` → `"meta_learning"`
- `"RandomForest"` → `"random_forest"`

### Registry Validation

Startup validation (`TRAINING/utils/registry_validation.py`) ensures:
- All keys are canonical (normalize(key) == key)
- No collisions after normalization
- Runs automatically on import (fail-fast)

### Feature Audit Output

Example audit report structure:
```
output/artifacts/feature_audits/
  feature_audit_fwd_ret_5d_summary.csv
  feature_audit_fwd_ret_5d_drops.csv
```

Summary CSV columns: `metric`, `count`, `stage`, `reason`
Drops CSV columns: `feature_name`, `stage`, `reason`, `dtype_polars`, `dtype_pandas`, `null_fraction`, `n_unique`, `sample_value`

