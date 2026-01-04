# RunIdentity Wiring Fixes and Path Organization

**Date**: 2026-01-04  
**Type**: Critical Bug Fixes, Path Organization  
**Impact**: High - Fixes null signatures in snapshots, simplifies SYMBOL_SPECIFIC directory structure

## Overview

Fixed critical bugs preventing `run_identity` signatures from appearing in TARGET_RANKING snapshots, and simplified SYMBOL_SPECIFIC path organization to be consistent with cohort paths.

## Critical Issues Fixed

### 1. `NameError: name 'run_identity' is not defined`

**Problem**: `_save_to_cohort()` in `reproducibility_tracker.py` tried to use `run_identity` and `prediction_fingerprint` variables that weren't defined in its scope.

**Error Message**:
```
⚠️ Diff telemetry failed (non-critical): name 'run_identity' is not defined
```

**Fix**:
- Added `run_identity` and `prediction_fingerprint` parameters to `_save_to_cohort()` function signature
- Updated all 4 call sites to pass these parameters through

**Files Modified**:
- `TRAINING/orchestration/utils/reproducibility_tracker.py` (lines 1056-1070, 3307, 3350, 3630, 3828)

---

### 2. `log_comparison()` Using Null Parameter Instead of Computed Identity

**Problem**: `evaluate_target_predictability()` computed `partial_identity` with real data (line 6291) but then ignored it, using the `run_identity` parameter (usually `None`) when calling `log_comparison()`.

**Before**:
```python
partial_identity = RunIdentity(...)  # Computed with real data
...
tracker.log_comparison(..., run_identity=run_identity)  # Uses param (None), not partial_identity!
```

**Fix**:
```python
tracker.log_comparison(..., run_identity=partial_identity)  # Use computed identity
```

**Files Modified**:
- `TRAINING/ranking/predictability/model_evaluation.py` (line 7697)

---

### 3. Missing `train_seed` and `hparams_signature` for TARGET_RANKING

**Problem**: `partial_identity` was created with `train_seed=None` (no fallback) and `hparams_signature=""` (empty string is falsy, gets skipped).

**Fix**:
- Added fallback chain for `train_seed`: `experiment_config.seed` → config loader → default 42
- Compute `hparams_signature` from evaluation model families using `compute_hparams_fingerprint()`

**Before** (snapshot.json):
```json
"comparison_group": {
  "train_seed": null,
  "hyperparameters_signature": null
}
```

**After**:
```json
"comparison_group": {
  "train_seed": 42,
  "hyperparameters_signature": "a1b2c3..."
}
```

**Files Modified**:
- `TRAINING/ranking/predictability/model_evaluation.py` (lines 6285-6310)

---

### 4. Inconsistent SYMBOL_SPECIFIC Path Organization

**Problem**: Different path patterns for the same view:
- Cohorts: `SYMBOL_SPECIFIC/symbol=AAPL/cohort=.../`
- Feature importances: `SYMBOL_SPECIFIC/universe=.../symbol=AAPL/feature_importances/`

The `universe=` prefix was redundant for SYMBOL_SPECIFIC since `symbol=` already uniquely identifies the scope.

**Fix**: Updated `OutputLayout.repro_dir()` to skip `universe=` for SYMBOL_SPECIFIC:

**Before**:
```
SYMBOL_SPECIFIC/universe=abc123/symbol=AAPL/feature_importances/
```

**After**:
```
SYMBOL_SPECIFIC/symbol=AAPL/feature_importances/
```

CROSS_SECTIONAL unchanged (still uses `universe=` to identify multi-symbol set).

**Files Modified**:
- `TRAINING/orchestration/utils/output_layout.py` (lines 189-206)

---

## Summary of Changes

| File | Change |
|------|--------|
| `reproducibility_tracker.py` | Added `run_identity`, `prediction_fingerprint` params to `_save_to_cohort()` |
| `model_evaluation.py` | Use `partial_identity` in `log_comparison()`, add seed fallback, compute hparams |
| `output_layout.py` | Simplify SS paths - remove redundant `universe=` prefix |

## Impact

- TARGET_RANKING snapshots now contain populated signatures for determinism tracking
- SYMBOL_SPECIFIC directory structure is now consistent between cohorts and feature importances
- No functionality broken - all changes are write-path only (no readers depend on old paths)

## Verification

```python
# Path logic verified:
CS: targets/.../reproducibility/CROSS_SECTIONAL/universe=abc123  # universe= for symbol set
SS: targets/.../reproducibility/SYMBOL_SPECIFIC/symbol=AAPL       # no redundant universe=
```
