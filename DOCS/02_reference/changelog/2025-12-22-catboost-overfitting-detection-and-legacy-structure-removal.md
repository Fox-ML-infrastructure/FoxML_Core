# CatBoost Overfitting Detection and Legacy REPRODUCIBILITY Structure Removal

**Date**: 2025-12-22  
**Type**: Performance Fix, Bug Fix, Code Cleanup  
**Impact**: High - Prevents 6-hour hangs, removes legacy directory creation  
**Breaking**: No - Backward compatible

## Summary

Fixed CatBoost feature selection taking 6+ hours by implementing config-driven overfitting detection, process-based timeout, and comprehensive timing diagnostics. Also removed legacy `RESULTS/REPRODUCIBILITY` directory structure creation, ensuring all writes go to target-first structure only.

## Problem

### CatBoost 6-Hour Feature Selection Runs

CatBoost training itself was fast (~57 seconds), but feature selection was taking 6+ hours inconsistently. The bottleneck was happening **AFTER** training completed, in expensive `PredictionValuesChange` importance computation:

- Overfitting check used hard threshold `>= 0.999` (99.9%)
- When `train_score` was 0.99-0.998, check didn't trigger
- PredictionValuesChange importance computation ran anyway (can take 2-3+ hours)
- No timeout, no fallback, no diagnostics
- Inconsistent behavior: 50 minutes when train_score < 0.99, 6 hours when 0.99-0.998

### Legacy REPRODUCIBILITY Structure

The legacy `RESULTS/REPRODUCIBILITY/FEATURE_SELECTION/...` structure was still being created alongside the target-first structure, causing:
- Duplicate directory structures
- Confusion about which location is authoritative
- Path resolution bugs stopping at `RESULTS/` instead of finding run directory

## Solution

### 1. Config-Driven Importance Policy (SST)

**File**: `CONFIG/pipeline/training/safety.yaml`

Added `feature_importance` section under `leakage_detection` with:
- `importance_max_wall_minutes: 30` - Maximum wall time for expensive importance computation (timeout)
- `overfit_train_acc_threshold: 0.99` - Skip expensive importance if train_acc >= this (default: 0.99 = 99%)
- `overfit_train_val_gap_threshold: 0.20` - Skip expensive importance if (train_acc - cv_acc) >= this (default: 0.20)
- `importance_fallback: "gain"` - Fallback method when skipping: "gain" | "split" | "none"
- `importance_skip_on_overfit: true` - Enable skipping expensive importance on overfitting detection
- `pvc_feature_count_cap: 250` - Optional: skip PVC if n_features > this (null to disable)

**Fix**: Moved `feature_importance` from incorrect location (after `auto_fixer`) to correct location under `leakage_detection` (after `ranking`).

### 2. Shared Overfitting Detection Helper

**File**: `TRAINING/ranking/utils/overfitting_detection.py` (NEW)

Created `should_skip_expensive_importance()` function with policy-based gating:
- Checks train accuracy threshold
- Checks train/CV gap
- Checks train/val gap
- Optional feature count cap
- Returns `(should_skip, reason, metadata)` tuple

Used by both feature selection and model evaluation for consistent behavior.

### 3. Policy-Based Overfitting Gating

**File**: `TRAINING/ranking/multi_model_feature_selection.py`

Replaced hard threshold (`>= 0.999`) with config-driven policy:
- Uses shared `should_skip_expensive_importance()` helper
- Logs decision with actual values: `train=... cv=... decision=SKIP|RUN, reason=...`
- Deterministic fallback importance when skipping:
  - `gain`: CatBoost native `FeatureImportance` (gain-based)
  - `split`: CatBoost `Split` importance
  - `none`: Zero importance
- Preserves comparability: same inputs → same outputs

### 4. Process-Based Timeout for PVC

**File**: `TRAINING/ranking/multi_model_feature_selection.py`

Implemented hard process timeout (not soft):
- Runs PVC in separate process and kills after N minutes
- Works even if CatBoost hangs in native code
- Falls back deterministically on timeout
- Logs `PVC_TIMEOUT` for audit

### 5. Comprehensive Timing Diagnostics

**File**: `TRAINING/ranking/multi_model_feature_selection.py`

Added `timed()` context manager with metadata:
- Wraps CV loop: `timed("catboost_cv", n_splits=..., n_features=..., n_samples=...)`
- Wraps final fit: `timed("catboost_fit", n_features=..., n_samples=...)`
- Wraps PVC: `timed("catboost_pvc", n_features=..., n_samples=..., trees=..., depth=...)`
- Logs: `⏱️ START ...` and `⏱️ END ...: X.XXs (X.XX minutes)`

Ends the "it took 6 hours somewhere" problem permanently.

### 6. Legacy REPRODUCIBILITY Structure Removal

**Files Changed**:
- `TRAINING/ranking/feature_selection_reporting.py`
- `TRAINING/ranking/feature_selector.py`
- `TRAINING/stability/feature_importance/io.py`
- `TRAINING/stability/feature_importance/hooks.py`

**Changes**:
- Removed all legacy `REPRODUCIBILITY/FEATURE_SELECTION/...` path construction
- Fixed path resolution to not stop at `RESULTS/` directory
- Now only stops when finding run directory (has `targets/`, `globals/`, or `cache/`)
- All writes now go to target-first structure: `RESULTS/runs/{run}/targets/{target}/reproducibility/...`
- Removed `repro_dir` fallback that could resolve to `RESULTS/`

## Files Changed

1. **`CONFIG/pipeline/training/safety.yaml`**:
   - Moved `feature_importance` section to correct location under `leakage_detection`
   - Added expensive importance computation policy settings

2. **`TRAINING/ranking/utils/overfitting_detection.py`** (NEW):
   - Shared helper for overfitting detection
   - Policy-based gating with multiple conditions

3. **`TRAINING/ranking/multi_model_feature_selection.py`**:
   - Added `timed()` context manager
   - Replaced overfitting check with policy-based gating
   - Added process-based timeout for PVC
   - Added fallback importance logic
   - Wrapped operations with timers
   - Added caching placeholder comment

4. **`TRAINING/ranking/feature_selection_reporting.py`**:
   - Removed legacy `REPRODUCIBILITY/FEATURE_SELECTION` path construction
   - Fixed path resolution to find run directory correctly

5. **`TRAINING/ranking/feature_selector.py`**:
   - Removed legacy path construction for snapshots
   - Now finds run directory and uses target-first structure

6. **`TRAINING/stability/feature_importance/io.py`**:
   - Fixed path resolution to not stop at `RESULTS/`
   - Now continues to find actual run directory

7. **`TRAINING/stability/feature_importance/hooks.py`**:
   - Fixed path resolution to not stop at `RESULTS/`
   - Now continues to find actual run directory

## Testing

### Acceptance Criteria

A run that previously took 6h now:
- Either finishes PVC under 30m **OR** times out and falls back deterministically
- Logs include: `train_score`, `cv_score`, `gap`, `decision=RUN|SKIP|TIMEOUT`
- Elapsed times for each stage (CV, fit, importance)
- Artifacts include: which importance path was used (PVC vs fallback), timeout flag if triggered
- No change in outputs when none of the gates trigger (comparability preserved)

### Legacy Structure

- No `RESULTS/REPRODUCIBILITY/` directory should be created
- All writes go to `RESULTS/runs/{run}/targets/{target}/reproducibility/...`
- Path resolution correctly finds run directory even when starting from deep paths

## Impact

- **Performance**: Prevents 6-hour hangs in feature selection
- **Reliability**: Deterministic fallback ensures runs complete even when overfitting detected
- **Observability**: Comprehensive timing diagnostics show exactly where time is spent
- **Code Quality**: Removed legacy structure creation, cleaner codebase
- **Configurability**: All thresholds and policies controlled via config (SST-compliant)

## Notes

- **Determinism**: Fallback importance ensures consistent outputs even when PVC is skipped
- **Comparability**: Same inputs → same outputs (unless timeout/overfitting triggers)
- **SST Compliance**: All thresholds and policies come from config
- **No Accuracy Degradation**: Fallback uses standard gain/split importance (default for tree models), multi-model consensus reduces single-model bias

