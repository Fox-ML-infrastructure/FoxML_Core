# Training Pipeline Integrity Fixes and Canonical Layout Migration

**Date**: 2025-12-23  
**Type**: Critical Bug Fixes, Structural Refactoring  
**Impact**: High - Fixes correctness bugs, removes competing hierarchies, enforces config-centered control

## Overview

Fixed 7 critical integrity issues in the training pipeline and removed the competing `training_results/` hierarchy that caused structural ambiguity. Standardized on target-first canonical layout: `run_root/targets/<target>/models/...` as Single Source of Truth (SST).

## Critical Issues Fixed

### 1. Routing Decisions Fingerprint Mismatch (Fail-Fast)

**Problem**: Routing decisions fingerprint mismatch logged ERROR but training continued anyway, breaking reproducibility and correctness.

**Fix**:
- **Production (dev_mode=False)**: Raises `ValueError` immediately on mismatch (no silent continue)
- **Development (dev_mode=True)**: Logs warning, attempts to regenerate from fresh candidates, returns marker `{"_STALE_DECISIONS_IGNORED": True}`
- Added invariant: Training must not proceed if routing decisions are stale and no regeneration occurred

**Files Modified**:
- `TRAINING/ranking/target_routing.py` (lines 537-545, 611-626)

**Impact**: Prevents training with stale routing decisions, ensuring reproducibility.

---

### 2. Feature Registry Bypass for Long Horizons

**Problem**: Long horizons (e.g., `fwd_ret_5d`) showed "0 features allowed" then "26 allowed" due to permissive fallback, giving false sense of leakage safety.

**Fix**:
- If registry returns 0 features explicitly and not in ranking mode:
  - **dev_mode=False**: Hard error (no permissive fallback)
  - **dev_mode=True**: Allow fallback but stamp metadata with "DEV_MODE_PERMISSIVE_REGISTRY"
- Only allow permissive fallback if explicitly marked `dev_mode=True` (and metadata records it)

**Files Modified**:
- `TRAINING/ranking/utils/leakage_filtering.py` (lines 670-692)

**Impact**: Enforces leakage-safe design - 0 allowed features means no features permitted unless dev mode.

---

### 3. Training Families Source Bug

**Problem**: Training used `feature_selection.model_families` instead of `training.model_families` from config, causing feature selectors to leak into training.

**Fix**:
- Extract `training.model_families` from experiment_config before calling routing
- Added assertion: `assert set(passed_families) == set(experiment_config.training.model_families)`
- Logs both FS and training families separately for debugging

**Files Modified**:
- `TRAINING/orchestration/intelligent_trainer.py` (lines 2040-2057)

**Impact**: Training now uses correct families from config, feature selectors cannot leak into training.

---

### 4. Routing Produces 0 Jobs (Metrics Aggregation + Thresholds)

**Problem**: Metrics aggregation found no symbol metrics → `sample_size=0`, `stability=UNKNOWN` → router disabled everything → 0 jobs.

**Fixes**:
- **Metrics Aggregation**: Fallback to CROSS_SECTIONAL cohort metadata for `sample_size` when symbol metrics missing
- **Routing Thresholds**: Auto-enable dev_mode thresholds when `max_samples_per_symbol < 10000` (unless explicitly disabled)

**Files Modified**:
- `TRAINING/orchestration/metrics_aggregator.py` (lines 336-375)
- `TRAINING/orchestration/training_router.py` (lines 214-224, 547-554)

**Impact**: Routing produces jobs even with small datasets or missing symbol metrics.

---

### 5. Stale Routing Decisions Loading

**Problem**: Training loaded routing decisions from wrong location (multiple legacy paths), causing fingerprint mismatches.

**Fix**:
- Enforce single known path: `globals/routing_decisions.json` (current run only)
- Removed legacy fallback paths (DECISION/, REPRODUCIBILITY/, etc.)
- Fail loudly if not found (when `validate_fingerprint=True`)

**Files Modified**:
- `TRAINING/ranking/target_routing.py` (lines 568-663)

**Impact**: Prevents loading stale decisions from wrong run, ensures fingerprint validation works correctly.

---

### 6. Output Layout Inconsistencies (Removed training_results/)

**Problem**: Two competing hierarchies writing models:
- `training_results/targets/<target>/...` (target-first, partial)
- `training_results/<family>/view=.../...` (family-first, partial)

This caused context-free models, stale routing decisions, and ambiguity.

**Fix**:
- **Removed `training_results/` directory entirely**
- **Canonical layout**: `run_root/targets/<target>/models/view=<view>/[symbol=<symbol>/]family=<family>/`
- Created `ArtifactPaths` builder (SST) for all artifact paths
- All model saves use `ArtifactPaths.model_dir()` → writes only to canonical location
- Model filenames don't encode target/symbol (path encodes it)
- Optional mirror generation for family-first browsing (symlinks or manifest)

**Files Modified**:
- `TRAINING/orchestration/utils/artifact_paths.py` (new file)
- `TRAINING/training_strategies/execution/training.py` (lines 714-1620)
- `TRAINING/orchestration/intelligent_trainer.py` (lines 873, 2316)
- `TRAINING/orchestration/utils/target_first_paths.py` (deprecated `training_results_root()`)

**New Files**:
- `TRAINING/orchestration/utils/artifact_paths.py` - ArtifactPaths builder (SST)
- `TRAINING/orchestration/utils/artifact_mirror.py` - Optional mirror generation

**Impact**: Single canonical layout eliminates ambiguity, ensures all models have context (target, view, symbol, family).

---

### 7. Router Pattern Miss

**Problem**: `fwd_ret_oc_same_day` targets defaulted to regression (no pattern matched).

**Fix**:
- Added pattern: `(r'^.*_oc_same_day.*$', TaskSpec('binary_classification', ...))`

**Files Modified**:
- `TRAINING/orchestration/routing/target_router.py` (line 110-112)

**Impact**: `*_oc_same_day` targets now correctly route to binary classification.

---

## Canonical Layout Structure

```
<run_root>/
  globals/
    manifests/
      models_manifest.parquet      # Index of all models (optional, for browsing)
    routing_plan/
      routing_plan.yaml
    routing/
      routing_candidates.parquet
      routing_decisions.json

  targets/
    <target>/
      reproducibility/
        CROSS_SECTIONAL/
          selected_features.txt
          feature_selection_summary.json
          target_confidence.json
          cohort=<cohort_id>/
            metadata.json
            metrics.json
        SYMBOL_SPECIFIC/
          symbol=<symbol>/
            selected_features.txt
            cohort=<cohort_id>/
              ...
      models/                        # NEW: All models here (canonical)
        view=CROSS_SECTIONAL/
          family=xgboost/
            model.joblib             # No target/symbol in filename
            model_meta.json
            metrics.json
            fingerprints.json
            scaler.joblib
            imputer.joblib
          family=lightgbm/
            ...
        view=SYMBOL_SPECIFIC/
          symbol=AAPL/
            family=xgboost/
              model.joblib
              model_meta.json
              ...
      decision/
        routing_decision.json
```

**Rules**:
- No filename encodes target/view/symbol - path already does
- Each model directory contains model + meta + metrics + fingerprints together
- `globals/` contains summaries and indexes, not primary artifacts
- **No `training_results/` directory** (removed entirely)

---

## ArtifactPaths Builder (SST)

Created `ArtifactPaths` class as single source of truth for all artifact paths:

```python
from TRAINING.orchestration.utils.artifact_paths import ArtifactPaths

# Get canonical model directory
model_dir = ArtifactPaths.model_dir(
    run_root=run_root,
    target="fwd_ret_5d",
    view="CROSS_SECTIONAL",
    family="xgboost",
    symbol=None
)
# Returns: targets/fwd_ret_5d/models/view=CROSS_SECTIONAL/family=xgboost/

# Get model file path
model_path = ArtifactPaths.model_file(model_dir, family="xgboost", extension="joblib")
# Returns: targets/fwd_ret_5d/models/view=CROSS_SECTIONAL/family=xgboost/model.joblib

# Get metadata file path
meta_path = ArtifactPaths.metadata_file(model_dir)
# Returns: targets/fwd_ret_5d/models/view=CROSS_SECTIONAL/family=xgboost/model_meta.json
```

**Usage**: All model saves/loads go through `ArtifactPaths` → ensures consistency.

---

## Optional Mirror Generation

If you want family-first browsing, use `artifact_mirror.py`:

```python
from TRAINING.orchestration.utils.artifact_mirror import generate_family_first_mirrors

# Generate manifest only (default)
manifest_path = generate_family_first_mirrors(run_root, create_symlinks=False)
# Creates: globals/manifests/models_manifest.parquet

# Generate symlinks (optional)
generate_family_first_mirrors(run_root, create_symlinks=True)
# Creates: training_results/<family>/view=<view>/target=<target> -> symlink to canonical location
```

**Decision**: Make this optional (config flag `training.generate_family_mirrors: false` by default).

---

## Migration Notes

### Backward Compatibility

- **Reading**: Code still reads from legacy paths if canonical location not found (graceful degradation)
- **Writing**: All new writes go to canonical location only
- **Deprecated**: `training_results_root()` function marked as deprecated with warning

### Breaking Changes

- **No `training_results/` directory created** - if your code expects it, it won't exist
- **Model paths changed** - old code reading from `training_results/<family>/...` will need to use `ArtifactPaths.model_dir()`
- **Routing decisions** - legacy fallback paths removed, only `globals/routing_decisions.json` is checked

---

## Testing Checklist

- [x] All models saved to `targets/<target>/models/view=.../family=.../`
- [x] No `training_results/` directory created
- [x] Model filenames don't encode target/symbol (path encodes it)
- [x] Routing fingerprint mismatch fails loudly (prod) or regenerates (dev)
- [x] Feature registry: 0 allowed features → error in prod, warning in dev
- [x] Training uses `training.model_families` from config (asserted)
- [x] Routing produces jobs when dataset is small (dev thresholds auto-enabled)
- [x] Stale routing decisions cannot be loaded (single known path enforced)
- [x] Router pattern added for `*_oc_same_day` targets

---

## Files Modified

### New Files
- `TRAINING/orchestration/utils/artifact_paths.py` - ArtifactPaths builder (SST)
- `TRAINING/orchestration/utils/artifact_mirror.py` - Optional mirror generation

### Modified Files
- `TRAINING/training_strategies/execution/training.py` - Use ArtifactPaths, remove training_results/ writes
- `TRAINING/orchestration/intelligent_trainer.py` - Remove training_results/ creation, fix families source
- `TRAINING/orchestration/utils/target_first_paths.py` - Deprecate training_results_root()
- `TRAINING/ranking/target_routing.py` - Fix fingerprint mismatch handling, enforce single path
- `TRAINING/ranking/utils/leakage_filtering.py` - Fix feature registry bypass
- `TRAINING/orchestration/metrics_aggregator.py` - Fix metrics aggregation fallback
- `TRAINING/orchestration/training_router.py` - Fix routing thresholds
- `TRAINING/orchestration/routing/target_router.py` - Add router pattern

### Removed/Deprecated
- `training_results/` directory creation (removed)
- `training_results_root()` helper (deprecated with warning)
- Family-first save logic (removed)

---

## Definition of Done

- [x] **No `training_results/` directory** - removed entirely
- [x] **All models in canonical location**: `targets/<target>/models/...`
- [x] **ArtifactPaths builder** - single SST for all paths
- [x] **Routing fingerprint mismatch** - fails loudly (prod) or regenerates (dev)
- [x] **Feature registry** - no permissive fallback unless dev_mode (metadata stamped)
- [x] **Training families** - uses `training.model_families` from config (asserted)
- [x] **Routing produces jobs** - dev thresholds auto-enabled for small datasets
- [x] **Stale decisions** - single known path, fingerprint validation
- [x] **Router pattern** - added for `*_oc_same_day` targets
- [x] **Optional mirrors** - manifest generated (symlinks optional)

---

## Related Changes

- See [2025-12-23-training-pipeline-organization-and-config-fixes.md](2025-12-23-training-pipeline-organization-and-config-fixes.md) for earlier path fixes
- See [2025-12-21-feature-selection-routing-and-training-view-tracking.md](2025-12-21-feature-selection-routing-and-training-view-tracking.md) for view tracking

