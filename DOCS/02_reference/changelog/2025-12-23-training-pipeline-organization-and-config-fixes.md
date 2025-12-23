# Training Pipeline Organization and Config Fixes (2025-12-23)

## Overview

Comprehensive refactoring to fix blocking correctness bugs, data integrity issues, and structural cleanup with centralized path SST (Single Source of Truth), phased implementation, and explicit invariants. This ensures reproducibility files are properly scoped by view/symbol, prevents unknown lookback features from causing RuntimeErrors, fixes model families config routing, and reorganizes globals/ directory structure.

## Impact

- **Correctness**: Prevents RuntimeError from unknown lookback features, fixes `cohort_metadata` initialization
- **Data Integrity**: Eliminates overwrites in reproducibility files and feature importances across views/symbols
- **Config Correctness**: Separates training.model_families from feature_selection.model_families
- **Structure**: Removes legacy METRICS/ creation, organizes globals/ into subfolders (routing/, training/, summaries/)

## Changes by Phase

### Phase 0: Centralized Path SST (Single Source of Truth)

**Goal**: Create single module that owns all output paths to prevent future regressions.

**Files Changed**:
- `TRAINING/orchestration/utils/target_first_paths.py`

**New Functions Added**:
- `run_root(output_dir: Path) -> Path`: Get run root directory (has targets/, globals/, cache/)
- `training_results_root(run_root: Path) -> Path`: Get training_results/ directory
- `globals_dir(run_root: Path, kind: Optional[str] = None) -> Path`: Get globals directory with optional subfolder (routing, training, summaries, rankings)
- `target_repro_dir(run_root: Path, target: str, view: Optional[str] = None, symbol: Optional[str] = None) -> Path`: Get reproducibility directory scoped by view/symbol
- `target_repro_file_path(run_root: Path, target: str, filename: str, view: Optional[str] = None, symbol: Optional[str] = None) -> Path`: Get file path in reproducibility directory (view/symbol-scoped)
- `model_output_dir(training_results_root: Path, family: str, view: str, symbol: Optional[str] = None) -> Path`: Get model output directory with view/symbol structure

**Design Principles**:
- All output paths must come from centralized helpers
- No freehand paths like `os.path.join(...)` for output directories
- View/symbol-aware paths prevent overwrites

---

### Phase 1: Blocking Correctness Fixes

#### 1. Quarantine Unknown Lookback Features Before Budget Call

**Problem**: Features with unknown lookback (`inf`) like `vwap_dev_high`, `vwap_dev_low` were passing through the gatekeeper and causing `RuntimeError` when `compute_budget()` was called.

**Invariant**: "Budget computation assumes sanitized features. Sanitizer must never call strict budget computation before filtering."

**Files Changed**:
- `TRAINING/ranking/shared_ranking_harness.py`

**Fix**:
- Pre-filter unknown lookback features using non-throwing lookup in `apply_cleaning_and_audit_checks()`
- Quarantine features with `inf` lookback BEFORE calling `compute_feature_lookback_max()`
- Remove quarantined features from X matrix and feature_names list
- Only pass safe features to budget computation

**Pattern**:
```python
# Step 1: Compute canonical map (non-throwing) to identify unknown features
lookback_result_precheck = compute_feature_lookback_max(...)
canonical_map = lookback_result_precheck.canonical_lookback_map

# Step 2: Identify and quarantine features with unknown lookback (inf)
unknown_features = []
safe_features = []
for feat_name in feature_names:
    feat_key = _feat_key(feat_name)
    lookback = canonical_map.get(feat_key)
    if lookback is None or lookback == float("inf"):
        unknown_features.append(feat_name)
    else:
        safe_features.append(feat_name)

# Step 3: Quarantine unknown features (remove from X and feature_names)
if unknown_features:
    X = np.delete(X, unknown_indices, axis=1)
    feature_names = safe_features

# Step 4: Only pass safe features to compute_feature_lookback_max (strict mode)
result = compute_feature_lookback_max(safe_features, ...)
```

**Impact**: Prevents RuntimeError from unknown lookback features, pipeline proceeds successfully.

#### 2. Fix `cohort_metadata` Initialization

**Problem**: `cohort_metadata` was referenced before assignment in some exception paths, causing `WARNING: local variable 'cohort_metadata' referenced before assignment`.

**Files Changed**:
- `TRAINING/ranking/feature_selector.py` (already fixed in previous work, verified)

**Fix**: Initialize `cohort_metadata = None`, `cohort_metrics = {}`, `cohort_additional_data = {}` at function scope (before any try blocks).

**Impact**: Exception handlers can safely reference these variables.

---

### Phase 2: Data Integrity (Stop Overwrites)

#### 3. Reproducibility Outputs Must Be View/Symbol-Scoped

**Problem**: Files like `selected_features.txt`, `feature_selection_rankings.csv`, `target_confidence.json`, `model_family_status.json`, `feature_selection_summary.json` were being saved directly to `targets/{target}/reproducibility/` instead of view/symbol subdirectories, causing overwrites when processing multiple symbols.

**Invariant**: "Reproducibility outputs must be view/symbol-scoped. No overwrites across views/symbols."

**Canonical Layout**:
- `targets/{target}/reproducibility/CROSS_SECTIONAL/{filename}`
- `targets/{target}/reproducibility/SYMBOL_SPECIFIC/symbol={symbol}/{filename}`

**Files Changed**:
- `TRAINING/ranking/feature_selection_reporting.py`
- `TRAINING/ranking/multi_model_feature_selection.py`

**Fix**:
- Use centralized path helper: `target_repro_file_path(run_root, target, filename, view, symbol)`
- Update all save paths to use view/symbol subdirectories
- Files affected: `selected_features.txt`, `feature_selection_rankings.csv`, `target_confidence.json`, `model_family_status.json`, `feature_selection_summary.json`

**Impact**: No overwrites across views/symbols, each view/symbol combination has its own files.

#### 4. Feature Importances Must Use View/Symbol Subdirectories

**Problem**: `save_feature_importances_for_reproducibility()` ignored `view` and `symbol` parameters, always saved to `reproducibility/feature_importances/`, causing overwrites.

**Files Changed**:
- `TRAINING/ranking/feature_selection_reporting.py`
- `TRAINING/ranking/feature_selector.py`

**Fix**:
- Updated `save_feature_importances_for_reproducibility()` to use view/symbol-scoped paths:
  - CROSS_SECTIONAL: `reproducibility/CROSS_SECTIONAL/feature_importances/`
  - SYMBOL_SPECIFIC: `reproducibility/SYMBOL_SPECIFIC/symbol={symbol}/feature_importances/`
- Added `aggregate_importances_cross_sectional()` helper for CROSS_SECTIONAL aggregation (mean of normalized importances per model family)
- Fixed fallback path: If view is CROSS_SECTIONAL but using per-symbol processing, aggregate importances across symbols and save once

**Impact**: Feature importances are properly scoped, no overwrites, CROSS_SECTIONAL view aggregates correctly.

---

### Phase 3: Functional Correctness (Config Routing)

#### 5. Split Model Families Config: Training vs Feature Selection

**Problem**: `training.model_families` from experiment config was being used for feature selection instead of training. Feature selection should use `feature_selection.model_families` (or fallback to `intelligent_training.model_families`).

**Files Changed**:
- `TRAINING/orchestration/intelligent_trainer.py`

**Fix**:
- Made config resolution explicit:
  - `training_families := experiment.training.model_families ?? experiment.intelligent_training.model_families ?? defaults`
  - `fs_families := experiment.feature_selection.model_families ?? experiment.intelligent_training.model_families ?? defaults`
- Thread separately:
  - Feature selection gets `fs_families` (passed via `IntelligentTrainer.fs_model_families`)
  - Training gets `training_families` (stored as `config_families`)
- Added log line: Print both resolved sets once per run
- Updated `IntelligentTrainer.__init__()` to accept `fs_model_families` parameter
- Updated feature selection config building to use `fs_model_families` when available

**Impact**: Training and feature selection use correct model families from config, no cross-contamination.

#### 6. Training Plan 0 Jobs + Dev Mode: Make It Impossible to Raise in Dev Mode

**Problem**: Training plan consumer was raising error even when `dev_mode=true`, preventing fallback to all targets.

**Files Changed**:
- `TRAINING/orchestration/training_plan_consumer.py`

**Fix**:
- Resolve `dev_mode` FIRST (before checking jobs)
- Log config source and resolved value
- If `not jobs`:
  - If `dev_mode`: Log warning + return fallback plan (don't raise)
  - Else: Raise with clear error

**Code Pattern**:
```python
# Resolve dev_mode FIRST
dev_mode = False
config_source = "default"
try:
    from CONFIG.config_loader import get_cfg
    routing_config = get_cfg("training_config.routing", default={}, config_name="training_config")
    dev_mode = routing_config.get("dev_mode", False)
    config_source = "training_config.routing.dev_mode"
except Exception as e:
    logger.debug(f"Failed to load dev_mode config: {e}")

logger.info(f"Training plan dev_mode={dev_mode} (from {config_source})")

if not jobs:
    if dev_mode:
        logger.warning(f"⚠️ Training plan has 0 jobs (dev_mode=true). Using fallback: all {len(targets)} targets.")
        return targets  # Don't raise!
    else:
        raise ValueError(...)  # Only raise if NOT dev_mode
```

**Impact**: Dev mode works correctly, allows testing with 0 jobs, production mode still hard-fails appropriately.

---

### Phase 4: Refactor/Cleanup

#### 7. Stop Creating `targets/` Under `training_results/`

**Status**: Already fixed in previous work (uses `base_run_dir` instead of `training_results/`)

**Files Verified**:
- `TRAINING/training_strategies/execution/training.py` (lines 652, 1401)

**Impact**: Models save to `training_results/{family}/view={view}/[symbol={symbol}/]`, metadata saves to `targets/{target}/models/` in base run directory.

#### 8. Stop Creating `METRICS/` But Keep Read Fallback

**Problem**: Code was still creating `METRICS/` directory for backward compatibility, but we want to stop creating it.

**Files Changed**:
- `TRAINING/orchestration/routing_integration.py`
- `TRAINING/orchestration/metrics_aggregator.py`

**Fix**:
- Removed `METRICS/` directory creation
- Removed legacy copy sections that wrote to `METRICS/`
- Readers still check both new and old locations (read-both, write-new strategy)

**Impact**: No more `METRICS/` directory creation, cleaner output structure.

#### 9. Reorganize `globals/` Into Subfolders

**Target Structure**:
```
globals/
  - routing/
    - routing_candidates.json
    - routing_candidates.parquet
    - routing_plan/
      - routing_plan.json
      - routing_plan.yaml
      - routing_plan.md
  - training/
    - training_plan/
      - master_training_plan.json
      - training_plan.json
      - training_plan.yaml
      - training_plan.md
  - summaries/
    - feature_selection_summary.json
    - model_family_status_summary.json
    - selected_features_summary.json
    - target_confidence_summary.json
    - performance_audit_report.json
  - rankings/
    - target_predictability_rankings.csv
  - stats.json
```

**Files Changed**:
- `TRAINING/orchestration/metrics_aggregator.py`: Save to `globals/routing/`
- `TRAINING/orchestration/routing_integration.py`: Save to `globals/routing/routing_plan/`
- `TRAINING/orchestration/training_router.py`: Save to `globals/routing/routing_plan/`
- `TRAINING/orchestration/training_plan_generator.py`: Save to `globals/training/training_plan/`
- `TRAINING/orchestration/intelligent_trainer.py`: Update summary save paths to `globals/summaries/`

**Reader Updates**:
- All readers check new locations first, then fall back to old flat structure (backward compatibility)

**Impact**: Better organization, easier navigation, maintains backward compatibility for reading.

---

## Files Changed Summary

### Core Path Infrastructure
- `TRAINING/orchestration/utils/target_first_paths.py` - Added view/symbol-aware path helpers

### Blocking Correctness
- `TRAINING/ranking/shared_ranking_harness.py` - Quarantine unknown lookback features
- `TRAINING/ranking/feature_selector.py` - Verified cohort_metadata initialization (already fixed)

### Data Integrity
- `TRAINING/ranking/feature_selection_reporting.py` - View/symbol-scoped reproducibility paths, feature importances paths, aggregation helper
- `TRAINING/ranking/multi_model_feature_selection.py` - View/symbol-scoped reproducibility paths
- `TRAINING/ranking/feature_selector.py` - CROSS_SECTIONAL aggregation for feature importances

### Functional Correctness
- `TRAINING/orchestration/intelligent_trainer.py` - Split model families config, pass fs_families to feature selection, update summary paths
- `TRAINING/orchestration/training_plan_consumer.py` - Fix dev_mode fallback

### Cleanup
- `TRAINING/orchestration/routing_integration.py` - Remove METRICS/ creation, update to globals/routing/
- `TRAINING/orchestration/metrics_aggregator.py` - Remove METRICS/ creation, update to globals/routing/
- `TRAINING/orchestration/training_router.py` - Update to globals/routing/routing_plan/
- `TRAINING/orchestration/training_plan_generator.py` - Update to globals/training/training_plan/

## Testing Recommendations

1. **Unknown Lookback Quarantine**: Run with features that have unknown lookback (e.g., `vwap_dev_high`, `vwap_dev_low`) - should quarantine and proceed without RuntimeError
2. **Reproducibility Scoping**: Run SYMBOL_SPECIFIC with 2+ symbols - verify both `SYMBOL_SPECIFIC/symbol={symbol}/` directories exist, files differ, no clobber
3. **CROSS_SECTIONAL Importances**: Force fallback per-symbol path for CROSS_SECTIONAL view - verify `save_feature_importances_for_reproducibility()` called once, output file created once in `CROSS_SECTIONAL/feature_importances/`
4. **Config Families Routing**: Set `training.model_families=["CatBoost"]`, `feature_selection.model_families=["LightGBM"]` - verify training uses CatBoost, FS uses LightGBM
5. **No METRICS + No training_results/targets**: Run minimal orchestration - verify `METRICS/` and `training_results/targets/` directories do not exist
6. **Globals Subfolders**: Verify `globals/routing/`, `globals/training/`, `globals/summaries/` exist with expected files

## Rollback Instructions

If rollback is needed:

```bash
# Check current branch and commit
git status
git log --oneline -1

# Rollback all changes
git reset --hard HEAD~1  # If changes are in single commit
# OR
git checkout <previous-commit-sha> -- TRAINING/orchestration/utils/target_first_paths.py TRAINING/ranking/shared_ranking_harness.py TRAINING/ranking/feature_selection_reporting.py TRAINING/ranking/multi_model_feature_selection.py TRAINING/ranking/feature_selector.py TRAINING/orchestration/intelligent_trainer.py TRAINING/orchestration/training_plan_consumer.py TRAINING/orchestration/routing_integration.py TRAINING/orchestration/metrics_aggregator.py TRAINING/orchestration/training_router.py TRAINING/orchestration/training_plan_generator.py
```

## Related Changes

- [2025-12-22-boruta-performance-optimizations.md](2025-12-22-boruta-performance-optimizations.md) - Boruta optimizations
- [2025-12-22-trend-analyzer-operator-precedence-fix.md](2025-12-22-trend-analyzer-operator-precedence-fix.md) - Trend analyzer path detection fix
- [2025-12-19-target-first-structure-migration.md](2025-12-19-target-first-structure-migration.md) - Target-first structure migration

