# Feature Selection Directory Structure Refactoring (2025-12-16)

## Summary

Reorganized feature selection output structure to match target ranking layout, eliminating scattered files and nested REPRODUCIBILITY directories.

## Motivation

Feature selection outputs were scattered in the base target directory, making navigation difficult and inconsistent with target ranking structure. Files were written directly to:
- `REPRODUCIBILITY/FEATURE_SELECTION/CROSS_SECTIONAL/{target}/` (scattered CSV/JSON files)
- `REPRODUCIBILITY/FEATURE_SELECTION/REPRODUCIBILITY/FEATURE_SELECTION/` (nested structure issue)

## Changes

### New Structure (Matches Target Ranking)

```
REPRODUCIBILITY/FEATURE_SELECTION/CROSS_SECTIONAL/{target}/
├── feature_importances/          # All CSV importance files
│   ├── {model}_importances.csv   # catboost, lightgbm, xgboost, random_forest, etc.
│   ├── feature_importance_multi_model.csv
│   ├── feature_importance_with_boruta_debug.csv
│   └── model_agreement_matrix.csv
├── metadata/                      # All JSON metadata files
│   ├── target_confidence.json
│   ├── target_routing.json
│   ├── cross_sectional_stability_metadata.json
│   ├── multi_model_metadata.json
│   └── model_family_status.json
├── artifacts/                     # Artifacts
│   └── selected_features.txt
├── DECISION/                      # Decision logs
└── REPRODUCIBILITY/               # Reproducibility artifacts
```

### File Organization

**Before:**
- All files scattered in base directory
- Mixed CSV and JSON files
- Nested REPRODUCIBILITY structure

**After:**
- `feature_importances/` - All CSV importance files (matches target ranking)
- `metadata/` - All JSON metadata files
- `artifacts/` - Selected features list
- Clean, organized structure matching target ranking

### Fixed Issues

1. **Nested REPRODUCIBILITY Structure**
   - Fixed path resolution in `save_feature_importances_for_reproducibility()`
   - Fixed snapshot saving logic in `feature_selector.py`
   - Prevents `REPRODUCIBILITY/FEATURE_SELECTION/REPRODUCIBILITY/FEATURE_SELECTION/` nesting

2. **File Naming Consistency**
   - Importance files now use `{model}_importances.csv` format (matches target ranking)
   - Previously used `importance_{model}.csv` format

3. **Backward Compatibility**
   - Load functions check new `metadata/` location first
   - Fallback to old location for existing runs
   - Ensures existing code continues to work

## Files Modified

- `TRAINING/ranking/multi_model_feature_selection.py`
  - `save_multi_model_results()` - Organized files into subdirectories
  - Updated file paths and logging

- `TRAINING/ranking/feature_selector.py`
  - Fixed nested REPRODUCIBILITY path resolution
  - Updated cross-sectional stability metadata location

- `TRAINING/ranking/feature_selection_reporting.py`
  - Fixed `save_feature_importances_for_reproducibility()` path resolution

- `TRAINING/orchestration/target_routing.py`
  - `save_target_routing_metadata()` - Writes to `metadata/` subdirectory

- `TRAINING/orchestration/routing_candidates.py`
  - Updated load functions to check new locations with backward compatibility

## Benefits

1. **Consistency**: Feature selection structure now matches target ranking exactly
2. **Organization**: Files grouped by type (importances, metadata, artifacts)
3. **Navigation**: Easier to find specific file types
4. **Maintainability**: Clear structure reduces confusion
5. **Backward Compatible**: Existing code continues to work

## Migration

No migration needed - new runs will use the organized structure automatically. Existing runs remain accessible via backward-compatible load functions.

## Related

- Target ranking structure: `REPRODUCIBILITY/TARGET_RANKING/CROSS_SECTIONAL/{target}/`
- See `DOCS/03_technical/implementation/REPRODUCIBILITY_STRUCTURE.md` for complete structure documentation

