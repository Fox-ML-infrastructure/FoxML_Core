# 2025-12-20: Untrack DATA_PROCESSING Folder

## Summary

Untracked the `DATA_PROCESSING/` folder from git and updated all dependencies. The TRAINING pipeline is completely independent of DATA_PROCESSING, so no core functionality was affected.

## Changes

### Git Tracking
- **Added to .gitignore**: `DATA_PROCESSING/` folder is now ignored by git
- **Removed from tracking**: 22 files untracked (features, pipeline, targets, utils modules)
- **Documentation removed**: 3 DATA_PROCESSING-specific documentation files removed

### Code Dependencies Fixed
- **`TRAINING/ranking/multi_model_feature_selection.py`** (line 3846):
  - Changed default `--output-dir` from `DATA_PROCESSING/data/features/multi_model` to `RESULTS/features/multi_model`
  - This is only a default for the standalone CLI script - users can override with `--output-dir` flag
  - Core training pipeline does not use this script directly

- **`CONFIG/ranking/features/config.yaml`** (line 7):
  - Updated `output_dir` from `DATA_PROCESSING/data/features` to `RESULTS/features`
  - Note: This config file is not actually loaded by the core TRAINING pipeline (verified via codebase search)

### Documentation Cleanup
- **Removed files**:
  - `DOCS/01_tutorials/pipelines/DATA_PROCESSING_README.md`
  - `DOCS/01_tutorials/pipelines/DATA_PROCESSING_WALKTHROUGH.md`
  - `DOCS/02_reference/api/DATA_PROCESSING_API.md`
- **Updated `DOCS/INDEX.md`**: Removed references to removed documentation files

## Verification

### Dependency Analysis
- ✅ **No Python imports**: Verified no code imports from DATA_PROCESSING modules
- ✅ **No runtime dependencies**: Core TRAINING pipeline does not call DATA_PROCESSING code
- ✅ **No data dependencies**: TRAINING pipeline does not read data files from DATA_PROCESSING
- ✅ **Config independence**: `CONFIG/ranking/features/config.yaml` is not loaded by core pipeline

### Impact Assessment
- **Core pipeline**: ✅ No impact - TRAINING is completely independent
- **Standalone scripts**: ⚠️ Minor - default output path changed (users can override)
- **Documentation**: ✅ Cleaned up - removed outdated DATA_PROCESSING-specific docs

## Files Changed

### Modified
- `.gitignore` - Added `DATA_PROCESSING/`
- `TRAINING/ranking/multi_model_feature_selection.py` - Updated default output path
- `CONFIG/ranking/features/config.yaml` - Updated output_dir reference
- `DOCS/INDEX.md` - Removed DATA_PROCESSING doc references

### Deleted (Untracked)
- `DATA_PROCESSING/features/` (5 files)
- `DATA_PROCESSING/pipeline/` (5 files)
- `DATA_PROCESSING/targets/` (6 files)
- `DATA_PROCESSING/utils/` (6 files)
- `DOCS/01_tutorials/pipelines/DATA_PROCESSING_README.md`
- `DOCS/01_tutorials/pipelines/DATA_PROCESSING_WALKTHROUGH.md`
- `DOCS/02_reference/api/DATA_PROCESSING_API.md`

**Total**: 29 files changed (4 modified, 25 deleted)

## Notes

- The `DATA_PROCESSING/` folder remains on disk but is now ignored by git
- Historical references in changelogs and some tutorials remain (documenting past functionality)
- All active dependencies have been updated to use `RESULTS/` instead of `DATA_PROCESSING/`

