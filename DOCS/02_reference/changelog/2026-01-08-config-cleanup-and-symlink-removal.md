# 2026-01-08: CONFIG Cleanup and Symlink Removal

## Summary

Removed all symlinks from CONFIG directory and updated all code to use canonical paths directly. This simplifies the codebase, makes the CONFIG folder easier to navigate, and ensures all configurable settings are fully accessible via config files.

## Changes

### CONFIG Structure

- **Removed all symlinks** (23 total):
  - 6 root-level symlinks (`excluded_features.yaml`, `feature_registry.yaml`, `feature_groups.yaml`, `feature_target_schema.yaml`, `logging_config.yaml`, `target_configs.yaml`)
  - 17 symlinks in `training_config/` directory (entire directory removed)
  - Legacy directory symlinks (`feature_selection/`, `target_ranking/`)
- **All paths are now canonical** - No symlinks remain in CONFIG directory

### Config Loader Updates

**File**: `CONFIG/config_loader.py`

- **`get_config_path()`**: 
  - Removed `old_path` fallback logic from all mappings
  - Now returns canonical paths directly
  - Simplified mapping structure (single path per config name)
- **`load_training_config()`**:
  - Removed fallback to `training_config/` directory
  - Only checks canonical `pipeline/training/` or `pipeline/` locations
  - Handles `system_config` specially (goes to `core/system.yaml`)
- **`list_available_configs()`**:
  - Removed check for `training_config/` directory
  - Lists from canonical `pipeline/training/` and `pipeline/` locations
  - Still checks both `models/` and `model_config/` for backward compatibility

### Code Updates

**File**: `TRAINING/orchestration/intelligent_trainer.py`

- Updated help text references from `CONFIG/training_config/intelligent_training_config.yaml` to `CONFIG/pipeline/training/intelligent.yaml`
- Updated fallback paths to use canonical locations only
- Removed references to `training_config/` directory

**File**: `CONFIG/config_builder.py`

- Removed fallback logic that checked `training_config/` or `feature_selection/` directories
- Updated to use canonical paths directly (`pipeline/pipeline.yaml`, `ranking/features/multi_model.yaml`)

### Documentation Updates

**File**: `CONFIG/README.md`

- Removed "Symlinks (Legacy Compatibility)" section
- Added "Path Resolution" section explaining canonical paths
- Updated "Recent Changes" to note symlink removal
- Updated changelog entries

**Files**: `DOCS/01_tutorials/SIMPLE_PIPELINE_USAGE.md`, `DOCS/01_tutorials/configuration/EXPERIMENT_CONFIG_GUIDE.md`, `DOCS/01_tutorials/configuration/CONFIG_BASICS.md`

- Updated all references from `CONFIG/training_config/intelligent_training_config.yaml` to `CONFIG/pipeline/training/intelligent.yaml`
- Updated references from `CONFIG/logging_config.yaml` to `CONFIG/core/logging.yaml`

## Impact Assessment

### ✅ No Breaking Changes

- **Run hash ID**: Unchanged - `config_fingerprint` is computed from resolved config content (canonical JSON), not file paths
- **Config tracking**: Unchanged - Resolved config contains actual config values, not paths
- **Snapshotting**: Unchanged - Snapshots store fingerprints and signatures, not config file paths
- **Metadata files**: Unchanged - Metadata doesn't store config file paths

### ✅ Improved Maintainability

- **Easier navigation**: CONFIG folder structure is now clear and unambiguous
- **No symlink confusion**: All paths are canonical, no need to resolve symlinks
- **Simpler code**: Removed fallback logic makes config loader easier to understand
- **Better documentation**: Clear canonical paths in all docs

### ✅ Fully Configurable Settings

All configurable settings are now accessible via config files:
- Data limits: `pipeline/pipeline.yaml` → `data_limits.*`
- Determinism: `pipeline/pipeline.yaml` → `determinism.*`
- Safety thresholds: `pipeline/training/safety.yaml` → `safety.*`
- Preprocessing: `pipeline/training/preprocessing.yaml` → `preprocessing.*`
- Strategy configs: `pipeline/training/intelligent.yaml` → `strategy_configs.*`
- Model hyperparameters: `models/*.yaml`

## Migration Guide

### For Users

**No action required** - All code uses the config loader API which handles path resolution automatically.

If you have external scripts that directly reference symlinked paths:
- Update to use canonical paths or the config loader API
- Root-level symlinks: Use `data/`, `core/`, `ranking/` paths directly
- `training_config/` paths: Use `pipeline/training/` or `pipeline/` paths directly

### For Developers

**Always use the config loader API** instead of hardcoded paths:

```python
# ✅ CORRECT
from CONFIG.config_loader import get_config_path, load_training_config
config_path = get_config_path("intelligent_training_config")
config = load_training_config("intelligent_training_config")

# ❌ DON'T
config_path = Path("CONFIG/training_config/intelligent_training_config.yaml")
```

## Files Modified

1. `CONFIG/config_loader.py` - Removed symlink fallback logic
2. `TRAINING/orchestration/intelligent_trainer.py` - Updated to canonical paths
3. `CONFIG/config_builder.py` - Removed fallback logic
4. `CONFIG/README.md` - Updated documentation
5. `DOCS/01_tutorials/SIMPLE_PIPELINE_USAGE.md` - Updated path references
6. `DOCS/01_tutorials/configuration/EXPERIMENT_CONFIG_GUIDE.md` - Updated path references
7. `DOCS/01_tutorials/configuration/CONFIG_BASICS.md` - Updated path references
8. `CHANGELOG.md` - Added entry for symlink removal

## Files/Directories Removed

1. All root-level symlinks (6 files)
2. `CONFIG/training_config/` directory (entire directory with 17 symlinks)
3. Legacy directory symlinks (if they existed)

## Verification

- ✅ All configs load successfully using canonical paths
- ✅ Config loader functions work correctly
- ✅ All canonical paths resolve correctly
- ✅ No symlinks remain in CONFIG directory
- ✅ Run hash and config tracking unchanged (fingerprints based on content, not paths)
- ✅ Documentation updated with canonical paths
