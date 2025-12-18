# Config Path Consolidation Status

## Overview

After migrating config files to the new structure, we need to update all code references to use the new paths or the config loaders (which already check both old and new locations).

## Status

### ✅ COMPLETE (2025-12-18): All Hardcoded Paths Migrated

**All critical code paths now use config loader API:**

- ✅ `CONFIG/config_loader.py` - Enhanced with `get_experiment_config_path()` and `load_experiment_config()`
- ✅ `CONFIG/config_builder.py` - Updated to check new locations first
- ✅ `TRAINING/orchestration/intelligent_trainer.py` - All 13 hardcoded paths replaced with config loader API
- ✅ `TRAINING/ranking/predictability/model_evaluation.py` - All hardcoded paths replaced
- ✅ `TRAINING/ranking/feature_selector.py` - All hardcoded paths replaced
- ✅ `TRAINING/ranking/target_ranker.py` - All hardcoded paths replaced
- ✅ `TRAINING/ranking/multi_model_feature_selection.py` - Uses centralized config loader
- ✅ `TRAINING/ranking/utils/leakage_filtering.py` - Enhanced to use `get_config_path()`

**Validation:**
- ✅ Created `CONFIG/tools/validate_config_paths.py` to scan for remaining hardcoded paths
- ✅ All active code paths use config loader API
- ✅ Remaining hardcoded paths are only in fallback code (when loader unavailable)

### ⚠️ Needs Update (Comments/Log Messages Only - Low Priority)

These are mostly in comments or log messages, not actual code paths:

- Model trainers (`TRAINING/model_fun/*.py`) - Log messages reference `model_config/` but code uses `load_model_config()` which already checks new locations
- Documentation strings - Reference old paths in help text
- Error messages - Reference old paths in user-facing messages

### 📝 Action Items

1. **High Priority** (Actual code paths):
   - ✅ Done - All critical loaders updated

2. **Medium Priority** (User-facing messages):
   - Update help text in argument parsers
   - Update error messages to reference new paths
   - Update documentation strings

3. **Low Priority** (Comments):
   - Update inline comments that reference old paths
   - Update README files in subdirectories

## Helper Functions

### Config Path Resolution

Use `CONFIG.config_loader.get_config_path(config_name)` to get the correct path for any config file. It automatically checks new locations first, then falls back to old.

```python
from CONFIG.config_loader import get_config_path

excluded_path = get_config_path("excluded_features")
# Returns: CONFIG/data/excluded_features.yaml (if exists)
#          or CONFIG/excluded_features.yaml (fallback)
```

### Experiment Config Helpers

**New functions (2025-12-18):**

```python
from CONFIG.config_loader import get_experiment_config_path, load_experiment_config

# Get experiment config path
exp_path = get_experiment_config_path("my_experiment")
# Returns: CONFIG_DIR / "experiments" / "my_experiment.yaml"

# Load experiment config (with proper precedence)
exp_config = load_experiment_config("my_experiment")
# Returns: Dict with experiment configuration
# Note: Experiment configs override intelligent_training_config and defaults
```

## Migration Checklist

- [x] Update config loaders to check new locations
- [x] Update critical code paths (leakage_filtering, intelligent_trainer, ranking)
- [x] Replace all hardcoded `Path("CONFIG/...")` patterns with config loader API
- [x] Add helper functions for experiment configs
- [x] Create validation script (`validate_config_paths.py`)
- [x] Verify SST compliance (defaults injection)
- [ ] Update help text and error messages (optional, low priority)
- [ ] Update comments and documentation (optional, low priority)

## Notes

- **Symlinks maintain backward compatibility** - Old paths still work
- **Config loaders check both locations** - Code using loaders is already compatible
- **Direct path references** - Only need updating if they bypass the loaders

