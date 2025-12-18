# Changelog — 2025-12-18

**Code Modularization & Large File Refactoring**

For a quick overview, see the [root changelog](../../../CHANGELOG.md).  
For other dates, see the [changelog index](README.md).

---

## Added

### Large File Modularization

**Major Code Reorganization - Split 7 Large Files into Modular Components**

- **Enhancement**: Refactored large monolithic files (2,000-6,800 lines) into smaller, maintainable modules
- **Total Impact**: ~2,000+ lines extracted, 23 new utility/module files created, 103 files changed

#### Model Evaluation Modularization
- **Split**: `TRAINING/ranking/predictability/model_evaluation.py` (6,801 lines → ~3,800 lines)
- **New Modules**:
  - `model_evaluation/config_helpers.py` - Configuration loading and processing
  - `model_evaluation/leakage_helpers.py` - Leakage detection and suspicion scoring
  - `model_evaluation/reporting.py` - Logging and file I/O utilities
- **Benefits**: Improved maintainability, clearer separation of concerns, easier testing

#### Reproducibility Tracker Modularization
- **Split**: `TRAINING/orchestration/utils/reproducibility_tracker.py` (4,187 lines → ~3,800 lines)
- **New Modules**:
  - `reproducibility/utils.py` - Environment info, tagged unions, logging helpers
  - `reproducibility/config_loader.py` - Configuration loading utilities
- **Benefits**: Centralized utility functions, cleaner main file, better organization

#### Diff Telemetry Modularization
- **Split**: `TRAINING/orchestration/utils/diff_telemetry.py` (3,858 lines → ~3,400 lines)
- **New Modules**:
  - `diff_telemetry/types.py` - All dataclasses and enums (ChangeSeverity, ComparabilityStatus, etc.)
- **Benefits**: Cleaner type definitions, easier to maintain and extend

#### Multi-Model Feature Selection Modularization
- **Split**: `TRAINING/ranking/multi_model_feature_selection.py` (3,769 lines → ~3,400 lines)
- **New Modules**:
  - `multi_model_feature_selection/types.py` - ModelFamilyConfig, ImportanceResult dataclasses
  - `multi_model_feature_selection/config_loader.py` - Configuration loading with deprecated path handling
  - `multi_model_feature_selection/importance_extractors.py` - Native, SHAP, and permutation importance extraction
- **Benefits**: Clear separation of types, config, and extraction logic

#### Intelligent Trainer Modularization
- **Split**: `TRAINING/orchestration/intelligent_trainer.py` (2,778 lines → ~2,600 lines)
- **New Modules**:
  - `intelligent_trainer/utils.py` - JSON serialization, sample size binning, cohort organization
- **Benefits**: Extracted utility functions, cleaner main orchestrator

#### Leakage Detection Modularization
- **Split**: `TRAINING/ranking/predictability/leakage_detection.py` (2,163 lines → ~2,000 lines)
- **New Modules**:
  - `leakage_detection/feature_analysis.py` - Feature analysis and leak detection
  - `leakage_detection/reporting.py` - Feature importance saving and suspicious feature logging
- **Benefits**: Separated analysis from reporting, better organization

### Common Utilities Centralization

**New Centralized Utility Modules**

- **`TRAINING/common/utils/file_utils.py`** - Atomic JSON file writing operations
  - Consolidates duplicated `_write_atomic_json` logic
  - Ensures data integrity even in case of crashes

- **`TRAINING/common/utils/cache_manager.py`** - Unified cache loading and saving
  - Provides consistent interface for caching
  - Reduces boilerplate across codebase

- **`TRAINING/common/utils/config_hashing.py`** - Deterministic configuration hashing
  - Centralizes config hash computation
  - Used for cache keys and reproducibility tracking

- **`TRAINING/common/utils/process_cleanup.py`** - Process cleanup utilities
  - Centralizes `loky` executor shutdown logic
  - Prevents resource leaks

- **`TRAINING/common/utils/path_setup.py`** - Project path setup
  - Standardizes `sys.path` and `PYTHONPATH` setup
  - Ensures consistent execution environment

- **`TRAINING/common/family_constants.py`** - Model family classifications
  - Centralizes `TF_FAMS`, `TORCH_FAMS`, `CPU_FAMS` definitions
  - Eliminates duplication across 7+ files

### Utils Folder Reorganization

**Major Utils Directory Restructure**

- **Reorganized**: `TRAINING/utils/` → domain-specific subdirectories
- **New Structure**:
  ```
  TRAINING/
  ├── ranking/utils/          # Ranking-specific utilities (24 files)
  ├── orchestration/utils/    # Orchestration utilities (8 files)
  └── common/utils/           # Shared/common utilities (16 files)
  ```
- **Backward Compatibility**: `TRAINING/utils/__init__.py` provides re-exports
- **Benefits**: Clear organization, easier to find utilities, better maintainability

---

## Changed

### Import Path Updates

- **Updated**: 25+ files to use new import paths
- **Files Affected**:
  - All `training_strategies/*.py` files (7 files)
  - Multiple ranking and orchestration modules
  - Updated to use centralized utilities

### Backward Compatibility

- **Maintained**: Full backward compatibility via `TRAINING/utils/__init__.py`
- **Re-exports**: All moved utilities still accessible from old import paths
- **Migration**: New code should use direct imports, old code continues to work

---

## Fixed

### Import Errors

- **Fixed**: All import errors from refactoring
  - Added missing `Path` import in `importance_extractors.py`
  - Fixed `_get_main_logger` alias in `reproducibility/utils.py`
  - Fixed `leakage_budget` module import in backward compatibility layer
  - Fixed circular import issues in modularized packages

### Module Exports

- **Fixed**: Missing exports in `__init__.py` files
  - Added exports for functions still in parent files (using `importlib`)
  - Fixed `detect_leakage`, `train_and_evaluate_models`, `process_single_symbol` exports
  - Ensured all refactored modules properly export their components

---

## Technical Details

### Files Changed
- **103 files changed**: 4,240 insertions(+), 1,772 deletions(-)
- **23 new files created**: Utility modules and subdirectories
- **48 files moved**: Reorganized into domain-specific directories

### Module Structure Created
```
TRAINING/
  common/
    utils/          # 6 new files (file_utils, cache_manager, config_hashing, etc.)
    family_constants.py
  ranking/
    predictability/
      model_evaluation/     # 3 new files
      leakage_detection/    # 2 new files
    multi_model_feature_selection/  # 3 new files
  orchestration/
    utils/
      reproducibility/      # 2 new files
      diff_telemetry/       # 1 new file
    intelligent_trainer/   # 1 new file
```

### Testing Status
- ✅ All imports verified and working
- ✅ Backward compatibility maintained
- ✅ Ready for integration testing

---

## Migration Notes

### For Developers

**New Import Paths (Recommended)**:
```python
# Old (still works via backward compatibility)
from TRAINING.utils import cache_manager

# New (recommended)
from TRAINING.common.utils.cache_manager import CacheManager
```

**Module-Specific Imports**:
```python
# Model evaluation utilities
from TRAINING.ranking.predictability.model_evaluation.config_helpers import get_importance_top_fraction

# Reproducibility utilities
from TRAINING.orchestration.utils.reproducibility.utils import collect_environment_info

# Multi-model feature selection
from TRAINING.ranking.multi_model_feature_selection.types import ModelFamilyConfig
```

### Backward Compatibility

- All old import paths continue to work
- No breaking changes for existing code
- Gradual migration recommended for new code

---

## Related

- **Branch**: `cleanup/phase2-bypass-and-utils-reorg`
- **Commit**: `d86f117` - "refactor: Split large files into modular components and fix import errors"
- **Testing**: Ready for integration testing with `e2e_full_targets_test.yaml`

