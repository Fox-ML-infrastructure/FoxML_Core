# TRAINING Folder Reorganization - 2025-12-18

## Summary

Comprehensive reorganization of the `TRAINING/` folder structure to improve modularity, reduce duplication, and clarify boundaries between components. All changes maintain backward compatibility through re-export wrappers.

## Major Changes

### Phase 1: Small Directory Consolidation

**Moved:**
- `TRAINING/features/` → `TRAINING/data/features/`
- `TRAINING/datasets/` → `TRAINING/data/datasets/`
- `TRAINING/memory/` → `TRAINING/common/memory/`
- `TRAINING/live/` → `TRAINING/common/live/`

**Backward Compatibility:**
- Created `__init__.py` wrappers in old locations that re-export from new locations
- All existing imports continue to work

### Phase 2: Overlapping Directory Merges

**Training Strategies:**
- Merged `TRAINING/strategies/` → `TRAINING/training_strategies/strategies/`
- Reorganized `TRAINING/training_strategies/` with new `execution/` subdirectory:
  - `main.py`, `training.py`, `data_preparation.py`, `family_runners.py`, `setup.py`, `train_sequential.py` → `execution/`
- Renamed `TRAINING/training_strategies/strategies.py` → `strategy_functions.py` (to avoid conflict with `strategies/` directory)

**Data Handling Consolidation:**
- `TRAINING/data_processing/` → `TRAINING/data/loading/`
- `TRAINING/preprocessing/` → `TRAINING/data/preprocessing/`
- `TRAINING/processing/` → `TRAINING/data/processing/`

**Core Utilities:**
- `TRAINING/core/` → `TRAINING/common/core/`

**Backward Compatibility:**
- All old import paths maintained via `__init__.py` wrappers

### Phase 3: Entry Point Reorganization

**Moved:**
- `TRAINING/unified_training_interface.py` → `TRAINING/orchestration/interfaces/unified_training_interface.py`
- `TRAINING/target_router.py` → `TRAINING/orchestration/routing/target_router.py`

**Backward Compatibility:**
- Root-level wrappers re-export from new locations

### Phase 4: Output Directory Cleanup

**Moved:**
- `TRAINING/modular_output/` → `RESULTS/modular_output/`
- `TRAINING/results/` → `RESULTS/training_results/`
- `TRAINING/EXPERIMENTS/` → `TRAINING/archive/experiments/`

## Import Path Updates

### New Import Paths (Recommended)

```python
# Data modules
from TRAINING.data.features import build_sequences_for_symbol
from TRAINING.data.datasets import SeqDataset
from TRAINING.data.loading import load_mtf_data
from TRAINING.data.preprocessing import MegaScriptPreprocessor
from TRAINING.data.processing import cross_sectional

# Common utilities
from TRAINING.common.memory import MemoryManager
from TRAINING.common.live import SeqRingBuffer
from TRAINING.common.core import determinism

# Training strategies
from TRAINING.training_strategies.strategies import SingleTaskStrategy
from TRAINING.training_strategies.strategy_functions import load_mtf_data
from TRAINING.training_strategies.execution import main

# Orchestration
from TRAINING.orchestration.interfaces import UnifiedTrainingInterface
from TRAINING.orchestration.routing import route_target
```

### Backward Compatible Paths (Still Work)

```python
# Old paths still work via re-export wrappers
from TRAINING.features import build_sequences_for_symbol
from TRAINING.datasets import SeqDataset
from TRAINING.memory import MemoryManager
from TRAINING.core import determinism
from TRAINING.strategies import SingleTaskStrategy
from TRAINING.data_processing import load_mtf_data_from_dir
from TRAINING.preprocessing import MegaScriptPreprocessor
from TRAINING.processing import cross_sectional
from TRAINING.unified_training_interface import UnifiedTrainingInterface
from TRAINING.target_router import route_target
```

## Config Loader Fixes

**Issue:** "Config loader not available" warnings appearing during imports.

**Root Cause:** `setup_all_paths()` already adds CONFIG to `sys.path`, but redundant path additions and warning-level logging were causing misleading messages.

**Fixes:**
- Removed redundant `sys.path.insert()` calls (CONFIG already added by `setup_all_paths()`)
- Changed log level from `warning` to `debug` for config loader import failures
- Config loader is now properly available and warnings are suppressed

**Files Updated:**
- All `training_strategies/` files that import `config_loader`
- Changed from `logging.warning()` to `logging.debug()` for import failures

## File Statistics

**Before:**
- 247 Python files, ~88,864 total lines
- Multiple overlapping directories
- Unclear boundaries between components

**After:**
- Same file count (no files deleted, only moved)
- Clear directory structure with single-purpose modules
- Better organization: `data/`, `common/`, `training_strategies/`, `orchestration/`

## New Directory Structure

```
TRAINING/
├── train.py                    # Primary entry point
├── train_with_strategies.py    # Backward compat wrapper
│
├── orchestration/              # Intelligent training pipeline
│   ├── intelligent_trainer/    # Main orchestrator
│   ├── interfaces/             # Unified training interface
│   ├── routing/                # Target routing logic
│   └── utils/                  # Orchestration utilities
│
├── ranking/                    # Target ranking system
│   ├── predictability/
│   ├── multi_model_feature_selection/
│   └── utils/
│
├── training_strategies/         # Training strategies (merged)
│   ├── strategies/              # Strategy classes
│   ├── execution/               # Execution code
│   └── utils/                   # Strategy utilities
│
├── data/                       # Consolidated data handling
│   ├── loading/                 # Data loading
│   ├── preprocessing/           # Preprocessing pipelines
│   ├── processing/              # Data transformations
│   ├── features/                # Feature engineering
│   └── datasets/                # Dataset classes
│
├── common/                     # Shared utilities
│   ├── core/                    # Core functionality
│   ├── memory/                  # Memory management
│   ├── live/                    # Live trading utilities
│   └── utils/                   # Common utilities
│
├── model_fun/                  # Model trainers
├── models/                     # Model wrappers
├── decisioning/                # Decision engine
├── stability/                  # Stability analysis
├── tools/                      # Development tools
└── tests/                      # Tests
```

## Testing

**Import Tests:**
- ✅ 25/25 key imports passing (100%)
- ✅ All backward compatibility wrappers working
- ✅ Config loader available and functional
- ✅ No import errors from reorganization

**Verification:**
- All core functionality imports working
- All backward compatibility imports working
- Config loader warnings resolved
- Empty folders removed

## Migration Notes

**For Developers:**
1. **Preferred**: Use new import paths for better clarity
2. **Compatible**: Old import paths still work but may be deprecated in future
3. **Strategy Functions**: Import from `TRAINING.training_strategies.strategy_functions` (not `strategies.py`)

**Breaking Changes:**
- None - all changes are backward compatible

**Deprecation:**
- Old import paths will continue to work but may show deprecation warnings in future versions

## Benefits

1. **Clearer Boundaries**: Each directory has a single, well-defined purpose
2. **Reduced Duplication**: Consolidated overlapping concerns
3. **Better Discoverability**: Related code grouped together
4. **Easier Maintenance**: Fewer directories to navigate
5. **Backward Compatibility**: All changes maintain existing imports via wrappers

## Related Changes

- See `2025-12-18-code-modularization.md` for earlier large file splits
- This reorganization builds on the modularization work

