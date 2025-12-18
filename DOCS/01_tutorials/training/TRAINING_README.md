# TRAINING Module

Core training infrastructure for FoxML Core.

## Documentation

**All training documentation is in the `DOCS/` folder.**

See the [Training Documentation](../../INDEX.md#tier-b-tutorials--walkthroughs) for complete guides.

## Quick Links

- [Intelligent Training Tutorial](INTELLIGENT_TRAINING_TUTORIAL.md) - Automated target ranking and feature selection
- [Model Training Guide](MODEL_TRAINING_GUIDE.md) - Manual training workflow
- [Feature Selection Tutorial](FEATURE_SELECTION_TUTORIAL.md) - Feature selection workflow
- [Walk-Forward Validation](WALKFORWARD_VALIDATION.md) - Validation workflow
- [Experiments Workflow](../../LEGACY/EXPERIMENTS_WORKFLOW.md) - 3-phase training workflow (Legacy)
- [Training Optimization](../../03_technical/implementation/TRAINING_OPTIMIZATION_GUIDE.md) - Optimization guide
- [Feature Selection Implementation](../../03_technical/implementation/FEATURE_SELECTION_GUIDE.md) - Implementation details

## Directory Structure

```
TRAINING/
├── train.py                    # Primary entry point (delegates to intelligent_trainer)
├── train_with_strategies.py    # Backward compat wrapper
│
├── orchestration/              # Intelligent training pipeline
│   ├── intelligent_trainer/    # Main orchestrator
│   ├── interfaces/             # Unified training interface
│   ├── routing/                # Target routing logic
│   └── utils/                  # Orchestration utilities
│
├── ranking/                    # Target ranking system
│   ├── predictability/         # Target predictability ranking (modular)
│   ├── multi_model_feature_selection/  # Feature selection
│   └── utils/                  # Ranking-specific utilities
│
├── training_strategies/         # Training strategies (merged from strategies/)
│   ├── strategies/              # Strategy classes (single-task, multi-task, cascade)
│   ├── execution/               # Execution code (main, training, data prep)
│   └── utils/                   # Strategy utilities
│
├── data/                       # Consolidated data handling
│   ├── loading/                 # Data loading (from data_processing)
│   ├── preprocessing/           # Preprocessing pipelines
│   ├── processing/              # Data transformations
│   ├── features/                # Feature engineering
│   └── datasets/                # Dataset classes
│
├── common/                     # Shared utilities
│   ├── core/                    # Core functionality (reproducibility, environment)
│   ├── memory/                  # Memory management
│   ├── live/                    # Live trading utilities
│   └── utils/                   # Common utilities
│
├── model_fun/                  # Model trainers (34 files)
├── models/                     # Model wrappers and registry (18 files)
├── decisioning/                # Decision engine (4 files)
├── stability/                  # Stability analysis (6 files)
├── tools/                      # Development tools (7 files)
├── tests/                      # Tests (7 files)
│
└── archive/                     # Historical files
```

## Entry Points

- **`train.py`** - Primary entry point, delegates to `IntelligentTrainer`
- **`train_with_strategies.py`** - Backward compatibility wrapper for training strategies
- **`orchestration/interfaces/unified_training_interface.py`** - Alternative unified interface

## Refactoring History

### 2025-12-18: TRAINING Folder Reorganization

- **Directory Consolidation**: 
  - Moved `features/` → `data/features/`
  - Moved `datasets/` → `data/datasets/`
  - Moved `memory/` → `common/memory/`
  - Moved `live/` → `common/live/`
  - Moved `core/` → `common/core/`
- **Overlapping Directory Merges**:
  - Merged `strategies/` → `training_strategies/strategies/`
  - Consolidated `data_processing/`, `preprocessing/`, `processing/` → `data/` (with subdirectories)
- **Entry Point Reorganization**:
  - Moved `unified_training_interface.py` → `orchestration/interfaces/`
  - Moved `target_router.py` → `orchestration/routing/`
- **Output Directory Cleanup**:
  - Moved `modular_output/` → `RESULTS/modular_output/`
  - Moved `results/` → `RESULTS/training_results/`
  - Archived `EXPERIMENTS/` → `archive/experiments/`
- **Config Loader Fixes**: Fixed misleading "Config loader not available" warnings (changed to debug level)
- **Backward Compatibility**: All old import paths maintained via `__init__.py` re-export wrappers
- **Import Status**: 100% of key imports passing
- See **[Detailed Changelog](../../02_reference/changelog/2025-12-18-training-folder-reorganization.md)** for complete details

### 2025-12-18: Code Modularization & Utils Reorganization

- **`TRAINING/utils/` reorganized** into domain-specific subdirectories:
  - `ranking/utils/` - Ranking-specific utilities (24 files)
  - `orchestration/utils/` - Orchestration utilities (8 files)  
  - `common/utils/` - Shared/common utilities (16 files)
- **Large files split** into modular components:
  - `reproducibility_tracker.py` → `reproducibility/` folder
  - `diff_telemetry.py` → `diff_telemetry/` folder
  - `multi_model_feature_selection.py` → `multi_model_feature_selection/` folder
  - `intelligent_trainer.py` → `intelligent_trainer/` folder
  - `leakage_detection.py` → `leakage_detection/` folder
- **Backward compatibility maintained** via `03_technical/implementation/training_utils/INTERNAL/)** for complete details

### 2025-12-09: Initial Large File Splits

- **`models/specialized_models.py`**: 4,518 → 82 lines (split into `models/specialized/`)
- **`ranking/rank_target_predictability.py`**: 3,454 → 56 lines (split into `ranking/predictability/`)
- **`train_with_strategies.py`**: 2,523 → 66 lines (split into `training_strategies/`)

**For detailed refactoring documentation, see:**
- **[Refactoring & Wrappers Guide](../../01_tutorials/REFACTORING_AND_WRAPPERS.md)** - User-facing guide explaining wrappers and import patterns
- **[Refactoring Summary](../../INTERNAL/REFACTORING_SUMMARY_INTERNAL.md)** - Internal technical details
- **[Module-Specific Docs](../../03_technical/refactoring/)** - Detailed structure for each refactored module

### Key Points

- ✅ **100% backward compatible** - All existing imports work unchanged
- ✅ **Original files preserved** in `TRAINING/archive/original_large_files/` (untracked)
- ✅ **Largest file now**: 2,542 lines (cohesive subsystem, not monolithic)
- ✅ **Most files**: 500-1,400 lines (focused responsibilities)

For detailed documentation on each component, see the [Training Documentation](../../INDEX.md#tier-b-tutorials--walkthroughs).
