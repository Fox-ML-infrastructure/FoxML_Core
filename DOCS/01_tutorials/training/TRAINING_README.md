# TRAINING Module

Core training infrastructure for FoxML Core.

## Documentation

**All training documentation has been moved to the `DOCS/` folder for better organization.**

See the [Training Documentation](../../DOCS/INDEX.md#tier-b-tutorials--walkthroughs) for complete guides.

## Quick Links

- [Intelligent Training Tutorial](../../DOCS/01_tutorials/training/INTELLIGENT_TRAINING_TUTORIAL.md) - Automated target ranking and feature selection
- [Model Training Guide](../../DOCS/01_tutorials/training/MODEL_TRAINING_GUIDE.md) - Manual training workflow
- [Feature Selection Tutorial](../../DOCS/01_tutorials/training/FEATURE_SELECTION_TUTORIAL.md) - Feature selection workflow
- [Walk-Forward Validation](../../DOCS/01_tutorials/training/WALKFORWARD_VALIDATION.md) - Validation workflow
- [Experiments Workflow](../../DOCS/01_tutorials/training/EXPERIMENTS_WORKFLOW.md) - 3-phase training workflow
- [Training Optimization](../../DOCS/03_technical/implementation/TRAINING_OPTIMIZATION_GUIDE.md) - Optimization guide
- [Feature Selection Implementation](../../DOCS/03_technical/implementation/FEATURE_SELECTION_GUIDE.md) - Implementation details

## Directory Structure

```
TRAINING/
├── orchestration/          # Intelligent training pipeline
├── ranking/                # Target ranking system
│   ├── predictability/     # Target predictability ranking (modular)
│   └── rank_target_predictability.py  # Backward-compat wrapper
├── model_fun/              # Model trainers (17 models)
├── models/                 # Model wrappers and registry
│   └── specialized/       # Specialized model implementations (modular)
│       └── specialized_models.py  # Backward-compat wrapper
├── training_strategies/    # Training strategies (modular)
│   └── train_with_strategies.py  # Backward-compat wrapper
├── strategies/             # Training strategies
├── utils/                  # Utilities (backward-compat re-exports)
│   ├── ranking/utils/      # Ranking-specific utilities (24 files)
│   ├── orchestration/utils/ # Orchestration utilities (8 files)
│   └── common/utils/       # Shared/common utilities (16 files)
├── common/                 # Common utilities (safety, threading, etc.)
├── preprocessing/          # Data preprocessing
├── processing/            # Data processing
├── features/               # Feature engineering
├── datasets/               # Dataset classes
├── EXPERIMENTS/            # 3-phase training workflow
└── train.py                # Main training entry point
```

## Refactoring History

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
- **Backward compatibility maintained** via `TRAINING/utils/__init__.py` re-exports
- See **[Detailed Changelog](../../02_reference/changelog/2025-12-18-code-modularization.md)** for complete details

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

For detailed documentation on each component, see the [Training Documentation](../../DOCS/INDEX.md#tier-b-tutorials--walkthroughs).
