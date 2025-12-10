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
├── utils/                  # Utilities (data loading, leakage filtering, etc.)
├── common/                 # Common utilities (safety, threading, etc.)
├── preprocessing/          # Data preprocessing
├── processing/            # Data processing
├── features/               # Feature engineering
├── datasets/               # Dataset classes
├── EXPERIMENTS/            # 3-phase training workflow
└── train.py                # Main training entry point
```

## Refactoring (2025-12-09)

Large monolithic files have been split into modular components:

- **`models/specialized_models.py`**: 4,518 → 82 lines (split into `models/specialized/`)
- **`ranking/rank_target_predictability.py`**: 3,454 → 56 lines (split into `ranking/predictability/`)
- **`train_with_strategies.py`**: 2,523 → 66 lines (split into `training_strategies/`)

**For detailed refactoring documentation, see:** [DOCS/03_technical/refactoring/](../../DOCS/03_technical/refactoring/)

### Key Points

- ✅ **100% backward compatible** - All existing imports work unchanged
- ✅ **Original files preserved** in `TRAINING/archive/original_large_files/` (untracked)
- ✅ **Largest file now**: 2,542 lines (cohesive subsystem, not monolithic)
- ✅ **Most files**: 500-1,400 lines (focused responsibilities)

For detailed documentation on each component, see the [Training Documentation](../../DOCS/INDEX.md#tier-b-tutorials--walkthroughs).
