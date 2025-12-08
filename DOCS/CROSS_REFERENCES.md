# Documentation Cross-References

This document tracks key cross-references between documentation files to ensure consistency.

## New Files (2025-12-08)

### Ranking and Selection Consistency
- **File**: `DOCS/01_tutorials/training/RANKING_SELECTION_CONSISTENCY.md`
- **References**:
  - `INTELLIGENT_TRAINING_TUTORIAL.md` - Main pipeline guide
  - `MODULAR_CONFIG_SYSTEM.md` - Config structure
  - `USAGE_EXAMPLES.md` - Practical examples
  - `CONFIG_LOADER_API.md` - Logging config utilities
  - `MODULE_REFERENCE.md` - Utility API reference
  - `README.md` (configuration) - Config overview

### Logging Configuration
- **File**: `CONFIG/logging_config.yaml`
- **Documented in**:
  - `MODULAR_CONFIG_SYSTEM.md` - Section 5
  - `CONFIG/README.md` - Directory structure
  - `README.md` (configuration) - Config file list
  - `CONFIG_LOADER_API.md` - API functions
  - `CONFIG_BASICS.md` - Example structure

### Utility Modules
- **Files**: 
  - `TRAINING/utils/target_utils.py`
  - `TRAINING/utils/sklearn_safe.py`
- **Documented in**:
  - `RANKING_SELECTION_CONSISTENCY.md` - Usage guide
  - `MODULE_REFERENCE.md` - API reference

## Key Cross-Reference Patterns

### Configuration Files
All config files should reference:
- `MODULAR_CONFIG_SYSTEM.md` - Main config guide
- `README.md` (configuration) - Overview
- `USAGE_EXAMPLES.md` - Practical examples

### Training Pipeline
All training docs should reference:
- `INTELLIGENT_TRAINING_TUTORIAL.md` - Main tutorial
- `RANKING_SELECTION_CONSISTENCY.md` - Pipeline behavior
- `MODULAR_CONFIG_SYSTEM.md` - Config system

### API References
All API docs should reference:
- `MODULE_REFERENCE.md` - Python API
- `INTELLIGENT_TRAINER_API.md` - Trainer API
- `CONFIG_LOADER_API.md` - Config loading

## Broken References Fixed

- Removed reference to non-existent `COMPREHENSIVE_FEATURE_RANKING.md` in `FEATURE_IMPORTANCE_METHODOLOGY.md`
- Updated all references to point to newer unified pipeline docs

## Preferred Documentation Order

When multiple docs cover similar topics, prefer:
1. **Newer unified docs** (RANKING_SELECTION_CONSISTENCY.md) over older scattered docs
2. **Modular config system** docs over legacy config references
3. **Intelligent training tutorial** over manual workflow docs
4. **Usage examples** with practical code over abstract descriptions
