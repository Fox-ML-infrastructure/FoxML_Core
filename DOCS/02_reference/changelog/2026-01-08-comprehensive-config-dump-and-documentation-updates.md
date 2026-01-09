# Comprehensive Config Dump and Documentation Updates

**Date**: 2026-01-08  
**Type**: Feature Addition, Documentation Enhancement  
**Impact**: Medium - Improves run reproducibility and user guidance

## Overview

Added comprehensive config dumping to run folders and enhanced documentation with hardware requirements and pipeline capabilities.

## Changes

### Comprehensive Config Dump to Globals Folder

**NEW**: `save_all_configs()` function in `TRAINING/orchestration/utils/manifest.py`
- Copies all YAML config files from `CONFIG/` directory to `globals/configs/` when a run is created
- Preserves original directory structure (core/, data/, pipeline/, ranking/, models/, etc.)
- Creates `INDEX.md` listing all configs organized by category
- Enables easy run recreation without needing access to original CONFIG folder
- Skips archive/ directory and non-config files (.py, .md)
- Integrated into `intelligent_trainer.py` after config resolution

**Benefits:**
- **Run Recreation**: All configs needed to recreate a run are in the run folder
- **Auditability**: Complete config snapshot for each run
- **Debugging**: Easy to see exactly what configs were used
- **Reproducibility**: No need to track down config files from CONFIG folder

**Files Modified:**
- `TRAINING/orchestration/utils/manifest.py` - Added `save_all_configs()` function
- `TRAINING/orchestration/intelligent_trainer.py` - Integrated config dump call

### Documentation Updates

**README.md Updates:**
- Added "System Requirements" section with detailed hardware guidance
- Added CPU recommendations (stable clocks, disable turbo boost, undervolting, newer CPUs, core count, base clock speed)
- Added GPU considerations (VRAM dependency, non-determinism note, strict mode behavior)
- Updated "Quick Overview" to highlight 3-stage pipeline and dual-view support
- Updated "Fingerprinting & Reproducibility" section to clarify 3-stage architecture
- Preserved all easter eggs and humor (OSRS comment, etc.)

**GETTING_STARTED.md Updates:**
- Updated system flow to show 3-stage pipeline (Target Ranking → Feature Selection → Training)
- Added dual-view evaluation mention
- Updated prerequisites with detailed RAM requirements (16-32GB for small experiments, 128GB+ for production)

**SIMPLE_PIPELINE_USAGE.md Updates:**
- Added pipeline overview explaining 3-stage architecture and dual-view support

**Hardware Requirements Details:**
- **Small Experiments**: 16-32GB RAM minimum (laptop-friendly)
- **Production/Ideal**: 128GB+ minimum, 512GB-1TB recommended
- **Scaling Factors**: Sample count, universe size, feature count directly affect RAM usage
- **Universe Batching**: Works but fewer batches = better results
- **CPU**: Stable clocks, disable turbo boost, slight undervolting recommended, newer CPUs better, more cores helpful but some operations single-threaded
- **GPU**: VRAM dependent, introduces slight non-determinism (acceptable tolerances), disabled in strict mode for tree models, more VRAM and newer GPU better

## Files Modified

- `README.md` - Added system requirements, updated pipeline capabilities
- `DOCS/00_executive/GETTING_STARTED.md` - Updated system flow and prerequisites
- `DOCS/01_tutorials/SIMPLE_PIPELINE_USAGE.md` - Added pipeline overview
- `TRAINING/orchestration/utils/manifest.py` - Added `save_all_configs()` function
- `TRAINING/orchestration/intelligent_trainer.py` - Integrated config dump

## Backward Compatibility

All changes are backward compatible:
- Config dump is additive (doesn't affect existing runs)
- Documentation updates are informational only
- No breaking changes to APIs or configs
