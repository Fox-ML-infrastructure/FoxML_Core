# Comprehensive Config Dump and Hardware Documentation Updates

**Date**: 2026-01-08  
**Type**: Feature Enhancement, Documentation  
**Impact**: Medium - Improves run reproducibility and hardware guidance

## Overview

Added comprehensive config dump functionality to automatically copy all configuration files to `globals/configs/` when a run is created, and enhanced hardware requirements documentation with CPU/GPU recommendations.

## Changes

### Comprehensive Config Dump

**NEW**: `save_all_configs()` function in `TRAINING/orchestration/utils/manifest.py`
- Automatically copies all `.yaml` config files from `CONFIG/` directory to `globals/configs/` preserving directory structure
- Creates `INDEX.md` listing all configs organized by category
- Skips archive directory and non-config files (README.md, .py files)
- Integrated into `intelligent_trainer.py` after config resolution
- Enables easy run recreation without needing access to original CONFIG folder

**Directory Structure Created:**
```
globals/configs/
├── core/
│   ├── identity_config.yaml
│   ├── logging.yaml
│   └── system.yaml
├── data/
├── pipeline/
├── ranking/
├── models/
├── defaults.yaml
└── INDEX.md
```

**Benefits:**
- Complete config snapshot for each run
- Full auditability - all configs used in run are preserved
- Easy run recreation - no need to track down config files
- No external dependencies for understanding run configuration

### Hardware Requirements Documentation

**UPDATED**: `README.md` System Requirements section

**CPU Recommendations Added:**
- Stable clocks: Disable turbo boost/overclocking for stability
- Undervolting: Slight undervolting recommended for stability
- Newer CPUs: Generally perform better
- Core count: More cores beneficial, but some operations are single-threaded
- Base clock speed: Faster base clocks improve performance
- Best practice: Disable turbo boost for reproducible results

**GPU Considerations Added:**
- VRAM dependent: Performance primarily limited by VRAM
- Non-determinism: GPU operations introduce slight non-determinism (within acceptable tolerances) due to parallel floating-point arithmetic
- Strict mode: GPU automatically disabled for tree models in strict mode for bitwise deterministic runs
- Best practice: More VRAM and newer GPU architectures provide better performance

**Files Modified:**
- `README.md` - Added CPU and GPU recommendations to System Requirements section

## Technical Details

### Config Dump Implementation

- Uses `Path.rglob("*.yaml")` to find all config files
- Preserves relative paths from CONFIG directory
- Creates parent directories as needed
- Handles errors gracefully (logs warnings, doesn't fail run)
- Creates categorized INDEX.md for easy navigation

### Integration

- Called in `intelligent_trainer.py` after `save_user_config()` and `save_overrides_config()`
- Passes experiment config name if available
- Wrapped in try/except to prevent run failures

## Backward Compatibility

- All changes are backward compatible
- Config dump is additive (doesn't affect existing functionality)
- Hardware docs are informational only

## Related Changes

- See `2026-01-08-hardware-requirements-documentation.md` for hardware docs details
- See `2026-01-08-config-cleanup-and-symlink-removal.md` for config structure context
