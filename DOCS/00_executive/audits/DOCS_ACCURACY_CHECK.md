# Documentation Accuracy Check Results

## Critical Issues Found

### 1. ❌ INCORRECT: "52+ model trainers" ✅ FIXED

**Location**: `README.md` line 37, multiple docs

**Claim**: "all 52+ model trainers use config-driven hyperparameters"

**Reality**: 
- There are **20 model families** (confirmed in code: `ALL_FAMILIES` list)
- There are **25 trainer files** (including base trainers: base_trainer.py, base_2d_trainer.py, base_3d_trainer.py)
- The "52+" number is incorrect

**Status**: ✅ Fixed in all files

### 2. ⚠️ TOO STRONG: "Full reproducibility guaranteed" / "Zero hardcoded values"

**Location**: Multiple docs (`ARCHITECTURE_OVERVIEW.md`, `CONFIG_BASICS.md`, `DETERMINISTIC_TRAINING.md`, etc.)

**Claim**: "Full reproducibility guaranteed" and "Zero hardcoded values"

**Reality Check**:
- ✅ **Mostly accurate**: The TRAINING pipeline does load from configs
- ⚠️ **Fallback defaults exist**: Some hardcoded fallbacks remain (e.g., `max_rows_per_symbol: 50000`, `min_cs: 10`) for cases where configs are unavailable
- ⚠️ **External factors**: Reproducibility can be affected by library versions, hardware differences, floating-point precision, etc.
- ⚠️ **CONFIG_AUDIT.md** documents remaining hardcoded thresholds in some modules

**Fix Needed**: Qualify these statements:
- "Full reproducibility guaranteed" → "Reproducibility ensured when using proper configs (same config → same results)"
- "Zero hardcoded values" → "All training parameters load from configs (with fallback defaults for edge cases)"

### 3. ⚠️ TOO STRONG: "Complete Single Source of Truth"

**Location**: Multiple docs

**Claim**: "Complete Single Source of Truth (SST)"

**Reality**: 
- ✅ **Mostly accurate**: The system does centralize configs
- ⚠️ **Fallbacks exist**: Some hardcoded fallback defaults remain
- ⚠️ **Not 100% complete**: Some modules still have hardcoded thresholds (documented in `CONFIG_AUDIT.md`)

**Fix Needed**: Qualify as "Single Source of Truth (SST) for all training parameters" rather than "Complete SST"

### 4. ✅ VERIFIED: Other Claims

- **GPU acceleration**: ✅ Accurate - LightGBM, XGBoost, CatBoost GPU support confirmed
- **20 model families**: ✅ Accurate - Confirmed in code
- **Config-driven SST**: ✅ Mostly accurate - All configs load from YAML (with fallbacks)
- **Backward compatibility**: ✅ Accurate - Symlinks and fallback logic confirmed
- **Experiment configs**: ✅ Accurate - `--experiment-config` flag exists and works
- **Training routing**: ✅ Accurate - Status clarified in README
- **Performance metrics**: ✅ Accurate - Benchmarks documented with ranges, not absolute claims
- **Development status warnings**: ✅ Accurate - Clear warnings about active development

## Recommendations

1. ✅ **Fix model count** - COMPLETED
2. ✅ **Clarify routing system status** - COMPLETED  
3. **Qualify reproducibility claims** - Use "ensures reproducibility" rather than "guarantees"
4. **Acknowledge fallbacks** - Note that fallback defaults exist for edge cases
5. **Be precise about "complete"** - Use "Single Source of Truth" without "Complete" qualifier, or clarify scope
6. **Document known limitations** - Reference `CONFIG_AUDIT.md` for remaining hardcoded values

