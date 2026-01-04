# GPU/CPU Determinism Config Fix

**Date**: 2026-01-04  
**Type**: Critical Bug Fix, Determinism Enhancement  
**Impact**: High - Fixes config settings being ignored, enables true deterministic runs

## Overview

Fixed critical disconnect between reproducibility config settings and actual GPU/CPU device selection. The pipeline had a well-designed determinism system but it was being bypassed by hardcoded calls and GPU detection logic that ignored strict mode settings.

## Critical Issues Fixed

### 1. Hardcoded `set_global_determinism()` Calls Ignoring Config

**Problem**: Four entry points had hardcoded `set_global_determinism()` calls with `strict_mode=False` and `prefer_cpu_tree_train=False`, completely bypassing config settings.

**Affected Files**:
- `TRAINING/ranking/predictability/main.py` (target ranking entry point)
- `TRAINING/ranking/multi_model_feature_selection.py` (feature selection entry point)
- `TRAINING/orchestration/intelligent_trainer.py` (intelligent trainer entry point)
- `TRAINING/training_strategies/utils.py` (training strategies)

**Before**:
```python
BASE_SEED = set_global_determinism(
    base_seed=base_seed,
    threads=None,
    deterministic_algorithms=False,
    prefer_cpu_tree_train=False,  # Always False!
    strict_mode=False  # Always False!
)
```

**Fix**: Created unified `init_determinism_from_config()` function that reads from config:
```python
BASE_SEED = init_determinism_from_config()  # Reads REPRO_MODE, respects config
```

**Files Modified**:
- `TRAINING/common/determinism.py` (added `init_determinism_from_config()`)
- All 4 entry points replaced hardcoded calls

---

### 2. GPU Device Selection Ignoring Strict Mode

**Problem**: LightGBM, XGBoost, and CatBoost GPU detection in `model_evaluation.py` read GPU settings from `gpu_config` but never checked `is_strict_mode()` to force CPU. Models were created directly with `device='cuda'` even when strict mode was enabled.

**Affected Code**:
- `model_evaluation.py`: LightGBM (line ~2376), XGBoost (line ~2920), CatBoost (line ~3135)
- `multi_model_feature_selection.py`: LightGBM (line ~1020)

**Fix**: Added strict mode check before GPU detection:
```python
from TRAINING.common.determinism import is_strict_mode

if is_strict_mode():
    logger.info("  ℹ️  Strict mode: forcing CPU for LightGBM (GPU disabled for determinism)")
    gpu_params = {}  # Skip GPU detection entirely
else:
    # Existing GPU detection logic...
```

**Files Modified**:
- `TRAINING/ranking/predictability/model_evaluation.py` (LightGBM, XGBoost, CatBoost)
- `TRAINING/ranking/multi_model_feature_selection.py` (LightGBM)

---

### 3. Training Phase GPU Detection Ignoring Strict Mode

**Problem**: Model trainers (XGBoost, PyTorch, TensorFlow) had GPU detection that didn't respect strict mode.

**Fix**: Added strict mode checks to:
- `xgboost_trainer.py`: Force `cpu_only=True` in strict mode
- `seq_torch_base.py`: Force `device=cpu` in strict mode
- `neural_network_trainer.py`: Force `cpu_only=True` in strict mode

**Files Modified**:
- `TRAINING/model_fun/xgboost_trainer.py`
- `TRAINING/model_fun/seq_torch_base.py`
- `TRAINING/model_fun/neural_network_trainer.py`

---

### 4. Environment-Level GPU Visibility Not Respected

**Problem**: `CUDA_VISIBLE_DEVICES` was always set to `"0"` in `set_global_determinism()`, even in strict mode.

**Fix**: Set `CUDA_VISIBLE_DEVICES="-1"` when strict mode or `prefer_cpu_tree_train` is enabled:
```python
"CUDA_VISIBLE_DEVICES": "-1" if (strict_mode or prefer_cpu_tree_train) else "0",
```

**Files Modified**:
- `TRAINING/common/determinism.py` (line ~129)

---

### 5. UnboundLocalError in `set_global_determinism()`

**Problem**: Redundant `import os` inside `set_global_determinism()` shadowed module-level import, causing `UnboundLocalError` when `os.environ.get()` was called before the import executed.

**Fix**: Removed redundant `import os` (line 80) since `os` is already imported at module level.

**Files Modified**:
- `TRAINING/common/determinism.py` (line 80)

---

## Summary of Changes

| File | Change |
|------|--------|
| `determinism.py` | Added `init_determinism_from_config()`, fixed `CUDA_VISIBLE_DEVICES` logic, removed redundant `import os` |
| `ranking/predictability/main.py` | Replace hardcoded call with `init_determinism_from_config()` |
| `ranking/multi_model_feature_selection.py` | Replace hardcoded call + add strict mode GPU check |
| `orchestration/intelligent_trainer.py` | Replace hardcoded call with `init_determinism_from_config()` |
| `training_strategies/utils.py` | Replace hardcoded call with `init_determinism_from_config()` |
| `ranking/predictability/model_evaluation.py` | Add strict mode GPU checks for LightGBM, XGBoost, CatBoost |
| `model_fun/xgboost_trainer.py` | Add strict mode CPU force check |
| `model_fun/seq_torch_base.py` | Add strict mode CPU force check for PyTorch |
| `model_fun/neural_network_trainer.py` | Add strict mode CPU force check for TensorFlow |

## Impact

### Before Fix
- `REPRO_MODE=strict` was ignored - GPU still used
- Config settings in `reproducibility.yaml` had no effect
- Mixed device execution (some runs GPU, others CPU) depending on detection success
- Non-deterministic results due to GPU parallel execution variability

### After Fix
- `REPRO_MODE=strict` now forces CPU for all models
- Config settings are respected across all phases
- Consistent device selection (CPU in strict mode, GPU in best_effort)
- Deterministic results when strict mode enabled

## How to Enable Strict Mode

**Option A: Environment variable (recommended for one-off runs)**
```bash
REPRO_MODE=strict python TRAINING/orchestration/intelligent_trainer.py ...
```

**Option B: Config file (for persistent strict mode)**
```yaml
# CONFIG/pipeline/training/reproducibility.yaml
reproducibility:
  mode: strict  # Change from best_effort
```

## Verification Logs

When strict mode is active, you should see:
```
🔒 Setting global determinism: seed=42, threads=1, deterministic=True
  ℹ️  Strict mode: forcing CPU for LightGBM (GPU disabled for determinism)
  ℹ️  Strict mode: forcing CPU for XGBoost (GPU disabled for determinism)
  ℹ️  Strict mode: forcing CPU for CatBoost (GPU disabled for determinism)
[PyTorch] Strict mode: forcing CPU (GPU disabled for determinism)
[NeuralNetwork] Strict mode: forcing CPU (GPU disabled for determinism)
```

## Testing

Verified with `bin/run_deterministic.sh`:
```bash
bin/run_deterministic.sh python TRAINING/orchestration/intelligent_trainer.py --experiment-config determinism_test
```

No more `UnboundLocalError` - strict mode now properly forces CPU across all phases.
