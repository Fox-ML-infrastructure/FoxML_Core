# Determinism Audit - TRAINING Pipeline

**Date**: 2025-12-10  
**Status**: In Progress

## Summary

This document tracks the audit and fixes for hardcoded `random_state=42` and other non-deterministic values across the TRAINING pipeline to ensure consistent, reproducible results.

## ✅ Fixed (Critical Path)

### 1. Base Trainer (`model_fun/base_trainer.py`)
- ✅ `safe_ridge_fit()`: Now uses `BASE_SEED` from determinism system
- ✅ `_get_test_split_params()`: Now uses `BASE_SEED` from determinism system

### 2. LightGBM Trainer (`model_fun/lightgbm_trainer.py`)
- ✅ `train()`: Now uses `_get_test_split_params()` for train_test_split
- ✅ `_build_model()`: Now uses `_get_random_state()` method
- ✅ Added `_get_random_state()` method that uses `BASE_SEED`

### 3. Training Strategies (`training_strategies/training.py`)
- ✅ Row downsampling: Now uses `stable_seed_from([target, 'downsample'])` for deterministic downsampling

### 4. Common Utilities
- ✅ `leakage_sentinels.py`: Shuffle now uses deterministic seed based on target name
- ✅ `ranking/predictability/leakage_detection.py`: Feature pruning uses `stable_seed_from([target_name, 'feature_pruning'])`

### 5. Model Evaluation (`ranking/predictability/model_evaluation.py`)
- ✅ Permutation importance: Now uses deterministic seed per feature
- ✅ Bootstrap sampling: Now uses deterministic seed per iteration

## ⚠️ Remaining (Lower Priority)

### Model Trainers (Still have hardcoded `random_state=42`)
These should be updated to use `BASE_SEED` or context-specific seeds:

- `model_fun/xgboost_trainer.py` - Line 82, 330
- `model_fun/gmm_regime_trainer.py` - Lines 82, 103, 130, 156, 173
- `model_fun/change_point_trainer.py` - Line 87
- `model_fun/reward_based_trainer.py` - Lines 69, 100
- `model_fun/ftrl_proximal_trainer.py` - Lines 70, 105
- `model_fun/mlp_trainer.py` - Line 94
- `model_fun/quantile_lightgbm_trainer.py` - Line 156
- `model_fun/comprehensive_trainer.py` - Lines 98, 106, 145, 153

**Fix Pattern**:
```python
# Before:
random_state=42

# After:
try:
    from TRAINING.common.determinism import BASE_SEED
    random_state = BASE_SEED if BASE_SEED is not None else 42
except:
    random_state = 42
```

### Strategies (Still have hardcoded `random_state=42`)
- `strategies/cascade.py` - Multiple instances (Lines 187, 190, 203, 206, 344, 349, 352, 366, 373)
- `strategies/single_task.py` - Multiple instances (Lines 68, 202, 207, 221, 226, 229, 330, 336)

**Fix Pattern**: Same as above, or use context-specific seeds:
```python
from TRAINING.common.determinism import stable_seed_from
seed = stable_seed_from([target_name, 'strategy', model_family])
```

### Common Utilities (Still have hardcoded `random_state=42`)
- `common/leakage_auto_fixer.py` - Lines 327, 333, 364, 372
  - These should use context-specific seeds based on target/feature being tested

### Other Files
- `unified_training_interface.py` - Line 149
- `train_crypto_models.py` - Line 34 (base_seed=42, but this is a fallback)
- `core/determinism.py` - Line 80 (fallback default, acceptable)

### Test Files (Acceptable)
- `tests/test_sequential_mode.py` - Lines 52, 243 (test files can use fixed seeds)

## 🔍 Pattern for Future Fixes

### For Model Trainers:
1. Add `_get_random_state()` method to base class (already done)
2. Replace all `random_state=42` with `random_state=self._get_random_state()`
3. For train_test_split, use `_get_test_split_params()` (already updated in base)

### For Context-Specific Operations:
Use `stable_seed_from()` for deterministic, context-aware seeds:
```python
from TRAINING.common.determinism import stable_seed_from

# Example: Feature-specific operation
seed = stable_seed_from([target_name, 'feature_selection', feature_name])

# Example: Model-specific operation
seed = stable_seed_from([target_name, model_family, 'training'])
```

### For Bootstrap/Permutation Operations:
```python
from TRAINING.common.determinism import stable_seed_from

# Bootstrap iteration
bootstrap_seed = stable_seed_from(['bootstrap', target_name, f'iter_{i}'])

# Permutation per feature
perm_seed = stable_seed_from(['permutation', target_name, f'feature_{i}'])
```

## 📊 Impact Assessment

### High Priority (Affects Reproducibility)
- ✅ Base trainer and LightGBM trainer (fixed)
- ✅ Training strategies downsampling (fixed)
- ✅ Ranking pipeline (fixed)
- ⚠️ Other model trainers (remaining)
- ⚠️ Strategy classes (remaining)

### Medium Priority (Affects Consistency)
- ⚠️ Leakage auto-fixer (remaining)
- ⚠️ Unified training interface (remaining)

### Low Priority (Acceptable)
- Test files (can use fixed seeds)
- Fallback defaults in determinism.py

## Next Steps

1. **Batch fix remaining model trainers** - Apply same pattern as LightGBM trainer
2. **Fix strategy classes** - Use context-specific seeds
3. **Fix leakage auto-fixer** - Use target/feature-specific seeds
4. **Verify all fixes** - Run reproducibility tests

## Notes

- All fixes maintain backward compatibility (fallback to 42 if determinism system unavailable)
- Context-specific seeds ensure different operations get different but deterministic seeds
- BASE_SEED from config ensures all components use the same base seed
