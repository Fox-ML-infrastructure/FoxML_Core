# SST Remaining Work - Concrete Action Plan

**Status**: Core infrastructure complete. Remaining items are incremental improvements.

---

## ✅ Completed

1. **SST Enforcement Test** - `TRAINING/tests/test_no_hardcoded_hparams.py`
   - Strict marker system: `FALLBACK_DEFAULT_OK`, `DESIGN_CONSTANT_OK`
   - Scans all Python files in `TRAINING/`
   - Flags hardcoded hyperparameters, thresholds, seeds

2. **Internal Documentation**
   - `SST_DETERMINISM_GUARANTEES.md` - Policy and guarantees
   - `SST_COMPLIANCE_CHECKLIST.md` - Pre-commit checklist

3. **Core Fixes**
   - Fixed config loading in `comprehensive_trainer.py`, `ngboost_trainer.py`
   - Added proper fallback markers throughout
   - Fixed test_size/random_state patterns in base trainers

---

## 🔧 Remaining Work (Prioritized)

### 1. "Top 10%" Patterns (HIGH PRIORITY)

**Location**: Multiple files in `TRAINING/ranking/predictability/`

**Pattern**:
```python
top_k = max(1, int(len(importances) * 0.1))  # Top 10% of features
```

**Fix**:

1. Add to `CONFIG/feature_selection/multi_model.yaml`:
```yaml
selection:
  importance_top_fraction: 0.10  # Top fraction of features by importance (default: 10%)
```

2. Update code:
```python
from CONFIG.config_loader import get_cfg

fraction = float(get_cfg("feature_selection.selection.importance_top_fraction", default=0.10, config_name="multi_model"))
top_k = max(1, int(len(importances) * fraction))
top_indices = np.argsort(importances)[-top_k:]
```

**Files to update**:
- `TRAINING/ranking/predictability/model_evaluation.py` (multiple instances)
- `TRAINING/ranking/predictability/leakage_detection.py` (multiple instances)

**Estimated effort**: 2-3 hours

---

### 2. `batch_size` in Neural Network Trainers (MEDIUM PRIORITY)

**Location**: `TRAINING/model_fun/neural_network_trainer.py`, `comprehensive_trainer.py`

**Current**:
```python
batch_size=256,
batch_size=32,
```

**Fix**:

1. Add training profiles to `CONFIG/training_config/optimizer_config.yaml`:
```yaml
training_profiles:
  default:
    batch_size: 256
    max_epochs: 50
  debug:
    batch_size: 32
    max_epochs: 5
  throughput_optimized:
    batch_size: 512
    max_epochs: 100
```

2. Update trainers:
```python
from CONFIG.config_loader import get_cfg, get_optimizer_config

profile = get_cfg("training.profile", default="default", config_name="optimizer_config")
profile_cfg = get_optimizer_config().get("training_profiles", {}).get(profile, {})
batch_size = profile_cfg.get("batch_size", 256)  # FALLBACK_DEFAULT_OK
max_epochs = profile_cfg.get("max_epochs", 50)  # FALLBACK_DEFAULT_OK
```

**Files to update**:
- `TRAINING/model_fun/neural_network_trainer.py`
- `TRAINING/model_fun/comprehensive_trainer.py`

**Estimated effort**: 1-2 hours

---

### 3. `n_estimators=1` Quick Test Models (LOW PRIORITY)

**Location**: `TRAINING/ranking/predictability/leakage_detection.py`, `model_evaluation.py`

**Current**:
```python
test_model = lgb.LGBMRegressor(device='cuda', n_estimators=1, ...)
```

**Decision**: These are diagnostic models, not production. Use `DESIGN_CONSTANT_OK` marker.

**Fix**:
```python
# DESIGN_CONSTANT_OK: n_estimators=1 for diagnostic leakage detection only, not production behavior
test_model = lgb.LGBMRegressor(device='cuda', n_estimators=1, verbose=lgbm_backend_cfg.native_verbosity)
```

**Files to update**:
- `TRAINING/ranking/predictability/leakage_detection.py` (lines ~903, ~911)
- `TRAINING/ranking/predictability/model_evaluation.py` (lines ~747, ~755)

**Estimated effort**: 30 minutes

---

### 4. Remaining Seed Values (MEDIUM PRIORITY)

**Location**: Various files

**Pattern**:
```python
seed = 42
base_seed = 42
leak_seed = 42
shuffle_seed = 42
```

**Fix**: All should use determinism system:

```python
from TRAINING.common.determinism import BASE_SEED, seed_for

# For component-specific seeds
seed = seed_for(target_name, fold_idx, "component_name")

# For base seed fallback
seed = BASE_SEED if BASE_SEED is not None else 42  # FALLBACK_DEFAULT_OK
```

**Files to check**:
- `TRAINING/common/leakage_auto_fixer.py` (line ~341: `leak_seed = 42`)
- `TRAINING/common/leakage_sentinels.py` (line ~329: `shuffle_seed = 42`)
- `TRAINING/model_fun/lightgbm_trainer.py` (line ~93: `split_seed = 42`)
- `TRAINING/unified_training_interface.py` (lines ~59, ~170)
- `TRAINING/core/determinism.py` (line ~80: `seed = 42`)

**Estimated effort**: 2-3 hours

---

### 5. Other Hardcoded Values (LOW PRIORITY)

**Location**: Various

- `correlation_threshold: float = 0.95` in `TRAINING/utils/feature_selection.py` (line ~176)
  - Should load from `safety_config.yaml` or `feature_selection_config.yaml`

- `confidence=0.95` in `TRAINING/common/leakage_auto_fixer.py` (line ~454)
  - Should load from `safety_config.yaml`

- `n_splits=5` in `TRAINING/utils/purged_time_series_split.py` (lines ~85, ~96)
  - Should load from `preprocessing_config.yaml`

**Estimated effort**: 1-2 hours

---

## 📋 Implementation Order

1. **Top 10% patterns** (highest impact, reusable pattern)
2. **Remaining seeds** (completes determinism story)
3. **batch_size profiles** (useful for dev/prod workflows)
4. **n_estimators=1 markers** (quick win)
5. **Other hardcoded values** (polish)

---

## 🧪 Testing After Each Fix

Run SST enforcement test:
```bash
pytest TRAINING/tests/test_no_hardcoded_hparams.py -v
```

Expected: Test should pass or only flag new items you're working on.

---

## 📝 Notes

- All fixes should maintain backward compatibility
- Fallback defaults should match config defaults
- Use strict markers: `FALLBACK_DEFAULT_OK` or `DESIGN_CONSTANT_OK`
- Update internal docs if adding new config sections
