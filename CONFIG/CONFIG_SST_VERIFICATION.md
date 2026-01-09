# CONFIG SST Verification Report

## Summary

Verification completed to ensure all configs in `CONFIG/` folder are being correctly passed through the pipeline as Single Source of Truth (SST).

**Status**: ✅ Config structure is well-organized, but there are **85 hardcoded values** in 29 files that should be replaced with config loading.

## Findings

### ✅ Good News

1. **Config structure is well-organized**:
   - Clear hierarchy: `pipeline/`, `ranking/`, `models/`, `experiments/`
   - Defaults properly injected via `defaults.yaml`
   - Config loader API is being used (85 files)

2. **Config loading is widespread**:
   - 342 occurrences of `get_cfg()` calls
   - 186 files importing from `CONFIG.config_loader`
   - 89 occurrences of `load.*config()` calls

3. **Config paths exist**:
   - `pipeline.determinism.base_seed` ✅ (line 73 in `pipeline.yaml`)
   - `pipeline.data_limits.min_cross_sectional_samples` ✅ (line 18)
   - `pipeline.data_limits.max_cs_samples` ✅ (line 24)

### ⚠️ Issues Found

**85 hardcoded values** in 29 files that should come from config:

1. **Seeds (39 issues)** - Most common issue
   - Pattern: `seed = 42`, `BASE_SEED = 42`, `random_state = 42`
   - Should use: `pipeline.determinism.base_seed`
   - Files affected: `model_evaluation.py`, `feature_selector.py`, `target_ranker.py`, etc.

2. **min_cs (24 issues)**
   - Pattern: `min_cs = 10`, `harness_min_cs = 10`, `min_cs=1`
   - Should use: `pipeline.data_limits.min_cross_sectional_samples` or `pipeline.data_limits.min_cs`
   - Files affected: `feature_selector.py`, `target_ranker.py`, `shared_ranking_harness.py`

3. **max_cs (10 issues)**
   - Pattern: `max_cs_samples = 1000`, `harness_max_cs_samples = 1000`
   - Should use: `pipeline.data_limits.max_cs_samples`
   - Files affected: `feature_selector.py`, `target_ranker.py`, `shared_ranking_harness.py`

4. **n_estimators (9 issues)**
   - Pattern: `n_estimators = 50`, `n_estimators = 1000`
   - Should use: `defaults.tree_models.n_estimators` or preprocessing config
   - Files affected: `model_evaluation.py`, `cross_sectional_feature_ranker.py`

5. **learning_rate (2 issues)**
   - Pattern: `learning_rate = 0.03`
   - Should use: `defaults.tree_models.learning_rate`
   - Files affected: `cross_sectional_feature_ranker.py`

6. **validation_split (1 issue)**
   - Pattern: `validation_split = 0.2`
   - Should use: `defaults.sampling.validation_split`
   - Files affected: `model_evaluation.py`

## Top 10 Files with Hardcoded Values

1. `TRAINING/models/specialized/trainers_extended.py`: 20 issues
2. `TRAINING/ranking/target_ranker.py`: 10 issues
3. `TRAINING/model_fun/base_trainer.py`: 3 issues
4. `TRAINING/ranking/feature_selector.py`: 3 issues
5. `TRAINING/models/specialized/trainers.py`: 4 issues
6. `TRAINING/ranking/cross_sectional_feature_ranker.py`: 2 issues
7. `TRAINING/ranking/multi_model_feature_selection.py`: 1 issue
8. `TRAINING/ranking/shared_ranking_harness.py`: 1 issue
9. `TRAINING/ranking/utils/cross_sectional_data.py`: 1 issue
10. `TRAINING/model_fun/base_trainer.py`: 3 issues

## Recommendations

### Priority 1: Replace Hardcoded Seeds

**Why**: Seeds are critical for reproducibility. All seeds should come from `pipeline.determinism.base_seed`.

**Action**:
```python
# ❌ DON'T:
seed = 42
BASE_SEED = 42

# ✅ DO:
from CONFIG.config_loader import get_cfg
seed = int(get_cfg("pipeline.determinism.base_seed", default=42, config_name="pipeline_config"))
BASE_SEED = int(get_cfg("pipeline.determinism.base_seed", default=42, config_name="pipeline_config"))
```

**Files to fix**:
- `TRAINING/ranking/predictability/model_evaluation.py` (multiple instances)
- `TRAINING/ranking/feature_selector.py` (lines 2343, 2434, 2553, 2554)
- `TRAINING/ranking/target_ranker.py` (multiple instances)

### Priority 2: Replace Hardcoded min_cs/max_cs

**Why**: Data limits should be configurable per experiment.

**Action**:
```python
# ❌ DON'T:
min_cs = 10
max_cs_samples = 1000

# ✅ DO:
from CONFIG.config_loader import get_cfg
min_cs = int(get_cfg("pipeline.data_limits.min_cross_sectional_samples", default=10, config_name="pipeline_config"))
max_cs_samples = int(get_cfg("pipeline.data_limits.max_cs_samples", default=1000, config_name="pipeline_config"))
```

**Files to fix**:
- `TRAINING/ranking/feature_selector.py` (lines 319, 320)
- `TRAINING/ranking/target_ranker.py` (multiple instances)
- `TRAINING/ranking/shared_ranking_harness.py`

### Priority 3: Replace Hardcoded Model Hyperparameters

**Why**: Model hyperparameters should come from model configs or defaults.

**Action**:
```python
# ❌ DON'T:
n_estimators = 50
learning_rate = 0.03

# ✅ DO:
from CONFIG.config_loader import get_cfg, load_model_config
# Option 1: From defaults
n_estimators = int(get_cfg("defaults.tree_models.n_estimators", default=1000))
# Option 2: From model config
model_config = load_model_config("lightgbm")
n_estimators = model_config.get("n_estimators", 1000)
```

**Files to fix**:
- `TRAINING/ranking/predictability/model_evaluation.py` (lines 672, 676)
- `TRAINING/ranking/cross_sectional_feature_ranker.py` (lines 52, 1068)

### Priority 4: Verify Config Passing Through Stages

**Action**: Ensure configs are explicitly passed as function parameters:

1. **TARGET_RANKING**: Verify `min_cs`, `max_cs_samples`, `multi_model_config` are passed from `intelligent_trainer.py` → `rank_targets()` → `evaluate_target_predictability()`

2. **FEATURE_SELECTION**: Verify `min_cs`, `max_cs_samples`, `multi_model_config` are passed from `intelligent_trainer.py` → `select_features_for_target()` → feature selection functions

3. **TRAINING**: Verify model configs are passed from `intelligent_trainer.py` → `train_models_for_interval_comprehensive()` → model trainers

## Verification Script

A verification script has been created at `CONFIG/tools/verify_config_sst.py`:

```bash
python CONFIG/tools/verify_config_sst.py
```

This script:
- Scans for hardcoded values that should come from config
- Verifies config loading patterns are consistent
- Checks that config paths exist
- Generates a report of issues

## Next Steps

1. **Fix hardcoded seeds** (Priority 1) - ~39 instances
2. **Fix hardcoded min_cs/max_cs** (Priority 2) - ~34 instances
3. **Fix hardcoded hyperparameters** (Priority 3) - ~12 instances
4. **Verify config passing** (Priority 4) - Audit function signatures
5. **Update changelog** - Document fixes

## Config Precedence (Reminder)

1. **CLI arguments** (highest priority)
2. **Experiment config** (`experiments/*.yaml`)
3. **Intelligent training config** (`pipeline/training/intelligent.yaml`)
4. **Pipeline configs** (`pipeline/training/*.yaml`)
5. **Defaults** (`defaults.yaml`) - lowest priority

All hardcoded values should respect this precedence chain.
