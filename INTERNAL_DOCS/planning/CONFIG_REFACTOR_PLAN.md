# Config Refactor Plan - Modular Pipeline Configs

## Current State Assessment

### What Exists
- ✅ `CONFIG/config_loader.py` - Centralized model config loader
- ✅ `CONFIG/training_config/` - Training pipeline configs (11 files)
- ✅ `CONFIG/model_config/` - Per-model hyperparameters (17 models)
- ✅ `CONFIG/multi_model_feature_selection.yaml` - Feature selection config
- ✅ `CONFIG/safety_config.yaml` - Leakage detection config
- ✅ `CONFIG/system_config.yaml` - System paths/settings

### Current Problems
1. **Configs get "crossed"**: Feature selection config used in training, etc.
2. **No experiment-level configs**: Can't easily reproduce a specific run
3. **No validation**: Wrong keys silently ignored or cause weird behavior
4. **Mixed concerns**: One config file controls multiple pipeline stages
5. **Hard to override**: No clean way to say "use this experiment config + these overrides"

## Difficulty Assessment: **MEDIUM** (2-3 days of focused work)

### Why It's Not Hard
- ✅ Config loading infrastructure already exists
- ✅ YAML parsing is straightforward
- ✅ Most modules already use config_loader or direct YAML loading
- ✅ No major architectural changes needed

### Why It's Not Trivial
- ⚠️ Need to update ~15-20 files that load configs
- ⚠️ Need to create typed config classes (dataclasses/Pydantic)
- ⚠️ Need migration path to avoid breaking existing code
- ⚠️ Need to test that nothing breaks

## Proposed Structure

```
CONFIG/
├── experiments/              # NEW: Experiment-level configs
│   ├── fwd_ret_60m_5m.yaml
│   └── y_will_peak_60m.yaml
│
├── feature_selection/        # NEW: Feature selection module configs
│   ├── multi_model.yaml      # (moved from root)
│   └── ranking.yaml
│
├── target_ranking/           # NEW: Target ranking module configs
│   └── multi_model.yaml
│
├── training/                 # NEW: Training module configs
│   ├── models.yaml           # Model hyperparameters (moved from model_config/)
│   └── pipeline.yaml         # (moved from training_config/pipeline_config.yaml)
│
├── leakage/                  # NEW: Leakage detection configs
│   ├── safety.yaml           # (moved from training_config/safety_config.yaml)
│   └── auto_fixer.yaml
│
├── system/                   # NEW: System-level configs
│   ├── paths.yaml            # (extracted from system_config.yaml)
│   ├── logging.yaml
│   └── gpu.yaml              # (moved from training_config/gpu_config.yaml)
│
├── data/                     # NEW: Data loading configs
│   └── loader.yaml
│
└── [legacy files kept for backward compat during migration]
    ├── multi_model_feature_selection.yaml
    └── training_config/
```

## Implementation Plan (Phased)

### Phase 1: Foundation (Day 1) - **EASY**

**Goal**: Create new structure + typed config classes

1. **Create typed config classes** (`CONFIG/config_schemas.py`):
   ```python
   from dataclasses import dataclass
   from typing import List, Dict, Any, Optional
   
   @dataclass
   class ExperimentConfig:
       name: str
       data_dir: Path
       symbols: List[str]
       target: str
       interval: str = "5m"
       max_samples_per_symbol: int = 5000
   
   @dataclass
   class FeatureSelectionConfig:
       top_n: int
       model_families: Dict[str, Dict[str, Any]]
       aggregation: Dict[str, Any]
   
   @dataclass
   class TrainingConfig:
       model_families: Dict[str, Dict[str, Any]]
       cv_folds: int = 5
   ```

2. **Create new directory structure** (empty files for now)

3. **Create config builder** (`CONFIG/config_builder.py`):
   ```python
   def build_feature_selection_config(
       experiment_cfg: ExperimentConfig,
       module_cfg: Dict[str, Any]
   ) -> FeatureSelectionConfig:
       # Merge experiment + module configs
       # Validate
       # Return typed object
   ```

**Effort**: 2-3 hours  
**Risk**: Low (just adding new files)

---

### Phase 2: Migrate Feature Selection (Day 1-2) - **MEDIUM**

**Goal**: Feature selection uses new config system

1. **Move** `multi_model_feature_selection.yaml` → `feature_selection/multi_model.yaml`
2. **Update** `TRAINING/ranking/feature_selector.py` to use new config builder
3. **Update** `TRAINING/ranking/multi_model_feature_selection.py` to use typed config
4. **Test** with `SCRIPTS/test_phase2_feature_selection.py`

**Effort**: 3-4 hours  
**Risk**: Medium (touching working code)

---

### Phase 3: Migrate Target Ranking (Day 2) - **MEDIUM**

**Goal**: Target ranking uses new config system

1. **Create** `target_ranking/multi_model.yaml` (extract from current usage)
2. **Update** `TRAINING/ranking/target_ranker.py`
3. **Update** `TRAINING/ranking/rank_target_predictability.py`
4. **Test** with target ranking

**Effort**: 2-3 hours  
**Risk**: Medium

---

### Phase 4: Migrate Training (Day 2-3) - **HARDER**

**Goal**: Training uses new config system

1. **Consolidate** model configs into `training/models.yaml`
2. **Update** `TRAINING/train_with_strategies.py`
3. **Update** model trainers to use new config
4. **Test** with full pipeline

**Effort**: 4-6 hours  
**Risk**: Higher (touching core training code)

---

### Phase 5: Add Experiment Configs (Day 3) - **EASY**

**Goal**: Enable experiment-level configs

1. **Create** example experiment config
2. **Update** `intelligent_trainer.py` to accept experiment config
3. **Test** end-to-end with experiment config

**Effort**: 2-3 hours  
**Risk**: Low

---

### Phase 6: Cleanup & Validation (Day 3) - **EASY**

**Goal**: Remove legacy configs, add validation

1. **Add** Pydantic/dataclass validation
2. **Deprecate** old config files (with warnings)
3. **Update** documentation
4. **Final** end-to-end test

**Effort**: 2-3 hours  
**Risk**: Low

---

## Total Estimate

- **Best case**: 2 days (if everything goes smoothly)
- **Realistic**: 3 days (with testing and fixes)
- **Worst case**: 4 days (if training migration is tricky)

## Migration Strategy

### Backward Compatibility

During migration, keep old configs working:

```python
def load_config_legacy_or_new(path: Path) -> Dict[str, Any]:
    """Try new structure first, fall back to legacy"""
    new_path = migrate_path(path)  # Map old → new
    if new_path.exists():
        return load_yaml(new_path)
    elif path.exists():
        logger.warning(f"Using legacy config: {path}")
        return load_yaml(path)
    else:
        raise FileNotFoundError(f"Config not found: {path}")
```

### Gradual Rollout

1. **Week 1**: Phase 1-2 (foundation + feature selection)
2. **Week 2**: Phase 3-4 (target ranking + training)
3. **Week 3**: Phase 5-6 (experiments + cleanup)

## Benefits After Migration

1. ✅ **No more crossed configs**: Each module has its own config
2. ✅ **Reproducible experiments**: One YAML file = one run
3. ✅ **Type safety**: Dataclasses catch errors at load time
4. ✅ **Easy overrides**: Experiment config can override module defaults
5. ✅ **Clear structure**: New contributors know where to look

## Example: After Migration

### Experiment Config (`CONFIG/experiments/fwd_ret_60m_test.yaml`)
```yaml
experiment:
  name: fwd_ret_60m_test
  description: "Quick test of fwd_ret_60m with 2 symbols"

data:
  data_dir: data/data_labeled/interval=5m
  symbols: [AAPL, MSFT]
  interval: 5m
  max_samples_per_symbol: 3000

targets:
  primary: fwd_ret_60m

feature_selection:
  top_n: 30
  model_families: [lightgbm, xgboost]  # Override module default

training:
  model_families: [lightgbm, xgboost]
  cv_folds: 5
```

### Usage
```python
from CONFIG.config_builder import build_configs_from_experiment

experiment_cfg = load_experiment_config("fwd_ret_60m_test")
fs_config = build_feature_selection_config(experiment_cfg)
train_config = build_training_config(experiment_cfg)

# Use typed configs
select_features(target=fs_config.target, config=fs_config)
train_models(target=train_config.target, config=train_config)
```

## Decision: Should We Do This?

### ✅ **YES, if:**
- You're planning to add more pipeline components
- You want reproducible experiments
- You're seeing config-related bugs
- You have 2-3 days to invest

### ❌ **NO, if:**
- Current system works fine for your needs
- You're in the middle of a critical deadline
- Config issues are rare and easy to fix

## Recommendation

**Do it, but phased:**

1. **Start with Phase 1-2** (foundation + feature selection) - **Low risk, high value**
   - Feature selection is already isolated
   - Proves the pattern works
   - Can stop here if needed

2. **Then Phase 3** (target ranking) - **Medium risk, medium value**
   - Similar to feature selection
   - Completes the "intelligence layer" configs

3. **Then Phase 4-6** (training + experiments) - **Higher risk, higher value**
   - Touches more code
   - But unlocks full benefits

This way you get value early and can stop at any phase if priorities change.

