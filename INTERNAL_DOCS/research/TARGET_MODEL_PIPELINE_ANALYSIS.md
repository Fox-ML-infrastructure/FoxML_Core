# Target/Model Pipeline Analysis & Refactoring Plan

**Date:** 2024-12-19
**Scope:** Unified regression/classification/ranking pipeline architecture

---

## Scope Clarification

**IMPORTANT**: This analysis focuses on the **target ranking script** (`SCRIPTS/rank_target_predictability.py`), NOT rebuilding the entire training pipeline.

The existing training pipeline (`TRAINING/`) works fine. The goal is to ensure the ranking script:
1. Uses correct task types (already done - uses `TaskType` enum)
2. Uses correct model objectives (already done - LGBMClassifier for classification, LGBMRegressor for regression)
3. ️ Computes full task-aware metrics (partially done - imports `evaluate_by_task` but only stores primary scores)

**What we're NOT doing:**
- Rebuilding the training pipeline
- Creating a new model registry (you already have `ModelRegistry` for loading trained models)
- Changing how models are trained in production

**What we ARE doing:**
- Ensuring ranking script uses correct metrics per task type
- Making sure models are instantiated with correct objectives
- Computing full metrics (IC, logloss, etc.) not just primary scores

---

## 1. Repo Map

### Data & Feature Engineering
- `DATA_PROCESSING/features/`: Feature generation (returns, volatility, technical indicators)
- `DATA_PROCESSING/pipeline/`: Data pipeline orchestration
- `DATA_PROCESSING/targets/`: Target generation modules
 - `barrier.py`: Barrier/first-passage targets (`y_will_peak`, `y_will_valley`, `y_first_touch`)
 - `excess_returns.py`: Market-adjusted return targets

### Target Definitions & Routing
- `CONFIG/target_configs.yaml`: Target configuration (63 targets, 7 enabled)
- `TRAINING/target_router.py`: **TaskSpec routing system** (maps target patterns → task types, objectives, metrics)
- `TRAINING/utils/target_resolver.py`: Target name resolution and validation
- `SCRIPTS/utils/task_types.py`: **NEW** - Unified TaskType enum and config classes
- `SCRIPTS/utils/target_validation.py`: Target validation utilities

### Model Zoo
- `TRAINING/model_fun/`: Model trainer implementations (for production training)
 - `base_trainer.py`: Base class with preprocessing
 - `lightgbm_trainer.py`, `xgboost_trainer.py`: Tree-based trainers
 - `mlp_trainer.py`, `cnn1d_trainer.py`, `lstm_trainer.py`, `transformer_trainer.py`: Neural network trainers
 - `ensemble_trainer.py`, `multi_task_trainer.py`: Ensemble/multi-task trainers
- `IBKR_trading/live_trading/model_predictor.py`: **ModelRegistry** - loads trained models from disk (for inference)
- `CONFIG/multi_model_feature_selection.yaml`: Model configs for feature selection (used by ranking script)
- `CONFIG/model_config/`: Per-model hyperparameter configs

### Training Strategies
- `TRAINING/strategies/base.py`: Base training strategy
- `TRAINING/strategies/single_task.py`: Single target per model
- `TRAINING/strategies/multi_task.py`: Multi-target shared encoder
- `TRAINING/train_with_strategies.py`: Main training entrypoint

### Evaluation & Ranking
- `SCRIPTS/rank_target_predictability.py`: **REFACTORED** - Target predictability ranking
- `SCRIPTS/utils/task_metrics.py`: **NEW** - Task-aware metric evaluation
- `SCRIPTS/utils/leakage_filtering.py`: **NEW** - Target-aware feature filtering

---

## 2. Current Target & Task-Type Inventory

### Target Categories Found

| Target Pattern | Task Type (Inferred) | Location | Notes |
|----------------|---------------------|----------|-------|
| `fwd_ret_*` | Regression | `target_configs.yaml`, `target_router.py` | Forward returns (continuous) |
| `y_will_peak_*`, `y_will_valley_*` | Binary Classification | `target_configs.yaml`, `barrier.py` | Barrier hit prediction (0/1) |
| `y_first_touch_*` | Multiclass Classification | `target_configs.yaml`, `barrier.py` | Direction {-1, 0, +1} |
| `y_will_swing_*` | Binary Classification | `target_configs.yaml` | Swing high/low prediction |
| `y_will_peak_mfe_*`, `y_will_valley_mdd_*` | Binary Classification | `target_configs.yaml` | MFE/MDD threshold events |
| `hit_asym_*` | Multiclass Classification | `target_router.py` | Asymmetric barrier hit {-1, 0, +1} |
| `ret_ord_*`, `mfe_ord_*`, `mdd_ord_*` | Multiclass/Ordinal | `target_router.py` | Ordinal buckets {-3..+3} |
| `regime_*` | Multiclass Classification | `target_router.py` | Regime classification (3-5 classes) |
| `xrank_*` | Ranking | `target_router.py` | Cross-sectional ranking (future) |
| `tth_*`, `tth_abs_*` | Regression | `target_router.py` | Time-to-hit (continuous) |
| `mfe_share_*`, `time_in_profit_*` | Regression | `target_router.py` | Path quality metrics |
| `idio_ret_*` | Regression | `target_router.py` | Idiosyncratic returns |

### Task Type Detection Logic (Current State)

**Multiple inconsistent implementations:**

1. **`TRAINING/target_router.py`** (Most comprehensive)
 - Uses regex patterns → `TaskSpec` with explicit task types
 - Supports: `regression`, `binary`, `multiclass`, `ranking`
 - Has objective mapping per model family

2. **`TRAINING/strategies/single_task.py`** (Ad-hoc)
   ```python
   if target_name.startswith('fwd_ret_'):
       return 'regression'
   elif any(target_name.startswith(prefix) for prefix in ['will_peak', 'will_valley', ...]):
       return 'classification'
   ```

3. **`TRAINING/strategies/multi_task.py`** (Similar ad-hoc logic)
 - Same pattern-based detection

4. **`SCRIPTS/rank_target_predictability.py`** (NEW - Unified)
 - Uses `TaskType.from_target_column()` from `task_types.py`
 - Supports: `REGRESSION`, `BINARY_CLASSIFICATION`, `MULTICLASS_CLASSIFICATION`

5. **`TRAINING/utils/target_resolver.py`** (Legacy)
 - Simple string matching, returns `'regression'` or `'classification'`

### Problems Identified

1. **Inconsistent task type detection** across 5+ locations
2. **No unified enum** - some use strings (`'regression'`, `'classification'`), some use `TaskType` enum
3. **Missing task type in target configs** - `target_configs.yaml` doesn't specify task type
4. **Ambiguous targets** - Some targets could be regression or classification depending on data
5. **Model compatibility not enforced** - Models can be instantiated with wrong task type

---

## 3. Current Model & Pipeline Inventory

### Model Families & Task Support

| Model Family | Trainer Class | Current Task Support | Location | Issues |
|--------------|---------------|---------------------|----------|--------|
| LightGBM | `LightGBMTrainer` | Both (via objective param) | `lightgbm_trainer.py` | No explicit task type validation |
| XGBoost | `XGBoostTrainer` | Both (via objective param) | `xgboost_trainer.py` | No explicit task type validation |
| Random Forest | `RandomForestRegressor/Classifier` | Both (separate classes) | `single_task.py` | Creates different classes based on target_type string |
| Neural Network (MLP) | `MLPTrainer` | Both (via loss function) | `mlp_trainer.py` | No explicit task type validation |
| Multi-Task NN | `MultiTaskTrainer` | Both (separate heads) | `multi_task_trainer.py` | Uses `target_types` dict (string keys) |
| Ensemble | `EnsembleTrainer` | Both (delegates to base models) | `ensemble_trainer.py` | No explicit task type validation |

### Model Instantiation Patterns

**Pattern 1: String-based task type** (`single_task.py`, `multi_task.py`)
```python
target_type = self._determine_target_type(target_name, y)  # Returns 'regression' or 'classification'
if target_type == 'classification':
    model = RandomForestClassifier(...)
else:
    model = RandomForestRegressor(...)
```

**Pattern 2: Objective-based** (`lightgbm_trainer.py`, `xgboost_trainer.py`)
```python
# No explicit task type - relies on objective string
model = lgb.LGBMRegressor(objective='regression', ...)
# or
model = lgb.LGBMClassifier(objective='binary', ...)
```

**Pattern 3: TaskSpec routing** (`target_router.py`)
```python
spec = spec_from_target(target_column)
objective = get_objective_for_family('LightGBM', spec)  # Returns 'regression', 'binary', etc.
```

### Training Pipeline Entry Points

1. **`TRAINING/train_with_strategies.py`**
 - Main entrypoint
 - Uses `SingleTaskStrategy` or `MultiTaskStrategy`
 - Task type determined ad-hoc in strategy classes

2. **`SCRIPTS/rank_target_predictability.py`**
 - Uses unified `TaskType` enum
 - Calls `train_and_evaluate_models()` with explicit `task_type` parameter
 - **GOOD**: Already refactored to use new architecture

3. **Feature selection scripts**
 - `SCRIPTS/rank_features_comprehensive.py`
 - Uses ad-hoc task type detection

### Metrics Computation

**Current state:**
- Regression: R², MSE, MAE (via sklearn `cross_val_score` with `scoring='r2'`)
- Classification: Accuracy, AUC (via sklearn `cross_val_score` with `scoring='roc_auc'` or `'accuracy'`)
- **NEW**: `SCRIPTS/utils/task_metrics.py` provides unified `evaluate_by_task()` function

**Problems:**
1. Metrics not consistently computed across all scripts
2. No IC (Information Coefficient) for regression in most places
3. No logloss/cross-entropy for classification in most places
4. Metrics hardcoded in training scripts instead of using task-aware functions

---

## 4. Problems, Smells, and Likely Breakages

### Critical Issues

1. **Task type detection inconsistency**
 - **Location**: `TRAINING/strategies/single_task.py:117`, `multi_task.py:96`, `target_resolver.py:239`
 - **Problem**: Same target can be classified differently in different places
 - **Impact**: Models may be instantiated with wrong objective/loss function
 - **Example**: `y_will_peak_60m_0.8` might be treated as regression in one place, classification in another

2. **No model-task compatibility checking**
 - **Location**: All model trainers
 - **Problem**: Models can be instantiated with incompatible task types
 - **Impact**: Runtime errors or silent failures (e.g., using `LGBMRegressor` for binary classification)
 - **Fix needed**: Add `is_compatible(target, model)` checks before training

3. **Target configs missing task type**
 - **Location**: `CONFIG/target_configs.yaml`
 - **Problem**: 63 targets defined but no `task_type` field
 - **Impact**: Task type must be inferred from column name (error-prone)
 - **Fix needed**: Add `task_type` field to each target config

4. **Metrics not task-aware**
 - **Location**: `TRAINING/strategies/*.py`, feature selection scripts
 - **Problem**: Regression metrics used for classification targets (or vice versa)
 - **Impact**: Incorrect evaluation, poor model selection
 - **Example**: Using R² for binary classification (works but suboptimal)

5. **Model constructors don't validate task type**
 - **Location**: `TRAINING/model_fun/*_trainer.py`
 - **Problem**: Trainers accept any `y` without validating it matches the model's expected task type
 - **Impact**: Models may train on wrong data format (e.g., continuous values for binary classifier)

### Design Smells

1. **String-based task types** instead of enum
 - Makes it easy to typo (`'regression'` vs `'Regression'`)
 - No IDE autocomplete
 - No type checking

2. **Ad-hoc pattern matching** for task type detection
 - Fragile - breaks when new target patterns added
 - Inconsistent across files

3. **No centralized model registry**
 - Model capabilities (supported tasks) not explicitly defined
 - Hard to query "which models support multiclass classification?"

4. **Target configs separate from task routing**
 - `target_configs.yaml` has target metadata
 - `target_router.py` has task type patterns
 - These should be unified

### Likely Breakages

1. **Adding new target patterns**
 - Must update multiple files: `target_router.py`, `single_task.py`, `multi_task.py`, `target_resolver.py`
 - Easy to miss one, causing inconsistency

2. **Adding new model families**
 - No clear place to register "this model supports regression and binary classification"
 - Must manually update all training strategies

3. **Changing task type of existing target**
 - Would break all existing models trained on that target
 - No migration path

---

## 5. Target/Model Abstraction Proposal

### Proposed Architecture

**Core Abstractions** (Already implemented in `SCRIPTS/utils/task_types.py`):

```python
# TaskType enum (already exists)
class TaskType(Enum):
    REGRESSION = auto()
    BINARY_CLASSIFICATION = auto()
    MULTICLASS_CLASSIFICATION = auto()
    # Future: RANKING, ORDINAL, SURVIVAL

# TargetConfig (already exists, needs enhancement)
@dataclass
class TargetConfig:
    name: str
    target_column: str
    task_type: TaskType  # REQUIRED - no inference needed
    horizon: Optional[int]
    description: str
    # ... other fields

# ModelConfig (already exists, needs integration)
@dataclass
class ModelConfig:
    name: str
    constructor: Callable[..., Any]
    supported_tasks: Set[TaskType]  # Explicit capability declaration
    default_params: Dict[str, Any]
    primary_metric_by_task: Dict[TaskType, str]
```

### Integration Points

**1. Update `target_configs.yaml` to include `task_type`:**

```yaml
targets:
  peak_60m:
    target_column: "y_will_peak_60m_0.8"
    task_type: "BINARY_CLASSIFICATION"  # NEW
    horizon: 60
    description: "Predict upward barrier hits"
    enabled: true
```

**2. Create unified target registry:**

```python
# SCRIPTS/utils/target_registry.py (NEW)
def load_target_configs() -> Dict[str, TargetConfig]:
    """Load targets from YAML and convert to TargetConfig objects"""
    # Read target_configs.yaml
    # Convert each target to TargetConfig with explicit task_type
    # Use TaskType.from_target_column() as fallback if task_type not in YAML
```

**3. Create model registry:**

```python
# SCRIPTS/utils/model_registry.py (NEW)
def get_model_configs_for_task(task_type: TaskType) -> List[ModelConfig]:
    """Get all models that support a given task type"""
    # Filter ModelConfig objects by supported_tasks
```

**4. Update `target_router.py` to use unified types:**

```python
# Replace TaskSpec.task (string) with TaskType enum
# Keep TaskSpec for backward compatibility but add TaskType field
```

### File Structure

```
SCRIPTS/utils/
  ├── task_types.py          #  EXISTS - TaskType, TargetConfig, ModelConfig
  ├── task_metrics.py        #  EXISTS - Task-aware metric evaluation
  ├── target_validation.py   #  EXISTS - Enhanced with TaskType
  ├── leakage_filtering.py  #  EXISTS - Target-aware filtering
  ├── target_registry.py     # 🆕 NEW - Load targets from YAML → TargetConfig
  └── model_registry.py      # 🆕 NEW - Model capability registry

TRAINING/
  ├── target_router.py       #  UPDATE - Use TaskType enum instead of strings
  ├── strategies/
  │   ├── single_task.py     #  UPDATE - Use TargetConfig, ModelConfig
  │   └── multi_task.py      #  UPDATE - Use TaskType enum
  └── model_fun/
      └── *_trainer.py       #  UPDATE - Validate task type compatibility
```

---

## 6. Training & Evaluation Pipeline Refactor Plan

### Unified Training Interface

**Proposed function signature:**

```python
def train_target_model(
    target_config: TargetConfig,
    model_config: ModelConfig,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    cv: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Unified training function that works for all task types.

    Returns:
        {
            'model': fitted_model,
            'metrics': task_aware_metrics_dict,
            'feature_importance': np.ndarray,
            'task_type': TaskType
        }
    """
    # 1. Validate compatibility
    if not is_compatible(target_config, model_config):
        raise ValueError(f"Model {model_config.name} does not support task {target_config.task_type}")

    # 2. Validate target data
    is_valid, error_msg = validate_target(y, task_type=target_config.task_type)
    if not is_valid:
        raise ValueError(f"Invalid target data: {error_msg}")

    # 3. Prepare labels (encoding, weights, etc.)
    # Use target_router.prepare_labels_for_task() or equivalent

    # 4. Instantiate model with correct objective
    model = model_config.constructor(**model_config.default_params)
    # Set objective based on task_type and model family

    # 5. Train
    model.fit(X, y)

    # 6. Evaluate with task-aware metrics
    y_pred = model.predict(X) if target_config.task_type == TaskType.REGRESSION else model.predict_proba(X)
    metrics = evaluate_by_task(target_config.task_type, y, y_pred)

    return {
        'model': model,
        'metrics': metrics,
        'feature_importance': model.get_feature_importance() if hasattr(model, 'get_feature_importance') else None,
        'task_type': target_config.task_type
    }
```

### Refactoring Steps

**Step 1: Update target configs to include task_type**
- File: `CONFIG/target_configs.yaml`
- Action: Add `task_type` field to each target (or auto-detect and document)
- Migration: Use `TaskType.from_target_column()` to populate if missing

**Step 2: Create target registry**
- File: `SCRIPTS/utils/target_registry.py` (NEW)
- Action: Load YAML → `TargetConfig` objects with explicit `task_type`
- Integration: Update `SCRIPTS/rank_target_predictability.py` to use registry

**Step 3: Create model registry**
- File: `SCRIPTS/utils/model_registry.py` (NEW)
- Action: Define `ModelConfig` objects for all model families
- Integration: Update `CONFIG/multi_model_feature_selection.yaml` → `ModelConfig` objects

**Step 4: Update target_router.py**
- File: `TRAINING/target_router.py`
- Action: Add `TaskType` field to `TaskSpec`, keep string for backward compat
- Integration: `spec_from_target()` returns `TaskSpec` with both `task` (string) and `task_type` (enum)

**Step 5: Update training strategies**
- Files: `TRAINING/strategies/single_task.py`, `multi_task.py`
- Action: Use `TargetConfig` and `ModelConfig` instead of ad-hoc detection
- Integration: Replace `_determine_target_type()` with `target_config.task_type`

**Step 6: Update model trainers**
- Files: `TRAINING/model_fun/*_trainer.py`
- Action: Add task type validation in `train()` method
- Integration: Check `is_compatible()` before training

**Step 7: Update evaluation logic**
- Files: All training/evaluation scripts
- Action: Use `evaluate_by_task()` from `task_metrics.py`
- Integration: Replace hardcoded metric computation

---

## 7. Migration Steps (Concrete TODO List)

### Phase 1: Foundation ( COMPLETE)

1. **DONE**: Create `TaskType` enum and config classes (`SCRIPTS/utils/task_types.py`)
2. **DONE**: Create task-aware metrics (`SCRIPTS/utils/task_metrics.py`)
3. **DONE**: Enhance target validation (`SCRIPTS/utils/target_validation.py`)
4. **DONE**: Create leakage filtering (`SCRIPTS/utils/leakage_filtering.py`)
5. **DONE**: Refactor `rank_target_predictability.py` to use TaskType enum and correct model objectives

### Phase 2: Use Full Task-Aware Metrics ( IN PROGRESS)

6. **Update ranking script to compute full metrics**
 - File: `SCRIPTS/rank_target_predictability.py`
 - Action: After training each model, call `evaluate_by_task()` to get full metrics (IC, logloss, etc.)
 - Status: Models already use correct objectives, just need to compute full metrics

### Phase 3: Optional Enhancements (Future - Not Required)

**Note**: These are optional improvements. The ranking script already works correctly with task types.

7. **Add `task_type` to `target_configs.yaml` (optional)**
 - Makes task type explicit in config (currently inferred from column name)
 - Low priority - inference works fine

8. **Create target registry helper (optional)**
 - `SCRIPTS/utils/target_registry.py` - convenience wrapper
 - Not required - ranking script already loads targets correctly

**We are NOT doing:**
- Updating production training pipeline (`TRAINING/strategies/*.py`)
- Creating new model registry (you already have `ModelRegistry` for inference)
- Changing how models are trained in production

---

## 8. Risk / Edge Cases

### Edge Cases

1. **Ambiguous targets**
 - Some targets could be regression or classification depending on data
 - **Example**: `mfe_15m` could be continuous (regression) or thresholded (binary)
 - **Resolution**: Explicit `task_type` in config overrides inference

2. **Multiclass vs Binary**
 - `y_first_touch` is multiclass {-1, 0, +1} but could be treated as binary (hit/no-hit)
 - **Resolution**: Use explicit `task_type` in config

3. **Model family variations**
 - Some models have separate regressor/classifier classes (RandomForest)
 - Others use objective parameter (LightGBM, XGBoost)
 - **Resolution**: `ModelConfig.constructor` handles this variation

4. **Backward compatibility**
 - Existing code uses string task types (`'regression'`, `'classification'`)
 - **Resolution**: Keep string fields in `TaskSpec` for backward compat, add enum field

### Uncertainties

1. **Ranking task type**
 - `target_router.py` has ranking support but not fully implemented
 - **Question**: Should ranking be a separate `TaskType` or handled differently?
 - **Recommendation**: Add `RANKING` to enum when ready

2. **Ordinal classification**
 - `target_router.py` mentions ordinal targets but uses `'multiclass'` task type
 - **Question**: Should ordinal be separate task type?
 - **Recommendation**: Keep as `MULTICLASS_CLASSIFICATION` for now, can specialize later

3. **Model config location**
 - Model configs are in `CONFIG/multi_model_feature_selection.yaml` (for feature selection)
 - Also in `CONFIG/model_config/` (for training)
 - **Question**: Should these be unified?
 - **Recommendation**: Keep separate for now, create `ModelConfig` objects from both sources

4. **Target discovery**
 - `discover_all_targets()` in `rank_target_predictability.py` infers task type from data
 - **Question**: Should discovered targets be added to `target_configs.yaml`?
 - **Recommendation**: Yes, but make it optional (can run discovery mode)

---

## 9. Immediate Action Items

### High Priority (Do First)

1. **Verify model-task compatibility**
 - Audit all model trainers to ensure they use correct objectives for task types
 - Test: Create test cases for each (target, model) pair
 - File: Create `tests/test_model_task_compatibility.py`

2. **Add task_type to target_configs.yaml**
 - Populate `task_type` for all 63 targets
 - Use `TaskType.from_target_column()` to auto-populate, then verify manually

3. **Create target_registry.py**
 - Unify target loading across all scripts
 - Single source of truth for target definitions

### Medium Priority

4. **Update training strategies**
 - Migrate `single_task.py` and `multi_task.py` to use `TargetConfig`
 - Remove ad-hoc task type detection

5. **Add model registry**
 - Explicitly declare model capabilities
 - Enable querying "which models support this task type?"

### Low Priority (Polish)

6. **Update all evaluation code**
 - Use `evaluate_by_task()` everywhere
 - Remove hardcoded metric computation

7. **Documentation**
 - Update all docs with new architecture
 - Add migration guide

---

## 10. Success Criteria

 **Definition of Done:**

1. All targets have explicit `task_type` in config (no inference needed)
2. All models validate task type compatibility before training
3. All training scripts use `TargetConfig` and `ModelConfig` objects
4. All evaluation uses task-aware metrics
5. No ad-hoc task type detection (string matching) remains
6. Single source of truth for target definitions (`target_registry.py`)
7. Single source of truth for model capabilities (`model_registry.py`)
8. All tests pass with new architecture
9. Backward compatibility maintained (existing scripts still work)

---

**Next Steps:** Start with Phase 2 (Target Registry) and Phase 3 (Model Registry) to establish the foundation, then proceed with training pipeline integration.

