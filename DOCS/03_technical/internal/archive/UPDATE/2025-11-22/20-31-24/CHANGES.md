# Target Ranking Script - Task-Aware Metrics Implementation

**Date:** 2025-11-22
**Time:** 20:31:24

## Summary

Implemented full task-aware metrics computation in the target ranking script, ensuring regression targets use IC/R²/MSE and classification targets use AUC/LogLoss/Accuracy. All models now compute comprehensive metrics while maintaining backward compatibility.

## Changes Made

### 1. Created Unified Task Type System (`SCRIPTS/utils/task_types.py`)

**New File:** Core abstractions for task-aware pipeline
- `TaskType` enum: `REGRESSION`, `BINARY_CLASSIFICATION`, `MULTICLASS_CLASSIFICATION`
- `TargetConfig` dataclass: Unified target configuration with explicit `task_type`
- `ModelConfig` dataclass: Model capabilities with `supported_tasks` set
- `is_compatible()`: Validates model-target compatibility
- `TaskType.from_target_column()`: Infers task type from column name and data

**Key Features:**
- Single source of truth for task types (no more string matching)
- Explicit model capability declarations
- Type-safe task type handling

### 2. Created Task-Aware Metrics (`SCRIPTS/utils/task_metrics.py`)

**New File:** Task-specific metric evaluation
- `eval_regression()`: IC (Information Coefficient), R², MSE, MAE
- `eval_binary_classification()`: ROC AUC, LogLoss, Accuracy
- `eval_multiclass()`: Cross-entropy, Accuracy, Macro F1
- `evaluate_by_task()`: Dispatches to correct metric function based on TaskType
- `compute_composite_score()`: Combines metrics into single ranking score

**Key Features:**
- IC computation for regression (correlation between predictions and actuals)
- Proper probability handling for classification (predict_proba)
- Handles NaN/inf gracefully

### 3. Enhanced Target Validation (`SCRIPTS/utils/target_validation.py`)

**Updated:** Added TaskType support
- `validate_target()`: Now accepts optional `task_type` parameter
- Task-specific validation:
 - Regression: Checks variance (rejects zero-variance targets)
 - Binary: Ensures exactly 2 classes with minimum samples per class
 - Multiclass: Validates class balance for CV compatibility
- `check_cv_compatibility()`: Task-aware CV fold validation

### 4. Implemented Target-Aware Leakage Filtering (`SCRIPTS/utils/leakage_filtering.py`)

**New File:** Prevents data leakage based on target type
- `filter_features_for_target()`: Main filtering function
- Target-aware rules:
 - Forward returns: Excludes overlapping forward returns, future path features
 - Barrier targets: Excludes barrier hit features (tth_*, hit_direction_*)
 - Always excludes: first_touch targets (leaked by definition)
- Horizon-aware filtering (excludes features with longer/overlapping horizons)

### 5. Refactored Ranking Script (`SCRIPTS/rank_target_predictability.py`)

**Major Updates:**

#### 5.1 Task Type Detection
- `discover_all_targets()`: Now returns `Dict[str, TargetConfig]` with explicit task types
- `prepare_features_and_target()`: Returns `TaskType` along with X, y, feature_names
- Uses `TaskType.from_target_column()` for consistent inference

#### 5.2 Model Training Updates
- `train_and_evaluate_models()`: Now accepts `task_type: TaskType` parameter
- All models use correct objectives:
 - LightGBM: `LGBMClassifier(objective='binary')` for binary, `LGBMRegressor` for regression
 - Random Forest: `RandomForestClassifier` vs `RandomForestRegressor`
 - XGBoost: `XGBClassifier` vs `XGBRegressor`
 - CatBoost: `CatBoostClassifier` vs `CatBoostRegressor`
 - Neural Network: `MLPClassifier` vs `MLPRegressor`

#### 5.3 Full Metrics Computation
- Added `_compute_and_store_metrics()` helper function:
 - Computes full task-aware metrics using `evaluate_by_task()`
 - Stores in `model_metrics` dict (full metrics)
 - Maintains `model_scores` dict (primary scores for backward compat)
- Updated all model sections:
 - LightGBM: Full metrics after training
 - Random Forest: Full metrics computation
 - Neural Network: Full metrics with scaled data handling
 - XGBoost: Full metrics computation
 - CatBoost: Full metrics computation
 - Histogram Gradient Boosting: Full metrics computation
 - Lasso: Full metrics (regression-only)

#### 5.4 Scoring Metric Selection
- Regression: Uses `'r2'` scoring for CV
- Binary Classification: Uses `'roc_auc'` scoring for CV
- Multiclass: Uses `'accuracy'` scoring for CV

### 6. Created Analysis Documentation (`docs/TARGET_MODEL_PIPELINE_ANALYSIS.md`)

**New File:** Comprehensive analysis of target/model pipeline
- Repo map and component inventory
- Target and model inventory with task type mapping
- Problem identification and solutions
- Migration plan (clarified scope - ranking script only, not full pipeline)

## Technical Details

### Metrics Now Computed

**For Regression Targets:**
- `r2`: Coefficient of determination
- `ic`: Information Coefficient (correlation between predictions and actuals)
- `mse`: Mean Squared Error
- `mae`: Mean Absolute Error

**For Binary Classification Targets:**
- `roc_auc`: Area Under ROC Curve
- `logloss`: Binary cross-entropy loss
- `accuracy`: Classification accuracy

**For Multiclass Classification Targets:**
- `cross_entropy`: Multiclass log loss
- `accuracy`: Classification accuracy
- `macro_f1`: Macro-averaged F1 score

### Backward Compatibility

- Primary scores (R², AUC, accuracy) still available in `model_scores` dict
- CLI interface unchanged - all existing arguments work
- Output format unchanged - rankings still work the same way
- Full metrics available in `model_metrics` dict for advanced analysis

## Files Modified

1. `SCRIPTS/rank_target_predictability.py` (major refactor)
 - Added task type parameter to `train_and_evaluate_models()`
 - Added `_compute_and_store_metrics()` helper
 - Updated all model training sections
 - Updated `discover_all_targets()` to return TargetConfig objects
 - Updated `prepare_features_and_target()` to return TaskType

2. `SCRIPTS/utils/task_types.py` (new file)
 - TaskType enum
 - TargetConfig and ModelConfig dataclasses
 - Compatibility checking functions

3. `SCRIPTS/utils/task_metrics.py` (new file)
 - Task-aware metric evaluation functions
 - Composite score computation

4. `SCRIPTS/utils/target_validation.py` (enhanced)
 - Added TaskType parameter support
 - Task-specific validation logic

5. `SCRIPTS/utils/leakage_filtering.py` (new file)
 - Target-aware feature filtering
 - Horizon-aware exclusion rules

6. `docs/TARGET_MODEL_PIPELINE_ANALYSIS.md` (new file)
 - Comprehensive analysis document
 - Migration plan and scope clarification

## Impact

### Benefits
- **Correct Metrics**: Regression targets now get IC (more informative than R² alone)
- **Better Evaluation**: Classification targets get LogLoss (better than accuracy alone)
- **Type Safety**: TaskType enum prevents typos and inconsistencies
- **Leakage Prevention**: Target-aware filtering prevents temporal leakage
- **Extensibility**: Easy to add new task types (ranking, ordinal, etc.)

### No Breaking Changes
- CLI interface identical
- Output format compatible
- Existing scripts continue to work
- Primary scores still available

## Testing Recommendations

1. **Verify metrics are computed:**
   ```python
   # Check that model_metrics contains full metrics
   assert 'ic' in model_metrics['lightgbm']  # for regression
   assert 'logloss' in model_metrics['lightgbm']  # for classification
   ```

2. **Verify task type detection:**
   ```python
   # Check that targets get correct task types
   assert target_config.task_type == TaskType.BINARY_CLASSIFICATION  # for y_will_peak_*
   assert target_config.task_type == TaskType.REGRESSION  # for fwd_ret_*
   ```

3. **Verify model objectives:**
   ```python
   # Check that models use correct objectives
   assert isinstance(model, LGBMClassifier)  # for classification
   assert isinstance(model, LGBMRegressor)  # for regression
   ```

## Next Steps (Optional)

- Add `task_type` field to `target_configs.yaml` (currently inferred)
- Create target registry helper (convenience wrapper)
- Add IC to composite score calculation for regression targets
