# Ranking and Selection Pipeline Consistency

This document explains how the ranking and selection pipelines maintain consistent behavior for interval handling, preprocessing, and model configuration.

## Overview

The target ranking and feature selection pipelines are now **behaviorally identical** in their handling of:
1. **Interval detection** - Respects `data.bar_interval` from experiment config
2. **Sklearn preprocessing** - Uses shared `make_sklearn_dense_X()` helper
3. **CatBoost configuration** - Auto-detects target type and sets loss function

This ensures consistent results and eliminates configuration drift between ranking and selection steps.

## Interval Handling

### Problem
Previously, interval detection could produce warnings when auto-detecting from timestamps, especially when `data.bar_interval` was already specified in config.

### Solution
The `explicit_interval` parameter is now wired through the entire call chain:

```
orchestrator (extracts from experiment_config.data.bar_interval)
  ↓
rank_targets(explicit_interval=...)
  ↓
evaluate_target_predictability(explicit_interval=...)
  ↓
train_and_evaluate_models(explicit_interval=...)
  ↓
prepare_features_and_target(explicit_interval=...)
  ↓
detect_interval_from_dataframe(explicit_interval=...)
```

### Usage

**With experiment config (recommended):**
```yaml
# CONFIG/experiments/my_experiment.yaml
data:
  bar_interval: "5m"  # Explicit interval
```

```bash
python TRAINING/train.py --experiment-config my_experiment --auto-targets --auto-features
```

**Result:** No interval auto-detection warnings in logs.

### Benefits
- ✅ No spurious warnings when interval is known
- ✅ Consistent interval handling across ranking and selection
- ✅ Proper horizon conversion for leakage filtering
- ✅ Backward compatible (defaults to auto-detection if not specified)

## Sklearn Preprocessing

### Problem
Previously, sklearn-based models (Lasso, Mutual Information, Univariate Selection, Boruta, Stability Selection) used ad-hoc `SimpleImputer` calls with inconsistent behavior.

### Solution
All sklearn models now use the shared `make_sklearn_dense_X()` helper from `TRAINING/utils/sklearn_safe.py`:

```python
from TRAINING.utils.sklearn_safe import make_sklearn_dense_X

# Consistent preprocessing for all sklearn models
X_dense, feature_names_dense = make_sklearn_dense_X(X, feature_names)
# Returns: dense float32 array, median-imputed, inf values handled
```

### Applied To

**In Ranking (`rank_target_predictability.py`):**
- Lasso
- Mutual Information
- Univariate Selection
- Boruta
- Stability Selection

**In Selection (`multi_model_feature_selection.py`):**
- Lasso
- Mutual Information
- Univariate Selection
- Boruta

### Benefits
- ✅ Consistent NaN handling (median imputation)
- ✅ Consistent dtype handling (float32)
- ✅ Consistent inf handling (replaced with NaN, then imputed)
- ✅ Same behavior in ranking and selection

### Tree Models (Not Affected)
Tree-based models (LightGBM, XGBoost, Random Forest, CatBoost) continue to use raw data as they handle NaNs natively.

## CatBoost Configuration

### Problem
Previously, CatBoost could use incorrect loss functions (e.g., `RMSE` for binary classification) if not explicitly configured.

### Solution
CatBoost now auto-detects target type and sets appropriate loss function:

```python
from TRAINING.utils.target_utils import is_classification_target, is_binary_classification_target

if "loss_function" not in params:
    if is_classification_target(y):
        if is_binary_classification_target(y):
            params["loss_function"] = "Logloss"
        else:
            params["loss_function"] = "MultiClass"
    else:
        params["loss_function"] = "RMSE"
```

### Loss Function Selection

| Target Type | Auto-Detected Loss Function |
|------------|----------------------------|
| Binary classification | `Logloss` |
| Multiclass classification | `MultiClass` |
| Regression | `RMSE` |

### YAML Override
You can still override in config if needed:

```yaml
model_families:
  catboost:
    enabled: true
    loss_function: "CrossEntropy"  # Override auto-detection
```

### Benefits
- ✅ Correct loss function for all target types
- ✅ No manual configuration needed
- ✅ Consistent behavior in ranking and selection
- ✅ YAML can still override for special cases

## Shared Utilities

### Target Type Detection

**New module:** `TRAINING/utils/target_utils.py`

Provides reusable helpers for detecting target types consistently across ranking and selection:

```python
from TRAINING.utils.target_utils import (
    is_classification_target,
    is_binary_classification_target,
    is_multiclass_target
)

# Used by CatBoost and other model builders
if is_classification_target(y):
    if is_binary_classification_target(y):
        # Binary classification
    elif is_multiclass_target(y):
        # Multiclass classification
else:
    # Regression
```

**Functions:**
- `is_classification_target(y, max_classes=20)` - Detects if target is classification (discrete) vs regression (continuous)
- `is_binary_classification_target(y)` - Detects if target is binary classification (exactly 2 classes, typically 0/1)
- `is_multiclass_target(y, max_classes=10)` - Detects if target is multiclass classification (3+ classes, but not too many)

**Used by:**
- CatBoost model builder (ranking and selection)
- Other model builders that need target type detection

### Sklearn Preprocessing

**Module:** `TRAINING/utils/sklearn_safe.py`

Provides consistent preprocessing for all sklearn-based models:

```python
from TRAINING.utils.sklearn_safe import make_sklearn_dense_X

# Consistent preprocessing for all sklearn models
X_dense, feature_names_dense = make_sklearn_dense_X(X, feature_names)
# Returns: dense float32 array, median-imputed, inf values handled
```

**Function:**
- `make_sklearn_dense_X(X, feature_names=None)` - Converts tabular data to dense float32 numpy array with:
  - Median imputation for NaNs
  - Inf values replaced with NaN, then imputed
  - Consistent dtype (float32)
  - Feature name mapping preserved

**Used by:**
- Lasso (ranking and selection)
- Mutual Information (ranking and selection)
- Univariate Selection (ranking and selection)
- Boruta (ranking and selection)
- Stability Selection (ranking)

**Note:** Tree-based models (LightGBM, XGBoost, Random Forest, CatBoost) continue to use raw data as they handle NaNs natively.

## Verification

### Check Interval Handling

Look for absence of warnings in logs:
```
# Should NOT see:
# WARNING: Auto-detection unclear (444000000000000.0m...)
```

### Check Sklearn Models

All sklearn models should complete without NaN/dtype errors:
```
# Should see successful completion for:
# - Lasso
# - Mutual Information
# - Univariate Selection
# - Boruta
# - Stability Selection
```

### Check CatBoost

CatBoost should run successfully for both classification and regression:
```
# Binary classification: Should use Logloss
# Multiclass: Should use MultiClass
# Regression: Should use RMSE
```

## Migration Guide

### For Existing Code

**No changes required** - all fixes are backward compatible:
- Interval auto-detection still works if `explicit_interval` not provided
- Sklearn models still work (just use shared helper now)
- CatBoost still works (just auto-detects loss function now)

### For New Code

**Use experiment configs** (recommended):
```yaml
data:
  bar_interval: "5m"  # Set explicitly
```

**Let CatBoost auto-detect:**
```yaml
model_families:
  catboost:
    enabled: true
    # Don't specify loss_function - let it auto-detect
```

## Related Documentation

- [Intelligent Training Tutorial](INTELLIGENT_TRAINING_TUTORIAL.md) - Complete pipeline guide
- [Modular Config System](../../02_reference/configuration/MODULAR_CONFIG_SYSTEM.md) - Config structure (includes `logging_config.yaml`)
- [Usage Examples](../../02_reference/configuration/USAGE_EXAMPLES.md) - Practical examples (includes interval config and CatBoost examples)
- [Config Loader API](../../02_reference/configuration/CONFIG_LOADER_API.md) - Logging config utilities
- [Module Reference](../../02_reference/api/MODULE_REFERENCE.md) - API reference for `target_utils.py` and `sklearn_safe.py`
- [Configuration System Overview](../../02_reference/configuration/README.md) - Complete config system overview

