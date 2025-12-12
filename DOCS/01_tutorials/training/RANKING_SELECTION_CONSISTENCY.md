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
Previously, interval detection could produce warnings when auto-detecting from timestamps, especially when `data.bar_interval` was already specified in config. Large gaps (overnight/weekend) were contaminating detection, causing false warnings.

### Solution
The `explicit_interval` parameter is now wired through the entire call chain. Additionally, interval detection has been improved with:
1. **Median-based gap filtering**: Ignores gaps > 10x median (configurable via `pipeline.data_interval.max_gap_factor`)
2. **Fixed interval mode**: Skip auto-detection entirely when interval is known (`interval_detection.mode=fixed`)
3. **Reduced noise**: Warnings downgraded to INFO level when default is used correctly

The call chain:

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

**With experiment config (recommended - fixed mode):**
```yaml
# CONFIG/experiments/my_experiment.yaml
data:
  bar_interval: "5m"  # Explicit interval
  interval_detection:
    mode: fixed  # Skip auto-detection, use bar_interval directly
```

```bash
python TRAINING/train.py --experiment-config my_experiment --auto-targets --auto-features
```

**Result:** No interval auto-detection warnings in logs. Interval is used directly from config.

**With auto-detection (improved):**
```yaml
# CONFIG/experiments/my_experiment.yaml
data:
  # bar_interval not set - will auto-detect
  interval_detection:
    mode: auto  # Default: auto-detect with gap filtering
```

**Result:** Auto-detection runs with improved gap filtering. Large gaps (overnight/weekend) are ignored. Warnings downgraded to INFO level when default is used correctly.

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

**In Ranking (`ranking/predictability/model_evaluation.py`):**
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

### Boruta Statistical Gatekeeper

#### Overview
Boruta is implemented as a **statistical gatekeeper**, not just another importance scorer. It uses ExtraTrees with stability-oriented hyperparams to test whether features are demonstrably better than noise, then modifies consensus scores via bonuses/penalties.

#### Implementation Details

**Base Estimator:**
- Uses `ExtraTreesClassifier/Regressor` (more random than RF, better for stability testing)
- Hyperparams optimized for stable importance: `n_estimators: 500`, `max_depth: 6`, `perc: 95`
- Configurable `class_weight`, `n_jobs`, `verbose` via YAML
- **Note**: Boruta's internal `ExtraTreesClassifier` is trained on a transformed subset of features (confirmed/rejected/tentative selection), not the full feature set. As a result, Boruta uses `train_score = math.nan` (not a numeric score) since it's a selector, not a predictor. This prevents feature count mismatch errors and false "failed" status.

**Gatekeeper Role:**
- **Excluded from base consensus** — Boruta does not contribute to `consensus_score_base`
- **Applied as modifier** — Boruta bonuses/penalties modify base consensus to produce `consensus_score`
- **Scoring system:**
  - Confirmed features: `+1.0` → `+boruta_confirm_bonus` (default: `+0.2`)
  - Tentative features: `+0.3` → no modifier (neutral)
  - Rejected features: `-1.0` → `+boruta_reject_penalty` (default: `-0.3`)

**Output Columns:**
- `consensus_score_base` — Base consensus (model families only, without Boruta)
- `consensus_score` — Final consensus (with Boruta gatekeeper effect) — used for sorting/selection
- `boruta_gate_effect` — Pure Boruta effect (final - base) for debugging
- `boruta_gate_score` — Raw Boruta scores (1.0/0.3/-1.0)
- `boruta_confirmed` — Boolean mask for confirmed features
- `boruta_rejected` — Boolean mask for rejected features
- `boruta_tentative` — Boolean mask for tentative features

**Magnitude Sanity Checks:**
- Warns if `max(|bonus|, |penalty|) / base_range > threshold` (default: 0.5)
- Helps ensure Boruta is a "gentle nudge" not "god-logic"
- Configurable via `boruta_magnitude_warning_threshold` in aggregation config

**Ranking Impact Metric:**
- Compares top-K sets: base consensus vs final consensus
- Logs: `"X features changed in top-K set (base vs final). Ratio: Y%"`
- Interpretation:
  - `changed ~ 0%` → gate is inert
  - `changed ~ 10-30%` → meaningful but not dominant
  - `changed ~ 50%+` → Boruta is dominating (reduce bonuses/penalties)

**Configuration:**
```yaml
# CONFIG/feature_selection/multi_model.yaml
boruta:
  enabled: true
  config:
    n_estimators: 500
    max_depth: 6
    perc: 95
    n_jobs: 1
    verbose: 0
    class_weight: "auto"  # auto/none/dict

aggregation:
  boruta_confirm_bonus: 0.2
  boruta_reject_penalty: -0.3
  boruta_confirmed_threshold: 0.9
  boruta_tentative_threshold: 0.0
  boruta_magnitude_warning_threshold: 0.5
```

**Debug Output:**
- `feature_importance_with_boruta_debug.csv` — Explicit debug view with all Boruta columns
- Sorted by final consensus score for easy inspection
- Includes base score, final score, gate effect, and Boruta flags

## CatBoost Configuration

### Problem
Previously, CatBoost could use incorrect loss functions (e.g., `RMSE` for binary classification) if not explicitly configured. Additionally, parameter conflicts could occur when defaults injection added synonyms like `n_estimators` while config already had `iterations`, or when both `random_state` and `random_seed` were present.

### Solution
CatBoost now auto-detects target type and sets appropriate loss function. Parameter sanitization automatically resolves conflicts:

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

### Parameter Sanitization

All model constructors use `clean_config_for_estimator()` to prevent parameter conflicts:

- **CatBoost iteration synonyms**: Removes `n_estimators`, `num_boost_round`, `num_trees` if `iterations` is present (CatBoost accepts only one)
- **CatBoost random_state/random_seed**: Converts `random_state` to `random_seed` or removes if duplicate
- **MLPRegressor verbose**: Sanitizes `verbose=-1` to `verbose=0` (sklearn requires `>= 0`)
- **RandomForest verbose**: Removes negative verbose values

This ensures models work correctly even when global defaults injection adds conflicting parameters. See [Config Cleaner API](../../02_reference/configuration/CONFIG_CLEANER_API.md) for details.

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

