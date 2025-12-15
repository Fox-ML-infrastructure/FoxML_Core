# CatBoost GPU Fixes (2025-12-15)

## Summary

Critical fixes for CatBoost GPU mode compatibility, sklearn cloning support, and feature importance output. These fixes enable CatBoost to work correctly with GPU acceleration and ensure all feature importances are properly saved to the results directory.

---

## Critical Bug Fixes

### 1. CatBoost GPU Requires Pool Objects (Not Numpy Arrays)

**Problem:** When CatBoost uses GPU mode (`task_type='GPU'`), it requires `catboost.Pool` objects instead of numpy arrays for the `fit()` method. The code was passing numpy arrays directly, causing:
```
Invalid data type=<class 'numpy.ndarray'>, must be catboost.Pool
```

**Impact:** CatBoost GPU training was completely broken - all GPU-enabled CatBoost training failed with type errors.

**Fix:**
- Created `CatBoostGPUWrapper` class that automatically converts numpy arrays to Pool objects when GPU mode is enabled
- Wrapper intercepts `fit()`, `predict()`, and `score()` calls
- Converts numpy arrays to Pool objects only when GPU is enabled (CPU mode unchanged)
- Works with both `cross_val_score()` (sklearn) and direct `model.fit()` calls

**Files Changed:**
- `TRAINING/ranking/predictability/model_evaluation.py`
- `TRAINING/ranking/multi_model_feature_selection.py`

**Result:** CatBoost GPU training now works correctly with automatic Pool conversion.

---

### 2. Sklearn Clone Compatibility for CatBoost Wrapper

**Problem:** The `CatBoostGPUWrapper` class was not sklearn-compatible, causing errors when sklearn's `cross_val_score()` tried to clone the model:
```
Cannot clone object... as the constructor either does not set or modifies parameter cat_features
```

**Impact:** Cross-validation with CatBoost GPU mode failed during model cloning.

**Fix:**
- Implemented `get_params()` and `set_params()` methods for sklearn compatibility
- Wrapper can be instantiated from kwargs (for sklearn cloning)
- Stores model class type (`_model_class`) to recreate correct model (Classifier vs Regressor)
- Handles `cat_features` parameter correctly for sklearn's clone validation
- Ensures round-trip validation: `get_params()` → `__init__(**params)` → `get_params()` matches

**Files Changed:**
- `TRAINING/ranking/predictability/model_evaluation.py`
- `TRAINING/ranking/multi_model_feature_selection.py`

**Result:** CatBoost wrapper is now fully sklearn-compatible and works with cross-validation.

---

### 3. CatBoost Feature Importances Not Being Saved

**Problem:** CatBoost was computing feature importance but not adding it to the `all_feature_importances` dictionary, which is used to generate CSV files. This meant `catboost_importances.csv` was never created in the results directory.

**Impact:** CatBoost feature importances were computed but not saved, making it impossible to analyze CatBoost feature rankings alongside other models (lightgbm, xgboost, random_forest).

**Fix:**
- Added code to store CatBoost importances in `all_feature_importances` dictionary
- Uses same format as other models (pandas Series with feature_names alignment)
- Handles GPU wrapper by accessing `base_model` when needed
- Converts data to Pool for GPU mode when computing importance
- Aligns importances with `feature_names` order (same pattern as other models)

**Files Changed:**
- `TRAINING/ranking/predictability/model_evaluation.py`

**Result:** CatBoost feature importances are now saved to `catboost_importances.csv` in:
```
RESULTS/.../REPRODUCIBILITY/TARGET_RANKING/CROSS_SECTIONAL/{target}/feature_importances/catboost_importances.csv
```

---

## Technical Details

### GPU Pool Conversion

When CatBoost GPU mode is enabled, the wrapper:
1. Detects GPU mode: `task_type='GPU'` in params
2. Creates wrapper instance with base model
3. Intercepts `fit()` calls: converts numpy arrays → Pool objects
4. Delegates to base model: base model receives Pool objects
5. Works transparently: sklearn's `cross_val_score()` works without changes

### Sklearn Compatibility

The wrapper implements sklearn's estimator protocol:
- `get_params(deep=True)`: Returns all parameters including base model params
- `set_params(**params)`: Updates both wrapper and base model params
- `__init__(**kwargs)`: Can be instantiated from params (for cloning)
- Model class inference: Uses `loss_function` or stored `_model_class` to determine Classifier vs Regressor

### Feature Importance Extraction

For GPU mode:
- Accesses `base_model` from wrapper
- Converts numpy array to Pool object with categorical features
- Calls `get_feature_importance()` on base model with Pool data
- Aligns results with feature_names order for consistency

---

## Files Changed

- `TRAINING/ranking/predictability/model_evaluation.py` - Added GPU wrapper, sklearn compatibility, feature importance saving
- `TRAINING/ranking/multi_model_feature_selection.py` - Added GPU wrapper and sklearn compatibility

---

## Testing

All fixes tested with:
- GPU mode enabled (`task_type='GPU'`)
- CPU mode (backward compatible)
- Cross-validation with sklearn
- Feature importance extraction and saving

---

## Related Issues

- CatBoost GPU mode was completely broken before these fixes
- Feature importances were missing from results directory
- Sklearn cross-validation failed with CatBoost GPU

---

## Migration Notes

No migration needed - these are bug fixes that restore expected functionality. CatBoost GPU mode should now work out of the box when `gpu.catboost.task_type: "GPU"` is set in config.
