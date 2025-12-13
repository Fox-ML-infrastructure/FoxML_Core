# Stability and Dtype Fixes — 2025-12-13

**Critical Fixes for Random Stability Metrics, CatBoost Dtype Issues, and Snapshot Grouping**

## Summary

Fixed three critical issues causing random stability metrics and fake performance:
1. **Stability snapshots using wrong method identifier** - was using `importance_method` instead of `model_family`
2. **CatBoost treating numeric columns as text/categorical** - causing fake perfect scores
3. **Missing documentation** - clarified that stability must be computed per-model-family

---

## 1. Stability Snapshots Using Wrong Method Identifier

### Root Cause
Snapshots were being saved with `method=importance_method` (e.g., "native", "shap", "permutation") instead of `method=model_family` (e.g., "lightgbm", "ridge", "elastic_net"). This caused stability analysis to compare snapshots from different model families, which naturally have low overlap because they use different importance definitions.

### Expected Random Overlap
With n=39 features and k=20:
- Expected intersection size = k²/n = 400/39 ≈ 10.26
- Expected overlap fraction = 10.26/20 ≈ 0.513

Observed overlap ~0.50 is **literally chance-level**, confirming snapshots were being compared incorrectly.

### Fix
**Files Modified:**
- `TRAINING/ranking/multi_model_feature_selection.py` (line 2383)
- `TRAINING/ranking/feature_selector.py` (line 314 - already correct)
- `TRAINING/stability/feature_importance/analysis.py` (added critical documentation)

**Changes:**
```python
# BEFORE (WRONG - uses importance_method):
save_snapshot_from_series_hook(
    target_name=target_column,
    method=method,  # ❌ This is "native", "shap", etc. - WRONG!
    importance_series=importance,
    ...
)

# AFTER (CORRECT - uses model_family):
save_snapshot_from_series_hook(
    target_name=target_column,
    method=family_name,  # ✅ This is "lightgbm", "ridge", etc. - CORRECT!
    importance_series=importance,
    ...
)
```

**Key Principle:**
- **Stability must be computed per-model-family** (comparing same family across runs)
- **NOT across different families** (RFE vs Boruta vs Lasso will naturally have low overlap)
- Each model family uses different importance definitions (gain vs split vs coef vs ranking)

---

## 2. CatBoost Treating Numeric Columns as Text/Categorical

### Root Cause
CatBoost was receiving features with `object` dtype (caused by NaN + mixed types in pandas), which it interprets as text/categorical. This allows CatBoost to "memorize" patterns and achieve perfect scores (1.0) while generalizing poorly.

### Symptoms
- CatBoost warnings: "Detected 39 text/object columns: ['volume', 'vwap', ...]"
- Perfect scores: `catboost: score=1.0000`
- Low stability: Random importance rankings (overlap ~0.50)

### Fix
**Files Modified:**
- `TRAINING/ranking/multi_model_feature_selection.py` (lines 1419-1427)

**Changes:**
```python
# BEFORE (CatBoost receives potentially object-dtype X):
model.fit(X, y)  # ❌ X may have object dtype

# AFTER (Hard-cast all numeric columns to float32):
# Convert X to DataFrame to check/fix dtypes
X_df = pd.DataFrame(X, columns=feature_names)

# Hard-cast all numeric columns to float32 (prevents object dtype)
for col in X_df.columns:
    if pd.api.types.is_numeric_dtype(X_df[col]):
        X_df[col] = X_df[col].astype('float32')  # ✅ Explicit float32
    elif X_df[col].dtype.name in ['object', 'string', 'category']:
        # Try to convert to numeric, drop if fails
        try:
            X_df[col] = pd.to_numeric(X_df[col], errors='coerce').astype('float32')
        except Exception:
            logger.warning(f"Dropping non-numeric column {col}")
            X_df = X_df.drop(columns=[col])
            feature_names = [f for f in feature_names if f != col]

X_catboost = X_df.values.astype('float32')  # ✅ Guaranteed float32

# Explicitly tell CatBoost there are no categorical features
if 'cat_features' not in cb_config:
    cb_config['cat_features'] = []  # ✅ No categoricals by default

model.fit(X_catboost, y)  # ✅ Now receives float32 numeric data
```

**Key Improvements:**
- ✅ Hard-cast all numeric columns to `float32` before CatBoost
- ✅ Explicitly set `cat_features=[]` to prevent CatBoost from guessing
- ✅ Drop or convert non-numeric columns (don't let them become object dtype)
- ✅ Prevents CatBoost from "memorizing" via text/categorical interpretation

---

## 3. Documentation: Stability Must Be Per-Model-Family

### Added Documentation
**Files Modified:**
- `TRAINING/stability/feature_importance/analysis.py` (docstrings)
- `TRAINING/stability/feature_importance/hooks.py` (already had per-method analysis)

**Key Points:**
1. **Stability is computed per-model-family** (e.g., LightGBM gain across runs)
2. **NOT across different families** (RFE vs Boruta will naturally have low overlap)
3. **Snapshots are grouped by `(target_name, method, universe_id)`** where `method` = model_family
4. **Overlap compares top-K by feature name** (not magnitude, since magnitudes aren't comparable)

---

## Expected Behavior After Fixes

### Stability Metrics
- **Per-model-family stability**: Should see overlap > 0.7 for stable methods (LightGBM, XGBoost, RF)
- **High-variance methods**: May see overlap ~0.5-0.6 (stability_selection, boruta, rfe, neural_network)
- **Random baseline**: With n=39, k=20, expected random overlap ≈ 0.51

### CatBoost Performance
- **Scores should be realistic**: No more perfect 1.0 scores (unless truly perfect signal)
- **Generalization gap**: Train score vs validation score should be reasonable
- **No dtype warnings**: Should not see "Detected X text/object columns" warnings

### Snapshot Grouping
- **Snapshots grouped by model_family**: `target_name/lightgbm/`, `target_name/ridge/`, etc.
- **Stability computed within each group**: Comparing LightGBM run1 vs LightGBM run2
- **Not comparing across groups**: Not comparing LightGBM vs RFE (different methods)

---

## Verification Steps

1. **Check snapshot directories**:
   ```bash
   ls artifacts/feature_importance/{target_name}/
   # Should see: lightgbm/, xgboost/, ridge/, elastic_net/, etc.
   # NOT: native/, shap/, permutation/
   ```

2. **Check stability reports**:
   ```bash
   cat artifacts/feature_importance/stability_reports/{target_name}_lightgbm.txt
   # Should show overlap > 0.7 for stable methods
   # Should show overlap ~0.5-0.6 for high-variance methods
   ```

3. **Check CatBoost logs**:
   - Should NOT see "Detected X text/object columns" warnings
   - Should NOT see perfect scores (1.0) unless truly perfect signal
   - Should see realistic train/validation score gaps

4. **Check snapshot method names**:
   ```python
   from TRAINING.stability.feature_importance.io import load_snapshots
   snapshots = load_snapshots(base_dir, target_name, method="lightgbm")
   # Should load snapshots from lightgbm/ directory
   # NOT from native/ or shap/ directories
   ```

---

## Related Issues

- **Random stability metrics**: Fixed by using model_family as method identifier
- **CatBoost fake performance**: Fixed by hard-casting to float32 and setting cat_features=[]
- **Perfect scores + random importances**: Should be resolved after dtype fix

---

## Migration Notes

### For Users
- **No action required** - All fixes are backward compatible
- **Existing snapshots**: Old snapshots with wrong method names will be in separate directories
- **New snapshots**: Will use correct model_family as method identifier

### For Developers
- **Always use model_family as method identifier** when saving snapshots
- **Never compare stability across different model families** (they use different importance definitions)
- **Always hard-cast numeric features to float32** before CatBoost (prevents object dtype issues)
- **Explicitly set cat_features=[]** for CatBoost unless you have real categoricals

---

## Files Modified

1. `TRAINING/ranking/multi_model_feature_selection.py`
   - Fixed snapshot method identifier (line 2383)
   - Added CatBoost dtype fix (lines 1419-1427)

2. `TRAINING/ranking/feature_selector.py`
   - Already correct (uses model_family)

3. `TRAINING/stability/feature_importance/analysis.py`
   - Added critical documentation about per-model-family stability

4. `TRAINING/stability/feature_importance/hooks.py`
   - Already had per-method analysis (no changes needed)

---

## Next Steps

1. **Re-run feature selection** to generate new snapshots with correct method identifiers
2. **Verify stability metrics** are now per-model-family (overlap should be > 0.7 for stable methods)
3. **Verify CatBoost scores** are realistic (no more perfect 1.0 scores)
4. **Monitor for dtype warnings** (should not see "Detected X text/object columns")

All fixes are complete and backward compatible.
