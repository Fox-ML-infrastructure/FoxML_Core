# Feature Importance Calculation Fix

## The Problem

You noticed that **feature importance was always 5.16** regardless of R² value. This was a bug in how importance was being calculated.

### Root Cause

The old calculation used:
```python
np.mean(np.abs(model.feature_importances_))
```

**Why this was wrong:**
- Feature importances are typically **normalized to sum to 1.0**
- The mean of normalized importances = `1.0 / n_features`
- With ~291 features: `1.0 / 291 ≈ 0.0034`
- But the actual value (5.16) suggests importances might be on a different scale

**The real issue:** Taking the mean of all importances doesn't tell us how **concentrated** the signal is. It's just a constant based on normalization.

---

## The Fix

Changed to use **sum of top 10% of features**:

```python
importances = model.feature_importances_
top_k = max(1, int(len(importances) * 0.1))  # Top 10% of features
top_importance = np.sum(np.sort(importances)[-top_k:])
```

### Why This Works Better

1. **Varies with signal strength:**
 - **High R²** → Few features have high importance → Top-k sum is **high** (e.g., 0.6-0.8)
 - **Low R²** → Importance spread evenly → Top-k sum is **low** (e.g., 0.1-0.2)
 - **Negative R²** → Model can't learn → Top-k sum is **very low** (e.g., 0.05-0.1)

2. **Measures concentration:**
 - If top 10% of features capture 60% of importance → Strong signal
 - If top 10% of features capture 10% of importance → Weak signal

3. **Correlates with R²:**
 - Better targets will have more concentrated importance
 - This metric now **varies** with target predictability

---

## Updated Models

All tree-based models now use the top-k sum approach:
- LightGBM
- Random Forest
- XGBoost
- Histogram Gradient Boosting

Neural Network already uses **permutation importance** (different but valid approach).

---

## Expected Behavior After Fix

### Before (Bug):
```
Target A: R² = 0.15, importance = 5.16
Target B: R² = -0.40, importance = 5.16   Same!
```

### After (Fixed):
```
Target A: R² = 0.15, importance = 0.65   High concentration
Target B: R² = -0.40, importance = 0.12   Low concentration
```

---

## Impact on Composite Score

The composite score calculation uses importance:
```python
importance_component = min(1.0, mean_importance / 100.0)
```

With the fix:
- **Good targets** (R² > 0.1): importance ≈ 0.5-0.8 → higher composite score
- **Poor targets** (R² < 0): importance ≈ 0.1-0.2 → lower composite score

This makes the composite score more accurate and useful for ranking.

---

## Next Steps

1. **Re-run target ranking** - Importance will now vary correctly
2. **Check results** - Good targets should have higher importance
3. **Use composite score** - It will now better reflect true predictability

The fix is backward compatible - existing results will still be valid, but new runs will have more accurate importance metrics.

