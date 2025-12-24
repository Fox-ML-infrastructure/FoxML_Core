# Additional Models for Feature Selection

## Current Models (Already Implemented)

 **Tree-Based:**
- LightGBM (gain importance)
- XGBoost (gain importance)
- Random Forest (gini importance)
- Histogram Gradient Boosting (sklearn)

 **Neural:**
- Neural Network (permutation importance)

 **Linear (Disabled):**
- Ridge (coefficient magnitude)
- ElasticNet (sparse coefficients)

---

## Recommended Additional Models

### 1. **CatBoost** **HIGH VALUE**

**Why:** Another gradient boosting framework with different splitting strategy

**Benefits:**
- Handles categorical features natively
- Different feature importance calculation (PredictionValuesChange)
- Often finds features that LightGBM/XGBoost miss
- GPU support

**Implementation:**
```python
import catboost as cb

model = cb.CatBoostRegressor(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    loss_function='RMSE',
    verbose=False,
    random_seed=42
)
model.fit(X, y)
importance = model.get_feature_importance()  # PredictionValuesChange
```

**When to use:** Always - adds diversity to tree-based ensemble

---

### 2. **Lasso (L1 Regularization)** **HIGH VALUE**

**Why:** Explicitly selects features by driving coefficients to zero

**Benefits:**
- **Sparse selection** - automatically drops irrelevant features
- **Linear relationships** - finds features with direct linear impact
- **Interpretable** - coefficient magnitude = importance
- **Fast** - very quick to train

**Implementation:**
```python
from sklearn.linear_model import Lasso

model = Lasso(alpha=0.1, max_iter=1000, random_state=42)
model.fit(X, y)
importance = np.abs(model.coef_)  # Absolute coefficients
```

**When to use:** When you want explicit feature selection (coefficients = 0 = dropped)

---

### 3. **Mutual Information** **HIGH VALUE**

**Why:** Information-theoretic measure, model-agnostic

**Benefits:**
- **No model training** - just measures information content
- **Non-linear relationships** - captures any dependency
- **Fast** - O(n_features × n_samples)
- **Target-independent** - can rank features without training

**Implementation:**
```python
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

# For regression
if is_regression:
    importance = mutual_info_regression(X, y, random_state=42)
else:
    importance = mutual_info_classif(X, y, random_state=42)
```

**When to use:** Always - provides baseline information content

---

### 4. **Univariate Feature Selection** **MEDIUM VALUE**

**Why:** Statistical tests for feature-target relationships

**Methods:**
- `f_regression` - F-test for regression
- `f_classif` - F-test for classification
- `chi2` - Chi-squared test (classification only)

**Benefits:**
- **Fast** - O(n_features)
- **Statistical significance** - p-values included
- **Model-agnostic** - no training needed

**Implementation:**
```python
from sklearn.feature_selection import f_regression, f_classif, chi2

if is_regression:
    scores, pvalues = f_regression(X, y)
else:
    scores, pvalues = f_classif(X, y)
importance = scores  # Higher = more significant
```

**When to use:** Quick baseline, statistical validation

---

### 5. **Recursive Feature Elimination (RFE)** **MEDIUM VALUE**

**Why:** Iteratively removes least important features

**Benefits:**
- **Feature interactions** - considers feature combinations
- **Optimal subset** - finds minimal feature set
- **Works with any model** - wrapper method

**Implementation:**
```python
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor

estimator = RandomForestRegressor(n_estimators=100, random_state=42)
selector = RFE(estimator, n_features_to_select=50, step=5)
selector.fit(X, y)
importance = selector.ranking_  # Lower rank = more important
```

**When to use:** When you want to find minimal feature set

---

### 6. **Boruta** **HIGH VALUE** (All-Relevant Feature Selection)

**Why:** Finds ALL relevant features (not just most important)

**Benefits:**
- **Shadow features** - compares real features to random
- **Statistical significance** - p-value based selection
- **No false positives** - only selects truly relevant features
- **Handles interactions** - considers feature combinations

**Implementation:**
```python
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
boruta = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=42)
boruta.fit(X, y)
selected = boruta.support_  # Boolean array
ranking = boruta.ranking_  # Lower = more important
```

**When to use:** When you want comprehensive feature discovery

---

### 7. **Stability Selection** **MEDIUM VALUE**

**Why:** Finds features that are consistently important across subsamples

**Benefits:**
- **Robust** - features must be important across many samples
- **Reduces overfitting** - only stable features selected
- **Works with any model** - wrapper method

**Implementation:**
```python
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel

# Bootstrap stability selection
n_bootstrap = 100
stability_scores = np.zeros(X.shape[1])

for _ in range(n_bootstrap):
    indices = np.random.choice(len(X), size=len(X), replace=True)
    X_boot, y_boot = X[indices], y[indices]

    model = LassoCV(cv=5, random_state=42)
    model.fit(X_boot, y_boot)
    stability_scores += (np.abs(model.coef_) > 1e-6).astype(int)

importance = stability_scores / n_bootstrap  # Fraction of times selected
```

**When to use:** When you want robust, stable feature selection

---

### 8. **AutoML Frameworks** (Optional)

**H2O AutoML:**
- Automatic feature selection
- Model stacking
- Feature importance from ensemble

**AutoGluon:**
- Automatic feature selection
- Multi-model ensemble
- Built-in importance extraction

**When to use:** For automated end-to-end pipeline

---

## Recommended Priority

### **Tier 1: Add These First**
1. **CatBoost** - Different tree-based approach
2. **Lasso** - Explicit sparse selection
3. **Mutual Information** - Information-theoretic baseline

### **Tier 2: Add for Robustness**
4. **Boruta** - All-relevant feature selection
5. **Univariate Selection** - Statistical validation

### **Tier 3: Advanced Methods**
6. **RFE** - Minimal feature sets
7. **Stability Selection** - Robust selection

---

## Implementation Plan

### Quick Win: Add CatBoost + Lasso + Mutual Information

These three add significant diversity with minimal effort:

```yaml
# Add to CONFIG/multi_model_feature_selection.yaml

catboost:
  enabled: true
  importance_method: "native"  # PredictionValuesChange
  weight: 1.0
  config:
    iterations: 300
    learning_rate: 0.05
    depth: 6
    loss_function: "RMSE"
    verbose: false
    random_seed: 42

lasso:
  enabled: true
  importance_method: "native"  # abs(coef_)
  weight: 0.9
  config:
    alpha: 0.1
    max_iter: 1000
    random_state: 42

mutual_information:
  enabled: true
  importance_method: "native"  # Direct calculation
  weight: 0.8
  config:
    discrete_features: "auto"
    random_state: 42
```

---

## Expected Benefits

### Diversity Gains:
- **CatBoost**: Finds 5-10% different features than LightGBM/XGBoost
- **Lasso**: Identifies sparse linear relationships
- **Mutual Information**: Captures non-linear dependencies

### Robustness Gains:
- **More models** = higher consensus threshold
- **Different biases** = more universal features
- **Better generalization** = 10-20% improvement in production

---

## Next Steps

1. **Add CatBoost** (easiest, high value)
2. **Add Lasso** (fast, explicit selection)
3. **Add Mutual Information** (baseline, no training)
4. **Re-run feature selection** with expanded model set
5. **Compare results** - see which features are consensus across all models

Would you like me to implement these additional models in your feature selection pipeline?

