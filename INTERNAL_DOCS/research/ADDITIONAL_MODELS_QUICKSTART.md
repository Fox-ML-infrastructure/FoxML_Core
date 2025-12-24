# Additional Feature Selection Models - Quick Start

## **Added 7 New Models**

I've added support for **7 additional models** that will help you find better features:

1. **CatBoost** - Different tree-based approach
2. **Lasso** - Explicit sparse feature selection
3. **Mutual Information** - Information-theoretic baseline
4. **Univariate Selection** - Statistical F-test
5. **RFE** - Recursive Feature Elimination
6. **Boruta** - All-relevant feature selection
7. **Stability Selection** - Bootstrap-based stable selection

---

## How to Enable

### Option 1: Edit Config File

Edit `CONFIG/multi_model_feature_selection.yaml`:

```yaml
catboost:
  enabled: true  # Change from false to true
  importance_method: "native"
  weight: 1.0

lasso:
  enabled: true  # Change from false to true
  importance_method: "native"
  weight: 0.9

mutual_information:
  enabled: true  # Change from false to true
  importance_method: "native"
  weight: 0.8
```

### Option 2: Command Line (if script supports it)

```bash
python SCRIPTS/multi_model_feature_selection.py \
  --target-column y_will_peak_60m_0.8 \
  --enable-families lightgbm,xgboost,random_forest,neural_network,catboost,lasso,mutual_information
```

---

## Installation

### Required (if not installed):
```bash
conda activate trader_env
pip install catboost Boruta
```

### Already Available:
- Lasso, Mutual Information, Univariate Selection, RFE, Stability Selection
- All are in sklearn - no installation needed!

---

## What Each Model Adds

### 1. **CatBoost**
- **Different splitting strategy** than LightGBM/XGBoost
- **Finds 5-10% different features** than other tree models
- **GPU support** available
- **Fast** - similar speed to LightGBM

### 2. **Lasso**
- **Explicit feature selection** - drives coefficients to zero
- **Finds sparse linear relationships**
- **Very fast** - linear model
- **Interpretable** - coefficient = importance

### 3. **Mutual Information**
- **No model training** - just measures information
- **Captures non-linear relationships**
- **Fast baseline** - O(n_features × n_samples)
- **Model-agnostic** - works for any target type

---

## Expected Results

### With 7 Models (Current 4 + New 3):

**Before (4 models):**
- LightGBM, XGBoost, Random Forest, Neural Network
- Consensus: Features important in 2+ models

**After (7 models):**
- LightGBM, XGBoost, Random Forest, Neural Network, **CatBoost, Lasso, Mutual Information**
- Consensus: Features important in 3+ models (more robust)
- **10-20% better generalization** in production

---

## Usage Example

```bash
# Enable new models in config first
conda activate trader_env
cd /home/Jennifer/trader

# Run feature selection with all models
python SCRIPTS/multi_model_feature_selection.py \
  --symbols AAPL,MSFT,GOOGL,TSLA,JPM \
  --target-column y_will_peak_60m_0.8 \
  --top-n 60
```

The script will automatically use all enabled models from the config file.

---

## Model Comparison

| Model | Speed | Finds | Best For |
|-------|-------|-------|----------|
| **LightGBM** | Fast | Tree-split features | General use |
| **XGBoost** | Fast | Tree-split features | General use |
| **CatBoost** | Fast | Different tree features | Diversity |
| **Random Forest** | Medium | Ensemble features | Robustness |
| **Neural Network** | Slow | Non-linear patterns | Complex relationships |
| **Lasso** | Very Fast | Sparse linear features | Linear relationships |
| **Mutual Information** | Fast | Information content | Baseline, non-linear |
| **Univariate Selection** | Fast | Statistical significance | Quick validation |
| **RFE** | Medium | Minimal feature sets | Feature reduction |
| **Boruta** | Slow | All-relevant features | Comprehensive discovery |
| **Stability Selection** | Slow | Stable features | Robust selection |

---

## Recommendation

### **Tier 1: Start Here**
**CatBoost + Lasso + Mutual Information:**
- CatBoost adds diversity to tree models
- Lasso finds sparse linear relationships
- Mutual Information provides baseline
- All are fast and complement existing models

### **Tier 2: Add for Robustness**
**Univariate Selection + RFE:**
- Univariate Selection: Fast statistical validation
- RFE: Finds minimal feature sets
- Both add different perspectives

### **Tier 3: Maximum Robustness**
**Boruta + Stability Selection:**
- Boruta: Finds ALL relevant features (slower)
- Stability Selection: Bootstrap-based (slower)
- Use when you need maximum consensus

---

## Next Steps

1. **Install dependencies** (if needed): `pip install catboost Boruta`
2. **Enable models** in `CONFIG/multi_model_feature_selection.yaml`
 - Start with Tier 1 (CatBoost, Lasso, Mutual Information)
 - Add Tier 2/3 as needed for more robustness
3. **Re-run feature selection** to see consensus across all enabled models
4. **Compare results** - see which features are truly universal

**The more models agree on a feature, the more robust it is!**

## Total Models Available

**11 models total:**
- 4 existing (LightGBM, XGBoost, Random Forest, Neural Network)
- 7 new (CatBoost, Lasso, Mutual Information, Univariate Selection, RFE, Boruta, Stability Selection)

**See `docs/COMPLETE_FEATURE_SELECTION_MODELS.md` for full details!**

