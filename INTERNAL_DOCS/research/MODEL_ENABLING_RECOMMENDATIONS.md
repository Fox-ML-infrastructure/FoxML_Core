# Model Enabling Recommendations

## Current Status

### **Currently Enabled (4 models)**
- `lightgbm`
- `xgboost`
- `random_forest`
- `neural_network`

### **Currently Disabled (7 new models)**
- `catboost`
- `lasso`
- `mutual_information`
- `univariate_selection`
- `rfe`
- `boruta`
- `stability_selection`

---

## **Recommendation: Enable Tier 1 Models**

### **For Target Ranking: YES, Enable More Models!**

**Why:** Target ranking benefits from diverse model perspectives. More models = more robust target predictability scores.

**Recommended Configuration:**

```yaml
# Enable these 7 models for target ranking (fast + diverse)
lightgbm: enabled: true      #  Already enabled
xgboost: enabled: true        #  Already enabled
random_forest: enabled: true  #  Already enabled
neural_network: enabled: true #  Already enabled
catboost: enabled: true       #  Enable (different tree approach)
lasso: enabled: true         #  Enable (sparse linear)
mutual_information: enabled: true  #  Enable (baseline)
```

**Time Impact:** +2-3 minutes per target (still reasonable)

---

### **For Feature Selection: Enable Based on Use Case**

#### **Option 1: Fast & Balanced (Recommended)**
```yaml
# 7 models total
lightgbm: enabled: true
xgboost: enabled: true
random_forest: enabled: true
neural_network: enabled: true
catboost: enabled: true
lasso: enabled: true
mutual_information: enabled: true
```
**Time:** ~5-10 min per symbol
**Best for:** Production feature selection

#### **Option 2: Comprehensive (Maximum Robustness)**
```yaml
# 11 models total (all enabled)
# Enable all models from Option 1, plus:
univariate_selection: enabled: true
rfe: enabled: true
boruta: enabled: true
stability_selection: enabled: true
```
**Time:** ~20-30 min per symbol
**Best for:** Research, maximum consensus

---

## **How Scripts Use Models**

### **`rank_target_predictability.py`**
- **Automatically uses all enabled models** from config
- **No code changes needed** - just enable in config
- **Loads config automatically** from `CONFIG/multi_model_feature_selection.yaml`

### **`rank_features_by_ic_and_predictive.py`**
- **Automatically uses all enabled models** from config
- **No code changes needed** - just enable in config
- **Same config file** - consistent across scripts

### **`multi_model_feature_selection.py`**
- **Uses all enabled models** from config
- **Same config file** - consistent across scripts

---

## **Action Plan**

### **Step 1: Enable Tier 1 Models for Target Ranking**

Edit `CONFIG/multi_model_feature_selection.yaml`:

```yaml
catboost:
  enabled: true  # Change from false

lasso:
  enabled: true  # Change from false

mutual_information:
  enabled: true  # Change from false
```

**Result:** Target ranking will now use **7 models** instead of 4.

### **Step 2: Re-run Target Ranking**

```bash
conda activate trader_env
cd /home/Jennifer/trader

python SCRIPTS/rank_target_predictability.py \
  --discover-all \
  --symbols AAPL,MSFT,GOOGL,TSLA,JPM
```

**Expected:** More robust target scores, better consensus.

### **Step 3: Enable Additional Models for Feature Selection (Optional)**

If you want maximum robustness for feature selection:

```yaml
univariate_selection:
  enabled: true

rfe:
  enabled: true

# Optional (slower):
boruta:
  enabled: true

stability_selection:
  enabled: true
```

---

## **Scripts Folder - What to Keep**

### **Essential Scripts (KEEP)**

1. **`rank_target_predictability.py`**
 - Ranks targets by predictability
 - Uses all enabled models from config
 - **KEEP**

2. **`rank_features_by_ic_and_predictive.py`**
 - Main feature ranking (IC + predictive power)
 - Uses all enabled models from config
 - **KEEP**

3. **`multi_model_feature_selection.py`**
 - Detailed multi-model feature importance
 - Uses all enabled models from config
 - **KEEP**

4. **`rank_features_comprehensive.py`**
 - Quality audit + predictive + redundancy
 - **KEEP** (optional but useful)

5. **`filter_leaking_features.py`**
 - Used by all main scripts
 - **KEEP** (DO NOT DELETE)

6. **`list_available_symbols.py`**
 - Helper utility
 - **KEEP**

7. **`compare_feature_sets.py`**
 - Compare different feature sets
 - **KEEP**

### **Utility Folder (KEEP)**

- **`SCRIPTS/utils/target_validation.py`**
 - Used by all ranking scripts
 - **KEEP** (DO NOT DELETE)

### ️ **Outdated Scripts (CAN DELETE)**

Based on `OUTDATED_SCRIPTS.md`:

```bash
cd /home/Jennifer/trader/scripts

# These are safe to delete (already identified as outdated):
# - identify_leaking_features.py (if exists)
# - test_quick_ranking.py (if exists)
# - LEAKAGE_FIX_README.md (if exists)
```

### **Helper Scripts (Optional - Keep if You Use Them)**

- `run_baseline_validation.sh`
- `run_comprehensive_feature_ranking.sh`
- `run_feature_ranking.sh`
- `run_regime_enhancement.sh`
- `run_target_ranking_demo.sh` (can delete if not used)
- `show_existing_output.sh` (can delete if not used)

### **Documentation (Keep)**

- `OUTDATED_SCRIPTS.md` - Useful reference
- `EXAMPLE_MULTI_MODEL_WORKFLOW.md` - Example workflow
- `MULTI_MODEL_QUICKSTART.sh` - Quick start helper
- `DEGENERATE_TARGET_HANDLING.md` - Reference doc

---

## **Summary**

### **Question 1: Are all models enabled?**
**Answer:** No - only 4 are enabled. **Recommendation:** Enable Tier 1 (CatBoost, Lasso, Mutual Information) for better target ranking.

### **Question 2: Should targets be evaluated with these models?**
**Answer:** **YES!** The `rank_target_predictability.py` script already uses all enabled models from the config. Just enable them in the config file.

### **Question 3: What's important to keep in scripts folder?**
**Answer:**
- **KEEP:** All `rank_*.py` scripts, `multi_model_feature_selection.py`, `filter_leaking_features.py`, `utils/target_validation.py`
- **DELETE:** Outdated scripts (if they exist): `identify_leaking_features.py`, `test_quick_ranking.py`
- **OPTIONAL:** Helper shell scripts (keep if you use them)

---

## **Quick Action**

```bash
# 1. Enable Tier 1 models in config
# Edit CONFIG/multi_model_feature_selection.yaml:
#   catboost: enabled: true
#   lasso: enabled: true
#   mutual_information: enabled: true

# 2. Re-run target ranking with more models
python SCRIPTS/rank_target_predictability.py --discover-all

# 3. Clean up outdated scripts (optional)
cd scripts
# Check what exists first:
ls identify_leaking_features.py test_quick_ranking.py 2>/dev/null
# Delete if they exist:
rm -f identify_leaking_features.py test_quick_ranking.py
```

---

## **Benefits of Enabling More Models**

### **For Target Ranking:**
- **More robust scores** - consensus across 7 models instead of 4
- **Better target selection** - targets that work across diverse architectures
- **Reduced bias** - less dependent on single model family

### **For Feature Selection:**
- **More universal features** - features that work across all model types
- **Higher confidence** - features important in 5+ models are extremely reliable
- **Better generalization** - 10-20% improvement in production

**The more models agree, the more robust the result!**

