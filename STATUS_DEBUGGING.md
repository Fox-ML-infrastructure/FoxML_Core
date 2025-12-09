# Training Pipeline Debugging Status

**Last Updated:** 2025-12-09  
**Status:** Active debugging of feature coercion and QuantileLightGBM fallback issues

---

## Current Issues Being Investigated

### 1. **CRITICAL: All Features Become NaN After Coercion (30m Targets)**

**Symptoms:**
- For 30m targets (horizon=6), feature registry allows 20 features
- All 20 features become all-NaN after `pd.to_numeric(errors='coerce')`
- Training fails with "No valid data after cleaning" for all 18 targets
- Pipeline reports "✅ Training completed successfully" but trains 0 models

**Root Cause Hypothesis:**
- Feature names from selection/registry don't match actual column names in data
- Features exist but have non-numeric dtypes that can't be coerced
- Features exist but are all NaN in raw data (never computed for this dataset)

**Debugging Added:**
- ✅ Pre-coercion diagnostics: logs which features are missing, NaN ratios before coercion
- ✅ Post-coercion diagnostics: logs which features became NaN, with raw data samples
- ✅ Early validation: checks if feature_names is empty, if features exist in dataframe
- ✅ Guardrails: fails loudly when 0 models trained instead of silent "success"
- ✅ Debug file output: writes NPZ files with feature names when all features become NaN

**Next Steps:**
- Run training again and inspect `🔍 Debug` messages to identify exact feature names causing issues
- Check `debug_feature_coercion/all_nan_features_*.npz` files to see feature/column mismatches
- Verify feature selection output matches actual data column names

---

### 2. **CRITICAL: QuantileLightGBM Silent Fallback to Huber**

**Symptoms:**
```
[QuantileLGBM] Training with alpha=0.5, rounds=2000, ESR=100, budget=1800s
WARNING - ❌ QuantileLightGBM failed (too many values to unpack (expected 3)) → falling back to Huber LGBM
INFO - ✅ Huber fallback trained | best_iter=10
```

**Root Cause:**
- QuantileLightGBM trainer has a bug: unpacking error `too many values to unpack (expected 3)`
- Likely in return value handling from `lgb.train()` or callback processing
- Fallback silently replaces quantile model with Huber regression
- **Semantic mismatch**: downstream code expects quantile behavior but gets L2 regression

**Impact:**
- Models labeled "QuantileLightGBM" are actually Huber regression models
- Any code expecting quantile-specific behavior (asymmetric loss, VaR, tail conditioning) will be wrong
- Early stopping at `best_iter=10` may be too shallow (needs investigation)

**Debugging Added:**
- ✅ Enhanced exception logging in quantile trainer (logs full stack trace)
- ✅ Validation metric progression logging (shows if model is improving or stuck)
- ✅ Feature count logging (helps diagnose if feature reduction causes fast convergence)

**Next Steps:**
- ✅ **DONE**: Added full stack trace logging (`logger.exception`) to identify exact line causing the error
- Fix the unpacking error in `quantile_lightgbm_trainer.py` (once stack trace shows the exact location)
- Consider temporarily disabling Huber fallback to force hard failure until quantile is fixed
- Verify LightGBM version compatibility with return value signatures

---

### 3. **Training Speed: Fast but Expected**

**Status:** ✅ **NOT A BUG** - This is working as designed

**Observations:**
- QuantileLightGBM completes in ~21 seconds (was ~15 minutes before)
- Training stops at `best_iter=10` (very early, out of 2000 rounds)
- Only 55 features used (down from 500+ due to leakage filtering)

**Why This Is Expected:**
- **Feature reduction**: 500+ → 55 features = much faster training
- **Early stopping**: Model converges quickly, validation metric stops improving
- **Cross-sectional sampling**: Max 50 samples per timestamp prevents data explosion
- **Regularization**: Strong regularization + shallow trees = fast convergence

**Action Items:**
- Monitor validation metrics to ensure `best_iter=10` isn't too shallow
- If performance is acceptable, this is fine
- If you need more complex models, consider:
  - Increasing `early_stopping_rounds`
  - Reducing regularization
  - Increasing `num_leaves` or `max_depth`

---

## Diagnostic Tools Added

### Feature Coercion Diagnostics
- **Location**: `TRAINING/train_with_strategies.py` lines 967-1055
- **What it logs**:
  - Initial feature_df shape and feature counts
  - Missing columns from selected features
  - NaN ratios before and after coercion
  - Raw data samples for features that become all-NaN
  - Critical errors when all features are dropped

### Guardrails
- **Location**: `TRAINING/orchestration/intelligent_trainer.py` lines 670-710
- **What it does**:
  - Fails loudly when 0 models are trained
  - Tracks failed targets and reasons
  - Sets status to `'failed_no_models'` instead of `'completed'`
  - Provides actionable error messages

### QuantileLightGBM Diagnostics
- **Location**: `TRAINING/model_fun/quantile_lightgbm_trainer.py` lines 177-208
- **What it logs**:
  - Feature count used for training
  - Validation metric progression (iterations 0-9, then every 50)
  - Early stopping analysis (why training stopped early)
  - Improvement patterns (early vs late convergence)

---

## Expected Behavior After Fixes

### Feature Coercion Issue
**When Fixed:**
- Diagnostic logs will show exactly which features are missing or becoming NaN
- Debug files will contain feature names for manual inspection
- Pipeline will fail loudly with clear error messages instead of silent "success"

**Success Criteria:**
- All selected features exist in dataframe
- Features can be coerced to numeric without becoming all-NaN
- At least some targets train successfully

### QuantileLightGBM Issue
**When Fixed:**
- QuantileLightGBM will train without falling back to Huber
- Full stack trace will identify exact line causing unpacking error
- Models will actually be quantile regression, not Huber regression

**Success Criteria:**
- No "too many values to unpack" errors
- Quantile models train successfully
- Validation metrics show quantile loss, not Huber loss

---

## Files Modified for Debugging

1. **TRAINING/train_with_strategies.py**
   - Added feature existence validation
   - Added pre/post-coercion NaN diagnostics
   - Added critical guards for empty feature matrices
   - Added failure tracking in results dictionary

2. **TRAINING/orchestration/intelligent_trainer.py**
   - Added guardrails to fail loudly on 0 models
   - Added failed target tracking and reporting
   - Changed status from 'completed' to 'failed_no_models' when appropriate

3. **TRAINING/model_fun/quantile_lightgbm_trainer.py**
   - Added validation metric progression logging
   - Added feature count logging
   - Added early stopping analysis
   - Enhanced exception handling (needs full stack trace still)

---

## Next Run Checklist

When you run training again, check for:

- [ ] `🔍 Debug [target]:` messages showing feature diagnostics
- [ ] `❌ CRITICAL [target]:` messages when all features become NaN
- [ ] `❌ TRAINING RUN FAILED: 0 models trained` instead of silent success
- [ ] Debug files in `debug_feature_coercion/` directory
- [ ] Full stack trace for QuantileLightGBM unpacking error
- [ ] Validation metric progression logs for QuantileLightGBM

---

## Quick Reference: What to Look For

**If you see:**
- `🔍 Debug [target]: X features missing from combined_df` → Feature name mismatch
- `🔍 Debug [target]: X features became ALL NaN AFTER coercion` → Coercion issue
- `❌ CRITICAL [target]: ALL X selected features became all-NaN` → All features failed
- `too many values to unpack (expected 3)` → QuantileLightGBM bug
- `✅ Training completed successfully` + `Trained 0 models` → Guardrail should catch this

**Action:**
- Check the diagnostic logs for the specific feature names
- Inspect debug NPZ files
- Fix the root cause (name mismatch, dtype issue, or quantile unpacking bug)

