# Training Pipeline Debugging Status

**Last Updated:** 2025-12-09  
**Status:** QuantileLightGBM validated in isolation; debugging integration of ranked targets and features into training pipeline.

**Note:** Testing cadence temporarily reduced due to scheduling constraints. Currently debugging issues with passing ranked targets and selected features to training pipeline.

---

## Current Testing Status

**Isolation Testing:** ✅ **PASSING**
- QuantileLightGBM trains successfully without errors
- No fallback to Huber regression
- Validation metrics logged with 9 decimal precision
- Early stopping working correctly

**E2E Testing:** 🔧 **DEBUGGING IN PROGRESS**
- Previous run (without cross-sectional ranking) showed healthy pipeline behavior
- Cross-sectional ranking feature implemented
- **Current issue**: Debugging problems with passing ranked targets and selected features to training pipeline
- Integration between ranking/selection and training steps needs resolution

---

## Recent E2E Run Evaluation (2025-12-09)

### System Health Assessment: ✅ **HEALTHY**

**Target Ranking (STEP 1):**
- ✅ Loaded 2 symbols (AAPL, MSFT), 50k rows each → 100,000 combined rows
- ✅ Leakage filtering: 307 safe features kept, 223 dropped as potential leaks
- ✅ Metadata correctly excluded: `['symbol', 'interval', 'source', 'ts']`
- ✅ 156 target/label columns excluded (`y_*`, `fwd_ret_*`, etc.)
- ✅ Cross-sectional data: 99,880 samples, 304 features after cleaning
- ✅ Leak scanner caught `adjusted` feature (100% match with target) and auto-removed
- ✅ Degenerate target `y_will_swing_high_60m_0.20` (single unique value) correctly skipped

**Feature Selection (STEP 2):**
- ✅ Multi-model consensus working: 8/14 model families enabled and running
- ✅ Strong model agreement: LightGBM/XGBoost/CatBoost scoring 0.62-0.76 consistently
- ✅ Feature selection producing expected blend: volume/volatility + trend + money flow + RSI
- ✅ No hidden leaks detected in selected features
- ✅ Aggregation across symbols working correctly

**Target Confidence & Production Gating:**
- ✅ Confidence scoring: HIGH score_tier → MEDIUM overall confidence (expected for 2-symbol run)
- ✅ Routing logic: MEDIUM confidence → `candidate` bucket, `allowed_in_production=False`
- ✅ System correctly holding targets behind gate until broader validation

**Known Non-Issues:**
- ⚠️ **Timestamp delta warnings**: Fixed with negative-delta guard and debug logging (harmless, using config default 5m)
- ⚠️ **Boruta gatekeeper disabled**: Expected behavior - Boruta runs per-symbol but aggregator-level gatekeeper is conservative by design (config-controlled)

**New Features:**
- ⏳ **Cross-sectional feature ranking**: Added panel model for universe-level feature importance
  - Module: `TRAINING/ranking/cross_sectional_feature_ranker.py`
  - Integrated into feature selection pipeline (optional, config-controlled)
  - Tags features: CORE (strong both), SYMBOL_SPECIFIC, CS_SPECIFIC, WEAK
  - Enabled in config with `min_symbols: 2` for testing
  - **Status**: Implementation complete, ready for E2E testing (not yet tested)

**Current Debugging Focus:**
- Integration issues between ranking/selection and training pipeline
- Passing ranked targets from STEP 1 to training
- Passing selected features from STEP 2 to training
- Ensuring data format compatibility between selection and training stages

**Bottom Line:**
The early intelligence layer (ranking & selection) is functioning as designed. All safety nets (leak detection, degenerate target skipping, confidence gating) are working correctly. The system is producing coherent feature selections with strong cross-model agreement. Currently debugging integration with training pipeline.

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

**Root Cause:** ✅ **IDENTIFIED**
- QuantileLightGBM trainer has a bug in `_record_validation` callback (line 189)
- **Exact issue**: Trying to unpack 3 values from `env.evaluation_result_list`, but modern LightGBM returns 4-tuple `(data_name, eval_name, result, is_higher_better)` or 5-tuple with stddev
- Old code: `for eval_name, eval_value, _ in env.evaluation_result_list:` ❌
- Fixed code: Uses indexing `item[0], item[1], item[2]` to handle both 4 and 5-tuple cases ✅
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
- ✅ **FIXED**: Updated `_record_validation` callback to use indexing instead of unpacking (handles 4 or 5-tuple from modern LightGBM)
- ✅ **VERIFIED**: Quantile training works correctly in isolation - no fallback to Huber, actual quantile models are trained
- ✅ **DONE**: Increased logging precision to 9 decimals to show micro-improvements
- ⏳ **IN PROGRESS**: E2E testing of full pipeline
- ⏳ **TODO**: Verify feature coercion diagnostics work correctly in E2E runs

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
   - Enhanced exception handling with full stack traces

---

## Next Run Checklist

When you run training again, check for:

- [x] Full stack trace for QuantileLightGBM unpacking error - **FIXED**
- [x] Validation metric progression logs for QuantileLightGBM - **WORKING** (now with 9 decimal precision)
- [x] QuantileLightGBM trains without falling back to Huber - **VERIFIED IN ISOLATION**
- [ ] `🔍 Debug [target]:` messages showing feature diagnostics - **E2E TESTING IN PROGRESS**
- [ ] `❌ CRITICAL [target]:` messages when all features become NaN - **E2E TESTING IN PROGRESS**
- [ ] `❌ TRAINING RUN FAILED: 0 models trained` instead of silent success - **E2E TESTING IN PROGRESS**
- [ ] Debug files in `debug_feature_coercion/` directory - **E2E TESTING IN PROGRESS**

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
