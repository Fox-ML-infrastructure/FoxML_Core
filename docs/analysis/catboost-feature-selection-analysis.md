# Deep Analysis: CatBoost Feature Selection Accuracy and Performance

**Date**: 2025-12-22  
**Focus**: Importance method impact on feature selection + 6-hour performance issue

## Executive Summary

### Key Findings

1. **Importance Method Impact**: Fallback importance (gain/split) vs PredictionValuesChange (PVC) **DOES affect feature rankings**, but multi-model consensus reduces the impact. The aggregation process normalizes importance scores, so relative rankings matter more than absolute values.

2. **Performance Issue**: 6 hours for 1000×100 is **excessive** - expected <5 minutes. Likely causes:
   - PVC computation taking 2-3 hours even with timeout
   - Stability snapshots triggering analysis hooks
   - Missing early exits when overfitting is detected
   - Potential nested operations in stability analysis

## 1. Importance Method Impact Analysis

### 1.1 How Importance Flows Through the System

```
train_model_and_get_importance()
  ↓
  Returns: (model, importance: pd.Series, method: str, train_score: float)
  ↓
ImportanceResult(
  model_family: str,
  symbol: str,
  importance_scores: pd.Series,  # ← Raw importance scores
  method: str,  # ← "PredictionValuesChange" or "FeatureImportance" or "Split"
  train_score: float
)
  ↓
aggregate_multi_model_importance()
  ↓
  For each family:
    - Concatenate importance_scores across symbols (pd.concat)
    - Aggregate: mean/median across symbols
    - Apply family weight
  ↓
  Combine across families:
    - weighted_mean / median / geometric_mean
  ↓
  Final consensus_score (used for ranking)
```

### 1.2 Critical Observations

**Observation 1: Importance is NOT normalized before aggregation**
- `normalize_importance()` is called during model training, but the **raw importance scores** are stored in `ImportanceResult`
- Aggregation uses raw scores directly: `pd.concat(importance_series_list).mean(axis=1)`
- This means **absolute values matter**, not just rankings

**Observation 2: Method field is informational only**
- `ImportanceResult.method` stores the method name but **is not used in aggregation**
- Aggregation doesn't distinguish between PVC and fallback importance
- No tracking of when fallback was used vs PVC

**Observation 3: Multi-model consensus reduces single-model bias**
- If CatBoost uses fallback but other models (LightGBM, XGBoost, etc.) use their native importance, consensus still works
- However, if CatBoost is weighted heavily or is the only tree model, fallback could significantly affect rankings

### 1.3 Impact Assessment

**Scenario A: CatBoost uses fallback, other models use native importance**
- **Impact**: Medium
- **Reason**: Multi-model consensus dilutes CatBoost's contribution
- **Risk**: If CatBoost weight is high (default: 1.0), rankings could shift

**Scenario B: All tree models use fallback (if all overfit)**
- **Impact**: High
- **Reason**: Tree models typically have high weight in consensus
- **Risk**: Feature rankings could be significantly different from PVC-based rankings

**Scenario C: CatBoost uses PVC, but other models fail**
- **Impact**: Low
- **Reason**: CatBoost PVC is still used, just fewer models in consensus
- **Risk**: Lower consensus confidence, but rankings are still valid

### 1.4 Ranking Correlation Analysis Needed

**Missing**: No comparison between PVC and fallback rankings
- Need to compute Spearman correlation when both methods are available
- Need to track top-K feature overlap
- Need to log when fallback is used with warning about potential ranking differences

## 2. Performance Analysis: 6 Hours for 1000×100

### 2.1 Expected Performance

For 1000 samples × 100 features:
- **CatBoost training**: <1 minute (with early stopping)
- **CV (3 folds)**: <2 minutes
- **PVC importance**: 40-80 minutes (if not skipped)
- **Total expected**: <5 minutes (if PVC skipped) or <90 minutes (if PVC runs)

**6 hours = 360 minutes** is **4× longer than worst-case PVC scenario**

### 2.2 Code Flow Analysis

```
process_single_symbol()
  ↓
  For each model_family (catboost, lightgbm, xgboost, etc.):
    ↓
    train_model_and_get_importance()
      ↓
      For CatBoost:
        - CV loop (3 folds): ~2 minutes ✅
        - Final fit: <1 minute ✅
        - Overfitting check: <1 second ✅
        - IF NOT SKIPPED:
          - PVC computation: 40-80 minutes ⚠️
        - IF SKIPPED:
          - Fallback (gain/split): <1 second ✅
      ↓
      Save stability snapshot (per model family)
        ↓
        save_snapshot_from_series_hook()
          ↓
          May trigger analyze_all_stability_hook() ⚠️
```

### 2.3 Potential Bottlenecks

**Bottleneck 1: PVC Timeout Not Working**
- Timeout is set to 30 minutes (`importance_max_wall_minutes: 30`)
- But if PVC takes 40-80 minutes, timeout should fire
- **Check**: Is process-based timeout actually killing the process?

**Bottleneck 2: Stability Analysis Hooks**
- `save_snapshot_from_series_hook()` may trigger `analyze_all_stability_hook()`
- This could analyze **all previous snapshots** for this target
- If there are 7 snapshots (from previous runs), this could multiply work
- **Check**: Does stability analysis run synchronously during feature selection?

**Bottleneck 3: Overfitting Detection Not Triggering**
- If `train_score < 0.99` but model is still overfitting, PVC runs anyway
- Gap threshold (`train_val_gap_threshold: 0.20`) might not catch all cases
- **Check**: Are the thresholds too permissive?

**Bottleneck 4: Multiple Calls to process_single_symbol**
- If called multiple times for same symbol (unlikely but possible)
- Or if called for multiple symbols sequentially
- **Check**: Is there a loop calling this multiple times?

**Bottleneck 5: Aggregation Overhead**
- `aggregate_multi_model_importance()` processes all results
- If called multiple times or with large result sets, could add overhead
- **Check**: Is aggregation called once or multiple times?

### 2.4 Missing Diagnostics

**Current timers:**
- CV time: ✅ Tracked
- Fit time: ✅ Tracked
- Importance time: ✅ Tracked
- **Missing**: Total `process_single_symbol` time
- **Missing**: Stability snapshot save time
- **Missing**: Aggregation time
- **Missing**: Data loading/preprocessing time

## 3. Recommendations

### 3.1 Accuracy/Correctness Fixes

1. **Track importance method in results**
   - Add `importance_method_used` field to `ImportanceResult`
   - Log when fallback is used with warning about potential ranking differences
   - Store in metadata for comparison

2. **Compare rankings when possible**
   - When fallback is used, optionally compute PVC anyway (if timeout allows) for comparison
   - Compute Spearman correlation between PVC and fallback rankings
   - Log top-K feature overlap
   - Store comparison metrics in metadata

3. **Add validation logging**
   - Log when fallback is used: `"⚠️ CatBoost: Using fallback {method} (overfitting detected) - feature rankings may differ from PVC"`
   - Add config option to require PVC (fail if overfitting detected) vs allow fallback

### 3.2 Performance Fixes

1. **Add comprehensive timing**
   - Wrap entire `process_single_symbol` with timer
   - Time stability snapshot save
   - Time aggregation
   - Time data loading/preprocessing
   - Log breakdown: "Total: X min (data: Y, training: Z, importance: W, stability: V, other: U)"

2. **Verify timeout enforcement**
   - Add logging when timeout fires: `"PVC_TIMEOUT after X minutes"`
   - Verify process is actually killed (not just timing out)
   - Add fallback importance computation time to total

3. **Check stability analysis**
   - Verify `save_snapshot_from_series_hook()` is non-blocking
   - Check if `analyze_all_stability_hook()` runs synchronously
   - If it does, make it async or skip during feature selection

4. **Tighten overfitting detection**
   - Lower `overfit_train_acc_threshold` from 0.99 to 0.95 (more aggressive)
   - Lower `overfit_train_val_gap_threshold` from 0.20 to 0.15
   - Add feature count check: skip PVC if `n_features > 200` (configurable)

5. **Add early exits**
   - Skip PVC if `n_features > pvc_feature_count_cap` (default: 250)
   - Skip PVC if estimated cost > budget (heuristic based on n_features × n_samples)
   - Log reason for skipping

### 3.3 Code Changes Needed

**File: `TRAINING/ranking/multi_model_feature_selection/types.py`**
- Add `importance_method_used: str` to `ImportanceResult` dataclass

**File: `TRAINING/ranking/multi_model_feature_selection.py`**
- Track `importance_method_used` when creating `ImportanceResult`
- Add timing around `process_single_symbol`
- Add timing around stability snapshot save
- Add comparison logic when fallback is used
- Add early exit checks before PVC computation
- Log comprehensive timing breakdown

**File: `TRAINING/ranking/utils/overfitting_detection.py`**
- Add feature count check to `should_skip_expensive_importance()`
- Add estimated cost heuristic

**File: `CONFIG/pipeline/training/safety.yaml`**
- Add `pvc_feature_count_cap: 200` (or lower)
- Consider lowering thresholds for more aggressive skipping

## 4. Next Steps

1. **Immediate**: Add comprehensive timing to identify where 6 hours is spent
2. **Short-term**: Fix timeout enforcement and add early exits
3. **Medium-term**: Add ranking comparison and validation logging
4. **Long-term**: Consider making stability analysis async or optional during feature selection

## 5. Questions to Answer

1. **Is PVC timeout actually working?** Check logs for "PVC_TIMEOUT" messages
2. **Is stability analysis blocking?** Check if `analyze_all_stability_hook()` runs synchronously
3. **Are thresholds too permissive?** Check actual train_score/val_score values when PVC runs
4. **Is process_single_symbol called multiple times?** Check call sites
5. **What's the actual time breakdown?** Need comprehensive timing to answer






