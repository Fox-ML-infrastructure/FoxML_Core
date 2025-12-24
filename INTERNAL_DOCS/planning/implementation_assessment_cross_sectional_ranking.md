# Cross-Sectional Feature Ranking: Implementation Assessment

**Date:** 2025-12-09  
**Status:** Feasibility Analysis

---

## Executive Summary

**Implementation Difficulty:** 🟢 **LOW-MEDIUM** (2-3 days of focused work)

**Risk to Current Functionality:** 🟢 **VERY LOW** (can be added as optional, config-controlled step)

**Recommendation:** ✅ **PROCEED** - High value, low risk, good reuse of existing infrastructure

---

## 1. Implementation Complexity Analysis

### What Already Exists (Reusable Infrastructure)

1. **Cross-Sectional Data Building:**
   - `TRAINING/utils/cross_sectional_data.py::prepare_cross_sectional_data_for_ranking()`
   - Already handles panel data construction, timestamp grouping, min_cs/max_cs_samples
   - Already used by target ranking pipeline
   - ✅ **100% reusable**

2. **Leakage Filtering:**
   - `TRAINING/utils/leakage_filtering.py` - already integrated into feature selection
   - Can reuse same filters for panel model
   - ✅ **100% reusable**

3. **Feature Selection Pipeline:**
   - `TRAINING/ranking/feature_selector.py` - modular structure
   - `TRAINING/ranking/multi_model_feature_selection.py` - aggregation logic
   - Results saved to structured CSV files
   - ✅ **Easy to extend**

4. **Model Training Infrastructure:**
   - LightGBM/XGBoost already used in per-symbol selection
   - Can reuse same model families config
   - ✅ **100% reusable**

### What Needs to Be Built

1. **New Module:** `TRAINING/ranking/cross_sectional_feature_ranker.py`
   - Function: `compute_cross_sectional_importance()`
   - Input: top_k features from per-symbol selection, panel data
   - Output: CS importance scores, feature tags (CORE/SYMBOL-SPECIFIC/WEAK)
   - **Estimated:** ~200-300 lines of code

2. **Integration Point:** `TRAINING/ranking/feature_selector.py`
   - Add optional call after `_aggregate_multi_model_importance()`
   - Merge CS scores into `summary_df`
   - Add feature tagging logic
   - **Estimated:** ~50-100 lines of code

3. **Config Extension:** `CONFIG/feature_selection/multi_model.yaml`
   - Add `cross_sectional_ranking` section:
     ```yaml
     cross_sectional_ranking:
       enabled: true
       min_symbols: 5  # Only run if >= 5 symbols
       model_families: [lightgbm, xgboost]  # Which models to use
       top_k_candidates: 50  # Use top 50 from per-symbol as candidates
       normalization: "zscore"  # Optional: per-date z-score normalization
     ```
   - **Estimated:** ~20 lines of config

4. **Output Extension:** `save_multi_model_results()`
   - Add `cs_importance_score` column to `summary_df`
   - Add `feature_category` column (CORE/SYMBOL-SPECIFIC/WEAK)
   - Save separate `cross_sectional_importance.csv` file
   - **Estimated:** ~30-50 lines of code

**Total Estimated Effort:** 300-500 lines of new code, mostly straightforward integration

---

## 2. Risk Assessment: Will It Break Current Functionality?

### ✅ **VERY LOW RISK** - Here's Why:

1. **Optional by Default:**
   - Can be disabled via config (`enabled: false`)
   - Only runs if `len(symbols) >= min_symbols` (default: 5)
   - With 2 symbols (current run), it won't execute at all
   - ✅ **Zero impact on existing runs**

2. **Additive, Not Modificative:**
   - Doesn't change existing per-symbol + aggregation logic
   - Runs **after** existing selection completes
   - Adds columns to `summary_df`, doesn't remove any
   - ✅ **Backward compatible**

3. **Isolated Module:**
   - New code in separate file (`cross_sectional_feature_ranker.py`)
   - Can be tested independently
   - Easy to disable/remove if issues arise
   - ✅ **Low blast radius**

4. **Reuses Existing Infrastructure:**
   - Uses same data loading (`load_mtf_data_for_ranking`)
   - Uses same leakage filters
   - Uses same model training code paths
   - ✅ **No new failure modes**

5. **Graceful Degradation:**
   - If CS ranking fails, log warning and continue with per-symbol results
   - Doesn't block main pipeline
   - ✅ **Fail-safe design**

### Potential Edge Cases (All Mitigatable):

1. **Memory Usage:**
   - Panel data can be large (N_symbols × N_timestamps)
   - **Mitigation:** Already have `max_cs_samples` limit (1000 per timestamp)
   - **Mitigation:** Only process `top_k_candidates` features (50), not all 300+

2. **Small Symbol Count:**
   - With 2 symbols, CS ranking adds little value
   - **Mitigation:** `min_symbols: 5` threshold prevents execution

3. **Feature Name Mismatches:**
   - Panel data might have different feature names than per-symbol
   - **Mitigation:** Use same feature discovery logic, same leakage filters

4. **Timestamp Alignment:**
   - Symbols might have different timestamp ranges
   - **Mitigation:** Existing `prepare_cross_sectional_data_for_ranking()` already handles this

---

## 3. Implementation Plan (Step-by-Step)

### Phase 1: Core Module (Day 1)

**File:** `TRAINING/ranking/cross_sectional_feature_ranker.py`

```python
def compute_cross_sectional_importance(
    candidate_features: List[str],
    target_column: str,
    symbols: List[str],
    data_dir: Path,
    model_families: List[str] = ['lightgbm'],
    min_cs: int = 10,
    max_cs_samples: int = 1000,
    normalization: Optional[str] = None  # 'zscore' or None
) -> pd.Series:
    """
    Compute cross-sectional feature importance using panel model.
    
    Returns:
        Series with feature -> CS importance score
    """
    # 1. Load panel data (reuse existing utility)
    from TRAINING.utils.cross_sectional_data import (
        load_mtf_data_for_ranking,
        prepare_cross_sectional_data_for_ranking
    )
    
    mtf_data = load_mtf_data_for_ranking(data_dir, symbols)
    
    # 2. Build panel with candidate features only
    X, y, feature_names, symbols_array, time_vals = (
        prepare_cross_sectional_data_for_ranking(
            mtf_data, target_column,
            min_cs=min_cs,
            max_cs_samples=max_cs_samples,
            feature_names=candidate_features  # Only candidate features
        )
    )
    
    # 3. Optional normalization (per-date z-score)
    if normalization == 'zscore':
        X = normalize_cross_sectional_per_date(X, time_vals)
    
    # 4. Train panel model(s) and get importance
    importances = {}
    for model_family in model_families:
        model, importance = train_panel_model(
            X, y, feature_names, model_family
        )
        importances[model_family] = importance
    
    # 5. Aggregate across model families (mean)
    cs_importance = pd.Series(0.0, index=feature_names)
    for imp in importances.values():
        cs_importance += imp
    cs_importance /= len(importances)
    
    return cs_importance
```

### Phase 2: Integration (Day 2)

**File:** `TRAINING/ranking/feature_selector.py`

Add after line 195 (after aggregation):

```python
# Optional: Cross-sectional ranking (if enabled and enough symbols)
cs_importance = None
if (aggregation_config.get('cross_sectional_ranking', {}).get('enabled', False) and
    len(symbols) >= aggregation_config.get('cross_sectional_ranking', {}).get('min_symbols', 5)):
    
    from TRAINING.ranking.cross_sectional_feature_ranker import (
        compute_cross_sectional_importance,
        tag_features_by_importance
    )
    
    top_k_candidates = aggregation_config.get('cross_sectional_ranking', {}).get('top_k_candidates', 50)
    candidates = selected_features[:top_k_candidates]
    
    logger.info(f"🔍 Computing cross-sectional importance for {len(candidates)} candidate features...")
    cs_importance = compute_cross_sectional_importance(
        candidate_features=candidates,
        target_column=target_column,
        symbols=symbols,
        data_dir=data_dir,
        model_families=aggregation_config.get('cross_sectional_ranking', {}).get('model_families', ['lightgbm']),
        min_cs=10,  # Could be configurable
        max_cs_samples=1000,  # Could be configurable
        normalization=aggregation_config.get('cross_sectional_ranking', {}).get('normalization')
    )
    
    # Merge CS scores into summary_df
    summary_df['cs_importance_score'] = summary_df['feature'].map(cs_importance).fillna(0.0)
    
    # Tag features
    summary_df['feature_category'] = tag_features_by_importance(
        symbol_importance=summary_df['consensus_score'],
        cs_importance=summary_df['cs_importance_score']
    )
else:
    summary_df['cs_importance_score'] = 0.0
    summary_df['feature_category'] = 'UNKNOWN'  # Or 'SYMBOL_ONLY' if CS not run
    if len(symbols) < 5:
        logger.debug(f"Skipping cross-sectional ranking: only {len(symbols)} symbols (min: 5)")
```

### Phase 3: Config & Output (Day 2-3)

1. **Add config section** to `CONFIG/feature_selection/multi_model.yaml`
2. **Extend `save_multi_model_results()`** to include CS columns
3. **Add unit tests** for small panel (2-3 symbols, 10 timestamps)
4. **Add integration test** with 5+ symbols

---

## 4. Testing Strategy

### Unit Tests:
- `test_cross_sectional_ranker.py`
  - Test with 2 symbols (should skip or return zeros)
  - Test with 5 symbols (should run)
  - Test feature tagging logic (CORE vs SYMBOL-SPECIFIC)
  - Test normalization (z-score per date)

### Integration Tests:
- Run full feature selection with CS ranking enabled
- Verify `summary_df` has new columns
- Verify CSV output includes CS scores
- Verify no regression in per-symbol selection

### Regression Tests:
- Run existing feature selection (CS disabled)
- Verify output identical to before
- Verify no performance degradation

---

## 5. Rollout Plan

### Phase 1: Development (Week 1)
- Implement core module
- Add integration point
- Unit tests

### Phase 2: Testing (Week 2)
- Integration tests
- Test with 2 symbols (should skip)
- Test with 10+ symbols (should run)
- Verify no regressions

### Phase 3: Deployment (Week 3)
- Merge to main
- **Default: `enabled: false`** (opt-in)
- Document in config
- Monitor for issues

### Phase 4: Enable by Default (Week 4+)
- After validation, enable for runs with 5+ symbols
- Keep disabled for small runs (2-4 symbols)

---

## 6. Expected Benefits

1. **Feature Categorization:**
   - Identify universe-core features vs symbol-specific
   - Better feature prioritization in training

2. **Quality Signal:**
   - Features that work both per-symbol AND cross-sectionally are higher confidence
   - Helps filter out AAPL-specific quirks

3. **Scalability:**
   - Becomes more valuable as symbol count grows (20-200 symbols)
   - Provides sanity check on per-symbol results

4. **No Downside:**
   - Optional, can be disabled
   - Doesn't break existing functionality
   - Low compute cost (only runs on top_k candidates, not all features)

---

## 7. Conclusion

**Implementation:** ✅ **FEASIBLE** - 2-3 days of focused work, mostly integration

**Risk:** ✅ **VERY LOW** - Optional, additive, isolated, graceful degradation

**Value:** ✅ **HIGH** - Better feature understanding, scales with symbol count

**Recommendation:** ✅ **PROCEED** - Start with Phase 1 (core module), test thoroughly, then integrate

---

## 8. Next Steps

1. **Create:** `TRAINING/ranking/cross_sectional_feature_ranker.py` skeleton
2. **Test:** With 2 symbols (should skip gracefully)
3. **Test:** With 5+ symbols (should run and produce scores)
4. **Integrate:** Add to `feature_selector.py` as optional step
5. **Config:** Add to `multi_model.yaml` with `enabled: false` default
6. **Document:** Update feature selection docs

**Estimated Timeline:** 2-3 days for MVP, 1 week for full integration + testing

