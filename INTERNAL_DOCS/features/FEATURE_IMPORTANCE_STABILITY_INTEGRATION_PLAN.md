# Feature Importance Stability Integration Plan

**Date:** 2025-12-10  
**Goal:** Wire feature importance stability tracking into feature selection and target ranking without UI dependencies

---

## Current Architecture Analysis

### 1. **Target Ranking** (`model_evaluation.py`)
**Current State:**
- Uses `quick_importance_prune()` from `feature_pruning.py`
- Pruning returns: `(X_pruned, feature_names_pruned, pruning_stats)`
- `pruning_stats` contains `top_10_features` and `top_10_importance` but not full importances
- After pruning, trains models and gets importances
- Saves importances via `_save_feature_importances()` to CSV files
- Location: `results/feature_importances/{target}/{symbol}/{model}_importances.csv`

**Integration Point:**
- After `quick_importance_prune()` completes (line ~337)
- After model training completes and importances are computed (line ~1779)
- Already has all the data needed, just needs snapshot save

**Difficulty: EASY** ⭐
- Importances already computed
- Just need to extract from `pruning_stats` or model importances
- Add snapshot save alongside existing CSV save

---

### 2. **Feature Selection** (`multi_model_feature_selection.py`)
**Current State:**
- Multiple methods: RFE, Boruta, Stability Selection, Mutual Information
- Each method calls `train_model_and_get_importance()` which returns `(model, importance_series, method_name)`
- Methods return `ImportanceResult` objects (from `feature_selector.py`)
- Final aggregation happens in `aggregate_multi_model_importance()` in `feature_selector.py`
- Location: `TRAINING/ranking/feature_selector.py:select_features_for_target()`

**Integration Points:**
1. **Per-method snapshots** (after each method completes):
   - After `train_model_and_get_importance()` returns (line ~605-970)
   - Each method (RFE, Boruta, etc.) has its own importance
   - Method: `"rfe"`, `"boruta"`, `"stability_selection"`, `"mutual_information"`

2. **Final aggregated snapshot** (after aggregation):
   - After `aggregate_multi_model_importance()` in `feature_selector.py`
   - This is the final selected features + aggregated importance
   - Method: `"multi_model_aggregated"`

**Difficulty: MEDIUM** ⭐⭐
- Need to identify where each method's importance is finalized
- Need to extract importance from `ImportanceResult` objects
- Need to determine `universe_id` (symbol list or "ALL")
- Multiple integration points (one per method + final aggregation)

---

### 3. **Quick Pruning** (`feature_pruning.py`)
**Current State:**
- Already computes full importances: `model.feature_importances_`
- Returns normalized importances in `pruning_stats['top_10_importance']`
- But doesn't return full feature->importance mapping
- Called from both target ranking and feature selection

**Integration Point:**
- At end of `quick_importance_prune()` function (line ~226)
- Already has `normalized_importance` array and `feature_names`
- Just need to create snapshot before returning

**Difficulty: EASY** ⭐
- Already has all data
- Just need to create dict mapping and save snapshot
- But: needs `target_name` and `universe_id` passed in (currently not available)

---

## Integration Scaffold

### Phase 1: Core Infrastructure (No Integration Yet)
**Files to Create:**
1. `TRAINING/utils/feature_importance_schema.py` (~30 LOC)
   - `FeatureImportanceSnapshot` dataclass
   - Helper to convert dict/Series to snapshot

2. `TRAINING/utils/feature_importance_io.py` (~50 LOC)
   - `save_importance_snapshot()` function
   - `load_snapshots()` function
   - Directory structure: `artifacts/feature_importance/{target}/{method}/{run_id}.json`

3. `scripts/analyze_importance_stability.py` (~200 LOC)
   - CLI script for stability analysis
   - No UI, just prints to stdout

**Effort:** ~2-3 hours  
**Risk:** Low (isolated, no core changes)

---

### Phase 2: Target Ranking Integration
**Files to Modify:**
1. `TRAINING/ranking/predictability/model_evaluation.py`
   - After `quick_importance_prune()` (line ~337)
     - Extract full importances from pruning model
     - Create snapshot with method="quick_pruner"
     - Save snapshot
   - After `_save_feature_importances()` (line ~1809)
     - For each model, create snapshot with method="lightgbm", "random_forest", etc.
     - Save snapshot alongside CSV

**Changes Needed:**
- Pass `target_column` to `quick_importance_prune()` (already available)
- Extract importances from pruning model before returning
- Add snapshot save calls (2-3 locations)

**Effort:** ~1-2 hours  
**Risk:** Low (additive, doesn't change existing logic)

---

### Phase 3: Feature Selection Integration
**Files to Modify:**
1. `TRAINING/ranking/multi_model_feature_selection.py`
   - After each `train_model_and_get_importance()` call (line ~605-970)
     - Create snapshot with method="rfe", "boruta", etc.
     - Save snapshot
   - Need to pass `target_column` and `universe_id` through call chain

2. `TRAINING/ranking/feature_selector.py`
   - After `aggregate_multi_model_importance()` (line ~202)
     - Create snapshot with method="multi_model_aggregated"
     - Save snapshot

**Changes Needed:**
- Add `target_column` parameter to `train_model_and_get_importance()`
- Add `universe_id` parameter (symbol list or "ALL")
- Add snapshot save after each method
- Add snapshot save after aggregation

**Effort:** ~2-3 hours  
**Risk:** Medium (needs parameter threading through call chain)

---

### Phase 4: Quick Pruning Integration (Optional)
**Files to Modify:**
1. `TRAINING/utils/feature_pruning.py`
   - Add optional `target_name` and `universe_id` parameters
   - At end of function, create snapshot if parameters provided
   - Return snapshot path in `pruning_stats`

**Changes Needed:**
- Add optional parameters (backward compatible)
- Create snapshot if parameters provided
- Update callers to pass parameters

**Effort:** ~1 hour  
**Risk:** Low (optional parameters, backward compatible)

---

## Integration Difficulty Summary

| Component | Difficulty | Effort | Risk | Priority |
|-----------|-----------|--------|------|----------|
| Core Infrastructure | ⭐ Easy | 2-3h | Low | **P0** (required) |
| Target Ranking | ⭐ Easy | 1-2h | Low | **P1** (high value) |
| Feature Selection | ⭐⭐ Medium | 2-3h | Medium | **P1** (high value) |
| Quick Pruning | ⭐ Easy | 1h | Low | **P2** (nice to have) |

**Total Estimated Effort:** 6-9 hours  
**Total Risk:** Low-Medium (mostly additive changes)

---

## Key Design Decisions

### 1. **Snapshot Location**
```
artifacts/feature_importance/
  {target_name}/
    {method}/
      {run_id}.json
```

**Rationale:**
- Matches existing CSV structure (`results/feature_importances/{target}/{symbol}/`)
- Easy to find all snapshots for a target+method
- `run_id` = UUID or timestamp-based (deterministic from run metadata)

### 2. **Method Names**
- Target Ranking: `"quick_pruner"`, `"lightgbm"`, `"random_forest"`, `"neural_network"`
- Feature Selection: `"rfe"`, `"boruta"`, `"stability_selection"`, `"mutual_information"`, `"multi_model_aggregated"`
- Cross-Sectional: `"cross_sectional"` (if we add it later)

### 3. **Universe ID**
- Target Ranking: `None` (single symbol) or symbol name
- Feature Selection: `"ALL"` or comma-separated symbol list
- Cross-Sectional: `"CROSS_SECTIONAL"`

### 4. **Run ID Generation**
- Use deterministic UUID from: `target_name + method + timestamp + git_commit`
- Or simple timestamp: `datetime.now().strftime("%Y%m%d_%H%M%S")`
- Store in snapshot for traceability

---

## Implementation Order

1. **Phase 1** (Core Infrastructure) - Build foundation
2. **Phase 2** (Target Ranking) - Quick win, high visibility
3. **Phase 3** (Feature Selection) - More complex but high value
4. **Phase 4** (Quick Pruning) - Optional polish

---

## Testing Strategy

1. **Unit Tests:**
   - Schema serialization/deserialization
   - IO save/load functions
   - Stability metrics calculations

2. **Integration Tests:**
   - Run target ranking → verify snapshots created
   - Run feature selection → verify snapshots created
   - Run stability script → verify metrics computed

3. **Manual Verification:**
   - Run pipeline on test data
   - Check `artifacts/feature_importance/` directory structure
   - Run stability script and verify output

---

## Potential Issues & Mitigations

### Issue 1: Parameter Threading
**Problem:** Need to pass `target_column` and `universe_id` through call chains  
**Mitigation:** Add as optional parameters with defaults, thread through gradually

### Issue 2: Duplicate Snapshots
**Problem:** Same run might create multiple snapshots (pruning + models)  
**Mitigation:** Use same `run_id` for all snapshots in a single run, or use method-specific run_ids

### Issue 3: Disk Space
**Problem:** Many snapshots could accumulate  
**Mitigation:** Add retention policy (keep last N per target+method) or compression

### Issue 4: Backward Compatibility
**Problem:** Adding parameters might break existing callers  
**Mitigation:** All new parameters optional with sensible defaults

---

## Success Criteria

✅ Can run target ranking and see snapshots in `artifacts/feature_importance/`  
✅ Can run feature selection and see snapshots for each method  
✅ Can run `scripts/analyze_importance_stability.py` and get stability metrics  
✅ No UI dependencies, no dashboard requirements  
✅ Backward compatible (existing code still works)  
✅ Minimal changes to core training logic (< 50 LOC per integration point)

---

## Automation Options

### Option A: Automatic Stability Analysis (Recommended)

**Design:** Run stability analysis automatically after each run completes, log results to stdout.

**Integration Points:**

1. **Target Ranking** (`main.py`):
   - After all targets evaluated (line ~304)
   - For each target that has 2+ snapshots, run stability analysis
   - Log stability metrics to console
   - Save stability report to `{output_dir}/stability_reports/{target_name}.txt`

2. **Feature Selection** (`feature_selector.py`):
   - After `select_features_for_target()` completes (line ~210)
   - If 2+ snapshots exist for this target+method, run stability analysis
   - Log stability metrics
   - Save report alongside selection results

**Implementation:**
```python
# TRAINING/utils/feature_importance_stability.py
def analyze_stability_auto(
    base_dir: Path,
    target_name: str,
    method: str,
    min_snapshots: int = 2,
    log_to_console: bool = True,
    save_report: bool = True,
    report_path: Optional[Path] = None
) -> Optional[Dict[str, float]]:
    """Automatically analyze stability if enough snapshots exist."""
    snapshots = load_snapshots(base_dir, target_name, method)
    
    if len(snapshots) < min_snapshots:
        return None  # Not enough data yet
    
    metrics = compute_stability_metrics(snapshots)
    
    if log_to_console:
        logger.info(f"📊 Stability for {target_name}/{method}:")
        logger.info(f"   Top-20 overlap: {metrics['mean_overlap']:.3f}")
        logger.info(f"   Kendall tau: {metrics['mean_tau']:.3f}")
        if metrics['mean_overlap'] < 0.7:
            logger.warning(f"   ⚠️  Low stability detected (overlap < 0.7)")
    
    if save_report:
        save_stability_report(metrics, report_path or base_dir / "stability_reports" / f"{target_name}_{method}.txt")
    
    return metrics
```

**Changes Needed:**
- Add call to `analyze_stability_auto()` after target ranking completes
- Add call after feature selection completes
- Config flag: `safety.feature_importance.auto_analyze_stability: true`

**Effort:** +1 hour (on top of base implementation)  
**Difficulty:** Easy (just add function calls)

---

### Option B: Background Analysis (Post-Run)

**Design:** Run stability analysis as a separate step after main pipeline completes.

**Implementation:**
- Add CLI command: `python scripts/analyze_all_stability.py --output-dir {dir}`
- Or add to Makefile: `make analyze-stability`
- Scans all snapshots in `artifacts/feature_importance/`
- Generates comprehensive report for all targets/methods

**Effort:** +0.5 hours  
**Difficulty:** Easy (standalone script)

---

### Option C: Config-Driven Thresholds with Warnings

**Design:** Automatically warn if stability drops below thresholds.

**Implementation:**
```python
# In stability analysis
stability_config = get_cfg("safety.feature_importance.stability_thresholds", default={})
min_overlap = stability_config.get('min_top_k_overlap', 0.7)
min_tau = stability_config.get('min_kendall_tau', 0.6)

if metrics['mean_overlap'] < min_overlap:
    logger.warning(f"⚠️  Low feature stability for {target_name}/{method}: "
                  f"overlap={metrics['mean_overlap']:.3f} < {min_overlap}")
    # Could also mark target as "unstable" in results
```

**Effort:** +0.5 hours  
**Difficulty:** Easy (just add threshold checks)

---

### Option D: Full Automation (All of the Above)

**Combined approach:**
1. Automatic snapshot saving (Phase 1-3)
2. Automatic stability analysis after each run (Option A)
3. Config-driven thresholds with warnings (Option C)
4. Optional: Background batch analysis script (Option B)

**Result:** 
- Run pipeline → automatically get stability metrics in logs
- Low stability → automatic warnings
- Can still run manual analysis script for deeper dives

**Total Effort:** +2 hours (on top of base 6-9h)  
**Total Difficulty:** Easy-Medium

---

## Recommended: Option D (Full Automation)

**Why:**
- Zero manual steps required
- Immediate feedback in logs
- Warnings catch issues early
- Still allows manual deep-dive analysis

**User Experience:**
```
$ python TRAINING/train.py --auto-targets

... training runs ...

📊 Stability for peak_60m_0.8/quick_pruner:
   Top-20 overlap: 0.852
   Kendall tau: 0.734
✅ Stability is good

📊 Stability for peak_60m_0.8/lightgbm:
   Top-20 overlap: 0.623
   Kendall tau: 0.512
⚠️  Low stability detected (overlap < 0.7) - feature importance may be unstable
```

**No UI needed, just automated logging and warnings.**

---

## Next Steps

1. Review this plan
2. Approve core infrastructure design + automation approach
3. Implement Phase 1 (core infrastructure)
4. Add automation hooks (Option D)
5. Test Phase 1 in isolation
6. Proceed with Phase 2 (target ranking)
7. Proceed with Phase 3 (feature selection)
8. Optional: Phase 4 (quick pruning)
