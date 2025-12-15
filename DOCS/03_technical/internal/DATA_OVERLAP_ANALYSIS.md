# Data Overlap Analysis: target_rankings/ vs REPRODUCIBILITY/

## Summary

**Yes, there is significant overlap**, but each location also has **unique data** that the other doesn't have.

## Data Comparison

### 1. `target_rankings/target_predictability_rankings.csv` & `.yaml`

**Contains:**
- ✅ **Aggregated summary** across ALL targets (one row per target)
- ✅ **Per-model scores**: `lightgbm_r2`, `xgboost_r2`, `random_forest_r2`, `neural_network_r2`, `catboost_r2`, `lasso_r2`, `mutual_information_r2`, `univariate_selection_r2`, `rfe_r2`, `boruta_r2` - **UNIQUE DATA**
- ✅ **Consistency metric** - **UNIQUE DATA**
- ✅ **Recommendations**: `PRIORITIZE`, `ENABLE`, `TEST` - **DECISION DATA**
- ✅ Core metrics: `composite_score`, `mean_score`, `std_score`, `mean_importance`, `n_models`, `leakage_flag`, `task_type`
- ✅ Ranking: `rank` (1, 2, 3, ...)

**Granularity:** One row per target (aggregated across all views/cohorts)

### 2. `REPRODUCIBILITY/TARGET_RANKING/{view}/{target}/cohort={cohort_id}/metrics.json`

**Contains:**
- ✅ Core metrics: `composite_score`, `mean_score`, `std_score`, `mean_importance`, `n_models`, `leakage_flag`, `task_type`
- ✅ Cohort-specific metadata: `N_effective_cs`, `n_features_pre`, `n_features_post_prune`, `pos_rate` - **UNIQUE DATA**
- ❌ **NO per-model scores** (lightgbm_r2, xgboost_r2, etc.)
- ❌ **NO recommendations**
- ❌ **NO consistency metric**
- ❌ **NO ranking**

**Granularity:** One file per target/view/cohort combination (much more granular)

### 3. `REPRODUCIBILITY/TARGET_RANKING/routing_decisions.json`

**Contains:**
- ✅ Routing decisions: Which targets route to which views (CROSS_SECTIONAL, SYMBOL_SPECIFIC, BOTH, BLOCKED) - **DECISION DATA**
- ✅ Routing rationale: `reason`, `cs_auc`, `symbol_auc_mean`, `symbol_auc_median`, etc. - **UNIQUE DATA**
- ✅ Summary statistics: `total_targets`, `cross_sectional_only`, `symbol_specific_only`, `both`, `blocked`

**Granularity:** One file for entire run (all targets)

## Overlap Summary

### Overlapping Data (same metrics, different granularity):
- `composite_score`
- `mean_score`
- `std_score`
- `mean_importance`
- `n_models`
- `leakage_flag`
- `task_type`

**But:** 
- `target_rankings/` = **aggregated** (one row per target, combines all views/cohorts)
- `REPRODUCIBILITY/` = **per-cohort** (one file per target/view/cohort)

### Unique to `target_rankings/`:
1. **Per-model scores** (`lightgbm_r2`, `xgboost_r2`, etc.) - Critical for model comparison
2. **Consistency metric** - Model agreement measure
3. **Recommendations** (`PRIORITIZE`, `ENABLE`, `TEST`) - Decision-making guidance
4. **Ranking** (1, 2, 3, ...) - Ordered list

### Unique to `REPRODUCIBILITY/`:
1. **Per-cohort granularity** - Can see metrics for each specific cohort
2. **Cohort metadata** (`N_effective_cs`, `n_features_pre`, `n_features_post_prune`, `pos_rate`)
3. **Full metadata.json** - Complete reproducibility context (date ranges, symbols, config hashes, etc.)
4. **Telemetry files** - Drift tracking, trend analysis
5. **Feature importances** - Per-model feature importance CSVs
6. **Feature exclusion lists** - What was excluded and why

## Recommendation

### Option 1: Keep Both (Recommended)
- **`target_rankings/` CSV**: Keep as **aggregated summary** with per-model scores (human-readable, Excel-friendly)
- **`REPRODUCIBILITY/`**: Keep for **detailed per-cohort analysis** and reproducibility auditing
- **Rationale**: They serve different purposes:
  - `target_rankings/` = "Quick summary: Which targets are best? Which models performed best?"
  - `REPRODUCIBILITY/` = "Deep dive: What happened in each specific cohort? Can I reproduce this?"

### Option 2: Consolidate (More Complex)
- Move `target_rankings/` CSV to `REPRODUCIBILITY/TARGET_RANKING/target_predictability_rankings.csv`
- Move `target_rankings/` YAML to `DECISION/TARGET_RANKING/target_prioritization.yaml`
- **Problem**: Per-model scores are not currently stored in REPRODUCIBILITY, so we'd need to add them
- **Problem**: Aggregated summary would be mixed with per-cohort data

## Decision

**For the reorganization plan:**

1. **`target_predictability_rankings.csv`** → `REPRODUCIBILITY/TARGET_RANKING/`
   - Contains unique per-model scores
   - Is a reproducibility artifact (metrics table)
   - But note: It's aggregated, not per-cohort

2. **`target_predictability_rankings.yaml`** → `DECISION/TARGET_RANKING/target_prioritization.yaml`
   - Contains recommendations (decisions)
   - Is a decision log

3. **`routing_decisions.json`** → `DECISION/TARGET_RANKING/`
   - Is a decision log
   - Also keep copy in `REPRODUCIBILITY/` for convenience

**Key Insight:** The CSV has **unique per-model score data** that's not in REPRODUCIBILITY. We should preserve this in the new structure.
