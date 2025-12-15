# Results Directory Reorganization Plan

## Current Structure Analysis

### Current Output Locations

1. **`target_rankings/`** (at run root):
   - `target_predictability_rankings.csv` - Rankings with metrics
   - `target_predictability_rankings.yaml` - Rankings with recommendations/decisions
   - **Written by**: `TRAINING/ranking/predictability/reporting.py::save_rankings()`
   - **Called from**: `TRAINING/ranking/target_ranker.py::rank_targets()` (line 977)

2. **`feature_exclusions/`** (at run root):
   - `{target}_exclusions.yaml` - Target-conditional exclusion lists
   - **Written by**: `TRAINING/utils/target_conditional_exclusions.py::generate_target_exclusion_list()` (line 315)
   - **Also written to**: `REPRODUCIBILITY/TARGET_RANKING/{view}/{target}/feature_exclusions/` (duplicate)
   - **Also written to**: `feature_selections/{target}/feature_exclusions/` (for feature selection stage)

3. **`REPRODUCIBILITY/TARGET_RANKING/`**:
   - `routing_decisions.json` - Routing decisions (which targets go to which views)
   - `CROSS_SECTIONAL/{target}/cohort={cohort_id}/` - Metrics, metadata, telemetry
   - `SYMBOL_SPECIFIC/{target}/symbol={symbol}/cohort={cohort_id}/` - Metrics, metadata, telemetry
   - `{view}/{target}/feature_exclusions/` - Exclusion lists (duplicate of root-level)

4. **`feature_selections/{target}/`**:
   - `feature_selection_rankings.csv` - Feature rankings with metrics
   - `feature_selection_rankings.yaml` - Feature rankings with recommendations
   - `selected_features.txt` - Selected features list
   - `feature_exclusions/{target}_exclusions.yaml` - Exclusion lists
   - `REPRODUCIBILITY/FEATURE_SELECTION/` - Metrics, metadata, telemetry

## Classification: Decision vs Reproducibility

### Decision Logs (should go to `DECISION/`)
These are **actionable decisions** made by the system:
- **`routing_decisions.json`** - Which targets route to which views (CROSS_SECTIONAL, SYMBOL_SPECIFIC, BOTH, BLOCKED)
- **`target_predictability_rankings.yaml`** - Recommendations about which targets to prioritize (PRIORITIZE, CONSIDER, SKIP)
- **`feature_selection_rankings.yaml`** - Recommendations about which features to use (SELECT, CONSIDER, REJECT)

### Reproducibility Artifacts (should go to `REPRODUCIBILITY/`)
These are **evidence/metrics** needed to reproduce or audit the run:
- **`target_predictability_rankings.csv`** - Full metrics table (composite_score, mean_score, std_score, etc.)
- **`feature_selection_rankings.csv`** - Full metrics table (consensus_score, n_models_agree, etc.)
- **`feature_exclusions/{target}_exclusions.yaml`** - What was excluded and why (reproducibility metadata)
- **`selected_features.txt`** - Final selected features list (reproducibility artifact)
- All existing `REPRODUCIBILITY/` structure (metrics.json, metadata.json, telemetry, etc.)

## Proposed New Structure

```
RESULTS/{run}/
├── DECISION/
│   ├── TARGET_RANKING/
│   │   ├── routing_decisions.json          # Which targets route where
│   │   └── target_prioritization.yaml      # Which targets to prioritize (renamed from target_predictability_rankings.yaml)
│   └── FEATURE_SELECTION/
│       └── {target}/
│           └── feature_prioritization.yaml # Which features to use (renamed from feature_selection_rankings.yaml)
│
└── REPRODUCIBILITY/
    ├── TARGET_RANKING/
    │   ├── target_predictability_rankings.csv  # Full metrics table
    │   ├── routing_decisions.json              # MOVE: Decision log (duplicate for convenience)
    │   ├── CROSS_SECTIONAL/
    │   │   └── {target}/
    │   │       ├── cohort={cohort_id}/
    │   │       │   ├── metrics.json
    │   │       │   ├── metadata.json
    │   │       │   ├── telemetry_*.json
    │   │       │   └── ...
    │   │       └── feature_exclusions/
    │   │           └── {target}_exclusions.yaml
    │   └── SYMBOL_SPECIFIC/
    │       └── {target}/
    │           └── symbol={symbol}/
    │               ├── cohort={cohort_id}/
    │               │   └── ...
    │               └── feature_exclusions/
    │                   └── {target}_exclusions.yaml
    │
    └── FEATURE_SELECTION/
        ├── feature_selection_rankings.csv  # Full metrics table (aggregated across targets)
        └── {target}/
            ├── feature_selection_rankings.csv  # Per-target metrics
            ├── selected_features.txt           # Final selected features
            ├── feature_exclusions/
            │   └── {target}_exclusions.yaml
            ├── CROSS_SECTIONAL/
            │   └── cohort={cohort_id}/
            │       └── ...
            └── SYMBOL_SPECIFIC/
                └── symbol={symbol}/
                    └── cohort={cohort_id}/
                        └── ...
```

## Implementation Plan

### Phase 1: Create New Directory Structure
1. Create `DECISION/` and `REPRODUCIBILITY/` at run root level
2. Create subdirectories: `DECISION/TARGET_RANKING/`, `DECISION/FEATURE_SELECTION/`

### Phase 2: Update Target Ranking Outputs

**Files to modify:**
- `TRAINING/ranking/predictability/reporting.py::save_rankings()`
  - Move `target_predictability_rankings.yaml` → `DECISION/TARGET_RANKING/target_prioritization.yaml`
  - Move `target_predictability_rankings.csv` → `REPRODUCIBILITY/TARGET_RANKING/target_predictability_rankings.csv`
  - Update function signature to accept base output_dir (not target_rankings subdir)

- `TRAINING/ranking/target_routing.py::_save_dual_view_rankings()`
  - Move `routing_decisions.json` → `DECISION/TARGET_RANKING/routing_decisions.json`
  - Also save a copy to `REPRODUCIBILITY/TARGET_RANKING/routing_decisions.json` (for convenience/reproducibility)

- `TRAINING/ranking/target_ranker.py::rank_targets()`
  - Update `output_dir` parameter handling (accept base dir, not target_rankings subdir)
  - Update calls to `save_rankings()` and `_save_dual_view_rankings()`

- `TRAINING/orchestration/intelligent_trainer.py`
  - Update `output_dir` passed to `rank_targets()` (remove `/target_rankings` suffix)
  - Update references to `target_rankings` subdirectory
  - Update telemetry rollup paths

### Phase 3: Update Feature Exclusion Outputs

**Files to modify:**
- `TRAINING/utils/target_conditional_exclusions.py::generate_target_exclusion_list()`
  - Remove writing to root-level `feature_exclusions/` directory
  - Keep only `REPRODUCIBILITY/TARGET_RANKING/{view}/{target}/feature_exclusions/` location
  - For feature selection stage: use `REPRODUCIBILITY/FEATURE_SELECTION/{target}/feature_exclusions/`

- `TRAINING/ranking/shared_ranking_harness.py::build_panel()`
  - Update `target_exclusion_dir` to use `REPRODUCIBILITY/` structure
  - Remove root-level `feature_exclusions/` directory creation

- `TRAINING/ranking/predictability/model_evaluation.py::evaluate_target_predictability()`
  - Already writes to `REPRODUCIBILITY/TARGET_RANKING/{view}/{target}/feature_exclusions/` (line 4371)
  - Verify this is the only location

### Phase 4: Update Feature Selection Outputs

**Files to modify:**
- `TRAINING/ranking/feature_selection_reporting.py::save_feature_selection_rankings()`
  - Move `feature_selection_rankings.yaml` → `DECISION/FEATURE_SELECTION/{target}/feature_prioritization.yaml`
  - Move `feature_selection_rankings.csv` → `REPRODUCIBILITY/FEATURE_SELECTION/{target}/feature_selection_rankings.csv`
  - Keep `selected_features.txt` in `REPRODUCIBILITY/FEATURE_SELECTION/{target}/`

- `TRAINING/ranking/feature_selector.py::select_features_for_target()`
  - Update `output_dir` handling to use new structure

- `TRAINING/orchestration/intelligent_trainer.py`
  - Update feature selection `output_dir` references

### Phase 5: Update Loading/Reading Code

**Files to check/update:**
- `TRAINING/utils/target_conditional_exclusions.py::load_target_exclusion_list()`
  - Update to read from `REPRODUCIBILITY/` structure only

- `TRAINING/ranking/target_routing.py::load_routing_decisions()`
  - Update to read from `DECISION/TARGET_RANKING/` (with fallback to `REPRODUCIBILITY/` for backward compatibility)

- `TRAINING/orchestration/intelligent_trainer.py`
  - Update any code that reads `target_rankings/` or `feature_exclusions/` directories

- `TRAINING/orchestration/metrics_aggregator.py`
  - Update paths that reference `target_rankings/` subdirectory

### Phase 6: Cleanup

1. Remove creation of old directories:
   - `target_rankings/` (at run root)
   - `feature_exclusions/` (at run root)

2. Update documentation:
   - `DOCS/01_tutorials/training/AUTO_TARGET_RANKING.md`
   - `DOCS/01_tutorials/training/INTELLIGENT_TRAINING_TUTORIAL.md`
   - `DOCS/03_technical/implementation/REPRODUCIBILITY_TRACKING.md`
   - Any other docs referencing old paths

3. Update changelog

## Backward Compatibility Considerations

1. **Loading code should check both old and new locations** (with deprecation warnings)
2. **Migration script** (optional): Move existing results to new structure
3. **Documentation**: Clearly state new structure and deprecation timeline

## Benefits

1. **Clear separation**: Decisions vs. Reproducibility artifacts
2. **Professional structure**: `DECISION/` and `REPRODUCIBILITY/` at same level
3. **No duplication**: Single source of truth for each artifact type
4. **Easier navigation**: Decision-makers go to `DECISION/`, auditors go to `REPRODUCIBILITY/`
5. **Consistent naming**: All decision YAMLs use `*_prioritization.yaml` naming

## Risk Assessment

- **Low risk**: Mostly path changes, no logic changes
- **Testing needed**: Verify all outputs are written to correct locations
- **Breaking change**: Old paths will no longer exist (but loading code can handle both)
