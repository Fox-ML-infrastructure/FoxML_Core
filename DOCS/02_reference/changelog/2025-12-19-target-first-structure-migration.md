# Target-First Directory Structure Migration

**Date**: 2025-12-19  
**Type**: Architecture / Refactoring  
**Impact**: High - Changes output directory structure  
**Breaking**: No - Backward compatible (reading logic supports both structures)

## Summary

Migrated all output artifacts from phase-first organization (`REPRODUCIBILITY/{STAGE}/...`) to target-first organization (`targets/<target>/...`). This makes it easier to find all information related to a specific target in one place and improves decision-making workflows.

## Changes

### New Structure

**Target-First Organization**:
```
{run_dir}/
├── manifest.json                    # Run-level manifest
├── globals/                         # Global summaries
│   ├── routing_decisions.json
│   ├── target_prioritization.yaml
│   ├── target_confidence_summary.json
│   └── stats.json
└── targets/                         # Per-target artifacts
    └── {target}/
        ├── metadata.json           # Per-target metadata
        ├── decision/               # Routing and prioritization
        ├── reproducibility/       # Reproducibility tracking
        ├── metrics/                # Performance metrics
        ├── models/                 # Trained models
        └── trends/                 # Trend analysis
```

### Key Improvements

1. **Target-Centric**: All artifacts for a target are in `targets/<target>/`
2. **Self-Contained**: Each target directory has everything needed for analysis
3. **Better Organization**: No duplicate structure (phase/mode/target → just target)
4. **Easier Navigation**: Find all target information without traversing multiple directories
5. **Cleaner Structure**: Global summaries in `globals/`, per-target in `targets/`

### Files Modified

#### Core Structure
- `TRAINING/orchestration/utils/target_first_paths.py` - **NEW**: Helper functions for target-first paths
- `TRAINING/orchestration/utils/manifest.py` - **NEW**: Manifest creation and per-target metadata

#### Reproducibility Tracking
- `TRAINING/orchestration/utils/reproducibility_tracker.py`
  - Writes `metadata.json` to `targets/<target>/reproducibility/<view>/cohort=<id>/`
  - Removed legacy `REPRODUCIBILITY/` directory creation
  - `stats.json` now goes to `globals/stats.json`

#### Metrics
- `TRAINING/common/utils/metrics.py`
  - Writes metrics to `targets/<target>/metrics/<view>/`
  - Also writes to `targets/<target>/reproducibility/<view>/cohort=<id>/`
  - Removed all legacy writes

#### Diff Telemetry
- `TRAINING/orchestration/utils/diff_telemetry.py`
  - Writes `snapshot.json`, `diff_prev.json`, `diff_baseline.json`, `metric_deltas.json` to target-first structure
  - Removed all legacy writes

#### Routing and Decisions
- `TRAINING/ranking/target_routing.py`
  - Writes routing decisions to `globals/routing_decisions.json` and `targets/<target>/decision/routing_decision.json`
  - `load_routing_decisions()` checks target-first first, then legacy

- `TRAINING/ranking/feature_selection_reporting.py`
  - Writes feature prioritization to `targets/<target>/decision/feature_prioritization.yaml`
  - Writes feature selection rankings to `targets/<target>/reproducibility/`

- `TRAINING/ranking/predictability/reporting.py`
  - Writes target rankings to `globals/target_prioritization.yaml` and `globals/target_predictability_rankings.csv`
  - Writes feature importances to `targets/<target>/reproducibility/feature_importances/`

- `TRAINING/orchestration/target_routing.py`
  - Writes confidence summaries to `globals/target_confidence_summary.json` and `.csv`

#### Feature Selection
- `TRAINING/ranking/multi_model_feature_selection.py`
  - Writes feature importance files to `targets/<target>/reproducibility/feature_importances/`

- `TRAINING/ranking/predictability/model_evaluation.py`
  - Writes featureset artifacts to `targets/<target>/reproducibility/featureset_artifacts/`
  - Writes feature exclusions to `targets/<target>/reproducibility/feature_exclusions/`

- `TRAINING/ranking/shared_ranking_harness.py`
  - Writes feature exclusions to `targets/<target>/reproducibility/feature_exclusions/`

#### Model Training
- `TRAINING/training_strategies/execution/training.py`
  - Saves models to `targets/<target>/models/<family>/`
  - Removed legacy model saving paths

#### Stability Tracking
- `TRAINING/stability/feature_importance/io.py`
  - `get_snapshot_base_dir()` returns `targets/<target>/reproducibility/feature_importance_snapshots/` when `target_name` provided

#### Trend Analysis
- `TRAINING/common/utils/trend_analyzer.py`
  - `load_artifact_index()` traverses target-first structure
  - Writes trend reports to `targets/<target>/trends/` (within-run) and `trend_reports/by_target/<target>/` (across-runs)

#### Metrics Aggregation
- `TRAINING/orchestration/metrics_aggregator.py`
  - Checks target-first structure first, then falls back to legacy
  - Updated `_load_cross_sectional_metrics()`, `_load_symbol_metrics()`, `_load_stability_metrics()`

#### Orchestration
- `TRAINING/orchestration/intelligent_trainer.py`
  - Calls `initialize_run_structure()` to create `targets/` and `globals/`
  - Calls `create_manifest()` and `create_target_metadata()` after training
  - Removed legacy directory creation

- `TRAINING/orchestration/routing_integration.py`
  - Updated to use run directory (MetricsAggregator handles path resolution)

## Backward Compatibility

### Reading Logic
All reading logic checks target-first structure first, then falls back to legacy:
- `load_routing_decisions()` - Checks `globals/routing_decisions.json`, then legacy
- `MetricsAggregator` - Checks `targets/<target>/reproducibility/`, then legacy
- `TrendAnalyzer.load_artifact_index()` - Traverses both structures
- `aggregate_metrics_facts()` - Reads from both structures

### Legacy Structure
- **No longer created**: New runs do not create `REPRODUCIBILITY/` directories
- **Preserved**: Old runs remain in legacy structure (not migrated)
- **Readable**: All readers support both structures

## Migration Notes

### For New Runs
- Automatically use target-first structure
- No configuration changes needed
- All artifacts written to `targets/<target>/` and `globals/`

### For Old Runs
- Legacy structure preserved but not written to
- Reading logic automatically finds artifacts in legacy structure
- No migration script needed (readers handle both)

### For Analysis Tools
- Updated to read from target-first structure first
- Fall back to legacy structure for old runs
- No breaking changes for existing analysis workflows

## Testing

- ✅ All writes go to target-first structure
- ✅ All reads check target-first first, then legacy
- ✅ No legacy directory creation in new runs
- ✅ Backward compatibility maintained
- ✅ All metadata tracking works
- ✅ All snapshots work
- ✅ All diff telemetry works
- ✅ All trend analysis works

## Documentation Updates

- Updated `DOCS/03_technical/implementation/REPRODUCIBILITY_STRUCTURE.md` with new target-first structure
- Updated `DOCS/03_technical/telemetry/DIFF_TELEMETRY.md` with new paths
- Updated `DOCS/01_tutorials/training/INTELLIGENT_TRAINING_TUTORIAL.md` with new structure

## Related Issues

- Better organization for decision-making workflows
- Easier navigation of target-specific artifacts
- Cleaner file structure
- Improved traceability (all target artifacts together)

