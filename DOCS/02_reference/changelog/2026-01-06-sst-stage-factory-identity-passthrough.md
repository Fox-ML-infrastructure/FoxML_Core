# SST Stage Factory & Identity Passthrough

**Date**: 2026-01-06  
**Category**: Reproducibility, Determinism Tracking  
**Impact**: Medium (improves observability, no computation changes)

## Summary

Added SST (Single Source of Truth) stage factory for stage-aware reproducibility tracking and fixed identity passthrough issues that caused null fingerprints in FEATURE_SELECTION snapshots.

## Problem

1. **Missing Stage Context**: No SST mechanism to track which pipeline stage (TARGET_RANKING, FEATURE_SELECTION, TRAINING) was currently executing
2. **Silent Identity Failures**: FEATURE_SELECTION identity creation logged failures at DEBUG level, silently resulting in null fingerprints
3. **No Fallback**: When identity finalization failed, partial identity signatures were discarded instead of being used as fallback
4. **Stage Separation**: No clear separation of reproducibility outputs between stages

## Changes

### SST Stage Factory (`TRAINING/orchestration/utils/run_context.py`)

New functions for stage tracking:
- `save_stage_transition(output_dir, stage, reason)` - Records stage to `run_context.json`
- `get_current_stage(output_dir)` - Retrieves current stage from SST
- `get_stage_history(output_dir)` - Gets full stage transition history
- `resolve_stage(explicit_stage, scope, output_dir)` - Priority chain for stage resolution

### Stage-Aware Paths (`TRAINING/orchestration/utils/target_first_paths.py`)

Modified path functions to include stage:
- `get_target_reproducibility_dir(output_dir, target, stage=None)` - Now accepts optional stage
- `get_scoped_artifact_dir(output_dir, target, view, symbol, artifact_type, stage=None)`
- `ensure_scoped_artifact_dir(...)` - Same signature update

New path helpers:
- `iter_stage_dirs(output_dir, target)` - Iterate over stage directories
- `find_cohort_dirs(base_dir, target, view, universe_sig)` - SST-aware cohort finder
- `parse_reproducibility_path(path)` - Parse paths with/without stage component

### Identity Passthrough Fixes

Applied to multiple files with same pattern:

**Files Fixed:**
- `TRAINING/ranking/feature_selector.py` (multi_model_aggregated + per-model family)
- `TRAINING/ranking/cross_sectional_feature_ranker.py` (cross_sectional_panel)
- `TRAINING/ranking/multi_model_feature_selection.py` (per-family in process_single_symbol)

**Fix Pattern:**
1. **Error Visibility**: Changed `logger.debug` to `logger.warning` for identity creation failures
2. **Partial Fallback**: When finalization fails, extract and pass partial identity signatures:
   - `dataset_signature` (from FEATURE_SELECTION stage, not TARGET_RANKING)
   - `target_signature`
   - `routing_signature`
   - `train_seed`
   - `hparams_signature` (computed for aggregated snapshot)
   - `feature_signature` (computed from selected features)
3. **Effective Identity**: Pass `effective_identity = finalized_identity or partial_identity_dict` to snapshot hooks

### Integration Points

- `intelligent_trainer.py` - Calls `save_stage_transition()` at phase boundaries
- `reproducibility_tracker.py` - Uses stage-aware paths via `scope.stage`
- `io.py` - Passes `stage` to path functions
- `metrics_aggregator.py` - Uses `find_cohort_dirs()` for scanning
- `diff_telemetry.py` - Updated path parsing for stage component

## New Output Structure

```
targets/{target}/reproducibility/{view}/
├── stage=TARGET_RANKING/
│   ├── cohort=.../
│   └── universe=.../feature_importance_snapshots/replicate/
├── stage=FEATURE_SELECTION/
│   ├── cohort=.../
│   └── universe=.../feature_importance_snapshots/{target}/multi_model_aggregated/
└── stage=TRAINING/
    └── cohort=.../
```

## Determinism Impact

**None.** All changes are:
- Observational (logging, metadata)
- Structural (file paths)
- Fallback mechanisms (data capture)

Actual computation, seeds, and model outputs are unchanged.

## Files Modified

| File | Change |
|------|--------|
| `TRAINING/orchestration/utils/run_context.py` | Added SST stage factory functions |
| `TRAINING/orchestration/utils/target_first_paths.py` | Added stage parameter and scanning helpers |
| `TRAINING/orchestration/intelligent_trainer.py` | Added stage transition calls |
| `TRAINING/ranking/feature_selector.py` | Fixed identity error handling and fallback |
| `TRAINING/ranking/cross_sectional_feature_ranker.py` | Fixed identity error handling and fallback for cross_sectional_panel |
| `TRAINING/ranking/multi_model_feature_selection.py` | Fixed identity error handling and fallback for per-family snapshots |
| `TRAINING/orchestration/utils/reproducibility_tracker.py` | Stage-aware path usage |
| `TRAINING/stability/feature_importance/io.py` | Stage parameter passthrough |
| `TRAINING/orchestration/metrics_aggregator.py` | SST-aware cohort scanning |
| `TRAINING/orchestration/utils/diff_telemetry.py` | Stage-aware path parsing |

## Verification

Run a full pipeline and check:
1. `run_context.json` contains `current_stage` and `stage_history`
2. `fs_snapshot_index.json` has populated fingerprints (not null)
3. Reproducibility paths include `stage=FEATURE_SELECTION/` subdirectories
