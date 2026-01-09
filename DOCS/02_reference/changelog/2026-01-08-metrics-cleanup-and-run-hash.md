# Metrics Cleanup, Delta Computation Updates, and Full Run Hash

**Date**: 2026-01-08  
**Category**: Metrics, Reproducibility, Determinism  
**Impact**: High (cleaner metrics output, better change detection, run-level reproducibility)

## Summary

Three major improvements:
1. **Metrics JSON Cleanup** - Restructured metrics output to be smaller, non-redundant, and semantically unambiguous
2. **Delta Computation Updates** - Updated to handle new grouped metrics structure with backward compatibility
3. **Full Run Hash with Change Detection** - Deterministic run identifier that aggregates all snapshots and summarizes changes

## Problem

1. **Redundant Metrics**: Same values stored under multiple keys (`auc`, `mean_r2`, `primary_metric_mean` all had same value)
2. **Mixed Semantics**: `auc` appeared for regression (should be task-gated)
3. **Bad Types**: Counts stored as floats (`11.0` instead of `11`)
4. **Flat Structure**: No logical grouping, hard to scan
5. **Delta Computation Broken**: Expected flat keys but new structure has nested paths
6. **No Run-Level Hash**: No single identifier representing entire run across all stages

## Changes

### Metrics JSON Cleanup and Restructuring

**Files Modified:**
- `TRAINING/ranking/predictability/metrics_schema.py` - Added clean metrics builders
- `TRAINING/ranking/predictability/model_evaluation.py` - Uses `build_clean_metrics_dict()`
- `TRAINING/ranking/feature_selector.py` - Uses `build_clean_feature_selection_metrics()`
- `TRAINING/ranking/cross_sectional_feature_ranker.py` - Uses `build_clean_feature_selection_metrics()`
- `TRAINING/training_strategies/reproducibility/schema.py` - Uses `build_clean_training_metrics()`
- `TRAINING/training_strategies/execution/training.py` - Uses `build_clean_training_metrics()`

**New Structure:**
```json
{
  "schema": {"metrics": "1.1", "scoring": "1.1"},
  "scope": {"view": "CROSS_SECTIONAL", "task_family": "regression"},
  "primary_metric": {
    "family": "regression",
    "name": "spearman_ic__cs__mean",
    "direction": "higher_is_better",
    "baseline": 0.0,
    "mean": 0.0583,
    "std": 0.1036,
    "se": 0.0312,
    "skill_mean": 0.0583,
    "skill_se": 0.0312
  },
  "coverage": {"n_cs_valid": 11, "n_cs_total": 11, "n_effective": 19980},
  "features": {"pre": 215, "post_prune": 189, "safe": 215},
  "y_stats": {"mean": 1.63e-05, "std": 0.0025, "min": -0.0768, "max": 0.0283, "finite_pct": 1.0},
  "models": {"n": 11, "scores": {"xgboost": 0.134, "lightgbm": 0.132, ...}},
  "score": {
    "composite": 0.357,
    "components": {
      "signal": 0.85,
      "mean_importance": 0.262,
      "consistency_penalty": -0.777
    }
  }
}
```

**Key Improvements:**
- **No Duplicates**: Single source of truth for each metric value
- **Task-Gated**: Regression doesn't include `auc`, classification doesn't include `r2`
- **Counts as Ints**: All count fields are integers (`n_models: 11` not `11.0`)
- **Grouped**: Logical sections for readability
- **Backward Compatible**: MetricsWriter handles nested structures, old snapshots still work

### Delta Computation Updates

**Files Modified:**
- `TRAINING/orchestration/utils/diff_telemetry.py`

**Changes:**
1. **New Helper Function**: `_flatten_metrics_dict()`
   - Recursively flattens nested metrics with dot-notation keys
   - Handles both old flat structure and new grouped structure
   - Example: `{"primary_metric": {"mean": 0.5}}` → `{"primary_metric.mean": 0.5}`

2. **Updated `_compute_metric_deltas()`**:
   - Flattens both `prev_metrics` and `current_metrics` before comparison
   - Uses backward compatibility mapping for old flat keys:
     - `auc` → `primary_metric.mean`
     - `std_score` → `primary_metric.std`
     - `n_models` → `models.n`
     - `composite_score` → `score.composite`
   - Updated z-score computation to use new paths:
     - `primary_metric.std` or `primary_metric.skill_se` instead of `std_score`
     - `models.n` instead of `n_models`
   - Updated score_metrics set to include both old and new paths

**Backward Compatibility:**
- Old flat metrics still work (mapped to new paths)
- New grouped metrics work directly
- Mixed structures handled gracefully

### Full Run Hash with Change Detection

**Files Modified:**
- `TRAINING/orchestration/utils/diff_telemetry.py` - Added run hash functions
- `TRAINING/orchestration/utils/diff_telemetry/__init__.py` - Exported new functions
- `TRAINING/orchestration/intelligent_trainer.py` - Integrated run hash computation
- `TRAINING/orchestration/utils/manifest.py` - Added run hash to manifest

**New Functions:**
1. **`compute_full_run_hash(output_dir, run_id=None)`**:
   - Loads all global snapshot indices (TARGET_RANKING, FEATURE_SELECTION, TRAINING)
   - Extracts deterministic fields from all snapshots:
     - Fingerprints: `config_fingerprint`, `data_fingerprint`, `feature_fingerprint`, `target_fingerprint`
     - Signatures: `scoring_signature`, `selection_signature`, `hyperparameters_signature`, `library_versions_signature`
     - Output digests: `metrics_sha256`, `artifacts_manifest_sha256`, `predictions_sha256`
     - Schema versions: `metrics_schema_version`, `scoring_schema_version`
     - Comparison group key fields
   - Sorts snapshots deterministically (by stage, target, view, symbol, model_family)
   - Computes SHA256 hash of canonical JSON representation
   - Returns 16-character hex digest

2. **`compute_run_hash_with_changes(output_dir, run_id=None, prev_run_id=None, diff_telemetry=None)`**:
   - Computes base run hash (same as above)
   - If previous run exists, computes diffs for all snapshots
   - Aggregates change information:
     - `changed_snapshots`: List of snapshots that changed with severity
     - `changed_keys_summary`: Aggregated list of all changed keys
     - `severity_summary`: Highest severity level (none, noise, minor, major, critical)
     - `metric_deltas_summary`: Counts by severity (noise, minor, major, none)
     - `excluded_factors_summary`: Summary of excluded factors that changed
   - Returns dict with `run_hash`, `run_id`, `prev_run_id`, `changes`, `snapshot_count`

3. **`save_run_hash(output_dir, run_id=None, prev_run_id=None, diff_telemetry=None)`**:
   - Computes run hash with changes
   - Saves to `globals/run_hash.json`
   - Includes timestamp (`computed_at`)
   - Returns path to saved file

**Integration:**
- **`intelligent_trainer.py`**: Automatically computes and saves run hash after pipeline completes
  - Finds previous run ID from existing `run_hash.json`
  - Creates `DiffTelemetry` instance for change detection
  - Logs run hash and change summary
- **`manifest.py`**: Includes run hash in manifest.json
  - `run_hash`: 16-char hex digest
  - `run_id`: Current run identifier
  - `run_changes`: Change summary (severity, changed_snapshots_count)

**Output Format (`globals/run_hash.json`):**
```json
{
  "run_hash": "abc123def4567890",
  "run_id": "run-2026-01-08-17-06-21",
  "prev_run_id": "run-2026-01-08-16-30-45",
  "snapshot_count": 42,
  "computed_at": "2026-01-08T17:30:00.123456",
  "changes": {
    "changed_snapshots": [
      {
        "stage": "TARGET_RANKING",
        "target": "fwd_ret_10m",
        "view": "CROSS_SECTIONAL",
        "symbol": null,
        "model_family": null,
        "severity": "minor",
        "changed_keys_count": 3,
        "metric_deltas_count": 2
      }
    ],
    "changed_keys_summary": ["primary_metric.mean", "score.composite", "coverage.n_cs_valid"],
    "severity_summary": "minor",
    "metric_deltas_summary": {"noise": 1, "minor": 1, "major": 0, "none": 0},
    "excluded_factors_summary": {"library_versions": 1}
  }
}
```

## Benefits

1. **Cleaner Metrics**: Smaller, grouped, task-aware metrics output
2. **Better Change Detection**: Delta computation works with both old and new structures
3. **Run-Level Reproducibility**: Single hash represents entire run state
4. **Easy Change Identification**: Change summary shows what changed and severity
5. **Backward Compatible**: Old snapshots and metrics still work

## Migration Notes

- Old metrics JSON files will still work (MetricsWriter handles nested structures)
- Delta computation automatically handles both old flat and new grouped structures
- Run hash is computed automatically after pipeline completion (no manual steps needed)
- Old runs without run hash will have `None` for previous run comparison (first run scenario)

## Testing

- Verify metrics JSON has grouped structure
- Verify counts are integers
- Verify task-gating (regression has no `auc`, classification has no `r2`)
- Verify delta computation works with both old and new structures
- Verify run hash is deterministic (same inputs → same hash)
- Verify change detection aggregates correctly across all snapshots
