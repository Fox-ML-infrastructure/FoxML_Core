# Reproducibility Directory Structure

## Overview

The reproducibility system uses a **target-first directory layout** that organizes all artifacts by target, making it easy to find all information related to a specific target in one place. This structure replaced the legacy phase-first organization in December 2025.

## Target-First Structure (Current)

All artifacts are organized under `targets/<target>/`, with global summaries in `globals/`:

```
{run_dir}/
├── manifest.json                    # Run-level manifest with experiment config, target index
├── globals/                         # Global summaries and run-level artifacts
│   ├── routing_decisions.json      # Global routing decisions (target -> route)
│   ├── target_prioritization.yaml  # Target ranking results
│   ├── target_predictability_rankings.csv
│   ├── target_confidence_summary.json
│   ├── target_confidence_summary.csv
│   └── stats.json                   # Run-level statistics
└── targets/                         # Per-target artifacts (target-first organization)
    └── {target}/                    # All artifacts for this target
        ├── metadata.json           # Per-target metadata aggregating all information
        ├── decision/                 # Routing and prioritization decisions
        │   ├── routing_decision.json
        │   └── feature_prioritization.yaml
        ├── reproducibility/         # Reproducibility tracking artifacts
        │   ├── CROSS_SECTIONAL/
        │   │   └── cohort={cohort_id}/
        │   │       ├── metadata.json      # Full cohort metadata
        │   │       ├── metrics.json       # Performance metrics
        │   │       ├── metrics.parquet    # Metrics in parquet format
        │   │       ├── snapshot.json      # Normalized snapshot for diff telemetry
        │   │       ├── diff_prev.json     # Diff vs previous run
        │   │       ├── diff_baseline.json # Diff vs baseline
        │   │       ├── metric_deltas.json # Metric deltas
        │   │       └── audit_report.json # Audit report
        │   ├── SYMBOL_SPECIFIC/
        │   │   └── symbol={symbol}/
        │   │       └── cohort={cohort_id}/
        │   │           ├── metadata.json
        │   │           ├── metrics.json
        │   │           └── ...
        │   ├── feature_importances/  # Feature importance files
        │   │   ├── feature_importance_multi_model.csv
        │   │   ├── feature_importance_with_boruta_debug.csv
        │   │   ├── {model}_importances.csv
        │   │   └── model_agreement_matrix.csv
        │   ├── feature_importance_snapshots/  # Stability tracking snapshots
        │   │   └── {target}/
        │   │       └── {method}/
        │   │           └── {run_id}.json
        │   ├── feature_exclusions/  # Feature exclusion tracking
        │   │   └── fwd_ret_{horizon}_exclusions.yaml
        │   ├── featureset_artifacts/ # Featureset artifacts
        │   │   ├── featureset_post_prune.json
        │   │   └── featureset_post_gatekeeper.json
        │   ├── feature_selection_rankings.csv
        │   └── selected_features.txt
        ├── metrics/                 # Performance metrics (organized by view)
        │   ├── view=CROSS_SECTIONAL/
        │   │   └── metrics.json
        │   └── view=SYMBOL_SPECIFIC/
        │       └── symbol={symbol}/
        │           └── metrics.json
        ├── models/                  # Trained models
        │   └── {family}/
        │       └── {model_files}
        └── trends/                  # Trend analysis reports
            └── {trend_reports}
```

## Key Benefits

1. **Target-Centric Organization**: All information for a target is in one place (`targets/<target>/`)
2. **Easy Navigation**: Find all artifacts related to a target without traversing multiple directories
3. **Self-Contained**: Each target directory contains everything needed for analysis
4. **Better Diffing**: All target artifacts together makes cross-run comparisons easier
5. **Cleaner Structure**: No duplicate organization by phase/mode/target - just target-first

## Global Artifacts

Global summaries live in `globals/`:
- **Routing decisions**: `globals/routing_decisions.json` - Global routing decisions for all targets
- **Target rankings**: `globals/target_prioritization.yaml` - Target ranking results
- **Confidence summaries**: `globals/target_confidence_summary.json` - Run-level confidence summary
- **Statistics**: `globals/stats.json` - Run-level statistics

## Per-Target Artifacts

Each target has a self-contained directory structure:

### Decision Files (`targets/<target>/decision/`)
- `routing_decision.json`: Routing decision for this target (CROSS_SECTIONAL, SYMBOL_SPECIFIC, etc.)
- `feature_prioritization.yaml`: Feature prioritization results

### Reproducibility (`targets/<target>/reproducibility/`)
- **Cohort directories**: `{view}/cohort={cohort_id}/` - Contains metadata, metrics, diffs, snapshots
- **Feature importances**: Aggregated feature importance files from multi-model feature selection
- **Feature importance snapshots**: Stability tracking snapshots for cross-run analysis
- **Feature exclusions**: YAML files tracking excluded features
- **Featureset artifacts**: JSON files tracking featureset state at various stages

### Metrics (`targets/<target>/metrics/`)
- Organized by view: `view=CROSS_SECTIONAL/` and `view=SYMBOL_SPECIFIC/`
- For SYMBOL_SPECIFIC, includes `symbol={symbol}/` subdirectories
- Contains `metrics.json` and `metrics.parquet` files

### Models (`targets/<target>/models/`)
- Organized by model family: `{family}/`
- Contains trained model files and metadata

### Trends (`targets/<target>/trends/`)
- Trend analysis reports for within-run analysis
- Across-runs trends are stored in `trend_reports/by_target/<target>/` (outside run directories)

## View Organization

The structure distinguishes between views:
- **CROSS_SECTIONAL**: Cross-sectional analysis (pooled across symbols)
  - Path: `targets/<target>/reproducibility/CROSS_SECTIONAL/cohort={cohort_id}/`
- **SYMBOL_SPECIFIC**: Symbol-specific analysis
  - Path: `targets/<target>/reproducibility/SYMBOL_SPECIFIC/symbol={symbol}/cohort={cohort_id}/`
  - Note: `symbol={symbol}` prevents overwriting when multiple symbols are processed

## Legacy Structure (Deprecated)

The legacy `REPRODUCIBILITY/` structure organized by phase first:

```
REPRODUCIBILITY/
  TARGET_RANKING/
    {target}/
      cohort={cohort_id}/
  FEATURE_SELECTION/
    CROSS_SECTIONAL/
      {target}/
        cohort={cohort_id}/
```

**Status**: Legacy structure is no longer created. Reading logic checks target-first structure first, then falls back to legacy for backward compatibility with old runs.

## Migration

- **New runs**: Automatically use target-first structure
- **Old runs**: Legacy structure is preserved but not written to
- **Reading logic**: All readers check target-first structure first, then fall back to legacy
- **No data migration**: Old runs remain in legacy structure (not migrated)

## File Contents

### manifest.json (Run Root)

Run-level manifest with experiment config and target index:

```json
{
  "run_id": "intelligent_output_20251219_225716",
  "git_sha": "abc123",
  "config_digest": "def456",
  "created_at": "2025-12-19T22:57:16",
  "targets": ["fwd_ret_10m", "fwd_ret_1d"],
  "experiment": {
    "name": "experiment_name",
    "data_dir": "/path/to/data",
    "symbols": ["AAPL", "MSFT"]
  },
  "run_metadata": {
    "data_dir": "/path/to/data",
    "symbols": ["AAPL", "MSFT"],
    "n_effective": 5000
  }
}
```

### targets/<target>/metadata.json

Per-target metadata aggregating all information:

```json
{
  "target": "fwd_ret_10m",
  "cohorts": {
    "CROSS_SECTIONAL": ["cohort=cr_2025Q3_..."],
    "SYMBOL_SPECIFIC": {
      "AAPL": ["cohort=cr_2025Q3_..."]
    }
  },
  "routing_decision": {
    "route": "CROSS_SECTIONAL",
    "confidence": "HIGH"
  },
  "metrics_summary": {
    "CROSS_SECTIONAL": {
      "mean_score": 0.75,
      "n_effective": 5000
    }
  }
}
```

### targets/<target>/reproducibility/<view>/cohort={cohort_id}/metadata.json

Full cohort metadata for reproducibility:

```json
{
  "schema_version": 2,
  "cohort_id": "cr_2025Q3_min_cs3_max1000_v1_e66a13aa",
  "run_id": "20251219T225716",
  "stage": "FEATURE_SELECTION",
  "route_type": "CROSS_SECTIONAL",
  "view": "CROSS_SECTIONAL",
  "target_name": "fwd_ret_10m",
  "n_effective": 5000,
  "n_symbols": 10,
  "symbols": ["AAPL", "MSFT", ...],
  "date_range_start": "2025-07-01T00:00:00Z",
  "date_range_end": "2025-09-30T23:59:59Z",
  "universe_id": "universeA",
  "min_cs": 3,
  "max_cs_samples": 1000,
  "seed": 42,
  "git_commit": "abc123",
  "created_at": "2025-12-19T22:57:16.123456"
}
```

### targets/<target>/reproducibility/<view>/cohort={cohort_id}/snapshot.json

Normalized snapshot for diff telemetry:

```json
{
  "run_id": "20251219T225716",
  "stage": "FEATURE_SELECTION",
  "target": "fwd_ret_10m",
  "view": "CROSS_SECTIONAL",
  "comparison_group": {
    "n_effective": 5000,
    "dataset_signature": "abc123",
    "task_signature": "def456",
    "routing_signature": "ghi789"
  },
  "metrics": {
    "mean_score": 0.75,
    "std_score": 0.02
  }
}
```

## Usage

### Automatic (Recommended)

The system automatically uses the target-first structure. All writes go to `targets/<target>/` and `globals/`.

### Reading Artifacts

All reading logic checks target-first structure first, then falls back to legacy:

```python
# Routing decisions
from TRAINING.ranking.target_routing import load_routing_decisions
decisions = load_routing_decisions(output_dir=output_dir)
# Checks: globals/routing_decisions.json, then legacy paths

# Metrics
from TRAINING.common.utils.metrics import MetricsWriter
metrics = MetricsWriter.load_metrics(target_dir)
# Checks: targets/<target>/metrics/, then legacy paths

# Reproducibility metadata
from TRAINING.orchestration.utils.reproducibility_tracker import ReproducibilityTracker
tracker = ReproducibilityTracker(output_dir)
# Writes to: targets/<target>/reproducibility/<view>/cohort=<id>/
```

## Benefits

1. **Target-Centric**: All information for a target in one place
2. **Self-Contained**: Each target directory has everything needed
3. **Better Organization**: No duplicate structure (phase/mode/target → just target)
4. **Easier Diffing**: All target artifacts together for cross-run comparisons
5. **Cleaner Navigation**: Find all target information without traversing multiple directories
6. **Backward Compatible**: Reading logic supports both new and legacy structures
