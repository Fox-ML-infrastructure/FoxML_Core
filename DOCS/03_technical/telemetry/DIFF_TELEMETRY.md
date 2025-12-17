# Diff Telemetry System

First-class telemetry with strict SST (Stable, Sortable, Typed) rules for tracking changes across runs.

## Overview

The diff telemetry system provides:
- **Normalized snapshots** for diffing (SST-compliant)
- **Delta tracking** (prev vs baseline)
- **Comparison groups** and comparability checks
- **Blame assignment** for drift
- **Regression detection**

Key principle: Only diff things that are **canonically normalized** and **hash-addressed**.

## Architecture

### Core Components

1. **NormalizedSnapshot**: SST-compliant snapshot of a run
   - Inputs (config, data, features, targets)
   - Process (splits, training regime, environment)
   - Outputs (metrics, stability, artifacts)

2. **ComparisonGroup**: Defines what makes runs comparable
   - `experiment_id`: Same experiment
   - `dataset_signature`: Same universe + time rules
   - `task_signature`: Same target + horizon + objective
   - `routing_signature`: Same routing config

3. **DiffResult**: Result of diffing two snapshots
   - Changed keys (canonical paths)
   - Severity (CRITICAL, MAJOR, MINOR, NONE)
   - Patch operations (JSON-Patch style)
   - Metric deltas (absolute + percent)

4. **BaselineState**: Baseline for regression detection
   - Established after N_min comparable runs
   - Updated when better runs are found
   - Used for regression detection

### Integration Points

The system is automatically integrated via `ReproducibilityTracker._save_to_cohort()`:

- **Feature Selection**: After `save_multi_model_results()`
- **Target Ranking**: After `train_and_evaluate_models()`
- **Model Training**: After model training completes

## File Structure

```
RESULTS/
  REPRODUCIBILITY/
    TELEMETRY/
      snapshot_index.json          # Global index of all snapshots (all runs)
      baseline_index.json          # Global index of baselines per comparison group
  {run_id}/
    REPRODUCIBILITY/
      TARGET_RANKING/.../
        cohort={cohort_id}/
          snapshot.json              # Normalized snapshot (per-run)
          diff_prev.json             # Diff vs previous run
          diff_baseline.json         # Diff vs baseline
      FEATURE_SELECTION/.../
        cohort={cohort_id}/
          snapshot.json
          diff_prev.json
          diff_baseline.json
```

## What Gets Tracked

### Inputs (What was fed to the run)

**Hard Invariants (CRITICAL)**
- Dataset split definition (purge/embargo params)
- Target definition + horizon + labeling rules
- Feature set membership + lookback budgets
- Leakage detector results / exceptions
- Train/test time ranges

**Soft but Important (MAJOR)**
- Hyperparameters
- Model version / library versions
- Sampling / weighting
- Calibration settings

### Process (What happened during execution)

- Split integrity (purge/embargo enforcement)
- Training regime (CV scheme, folds, early stopping)
- Compute environment (library versions, GPU/CPU, threads)
- Warnings as structured events

### Outputs (What was produced)

- Performance metrics (mean, std, distribution)
- Stability metrics (variance across folds/time)
- Model artifacts fingerprint
- Interpretability / diagnostics

## Comparison Groups

Runs are comparable if they share:
- Same `stage` (TARGET_RANKING, FEATURE_SELECTION, TRAINING)
- Same `view` (CROSS_SECTIONAL, SYMBOL_SPECIFIC)
- Same `target` (if specified)
- Same `comparison_group` key (experiment + dataset + task + routing)

## Baseline Logic

1. **Warmup**: Until N_min comparable runs (default: 5), no baseline
2. **Establishment**: Baseline = best metric run in rolling window
3. **Updates**: Baseline updates when better runs are found
4. **Regression Detection**: Compare current run to baseline

## Severity Levels

- **CRITICAL**: Hard invariants changed (splits, targets, features, leakage)
- **MAJOR**: Important config changed (hyperparams, versions, training regime)
- **MINOR**: Metrics only changed
- **NONE**: No meaningful changes

## Usage

The system is automatically called via `ReproducibilityTracker`. No manual calls needed.

To access diffs:

```python
from pathlib import Path
import json

# Load diff from cohort directory
cohort_dir = Path("RESULTS/.../cohort=.../")
with open(cohort_dir / "diff_prev.json") as f:
    diff = json.load(f)

print(f"Changes: {diff['changed_keys']}")
print(f"Severity: {diff['severity']}")
print(f"Metric deltas: {diff['metric_deltas']}")
```

## Normalization Rules

To ensure SST compliance:

1. **Floats**: Rounded to 6 decimal places
2. **Lists**: Sorted (if comparable)
3. **Dicts**: Keys sorted
4. **Timestamps**: Excluded from diff (use fingerprints instead)
5. **Paths**: Excluded from diff (use fingerprints instead)

## Future Enhancements

- Regression chain tracking
- Automated alerting on critical changes
- Blame attribution heuristics
- Integration with monitoring systems

