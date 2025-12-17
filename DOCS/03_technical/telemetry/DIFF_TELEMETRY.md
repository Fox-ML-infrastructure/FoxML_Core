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

Runs are **automatically organized** by comparison group metadata after the first snapshot is created.

**Initial Location** (before first snapshot):
```
RESULTS/
  sample_5k-10k/                  # Sample size bin (temporary organization)
    {run_name}/
      ...
```

**Final Location** (after first snapshot):
```
RESULTS/
  {comparison_group_dir}/          # Organized by all outcome-influencing metadata
    {run_name}/
      REPRODUCIBILITY/
        METRICS/
          snapshot_index.json      # Run-specific snapshot index
        TARGET_RANKING/.../
          cohort={cohort_id}/
            snapshot.json          # Normalized snapshot (per-run)
            diff_prev.json         # Diff vs previous run
            diff_baseline.json     # Diff vs baseline
            baseline.json          # Baseline for this exact comparison_group (stored per-cohort)
        FEATURE_SELECTION/.../
          cohort={cohort_id}/
            snapshot.json
            diff_prev.json
            diff_baseline.json
            baseline.json          # Baseline for this exact comparison_group
```

**Comparison Group Directory Naming:**
- Format: `{exp}-{experiment_id}_data-{hash}_task-{hash}_n-{N_effective}_family-{model_family}_features-{hash}`
- Example: `n-5000_family-lightgbm_features-abc12345`
- Filesystem-safe: Special characters are sanitized, length is limited to 200 chars
- Human-readable: All outcome-influencing metadata is visible in the directory name

**Automatic Organization:**
- Runs start in sample size bins (or `_pending/` if N_effective is unknown)
- On first snapshot creation, the run is automatically moved to the comparison group directory
- This ensures runs with exactly the same metadata are stored together for easy auditing

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

Runs are comparable **ONLY** if they have **EXACTLY** the same outcome-influencing metadata.

**CRITICAL PRINCIPLE**: Only metadata that **directly influences outcomes** is included in comparison groups. This ensures **audit-oriented cohorting** - runs are only compared when they are truly apples-to-apples.

### Outcome-Influencing Metadata (Included in Comparison Group)

**Required Exact Matches:**
1. **Exact N_effective** (sample size) - 5k runs only compare against 5k runs
   - Different sample sizes produce different statistical properties
   - Even small differences (5k vs 5.1k) are considered different
   
2. **Same dataset** (universe, date range, min_cs, max_cs_samples, **data identity**)
   - Hash of: `n_symbols`, `date_range_start`, `date_range_end`, `min_cs`, `max_cs_samples`
   - **Data identity**: Hash of actual row IDs / file manifest / parquet metadata (if available)
   - Different data = different outcomes
   - **Why data identity?** "Same date range" can still mean different data if files changed
   
3. **Same task** (target, horizon, objective, **labeling implementation**)
   - Hash of: target name, horizon, objective
   - **Labeling implementation signature**: Hash of labeling code/config (not just target name)
   - Different tasks = different outcomes
   - **Why labeling signature?** Same target name can have different labeling implementations
   
4. **Same routing/view** configuration
   - Hash of: view type (CROSS_SECTIONAL, SYMBOL_SPECIFIC)
   - Different routing = different outcomes
   
5. **Same model_family** (LightGBM, XGBoost, etc.)
   - Different families = fundamentally different algorithms = different outcomes
   - LightGBM vs XGBoost are **never** comparable
   
6. **Same feature set** (feature count, names, **feature pipeline signature**)
   - Hash of: feature count, feature names (if available)
   - **Feature pipeline signature**: Hash of transforms, lookbacks, normalization, winsorization, missing-value policy
   - Different features = different inputs = different outcomes
   - **Why pipeline signature?** Same feature names can have different transforms/normalization
   
7. **Same split protocol** (CV scheme, folds, purge/embargo, leakage guards)
   - Hash of: `cv_method`, `cv_folds`, `purge_minutes`, `embargo_minutes`, `leakage_filter_version`, `horizon_minutes`
   - Different split protocols = different train/test boundaries = different outcomes
   - **Why?** CV scheme and time-safety settings directly affect which samples are in train vs test

### Not Part of Comparability Key (Tracked Separately)

These are **outcome-influencing** but **intentionally excluded** from the comparability key to allow controlled diffs across sweeps:

- **Hyperparameters** (learning_rate, max_depth, etc.)
  - **Why excluded?** We intentionally allow diffs across hyperparameter sweeps
  - Still tracked in snapshots and shown in diffs
  - Allows comparing "same setup, different hyperparams"
  
- **Random seeds**
  - **Why excluded?** Allows comparing "same setup, different seeds" for variance analysis
  - Still tracked for reproducibility diagnostics
  - **Note:** Seeds CAN change outcomes (stochasticity), but we want to diff across them
  
- **Library/compute versions**
  - **Why excluded?** Allows comparing "same setup, different library versions" for regression detection
  - Still tracked for reproducibility diagnostics
  - **Note:** Versions CAN change outcomes (nondeterminism, numeric drift), but we want to diff across them
  
- **Training timestamps**
  - **Why excluded?** Not outcome-influencing, just metadata
  - Tracked for audit trail

**Design Philosophy:** The comparability key defines a **cohort** (experimentally defensible grouping), not a claim that excluded factors don't matter. We intentionally allow diffs across hyperparameters, seeds, and versions to enable controlled experiments and regression detection.

### Comparability Enforcement

The system **strictly enforces** that only runs with identical comparison groups can be compared:

```python
def _check_comparability(current, prev):
    # Must be same stage
    if current.stage != prev.stage:
        return False, "Different stages"
    
    # Must be same view
    if current.view != prev.view:
        return False, "Different views"
    
    # CRITICAL: Must have identical comparison groups
    if current.comparison_group.to_key() != prev.comparison_group.to_key():
        return False, "Different comparison groups"
    
    return True, None
```

**Examples:**
- ❌ 5k run vs 10k run → **INCOMPARABLE** (different N_effective)
- ❌ LightGBM vs XGBoost → **INCOMPARABLE** (different model_family)
- ❌ 100 features vs 50 features → **INCOMPARABLE** (different feature set)
- ❌ Different date ranges → **INCOMPARABLE** (different dataset)
- ❌ Different targets → **INCOMPARABLE** (different task)
- ✅ Same N_effective + same family + same features + same dataset + same task → **COMPARABLE**

### Benefits for True Auditability

1. **No false comparisons**: Runs are only compared when truly identical on outcome-influencing factors
2. **Clear audit trail**: Directory names show exactly what makes runs comparable
3. **Accurate baselines**: Baselines are established only from identical runs
4. **No data skew**: Sample size differences don't pollute comparisons
5. **Algorithm isolation**: Different model families are never mixed
6. **Feature consistency**: Only runs with same features and pipeline are compared
7. **Split protocol consistency**: Only runs with same CV scheme and time-safety are compared
8. **Controlled diffs**: Hyperparameters, seeds, and versions are intentionally excluded to enable sweeps

This ensures that every comparison is **experimentally defensible** (cohorting) and **audit-oriented**. The system provides **strict cohort gating** with **intentional allowance for controlled diffs** across hyperparameters, seeds, and versions.

### Excluded Factor Reporting

When excluded factors (hyperparameters, seeds, versions) differ between comparable runs, the diff **surfaces them loudly with actual values**:

```json
{
  "excluded_factors_changed": {
    "hyperparameters": {
      "learning_rate": {"prev": 0.01, "curr": 0.05},
      "max_depth": {"prev": 5, "curr": 7},
      "n_estimators": {"prev": 100, "curr": 200}
    },
    "train_seed": {"prev": 42, "curr": 1337},
    "versions": {
      "python_version": {"prev": "3.9.0", "curr": "3.10.0"},
      "cuda_version": {"prev": "12.2", "curr": "12.3"},
      "library_versions": {
        "prev": {"xgboost": "2.0.0", "numpy": "1.24.0"},
        "curr": {"xgboost": "2.0.1", "numpy": "1.24.0"}
      }
    }
  },
  "summary": {
    "excluded_factors_changed": true
  }
}
```

**Key Points:**
- Includes **actual values** (e.g., `learning_rate: 0.01 → 0.05`, `cuda: 12.2 → 12.3`)
- Not just a flag - reviewers can see exactly what changed
- All hyperparameters are included (not just a predefined list)
- All version fields are included (python, cuda, library versions)

This ensures reviewers are aware of confounders with full transparency, even when comparisons are allowed.

### Fingerprint Schema Versioning

Snapshots include `fingerprint_schema_version` to ensure compatibility:

- **Policy**: Different schema versions are **never comparable** (strict equality required)
- If fingerprint computation changes, old runs are explicitly marked as incomparable
- Prevents silent failures when fingerprint logic evolves
- Provides clear error messages: "Different fingerprint schema versions: 1.0 vs 2.0"
- **Rationale**: Fingerprint computation changes mean the comparability criteria changed, so old and new runs cannot be meaningfully compared

### Canonicalization

All fingerprints use **canonical serialization** with:
- **Sorted dictionary keys** (dicts are unordered by definition)
- **Preserved list order** (order may be semantic - only sort when explicitly unordered, e.g., feature sets)
- **Float representation via `repr()`** (preserves exact values, avoids 1e-7 vs 0.0 collapse)
- **Deterministic NaN/inf/-0.0 handling**:
  - `NaN` → "nan"
  - `inf` → "inf", `-inf` → "-inf"
  - `0.0` vs `-0.0` are distinguished
- **Consistent null/None handling** → "None"

**Critical Rules:**
- **Feature lists**: Sorted before hashing (features are unordered set)
- **Pipeline steps/layer lists**: Order preserved (order is semantic)
- **Floats**: Use `repr()` not rounding (preserves 1e-7, avoids precision loss)

This ensures stable hashing across dict ordering, feature ordering, and floating-point representation differences.

### Split Seed vs Train Seed

**CRITICAL**: The system separates:
- **split_seed** (in comparability key) - affects fold assignment
- **fold_assignment_hash** (in comparability key) - hash of actual per-row fold IDs
- **train_seed** (excluded) - affects model training only

**Why both split_seed and fold_assignment_hash?**
- `split_seed` captures seed-dependent fold assignment
- `fold_assignment_hash` captures actual fold assignment (from per-row fold IDs)
- If fold logic changes but seed stays same, `fold_assignment_hash` will differ → breaks comparability
- This prevents "fold drift" where changing seeds or fold logic invalidates comparisons

**fold_assignment_hash Source:**
- Stored in `fingerprint_sources['fold_assignment_hash']` in snapshots
- Default description: "hash over row_id→fold_id mapping"
- Can be customized via `additional_data['fold_assignment_hash_source']`
- Ensures auditability: reviewers can verify what the hash represents

**Example:**
- Same `split_seed=42` but different CV splitter implementation → different `fold_assignment_hash` → incomparable
- Different `split_seed` → different `fold_assignment_hash` → incomparable
- Same `split_seed` and same fold logic → same `fold_assignment_hash` → comparable

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

The system is **automatically integrated** for all stages:
- **TARGET_RANKING**: Via `ReproducibilityTracker.log_comparison()` → `DiffTelemetry.finalize_run()`
- **FEATURE_SELECTION**: Via `ReproducibilityTracker.log_comparison()` → `DiffTelemetry.finalize_run()`
- **TRAINING**: Via `ReproducibilityTracker.log_comparison()` → `DiffTelemetry.finalize_run()`

No manual calls needed - diff telemetry is automatically applied to all metadata outputs and metrics.

## Data Persistence

Diff telemetry data is persisted in two locations with appropriate data split:

### Metadata.json (Full Audit Trail)

Stored in `{cohort_dir}/metadata.json` under `diff_telemetry` key:
- **Full detail**: All fingerprints, sources, comparison groups, excluded factor changes
- **Purpose**: Complete audit trail for reviewers
- **Structure**: See "Metadata Integration" section below

### Metrics.json (Lightweight Queryable Fields)

Stored in `{cohort_dir}/metrics.json` under `diff_telemetry` key:
- **Lightweight**: Flags, counts, summaries only
- **Purpose**: Fast queries and aggregation without cardinality blowups
- **Structure**: See "Metrics Integration" section below

### Count Rule

`excluded_factors_changed_count` counts the **number of keys whose value differs** (one key = one change):
- Each hyperparameter key = 1 change
- `train_seed` = 1 change (if present)
- Each version key = 1 change

This matches the summary formatter counting logic, ensuring consistency between count and summary display.

### Accessing Diffs

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
print(f"Excluded factors: {diff.get('excluded_factors_changed', {})}")
print(f"Summary: {diff['summary'].get('excluded_factors_summary')}")
```

### Integration Points

**Target Ranking:**
- Called after `train_and_evaluate_models()` completes
- Tracks: target, dataset, feature set, split protocol
- Stores: `REPRODUCIBILITY/TARGET_RANKING/{view}/{target}/cohort={cohort_id}/`

**Feature Selection:**
- Called after `save_multi_model_results()` completes
- Tracks: target, dataset, feature set, model families
- Stores: `REPRODUCIBILITY/FEATURE_SELECTION/{view}/{target}/cohort={cohort_id}/`

**Training:**
- Called after model training completes
- Tracks: target, dataset, feature set, model family, split protocol
- Stores: `REPRODUCIBILITY/TRAINING/{view}/{target}/model_family={family}/cohort={cohort_id}/`
- Persists: `metadata.json` (full) + `metrics.json` (lightweight)

## Metadata Integration

Full diff telemetry is stored in `metadata.json` under `diff_telemetry` key:

```json
{
  "diff_telemetry": {
    "fingerprint_schema_version": "1.0",
    "comparison_group_key": "exp=test|data=abc12345|task=def67890|n=5000|family=lightgbm|features=ghi11111",
    "comparison_group": { ... },
    "fingerprints": {
      "config_fingerprint": "...",
      "data_fingerprint": "...",
      "feature_fingerprint": "...",
      "target_fingerprint": "..."
    },
    "fingerprint_sources": {
      "fold_assignment_hash": "hash over row_id→fold_id mapping"
    },
    "comparability": {
      "comparable": true,
      "comparability_reason": null,
      "prev_run_id": "2025-12-15T10:30:00Z"
    },
    "excluded_factors": {
      "changed": true,
      "summary": "learning_rate: 0.01→0.05, max_depth: 5→7, train_seed: 42→1337 (+2 more)",
      "changes": {
        "hyperparameters": {
          "learning_rate": {"prev": 0.01, "curr": 0.05},
          "max_depth": {"prev": 5, "curr": 7},
          ...
        },
        "train_seed": {"prev": 42, "curr": 1337},
        "versions": { ... }
      }
    }
  }
}
```

## Metrics Integration

Lightweight diff telemetry is stored in `metrics.json` under `diff_telemetry` key:

```json
{
  "diff_telemetry": {
    "comparable": 1,
    "excluded_factors_changed": 1,
    "excluded_factors_changed_count": 5,
    "excluded_factors_summary": "learning_rate: 0.01→0.05, max_depth: 5→7, train_seed: 42→1337 (+2 more)"
  }
}
```

**Note**: Full payloads (excluded_factors.changes, comparison_group, fingerprints) are **not** included in metrics to keep cardinality low. Use metadata.json for full audit trail.

## Backwards Compatibility

- `diff_telemetry` is **optional** in metadata (older runs without it are handled gracefully)
- Schema mismatches produce valid `diff_telemetry` structure with `comparability.comparable = false` and clear reason
- First run / no previous run returns stable shape with empty excluded factors

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

