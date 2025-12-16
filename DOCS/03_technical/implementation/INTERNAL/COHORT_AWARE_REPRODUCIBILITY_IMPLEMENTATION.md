# Cohort-Aware Reproducibility Implementation

## Summary

The reproducibility tracking system has been enhanced with **cohort-aware** functionality that:

1. **Organizes runs by cohort** (sample size, symbols, date range, config)
2. **Only compares runs within the same cohort**
3. **Uses sample-adjusted statistical tests** for drift detection
4. **Provides clear INCOMPARABLE labels** when cohorts differ

## What Changed

### Configuration (`CONFIG/training_config/safety_config.yaml`)

Added cohort-aware settings:

```yaml
safety:
  reproducibility:
    cohort_aware: true  # Enable cohort-aware tracking
    n_ratio_threshold: 0.90  # Min ratio for comparability (90% overlap)
    cohort_config_keys:      # Keys to include in cohort hash
      - min_cs
      - max_cs_samples
      - leakage_filter_version
      - universe_id
```

### Code Changes (`TRAINING/utils/reproducibility_tracker.py`)

**New methods**:
- `_extract_cohort_metadata()` - Extracts cohort info from metrics/additional_data
- `_compute_cohort_id()` - Computes deterministic hash for cohort
- `_get_cohort_dir()` - Gets directory path for a cohort
- `_save_to_cohort()` - Saves run to cohort-specific directory
- `_load_cohort_runs()` - Loads all runs for a cohort
- `_find_matching_cohort()` - Finds matching cohort by metadata
- `_compare_within_cohort()` - Sample-adjusted comparison within cohort

**Modified methods**:
- `log_comparison()` - Now checks for cohort metadata and uses cohort-aware path if available

## Storage Structure

Runs are now organized by cohort:

```
reproducibility/
  comparisons/
    {stage}/
      {item_name}/
        cohorts/
          {cohort_id}/
            runs/
              {run_id}.json
            latest.json
            summary.json
        index.json
```

**Example**:
```
reproducibility/
  comparisons/
    target_ranking/
      y_will_peak_60m_0.8/
        cohorts/
          abc123def456/  # Cohort: N=52k, 10 symbols, 2023-01→2023-06
            runs/
              2025-12-11T14-30-22.json
              2025-12-12T09-15-45.json
            latest.json → runs/2025-12-12T09-15-45.json
            summary.json
          xyz789uvw012/  # Cohort: N=11k, 5 symbols, 2024-01→2024-03
            runs/
              2025-12-12T12-00-00.json
            latest.json → runs/2025-12-12T12-00-00.json
            summary.json
        index.json
```

## How It Works

### 1. Cohort Extraction

When `log_comparison()` is called, it extracts cohort metadata from:

- `metrics`: `N_effective_cs`, `n_samples`, `sample_size`
- `additional_data`: `n_symbols`, `date_range`, `cs_config`

**Cohort definition**:
```python
{
    "N_effective_cs": int,        # Number of rows in final training set
    "n_symbols": int,              # Number of unique symbols
    "date_range": {                # Time coverage
        "start_ts": str,
        "end_ts": str
    },
    "cs_config": {                 # Config hash components
        "min_cs": int,
        "max_cs_samples": int,
        "leakage_filter_version": str,
        "universe_id": str
    }
}
```

### 2. Cohort Matching

When comparing runs:

1. **Extract cohort metadata** from current run
2. **Find matching cohort** by:
   - Exact match on cohort_id (hash of all metadata)
   - Or close match (same config, similar N within `n_ratio_threshold`)
3. **If no match** → save as new cohort baseline
4. **If match found** → load previous runs from that cohort

### 3. Sample-Adjusted Comparison

Within the same cohort, uses sample-adjusted z-scores:

```python
# Variance estimation
var_prev = auc_prev * (1 - auc_prev) / N_prev
var_curr = auc_curr * (1 - auc_curr) / N_curr

# Z-score
delta = auc_curr - auc_prev
sigma = sqrt(var_prev + var_curr)
z = delta / sigma

# Classification
|z| < 1  → STABLE (sample-adjusted)
1 ≤ |z| < 2 → DRIFTING (sample-adjusted)
|z| ≥ 2 → DIVERGED (sample-adjusted)
```

## Usage

### Automatic (Recommended)

The system automatically uses cohort-aware tracking when cohort metadata is available. Just pass it in `additional_data`:

```python
tracker.log_comparison(
    stage="target_ranking",
    item_name="y_will_peak_60m_0.8",
    metrics={
        "mean_score": 0.751,
        "std_score": 0.029,
        "N_effective_cs": 51889  # Sample size
    },
    additional_data={
        "n_symbols": 10,
        "date_range": {
            "start_ts": "2023-01-01T00:00:00",
            "end_ts": "2023-06-30T23:59:59"
        },
        "cs_config": {
            "min_cs": 10,
            "max_cs_samples": 1000,
            "leakage_filter_version": "v1.2",
            "universe_id": "SP500"
        }
    }
)
```

### Backward Compatibility

If cohort metadata is **not** provided, the system falls back to the legacy flat structure. This ensures existing code continues to work.

## Example Output

### New Cohort (First Run)

```
📊 Reproducibility: First run for target_ranking:y_will_peak_60m_0.8 (new cohort: N=51889, symbols=10, date_range=2023-01-01T00:00:00→2023-06-30T23:59:59)
```

### Within Same Cohort

```
ℹ️ Reproducibility: STABLE [cohort: N=51889, symbols=10, sample-adjusted] (Δ ROC-AUC=+0.0019 (+0.25%, z=0.62); within tolerance)
   Previous: ROC-AUC=0.742±0.029, N=51889, importance=0.23, composite=0.764
   Current:  ROC-AUC=0.744±0.029, N=52105, importance=0.23, composite=0.765
   Diff:     ROC-AUC=+0.0019 (+0.25%, z=0.62) [STABLE], composite=+0.0010 (+0.13%) [STABLE], importance=+0.00 (+0.00%) [STABLE]
```

### Different Cohorts (INCOMPARABLE)

If runs have different sample sizes (n_ratio < 0.90), they are saved to different cohorts and not compared. The system will log:

```
📊 Reproducibility: First run for target_ranking:y_will_peak_60m_0.8 (new cohort: N=10802, symbols=5, date_range=2024-01-01T00:00:00→2024-03-31T23:59:59)
```

## Integration Points

To enable cohort-aware tracking, update call sites to pass cohort metadata:

### Target Ranking

```python
# In TRAINING/ranking/predictability/model_evaluation.py
tracker.log_comparison(
    stage="target_ranking",
    item_name=target_name,
    metrics={
        "mean_score": result.mean_score,
        "std_score": result.std_score,
        "N_effective_cs": len(X_train)  # Add sample size
    },
    additional_data={
        "n_symbols": len(symbols),
        "date_range": {
            "start_ts": data_df['timestamp'].min().isoformat(),
            "end_ts": data_df['timestamp'].max().isoformat()
        },
        "cs_config": {
            "min_cs": min_cs,
            "max_cs_samples": max_cs_samples
        }
    }
)
```

### Feature Selection

```python
# In TRAINING/ranking/feature_selector.py
tracker.log_comparison(
    stage="feature_selection",
    item_name=target_column,
    metrics={
        "mean_score": mean_consensus,
        "std_score": std_consensus,
        "N_effective_cs": cs_data.shape[0]  # Add sample size
    },
    additional_data={
        "n_symbols": len(symbols),
        "date_range": {...},  # Extract from data
        "cs_config": {...}
    }
)
```

### Model Training

```python
# In TRAINING/training_strategies/training.py
tracker.log_comparison(
    stage="model_training",
    item_name=f"{target}:{family}",
    metrics={
        "mean_score": float(np.mean(cv_scores)),
        "std_score": float(np.std(cv_scores)),
        "N_effective_cs": len(X_train)  # Add sample size
    },
    additional_data={
        "n_symbols": len(mtf_data),
        "date_range": {...},  # Extract from mtf_data
        "cs_config": {
            "min_cs": min_cs,
            "max_cs_samples": max_cs_samples
        }
    }
)
```

## Benefits

1. **Statistically meaningful comparisons**: Only compare apples to apples
2. **Sample-adjusted drift**: Z-scores account for N differences
3. **Clear organization**: Easy to find related runs by cohort
4. **Backward compatible**: Falls back to legacy mode if no cohort metadata
5. **Defensible logs**: Can explain why comparisons are valid/invalid

## Next Steps

1. **Update call sites** to pass cohort metadata (see Integration Points above)
2. **Test** with real runs to verify cohort matching works correctly
3. **Monitor** logs to ensure INCOMPARABLE labels appear when expected
4. **Migrate** existing runs to new structure (optional, for historical analysis)
