# Feature Importance Stability Tracking

Non-invasive hooks for tracking and analyzing feature importance stability across runs.

## Structure

```
TRAINING/stability/feature_importance/
├── __init__.py          # Public API exports
├── schema.py            # FeatureImportanceSnapshot dataclass
├── io.py                # Save/load snapshots to/from disk
├── analysis.py          # Stability metrics computation
└── hooks.py             # Pipeline integration hooks
```

## Usage

### From Pipeline Code

```python
from TRAINING.stability.feature_importance import save_snapshot_hook

# After computing feature importance
importance_dict = {"feature1": 0.5, "feature2": 0.3, ...}

save_snapshot_hook(
    target_name="peak_60m_0.8",
    method="lightgbm",
    importance_dict=importance_dict,
    universe_id="AAPL",  # or "ALL" for cross-sectional
    output_dir=output_dir,  # Optional
    auto_analyze=True,  # Automatically analyze stability
)
```

### From Pandas Series

```python
from TRAINING.stability.feature_importance import save_snapshot_from_series_hook

# If you have a pandas Series
importance_series = pd.Series({...})  # feature names as index

save_snapshot_from_series_hook(
    target_name="peak_60m_0.8",
    method="quick_pruner",
    importance_series=importance_series,
    auto_analyze=True,
)
```

### Analyze All Stability

```python
from TRAINING.stability.feature_importance import analyze_all_stability_hook

# At end of pipeline run
all_metrics = analyze_all_stability_hook(output_dir=output_dir)
# Automatically logs stability for all targets/methods
```

### CLI Analysis

```bash
# Analyze specific target/method
python scripts/analyze_importance_stability.py \
    --target peak_60m_0.8 \
    --method lightgbm \
    --top-k 20

# Use custom snapshot directory
python scripts/analyze_importance_stability.py \
    --target peak_60m_0.8 \
    --method quick_pruner \
    --base-dir artifacts/feature_importance
```

## Snapshot Storage

Snapshots are stored as JSON files:

```
artifacts/feature_importance/
  {target_name}/
    {method}/
      {run_id}.json
```

Each snapshot contains:
- Target name
- Method name
- Universe ID (optional)
- Run ID (UUID)
- Creation timestamp
- Feature names (sorted by importance)
- Importance values (same order)

## Stability Metrics

- **Top-K Overlap**: Jaccard similarity of top-K features between runs
- **Kendall Tau**: Rank correlation of feature importance
- **Selection Frequency**: How often each feature appears in top-K

## Configuration

Add to `CONFIG/training_config/safety_config.yaml`:

```yaml
safety:
  feature_importance:
    auto_analyze_stability: true  # Auto-analyze after each snapshot
    stability_thresholds:
      min_top_k_overlap: 0.7      # Warn if overlap < 0.7
      min_kendall_tau: 0.6        # Warn if tau < 0.6
```

## Integration Points

1. **Target Ranking**: After model training completes
2. **Feature Selection**: After each method (RFE, Boruta, etc.) and after aggregation
3. **Quick Pruning**: After pruning completes

See `INTERNAL_DOCS/FEATURE_IMPORTANCE_STABILITY_INTEGRATION_PLAN.md` for detailed integration plan.
