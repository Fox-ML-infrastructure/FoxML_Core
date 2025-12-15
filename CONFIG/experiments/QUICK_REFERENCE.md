# Experiment Config Quick Reference

## Most Common Settings

### Change Target Count
```yaml
intelligent_training:
  top_n_targets: 5              # Select top 5 targets
  max_targets_to_evaluate: 10    # Rank top 10 targets
```

### Change Feature Count
```yaml
intelligent_training:
  top_m_features: 50  # Select 50 features per target
```

### Change Data Limits
```yaml
data:
  max_samples_per_symbol: 1000  # Limit data per symbol
  max_rows_train: 20000         # Limit training data
  min_cs: 3                     # Minimum cross-sectional samples
```

### Change Model Families
```yaml
feature_selection:
  model_families: [lightgbm, xgboost, random_forest]

training:
  model_families: [lightgbm, xgboost, random_forest]
```

### Enable/Disable Parallel Execution
```yaml
multi_target:
  parallel_targets: true  # or false

threading:
  parallel:
    enabled: true  # or false
```

## That's It!

Everything else uses sensible defaults. You only need to change what you want to override.
