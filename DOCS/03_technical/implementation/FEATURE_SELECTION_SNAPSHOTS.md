# Feature Selection Snapshot Structure

## Overview

The FEATURE_SELECTION stage creates multiple snapshots for different purposes. This document explains which snapshots exist, their purposes, and which one to use.

## Snapshot Types

### 1. `multi_model_aggregated` (Source of Truth)

**Location**: `targets/{target}/reproducibility/stage=FEATURE_SELECTION/{view}/universe={universe_sig}/feature_importance_snapshots/{target}/multi_model_aggregated/fs_snapshot.json`

**Purpose**: **This is the source of truth for feature selection results.**

- Contains aggregated importance scores from all model families (lightgbm, xgboost, catboost, etc.)
- Represents the consensus feature ranking used for actual feature selection
- Always created (not optional)
- Used by downstream stages (TRAINING) to determine which features to use

**When to use**: 
- ✅ **Use this snapshot** to see which features were selected
- ✅ **Use this snapshot** for reproducibility tracking
- ✅ **Use this snapshot** to understand feature selection results

**Example path**:
```
targets/fwd_ret_10m/reproducibility/stage=FEATURE_SELECTION/CROSS_SECTIONAL/universe=ef91e9db233a/feature_importance_snapshots/fwd_ret_10m/multi_model_aggregated/fs_snapshot.json
```

---

### 2. `cross_sectional_panel` (Optional Stability Analysis)

**Location**: `targets/{target}/reproducibility/stage=FEATURE_SELECTION/{view}/universe={universe_sig}/feature_importance_snapshots/{target}/cross_sectional_panel/fs_snapshot.json`

**Purpose**: **Optional cross-sectional stability analysis snapshot.**

- Contains cross-sectional importance scores from panel models (trained across all symbols simultaneously)
- Used for stability analysis across runs (detecting drift in cross-sectional feature importance)
- Only created if `cross_sectional_ranking.enabled=True` in config
- Separate computation from main feature selection (uses different models/methods)

**When to use**:
- ✅ **Use this snapshot** for stability analysis (comparing CS importance across runs)
- ✅ **Use this snapshot** to understand cross-sectional feature behavior
- ❌ **Do NOT use this** as the source of truth for feature selection results

**Configuration**:
```yaml
# CONFIG/preprocessing_config.yaml
preprocessing:
  multi_model_feature_selection:
    cross_sectional_ranking:
      enabled: true  # Set to false to disable (snapshot won't be created)
      min_cs: 10
      max_cs_samples: 1000
      top_k_candidates: 50  # Top features from main selection to re-evaluate
      model_families: ['lightgbm']
```

**Example path**:
```
targets/fwd_ret_10m/reproducibility/stage=FEATURE_SELECTION/CROSS_SECTIONAL/universe=ef91e9db233a/feature_importance_snapshots/fwd_ret_10m/cross_sectional_panel/fs_snapshot.json
```

---

### 3. Per-Model Snapshots (Individual Model Families)

**Location**: `targets/{target}/reproducibility/stage=FEATURE_SELECTION/{view}/universe={universe_sig}/feature_importance_snapshots/{target}/{model_family}/fs_snapshot.json`

**Purpose**: **Individual model family importance snapshots.**

- One snapshot per model family (lightgbm, xgboost, catboost, ridge, etc.)
- Used for per-model stability analysis (comparing same model across runs)
- Always created for each enabled model family
- Aggregated into `multi_model_aggregated` snapshot

**When to use**:
- ✅ **Use these snapshots** to see importance from individual models
- ✅ **Use these snapshots** for per-model stability analysis
- ✅ **Use these snapshots** to debug model-specific issues

**Example paths**:
```
targets/fwd_ret_10m/reproducibility/stage=FEATURE_SELECTION/CROSS_SECTIONAL/universe=ef91e9db233a/feature_importance_snapshots/fwd_ret_10m/lightgbm/fs_snapshot.json
targets/fwd_ret_10m/reproducibility/stage=FEATURE_SELECTION/CROSS_SECTIONAL/universe=ef91e9db233a/feature_importance_snapshots/fwd_ret_10m/xgboost/fs_snapshot.json
```

---

## Quick Reference

| Snapshot | Purpose | Always Created? | Source of Truth? |
|----------|---------|------------------|-------------------|
| `multi_model_aggregated` | Main feature selection results | ✅ Yes | ✅ **YES** |
| `cross_sectional_panel` | CS stability analysis | ❌ Only if enabled | ❌ No |
| `{model_family}` | Per-model importance | ✅ Yes (per family) | ❌ No (use aggregated) |

## Directory Structure

### Feature Importance Snapshots

```
targets/{target}/reproducibility/stage=FEATURE_SELECTION/{view}/universe={universe_sig}/feature_importance_snapshots/{target}/
├── multi_model_aggregated/          # ✅ Source of truth
│   └── fs_snapshot.json
├── cross_sectional_panel/            # Optional (if enabled)
│   └── fs_snapshot.json
├── lightgbm/                          # Per-model snapshots
│   └── fs_snapshot.json
├── xgboost/
│   └── fs_snapshot.json
├── catboost/
│   └── fs_snapshot.json
└── ...
```

### Cohort Directory (Metrics)

```
targets/{target}/reproducibility/stage=FEATURE_SELECTION/{view}/cohort={cohort_id}/
├── metadata.json                     # Full cohort metadata
├── metrics.json                       # Main multi-model feature selection metrics
├── metrics_cs_panel.json             # Cross-sectional panel metrics (if enabled)
├── metrics.parquet                    # Canonical metrics (parquet format)
├── snapshot.json                      # Normalized snapshot for diff telemetry
├── diff_prev.json                     # Diff vs previous run
└── ...
```

**Note**: Both `metrics.json` (main feature selection) and `metrics_cs_panel.json` (cross-sectional panel) are stored in the **same cohort directory** to consolidate all FEATURE_SELECTION metrics in one place.

## Common Questions

### Q: Which snapshot should I use for feature selection results?

**A**: Use `multi_model_aggregated` - it's the source of truth containing consensus importance from all model families.

### Q: Why are there two snapshots with different `universe_sig` values?

**A**: This was a bug in older runs. Both snapshots should use the same SST `universe_sig` (fixed in 2026-01-08). Old snapshots with `universe_sig="ALL"` can be ignored.

### Q: Do I need the `cross_sectional_panel` snapshot?

**A**: Only if you're doing cross-sectional stability analysis. It's optional and can be disabled via config (`cross_sectional_ranking.enabled=false`).

### Q: What's the difference between `multi_model_aggregated` and `cross_sectional_panel`?

**A**: 
- `multi_model_aggregated`: Aggregated importance from per-symbol models (main feature selection)
- `cross_sectional_panel`: Importance from panel models trained across all symbols (stability analysis)

### Q: Why don't I see per-model snapshots?

**A**: Per-model snapshots are created but may not be visible if:
- Snapshot creation failed (check logs for warnings)
- Models were filtered out (task incompatibility, etc.)
- Output directory structure differs

Check `globals/fs_snapshot_index.json` for a complete list of all snapshots.

## Related Documentation

- [Feature Selection Guide](FEATURE_SELECTION_GUIDE.md) - Main feature selection documentation
- [Reproducibility Structure](REPRODUCIBILITY_STRUCTURE.md) - Overall reproducibility directory structure
- [Feature Importance Stability](FEATURE_IMPORTANCE_STABILITY.md) - Stability analysis system
