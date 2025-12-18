# EXPERIMENTS Quick Start

> **⚠️ Note**: This 3-phase EXPERIMENTS workflow is a legacy approach. For new projects, consider using the [Intelligent Training Pipeline](INTELLIGENT_TRAINING_TUTORIAL.md) which provides automated target ranking and feature selection with a unified configuration system.

## Setup

### Step 1: Customize Data Loading

Edit `phase1_feature_engineering/run_phase1.py`, line ~44:

```python
def load_data(data_dir):
    """Replace this with your data loading code"""
    import pandas as pd

    # Your code here
    df = pd.read_parquet(f"{data_dir}/training_data.parquet")

    X = df[feature_columns].values
    y_dict = {target: df[target].values for target in targets}
    feature_names = feature_columns

    return X, y_dict, feature_names
```

### Step 2: Prepare Data

```bash
# Data directory already created at /home/Jennifer/trader/data
# Place your data there, or use environment variable to override

# Option 1: Use default location
cd /home/Jennifer/trader/data
# Place your data files here

# Option 2: Override with environment variable
export DATA_DIR=/path/to/your/custom/data/location
```

### Step 3: Run Phase 1

```bash
# Test on small sample first
python phase1_feature_engineering/run_phase1.py \
    --data-dir /path/to/small/sample \
    --config phase1_feature_engineering/feature_selection_config.yaml \
    --output-dir metadata
```

## Expected Output

After Phase 1 completes (~15-30 minutes):

```
metadata/
├── top_50_features.json          # Selected features
├── feature_importance_report.csv # Full rankings
├── vae_encoder.joblib            # VAE model
├── gmm_model.joblib              # GMM model
└── phase1_summary.json           # Summary
```

Verify:
```bash
ls metadata/
cat metadata/phase1_summary.json
head -20 metadata/feature_importance_report.csv
```

## Workflow Comparison

### Previous Workflow
```bash
# Train on ALL 421 features
./train_all_symbols.sh
  ├─ Phase 1: Cross-sectional (421 features) → Takes 4 hours, overfits
  └─ Phase 2: Sequential (421 features) → Takes 4 hours, overfits
```

### Current Workflow
```bash
# Train on SELECTED features
./run_all_phases.sh
  ├─ Phase 1: Feature selection (421 → 61 features) → Takes 30 min
  ├─ Phase 2: Core models (61 features) → Takes 1 hour, no overfitting
  └─ Phase 3: Sequential (61 features) → Takes 1 hour, no overfitting
```

Result: 8 hours → 2.5 hours, better accuracy.

## Command Reference

### Run Everything
```bash
./run_all_phases.sh
```

### Run Individual Phases
```bash
# Phase 1 only
python phase1_feature_engineering/run_phase1.py \
    --data-dir $DATA_DIR \
    --config phase1_feature_engineering/feature_selection_config.yaml \
    --output-dir metadata

# Phase 2 only (after Phase 1)
python phase2_core_models/run_phase2.py \
    --metadata-dir metadata \
    --config phase2_core_models/core_models_config.yaml

# Phase 3 only (after Phase 1)
python phase3_sequential_models/run_phase3.py \
    --metadata-dir metadata \
    --config phase3_sequential_models/sequential_config.yaml
```

### Monitor Progress
```bash
# Watch Phase 1 log
tail -f logs/phase1_*.log

# Check outputs
ls metadata/
ls output/core_models/
```

## Configuration Changes

### Try Different Feature Counts

Edit `phase1_feature_engineering/feature_selection_config.yaml`:
```yaml
feature_selection:
  n_features: 30  # Try 30, 40, 50, or 60
```

### Disable VAE or GMM

```yaml
feature_engineering:
  vae:
    enabled: false  # Set to false to skip
  gmm:
    enabled: false  # Set to false to skip
```

### Change Primary Target

```yaml
feature_selection:
  primary_target: fwd_ret_10m  # Use different target
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `FileNotFoundError` | Check `--data-dir` path |
| `No feature importance` | Check data has valid targets |
| `VAE training fails` | Set `vae: enabled: false` in config |
| `Phase 2/3 not found` | Create scripts or skip (Phase 1 works standalone) |

## Related Documentation

- [Experiments Workflow](EXPERIMENTS_WORKFLOW.md) - Complete workflow overview
- [Experiments Operations](../01_tutorials/training/EXPERIMENTS_OPERATIONS.md) - Detailed step-by-step guide
- [Phase 1: Feature Engineering](../01_tutorials/training/PHASE1_FEATURE_ENGINEERING.md) - Phase 1 documentation
- [Experiments Implementation](EXPERIMENTS_IMPLEMENTATION.md) - Implementation details

## Performance Comparison

| Metric | Before | After |
|--------|--------|-------|
| Training time | 8 hours | 2.5 hours |
| Feature count | 421 | 61 |
| Train vs Val gap | 0.40 (bad) | 0.04 (good) |
| Overfitting | Yes | No |

## Success Criteria

Workflow is working when:

1. Phase 1 completes and creates metadata files
2. Feature report shows declining importance
3. Train vs validation scores are close (gap < 0.1)
4. Training is faster than before
5. Models generalize better on unseen data
