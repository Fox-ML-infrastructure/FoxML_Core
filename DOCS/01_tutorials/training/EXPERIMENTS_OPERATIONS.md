# Operations Guide - 3-Phase Training Workflow

> **⚠️ Note**: This 3-phase EXPERIMENTS workflow is a legacy approach. For new projects, consider using the [Intelligent Training Pipeline](INTELLIGENT_TRAINING_TUTORIAL.md) which provides automated target ranking and feature selection with a unified configuration system.

Step-by-step instructions for running the optimized 3-phase training workflow.

## Prerequisites

### 1. Environment Setup
```bash
# Ensure you're in the TRAINING directory
cd /home/Jennifer/trader/TRAINING

# Verify required packages
python -c "import lightgbm, xgboost, sklearn, numpy, pandas; print('All packages available')"

# Create output directories if they don't exist
mkdir -p EXPERIMENTS/{metadata,logs,output}
```

### 2. Data Preparation
- Ensure data files are in the correct location
- Data should have ~421 features
- Targets should be properly labeled (e.g., `fwd_ret_5m`, `mdd_5m_0.001`, etc.)

## Phase 1: Feature Engineering & Selection

### Purpose
Reduce dimensionality from 421 features to ~50-60 features.

### Process
1. Loads all 421 features
2. Trains LightGBM on `fwd_ret_5m` to get feature importance
3. Selects top 50 most important features
4. Trains VAE to create 10 latent features
5. Trains GMM to create 1 regime feature
6. Saves artifacts to `metadata/` for use in Phase 2-3

### Run Command
```bash
cd EXPERIMENTS

python phase1_feature_engineering/run_phase1.py \
    --data-dir /path/to/your/data \
    --config phase1_feature_engineering/feature_selection_config.yaml \
    --output-dir metadata \
    --log-dir logs
```

### Configuration (`phase1_feature_engineering/feature_selection_config.yaml`)
```yaml
feature_selection:
  n_features: 50
  primary_target: fwd_ret_5m
  method: importance  # or 'correlation'

feature_engineering:
  vae:
    enabled: true
    latent_dim: 10
  gmm:
    enabled: true
    n_components: 3

lightgbm:
  max_depth: 8
  learning_rate: 0.03
  n_estimators: 1000
  early_stopping_rounds: 50
```

### Expected Outputs
```
metadata/
├── top_50_features.json          # List of selected feature names
├── feature_importance_report.csv # Importance scores for all features
├── vae_encoder.joblib            # Trained VAE encoder
├── gmm_model.joblib              # Trained GMM model
└── phase1_summary.json           # Summary statistics
```

### Verification
```bash
# Check outputs exist
ls metadata/top_50_features.json
ls metadata/vae_encoder.joblib
ls metadata/gmm_model.joblib

# View selected features
cat metadata/top_50_features.json | python -m json.tool | head -20

# View feature importance report
head -20 metadata/feature_importance_report.csv
```

### Troubleshooting

Issue: `FileNotFoundError: No such file or directory`
- Check `--data-dir` path is correct
- Ensure data files exist and are readable

Issue: `No feature importance available`
- Verify target exists in data
- Check for NaN values in features/target
- Ensure LightGBM training succeeded (check logs)

Issue: `VAE training fails`
- Reduce `latent_dim` from 10 to 5
- Check for NaN/Inf values in selected features
- Increase `n_epochs` if needed

## Phase 2: Core Model Training

### Purpose
Train LightGBM, MultiTask, and Ensemble models on selected features.

### Process
1. Loads Phase 1 artifacts (top features, VAE, GMM)
2. Transforms 421 features → 61 features (50 + 10 VAE + 1 GMM)
3. Trains models with proper regularization and early stopping
4. Saves trained models to `output/core_models/`

### Run Command
```bash
cd EXPERIMENTS

python phase2_core_models/run_phase2.py \
    --data-dir /path/to/your/data \
    --metadata-dir metadata \
    --config phase2_core_models/core_models_config.yaml \
    --output-dir output/core_models \
    --log-dir logs
```

### Configuration (`phase2_core_models/core_models_config.yaml`)
```yaml
strategy: single_task  # or multi_task

models:
  lightgbm:
    enabled: true
    max_depth: 8
    num_leaves: 96
    learning_rate: 0.03
    subsample: 0.75
    colsample_bytree: 0.75
    reg_alpha: 0.1
    reg_lambda: 0.1

  multitask:
    enabled: true
    targets: [tth_5m_0.5, mdd_5m_0.001, mfe_5m_0.001, fwd_ret_5m]
    learning_rate: 0.0001
    patience: 10
    hidden_dim: 256

  ensemble:
    enabled: true
    use_stacking: true
    stacking_cv: 5

targets:
  - fwd_ret_5m
  - fwd_ret_10m
  - mdd_5m_0.001
  - mfe_5m_0.001
  - tth_5m_0.5

early_stopping_rounds: 50
```

### Expected Outputs
```
output/core_models/
├── lightgbm_fwd_ret_5m.joblib
├── lightgbm_fwd_ret_10m.joblib
├── multitask_model.joblib
├── ensemble_model.joblib
├── training_summary.json
└── validation_scores.csv
```

### Verification
```bash
# Check models exist
ls output/core_models/*.joblib

# View training summary
cat output/core_models/training_summary.json | python -m json.tool

# Check validation scores
cat output/core_models/validation_scores.csv

# Verify overfitting is reduced (train vs val scores should be close)
```

### Troubleshooting

Issue: `Metadata not found`
- Ensure Phase 1 completed successfully
- Check `--metadata-dir` path

Issue: `Models still overfitting`
- Reduce `n_features` in Phase 1 to 30-40
- Increase regularization (`reg_alpha`, `reg_lambda`)
- Reduce `max_depth` to 6-7

Issue: `MultiTask training slow`
- Reduce `hidden_dim` to 128
- Reduce `n_epochs` to 50
- Use GPU if available

## Phase 3: Sequential Model Training

### Purpose
Train LSTM, Transformer, and CNN1D models on selected features.

### Process
1. Loads Phase 1 artifacts
2. Transforms features to sequences for sequential models
3. Trains sequential models with reduced dimensionality
4. Saves models to `output/sequential_models/`

### Run Command
```bash
cd EXPERIMENTS

python phase3_sequential_models/run_phase3.py \
    --data-dir /path/to/your/data \
    --metadata-dir metadata \
    --config phase3_sequential_models/sequential_config.yaml \
    --output-dir output/sequential_models \
    --log-dir logs
```

### Configuration (`phase3_sequential_models/sequential_config.yaml`)
```yaml
sequential:
  lookback_T: 60
  stride: 1

models:
  lstm:
    enabled: true
    hidden_dim: 128
    num_layers: 2
    dropout: 0.3

  transformer:
    enabled: false  # Disable initially, enable after LSTM works
    d_model: 128
    nhead: 8

  cnn1d:
    enabled: true
    hidden_dims: [128, 64]
    kernel_size: 5

targets:
  - tth_5m_0.5
  - mfe_share_5m
  - is_peak_5m_0.5

early_stopping:
  patience: 15
  min_delta: 0.001
```

### Expected Outputs
```
output/sequential_models/
├── lstm_tth_5m_0.5.joblib
├── cnn1d_tth_5m_0.5.joblib
├── training_history.csv
└── sequential_summary.json
```

### Verification
```bash
# Check models exist
ls output/sequential_models/*.joblib

# View training history
head -20 output/sequential_models/training_history.csv
```

### Troubleshooting

Issue: `Sequential models training too slow`
- Reduce `lookback_T` from 60 to 30
- Reduce `hidden_dim` to 64
- Use fewer `num_layers` (1 instead of 2)

Issue: `Out of memory`
- Reduce `batch_size` to 16 or 32
- Reduce `lookback_T`
- Train on subset of data first

Issue: `Models not learning`
- Check for NaN values in sequences
- Verify `lookback_T` is appropriate for your data frequency
- Increase learning rate slightly

## Running All Phases Together

### Master Script (`run_all_phases.sh`)

```bash
#!/bin/bash

cd EXPERIMENTS

# Configuration
DATA_DIR="/path/to/your/data"
LOG_DIR="logs"
METADATA_DIR="metadata"
OUTPUT_DIR="output"

# Create directories
mkdir -p $LOG_DIR $METADATA_DIR $OUTPUT_DIR

# Phase 1: Feature Engineering & Selection
python phase1_feature_engineering/run_phase1.py \
    --data-dir $DATA_DIR \
    --config phase1_feature_engineering/feature_selection_config.yaml \
    --output-dir $METADATA_DIR \
    --log-dir $LOG_DIR \
    2>&1 | tee $LOG_DIR/phase1_$(date +%Y%m%d_%H%M%S).log

# Phase 2: Core Model Training
python phase2_core_models/run_phase2.py \
    --data-dir $DATA_DIR \
    --metadata-dir $METADATA_DIR \
    --config phase2_core_models/core_models_config.yaml \
    --output-dir $OUTPUT_DIR/core_models \
    --log-dir $LOG_DIR \
    2>&1 | tee $LOG_DIR/phase2_$(date +%Y%m%d_%H%M%S).log

# Phase 3: Sequential Model Training
python phase3_sequential_models/run_phase3.py \
    --data-dir $DATA_DIR \
    --metadata-dir $METADATA_DIR \
    --config phase3_sequential_models/sequential_config.yaml \
    --output-dir $OUTPUT_DIR/sequential_models \
    --log-dir $LOG_DIR \
    2>&1 | tee $LOG_DIR/phase3_$(date +%Y%m%d_%H%M%S).log
```

### Usage
```bash
# Make executable
chmod +x run_all_phases.sh

# Run all phases
./run_all_phases.sh
```

## Monitoring Progress

### Real-time Logs
```bash
# Watch Phase 1 progress
tail -f logs/phase1_*.log

# Watch Phase 2 progress
tail -f logs/phase2_*.log

# Watch Phase 3 progress
tail -f logs/phase3_*.log
```

### Check Training Status
```bash
# Count completed models
find output/ -name "*.joblib" | wc -l

# View last training summary
cat output/core_models/training_summary.json | python -m json.tool
```

## Common Workflows

### Workflow 1: First-Time Complete Training
```bash
# Clean start
rm -rf metadata/* output/* logs/*

# Run all phases
./run_all_phases.sh
```

### Workflow 2: Re-run Phase 2 with Different Configs
```bash
# Phase 1 is already done, just modify Phase 2 config
vim phase2_core_models/core_models_config.yaml

# Run Phase 2 only
python phase2_core_models/run_phase2.py \
    --metadata-dir metadata \
    --config phase2_core_models/core_models_config.yaml
```

### Workflow 3: Experiment with Different Feature Counts
```bash
# Modify Phase 1 config
vim phase1_feature_engineering/feature_selection_config.yaml
# Change n_features: 50 to n_features: 30

# Re-run Phase 1
python phase1_feature_engineering/run_phase1.py ...

# Re-run Phase 2 with new features
python phase2_core_models/run_phase2.py ...
```

## Performance Benchmarks

### Expected Timings (Approximate)

Hardware: CPU-only, 12 cores, 32GB RAM

| Phase | Time | Bottleneck |
|-------|------|------------|
| Phase 1 | 15-30 min | Feature importance calculation |
| Phase 2 | 30-60 min | LightGBM training |
| Phase 3 | 60-120 min | LSTM training |
| Total | 2-3 hours | |

Hardware: GPU-enabled, 12 cores, 32GB RAM, RTX 3080

| Phase | Time | Bottleneck |
|-------|------|------------|
| Phase 1 | 15-30 min | Feature importance (CPU-bound) |
| Phase 2 | 20-40 min | LightGBM (CPU-bound) |
| Phase 3 | 20-30 min | LSTM (GPU-accelerated) |
| Total | 1-2 hours | |

## Validation Checklist

After each phase:

### Phase 1
- [ ] `metadata/top_50_features.json` exists and contains ~50 features
- [ ] `metadata/vae_encoder.joblib` exists
- [ ] `metadata/gmm_model.joblib` exists
- [ ] Feature importance report shows declining importance

### Phase 2
- [ ] All target models exist in `output/core_models/`
- [ ] Training logs show early stopping messages
- [ ] Validation scores are close to training scores (gap < 0.1)
- [ ] No NaN predictions in validation

### Phase 3
- [ ] Sequential models exist in `output/sequential_models/`
- [ ] Training history shows learning (loss decreasing)
- [ ] Early stopping triggered appropriately

## Related Documentation

- ⚠️ **Legacy**: [Experiments Workflow](../../LEGACY/EXPERIMENTS_WORKFLOW.md) - **DEPRECATED**: Use [Intelligent Training Pipeline](INTELLIGENT_TRAINING_TUTORIAL.md) instead
- ⚠️ **Legacy**: [Experiments Quick Start](../../LEGACY/EXPERIMENTS_QUICK_START.md) - **DEPRECATED**: Use [Intelligent Training Tutorial](INTELLIGENT_TRAINING_TUTORIAL.md) instead
- [Phase 1: Feature Engineering](PHASE1_FEATURE_ENGINEERING.md) - Phase 1 documentation
- [Training Optimization](../../03_technical/implementation/TRAINING_OPTIMIZATION_GUIDE.md) - Optimization guide
- [Feature Selection Implementation](../../03_technical/implementation/FEATURE_SELECTION_GUIDE.md) - Feature selection guide
