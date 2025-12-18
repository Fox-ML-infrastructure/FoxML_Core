# Phase 1: Feature Engineering & Selection

> **⚠️ Note**: This is part of the legacy 3-phase EXPERIMENTS workflow. For new projects, consider using the [Intelligent Training Pipeline](INTELLIGENT_TRAINING_TUTORIAL.md) which provides automated target ranking and feature selection with a unified configuration system.

## Purpose

Reduce feature dimensionality from 421 to ~50-60 features by:
1. Selecting top 50 most important features using LightGBM
2. Creating 10 latent features using VAE
3. Creating 1 regime feature using GMM

Reduces overfitting and speeds up training in Phases 2-3.

## Quick Start

```bash
python run_phase1.py \
    --data-dir /path/to/your/data \
    --config feature_selection_config.yaml \
    --output-dir ../metadata
```

## Process

### Step 1: Feature Selection
- Trains LightGBM on `fwd_ret_5m` target
- Extracts feature importance scores
- Selects top 50 features
- Saves `top_50_features.json`

### Step 2: VAE Feature Engineering (Optional)
- Trains Variational Autoencoder on selected features
- Extracts 10 latent features
- Saves `vae_encoder.joblib`

### Step 3: GMM Regime Detection (Optional)
- Trains Gaussian Mixture Model on first 5 features
- Creates 3 regime labels (e.g., low/mid/high volatility)
- Saves `gmm_model.joblib`

## Configuration

Edit `feature_selection_config.yaml`:

```yaml
feature_selection:
  n_features: 50              # Adjust to 30, 40, or 60 as needed
  primary_target: fwd_ret_5m  # Change to your main target

feature_engineering:
  vae:
    enabled: true             # Disable if you don't want VAE features
    latent_dim: 10            # Adjust latent space size

  gmm:
    enabled: true             # Disable if you don't want regime features
    n_components: 3           # Try 2, 3, or 4 regimes
```

## Outputs

All saved to `metadata/`:

1. `top_50_features.json`: List of selected feature names
2. `feature_importance_report.csv`: Full importance rankings
3. `vae_encoder.joblib`: Trained VAE encoder (if enabled)
4. `gmm_model.joblib`: Trained GMM model (if enabled)
5. `phase1_summary.json`: Summary statistics

## Customization

### To modify data loading:

Edit `run_phase1.py`, function `load_data()`:

```python
def load_data(data_dir):
    # Replace with your data loading logic
    import pandas as pd
    df = pd.read_parquet(f"{data_dir}/training_data.parquet")

    # Extract features and targets
    feature_cols = [col for col in df.columns if col.startswith('feat_')]
    target_cols = [col for col in df.columns if col.startswith('fwd_ret_')]

    X = df[feature_cols].values
    y_dict = {target: df[target].values for target in target_cols}
    feature_names = feature_cols

    return X, y_dict, feature_names
```

## Troubleshooting

Issue: Feature importance all zeros
- Check for NaN values in features or target
- Verify LightGBM is training (check logs)
- Try a different target that has more signal

Issue: VAE training fails
- Reduce `latent_dim` from 10 to 5
- Increase `n_epochs`
- Check for Inf/NaN in features

Issue: GMM gives strange regimes
- Try different `n_components` (2, 3, or 4)
- Use different features for GMM training
- Check feature distributions

## Next Steps

After Phase 1 completes:

1. Verify outputs exist:
   ```bash
   ls ../metadata/
   ```

2. Review feature report:
   ```bash
   head -20 ../metadata/feature_importance_report.csv
   ```

3. Proceed to Phase 2:
   ```bash
   cd ../phase2_core_models
   python run_phase2.py --metadata-dir ../metadata ...
   ```

## Related Documentation

- ⚠️ **Legacy**: [Experiments Workflow](../../LEGACY/EXPERIMENTS_WORKFLOW.md) - **DEPRECATED**: Use [Intelligent Training Pipeline](INTELLIGENT_TRAINING_TUTORIAL.md) instead
- ⚠️ **Legacy**: [Experiments Quick Start](../../LEGACY/EXPERIMENTS_QUICK_START.md) - **DEPRECATED**: Use [Intelligent Training Tutorial](INTELLIGENT_TRAINING_TUTORIAL.md) instead
- [Experiments Operations](EXPERIMENTS_OPERATIONS.md) - Detailed operations guide
- [Feature Selection Tutorial](FEATURE_SELECTION_TUTORIAL.md) - General feature selection tutorial
