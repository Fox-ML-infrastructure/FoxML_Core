# E2E Test Command

Simple command to run a complete end-to-end test: **target ranking → feature selection → model training**.

## Quick Command

```bash
python -m TRAINING.orchestration.intelligent_trainer \
    --data-dir "data/data_labeled/interval=5m" \
    --symbols AAPL MSFT GOOGL TSLA NVDA \
    --output-dir "test_e2e" \
    --families lightgbm xgboost random_forest
```

## What It Does

This single command runs all three steps:

1. **Target Ranking** - Automatically discovers and ranks targets, selects top 23
2. **Feature Selection** - Automatically selects top 50 features per target
3. **Model Training** - Trains models on selected targets/features

## Test Settings (Auto-Applied)

When `--output-dir` contains "test", the system automatically uses test-friendly settings:

- `top_n_targets: 23` (from `test.intelligent_training`)
- `max_targets_to_evaluate: 23` (limits ranking to 23 targets for speed)
- `top_m_features: 50` (selects 50 features per target)
- `min_cs: 3` (minimum cross-sectional samples)
- `max_rows_per_symbol: 5000` (limits data per symbol)
- `max_rows_train: 10000` (limits training data)

**No CLI arguments needed for these!** All from config.

## Minimal Test (Faster)

For even faster testing with fewer models:

```bash
python -m TRAINING.orchestration.intelligent_trainer \
    --data-dir "data/data_labeled/interval=5m" \
    --symbols AAPL MSFT \
    --output-dir "test_e2e_minimal" \
    --families lightgbm
```

## Full Test (More Models)

For comprehensive testing with all model families:

```bash
python -m TRAINING.orchestration.intelligent_trainer \
    --data-dir "data/data_labeled/interval=5m" \
    --symbols AAPL MSFT GOOGL TSLA NVDA \
    --output-dir "test_e2e_full" \
    --families lightgbm xgboost random_forest catboost neural_network lasso mutual_information univariate_selection
```

## Output

Results are saved to timestamped directory:
```
test_e2e_20251211_130342/
├── target_rankings/          # Step 1: Target rankings
│   └── target_predictability_rankings.csv
├── feature_selections/        # Step 2: Feature selections
│   └── {target}/
│       └── selected_features.txt
└── training_results/          # Step 3: Trained models
    └── (model artifacts, metrics, predictions)
```

## Customizing Test Settings

Edit `CONFIG/training_config/pipeline_config.yaml`:

```yaml
test:
  intelligent_training:
    top_n_targets: 10          # Change number of targets
    top_m_features: 30         # Change number of features
    min_cs: 3                  # Change min cross-sectional samples
    max_rows_per_symbol: 3000  # Change data limit
```

Then run the same command - it uses your updated settings.

## Troubleshooting

**No targets found?**
- Check that data exists in `--data-dir`
- Verify symbols have data files
- Check `CONFIG/excluded_features.yaml` isn't too restrictive

**Out of memory?**
- Reduce `max_rows_per_symbol` in test config
- Use fewer symbols: `--symbols AAPL MSFT`
- Use fewer model families: `--families lightgbm`

**Takes too long?**
- Reduce `max_targets_to_evaluate` in test config
- Use fewer symbols
- Use fewer model families

## See Also

- [Testing Quick Reference](TESTING_QUICK_REFERENCE.md) - More test examples
- [Intelligent Training Tutorial](INTELLIGENT_TRAINING_TUTORIAL.md) - Complete guide
- [Config Examples](../configuration/CONFIG_EXAMPLES.md) - Configuration examples
