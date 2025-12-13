# Auto Target Ranking Tutorial

## Overview

The intelligent training pipeline can automatically discover, rank, and select the best targets from your dataset. This guide shows you how to configure and run target ranking to find the top 5 most predictable targets.

## Quick Start

### Step 1: Create or Edit Experiment Config

Create a new experiment config file (or edit an existing one) in `CONFIG/experiments/`:

```yaml
experiment:
  name: my_target_ranking_test
  description: "Rank all targets and select top 5"

data:
  data_dir: data/data_labeled_v2/interval=5m
  symbols: [AAPL, MSFT, GOOGL, TSLA, NVDA]
  interval: 5m
  max_samples_per_symbol: 10000  # Adjust for your needs
  max_rows_per_symbol: 10000
  max_rows_train: 20000
  min_cs: 3

# Enable auto target discovery and ranking
intelligent_training:
  auto_targets: true           # Enable auto-discovery
  top_n_targets: 5             # Select top 5 after ranking
  max_targets_to_evaluate: 100 # Evaluate up to 100 targets (or all available)
  auto_features: true
  top_m_features: 50
  strategy: single_task
  run_leakage_diagnostics: false

# Feature selection
feature_selection:
  model_families:
    - lightgbm
    - xgboost
    - random_forest

# Training
training:
  model_families:
    - lightgbm
    - xgboost
    - random_forest

# Enable parallel execution for faster ranking
multi_target:
  parallel_targets: true
  skip_on_error: true
  save_summary: true

threading:
  parallel:
    max_workers_process: 8
    max_workers_thread: 8
    enabled: true
```

### Step 2: Run Target Ranking

```bash
python -m TRAINING.orchestration.intelligent_trainer \
  --output-dir "target_ranking_test" \
  --experiment-config my_target_ranking_test
```

## Configuration Options

### Key Settings for Target Ranking

| Setting | Description | Recommended Value |
|---------|-------------|-------------------|
| `auto_targets` | Enable auto-discovery of targets | `true` |
| `top_n_targets` | Number of top targets to select after ranking | `5` (or your desired number) |
| `max_targets_to_evaluate` | Maximum targets to evaluate (set high to get all) | `100` (or higher) |
| `parallel_targets` | Enable parallel target evaluation | `true` (faster) |
| `skip_on_error` | Continue if one target fails | `true` (recommended) |

### Data Limits (Adjust for Speed vs Coverage)

| Setting | Fast Test | Comprehensive |
|---------|-----------|---------------|
| `max_samples_per_symbol` | `10000` | `50000` |
| `max_rows_per_symbol` | `10000` | `50000` |
| `max_rows_train` | `20000` | `100000` |
| `min_cs` | `3` | `10` |

## How It Works

### Step 1: Target Discovery

The pipeline scans your data directory and discovers all available targets:
- `fwd_ret_*` targets (forward returns - non-repainting)
- `y_*` targets (various classification targets)
- Filters out degenerate targets (single class, zero variance, etc.)
- Filters out known leaked targets (e.g., `first_touch`)

**Example output:**
```
Discovered 48 valid targets
- y_* targets: 38
- fwd_ret_* targets: 10
Skipped 12 degenerate targets
Skipped 1 first_touch targets (leaked)
```

### Step 2: Target Ranking

For each discovered target, the pipeline:
1. Loads data for all symbols
2. Applies feature filtering (leakage prevention)
3. Performs feature selection (multi-model consensus)
4. Trains models and evaluates performance
5. Computes cross-validation scores (ROC-AUC for classification, R² for regression)

**Ranking metrics:**
- **Cross-sectional ROC-AUC** (for classification targets)
- **Cross-sectional R²** (for regression targets)
- **Symbol-specific scores** (optional, for comparison)

### Step 3: Target Selection

After ranking, the pipeline:
1. Sorts targets by score (highest first)
2. Selects top N targets (`top_n_targets`)
3. Saves ranking results to `target_rankings/`
4. Proceeds to feature selection and training for selected targets

## Output Structure

After running, you'll find:

```
RESULTS/
  sample_500k-1M/
    target_ranking_test_YYYYMMDD_HHMMSS/
      target_rankings/
        ranking_summary.json          # Overall ranking summary
        cross_sectional_rankings.json  # Cross-sectional view rankings
        symbol_specific_rankings.json  # Symbol-specific view rankings (if enabled)
        feature_exclusions/            # Per-target exclusion lists
          target_name_exclusions.yaml
      feature_selections/              # Feature selection results
        target_name/
          selected_features.json
      ...
```

### Ranking Summary Format

```json
{
  "cross_sectional": [
    {
      "target": "fwd_ret_60m",
      "score": 0.523,
      "task_type": "regression",
      "view": "CROSS_SECTIONAL",
      "n_samples": 249920,
      "n_features": 50
    },
    {
      "target": "y_will_peak_60m_0.8",
      "score": 0.763,
      "task_type": "binary_classification",
      "view": "CROSS_SECTIONAL",
      "n_samples": 249920,
      "n_features": 50
    }
    // ... more targets
  ],
  "top_n_selected": [
    "fwd_ret_60m",
    "y_will_peak_60m_0.8",
    // ... top 5 targets
  ]
}
```

## Interpreting Results

### Score Interpretation

| Score Range | Interpretation | Action |
|-------------|----------------|--------|
| **0.52 - 0.55** | Honest baseline (non-repainting) | ✅ Good for production |
| **0.55 - 0.60** | Moderate predictability | ✅ Promising |
| **0.60 - 0.70** | Good predictability | ✅ Very promising |
| **> 0.70** | Suspicious (possible repainting) | ⚠️ Investigate for leakage |

### Target Types

**Non-Repainting Targets (Honest Baselines):**
- `fwd_ret_*` - Forward returns (mathematically hard to predict)
- `y_first_touch_*` - Triple barrier (which barrier hits first)
- **Expected scores:** ~0.52-0.54

**Potentially Repainting Targets:**
- `y_will_peak_*` - Peak detection (may repaint)
- `y_will_valley_*` - Valley detection (may repaint)
- `y_will_swing_*` - Swing detection (may repaint)
- **Expected scores:** > 0.70 (suspicious if too high)

## Examples

### Example 1: Quick Test (Top 5 Targets)

```yaml
intelligent_training:
  auto_targets: true
  top_n_targets: 5
  max_targets_to_evaluate: 50
```

**Runtime:** ~30-60 minutes (depending on data size)

### Example 2: Comprehensive Ranking (All Targets)

```yaml
intelligent_training:
  auto_targets: true
  top_n_targets: 10
  max_targets_to_evaluate: 100  # Evaluate all available
```

**Runtime:** ~2-4 hours (depending on data size and parallelization)

### Example 3: Focused Test (Specific Target Types)

If you want to test only forward returns:

```yaml
intelligent_training:
  auto_targets: false
  manual_targets:
    - fwd_ret_5m
    - fwd_ret_30m
    - fwd_ret_60m
    - fwd_ret_120m
    - fwd_ret_240m
  top_n_targets: 5
```

## Troubleshooting

### Problem: Only 1 target is being evaluated

**Solution:** Check that `auto_targets: true` and remove or comment out `targets.primary` in your config. The fallback logic only uses `targets.primary` when `auto_targets: false`.

### Problem: Ranking is too slow

**Solutions:**
1. Reduce `max_samples_per_symbol` (e.g., 10000 instead of 50000)
2. Enable `parallel_targets: true`
3. Increase `max_workers_process` (e.g., 8 or 16)
4. Reduce `top_m_features` (e.g., 30 instead of 50)

### Problem: Some targets are failing

**Solution:** Enable `skip_on_error: true` in `multi_target` section. This allows the pipeline to continue even if some targets fail.

### Problem: Scores seem too high (> 0.90)

**Warning:** This likely indicates data leakage. Check:
1. Feature filtering is working (check logs for "Final Gatekeeper")
2. Purge window is sufficient (check `purge_minutes` in config)
3. Target definition doesn't repaint (forward-looking logic)

## Best Practices

1. **Start with reduced data limits** for initial testing, then increase for production
2. **Use parallel execution** (`parallel_targets: true`) for faster ranking
3. **Compare scores** - honest baselines (~0.52-0.54) vs suspicious targets (>0.70)
4. **Check ranking summary** - look for consistent top performers across runs
5. **Validate top targets** - manually inspect top 5 targets for leakage before production use

## Related Documentation

- [Intelligent Training Tutorial](./INTELLIGENT_TRAINING_TUTORIAL.md) - Full pipeline overview
- [Feature Selection Guide](./FEATURE_SELECTION.md) - How feature selection works
- [Configuration Reference](../../02_reference/configuration/) - Detailed config options
- [Execution Order Documents](../../03_technical/implementation/) - Pipeline execution details

## Example Config Files

- `CONFIG/experiments/e2e_full_targets_test.yaml` - Comprehensive test config
- `CONFIG/experiments/e2e_ranking_test.yaml` - Quick ranking test
- `CONFIG/experiments/_template.yaml` - Template for creating new configs
