# Intelligent Training Tutorial

Complete guide to using the intelligent training pipeline that automatically ranks targets, selects features, and trains models.

## Overview

The intelligent training pipeline (`TRAINING/train.py`) automates the entire model training workflow:

```
Data
  ↓
[1] Automatic Target Ranking → Select top N most predictable targets
  ↓
[2] Automatic Feature Selection → Select top M features per target
  ↓
[3] Model Training → Train all models with selected targets/features
  ↓
Trained Models
```

**Benefits:**
- **No manual steps**: Everything automated in one command
- **Intelligent selection**: Multi-model consensus for ranking/selection
- **Cached results**: Rankings/selections cached for faster reruns
- **Leakage-free**: All existing safeguards preserved
- **Unified behavior**: Ranking and selection use consistent preprocessing and configuration

**Pipeline Consistency:**
- **Interval handling**: Both ranking and selection respect `data.bar_interval` from experiment config (no spurious auto-detection warnings)
- **Sklearn preprocessing**: All sklearn-based models use shared `make_sklearn_dense_X()` helper for consistent NaN/dtype/inf handling
- **CatBoost configuration**: Auto-detects target type (classification vs regression) and sets appropriate loss function

## Quick Start

### Using Experiment Configs (Recommended)

The **preferred way** to use the intelligent training pipeline is with experiment configs. This keeps all settings in one file and prevents config "crossing" between modules:

**1. Create an experiment config** (`CONFIG/experiments/my_experiment.yaml`):
```yaml
experiment:
  name: my_experiment

data:
  data_dir: data/data_labeled/interval=5m
  symbols: [AAPL, MSFT, GOOGL]
  max_samples_per_symbol: 5000
  bar_interval: "5m"  # Explicit interval (prevents auto-detection warnings)

targets:
  primary: fwd_ret_60m

feature_selection:
  top_n: 100
  model_families: [lightgbm, xgboost]

training:
  model_families: [lightgbm, xgboost]
```

**Note on `bar_interval`**: Setting `data.bar_interval` explicitly prevents interval auto-detection warnings in both ranking and selection pipelines. The interval is used for horizon conversion and leakage filtering. Supported formats: `"5m"`, `"15m"`, `"1h"`, `"1d"`, or integer minutes (e.g., `5`).

**2. Run with experiment config:**
```bash
python TRAINING/train.py \
    --experiment-config my_experiment \
    --auto-targets \
    --top-n-targets 5 \
    --auto-features
```

**Benefits:**
- All settings in one file
- Type-safe configs with validation
- No config "crossing" between modules
- Easy to version and share

See [Modular Config System](../../02_reference/configuration/MODULAR_CONFIG_SYSTEM.md) for complete details.

### Basic Usage (Fully Automatic with CLI Args)

You can also use CLI arguments directly:

```bash
python TRAINING/train.py \
    --data-dir data/data_labeled/interval=5m \
    --symbols AAPL MSFT GOOGL \
    --auto-targets \
    --top-n-targets 5 \
    --auto-features \
    --top-m-features 100
```

This will:
1. Rank all available targets and select top 5
2. Select top 100 features for each target
3. Train all model families on the selected targets/features

### With Custom Model Families

```bash
python TRAINING/train.py \
    --data-dir data/data_labeled/interval=5m \
    --symbols AAPL MSFT GOOGL \
    --auto-targets \
    --top-n-targets 5 \
    --auto-features \
    --top-m-features 50 \
    --families LightGBM XGBoost MLP
```

### Manual Targets, Auto Features

```bash
python TRAINING/train.py \
    --data-dir data/data_labeled/interval=5m \
    --symbols AAPL MSFT \
    --targets fwd_ret_5m fwd_ret_15m \
    --auto-features \
    --top-m-features 50
```

### Faster E2E Testing

For faster end-to-end testing, use `--max-targets-to-evaluate` to limit the number of targets evaluated during ranking (this is different from `--top-n-targets`, which limits the final selection):

```bash
python TRAINING/train.py \
    --data-dir data/data_labeled/interval=5m \
    --symbols AAPL MSFT \
    --auto-targets \
    --top-n-targets 3 \
    --max-targets-to-evaluate 23 \
    --auto-features \
    --top-m-features 50
```

This will:
- Evaluate only 23 targets (instead of all discovered targets)
- Select top 3 from those evaluated
- Much faster for testing without running full ranking

### Using Cached Rankings (Faster)

```bash
python TRAINING/train.py \
    --data-dir data/data_labeled/interval=5m \
    --symbols AAPL MSFT GOOGL \
    --auto-targets \
    --top-n-targets 5 \
    --auto-features \
    --top-m-features 100 \
    --no-refresh-cache
```

## Step-by-Step Workflow

### Step 1: Target Ranking

The pipeline automatically discovers and ranks all available targets using multiple model families:

- **LightGBM** - Gradient boosting (handles NaNs natively)
- **XGBoost** - Gradient boosting (handles NaNs natively)
- **Random Forest** - Ensemble method (handles NaNs natively)
- **CatBoost** - Gradient boosting (handles NaNs natively, auto-detects classification vs regression)
- **Neural Network** - Deep learning (preprocessed for sklearn compatibility)
- **Lasso** - Linear model (uses shared sklearn preprocessing)
- **Mutual Information** - Statistical feature selection (uses shared sklearn preprocessing)
- **Univariate Selection** - F-test based selection (uses shared sklearn preprocessing)
- **Boruta** - Statistical gatekeeper (ExtraTrees-based, uses shared sklearn preprocessing, modifies consensus via bonuses/penalties)
- **Stability Selection** - Bootstrap-based selection (uses shared sklearn preprocessing)

**Ranking Criteria:**
- Cross-validated R²/ROC-AUC scores
- Feature importance magnitude
- Consistency across models
- Leakage detection flags

**Preprocessing Consistency:**
- **Tree-based models** (LightGBM, XGBoost, RF, CatBoost): Use raw data (handle NaNs natively)
- **Sklearn-based models** (Lasso, MI, Univariate, Boruta, Stability): Use `make_sklearn_dense_X()` helper for consistent preprocessing (dense float32, median imputation, inf handling)
- **CatBoost**: Auto-detects target type and sets `loss_function` appropriately (`Logloss` for binary, `MultiClass` for multiclass, `RMSE` for regression). YAML config can override if needed.

**Automatic Leakage Detection:**
The pipeline automatically detects and fixes data leakage:
- **Pre-training scan**: Detects near-copy features before model training
- **During training**: Detects perfect scores (≥99% CV, ≥99.9% training accuracy)
- **Auto-fixer**: Identifies leaking features and auto-updates exclusion configs
- **Configurable**: All thresholds configurable in `training_config/safety_config.yaml` (see [Safety & Leakage Configs](../../02_reference/configuration/SAFETY_LEAKAGE_CONFIGS.md))

**Configuration Options:**
- Pre-scan thresholds (min_match, min_corr)
- Feature count requirements (min_features_required, min_features_for_model)
- Warning thresholds (classification, regression)
- Auto-fixer settings (enabled, min_confidence, max_features_per_run)

See [Leakage Analysis](../../03_technical/research/LEAKAGE_ANALYSIS.md) for complete configuration details.

**Output:**
- `output_dir/target_rankings/target_predictability_rankings.csv` - Full rankings
- `output_dir/cache/target_rankings.json` - Cached results

**Example:**
```bash
# Rank targets and select top 5
python TRAINING/train.py \
    --data-dir data/data_labeled/interval=5m \
    --symbols AAPL MSFT GOOGL \
    --auto-targets \
    --top-n-targets 5
```

### Step 2: Feature Selection

For each selected target, the pipeline automatically selects the best features using multi-model consensus:

**Selection Method:**
- Trains multiple model families (LightGBM, XGBoost, Random Forest, CatBoost, Neural Network, Lasso, Mutual Information, Univariate Selection, Boruta, Stability Selection)
- Extracts feature importance (native/SHAP/permutation/coefficients)
- Aggregates importance across models and symbols
- **Boruta acts as statistical gatekeeper**: Excluded from base consensus, applied as modifier via bonuses/penalties (confirmed features get +0.2, rejected get -0.3, tentative neutral)
- Ranks features by consensus score

**Preprocessing Consistency:**
- Same preprocessing behavior as ranking: tree models use raw data, sklearn models use `make_sklearn_dense_X()` helper
- CatBoost uses same auto-detection logic as ranking
- Interval handling respects `data.bar_interval` from experiment config (same as ranking)

**Output:**
- `output_dir/feature_selections/{target}/selected_features.txt` - Feature list
- `output_dir/feature_selections/{target}/feature_importance_multi_model.csv` - Full rankings
- `output_dir/feature_selections/{target}/target_confidence.json` - Confidence metrics (HIGH/MEDIUM/LOW)
- `output_dir/feature_selections/{target}/target_routing.json` - Routing decision (core/candidate/experimental)
- `output_dir/cache/feature_selections/{target}.json` - Cached results

**Example:**
```bash
# Select top 100 features per target
python TRAINING/train.py \
    --data-dir data/data_labeled/interval=5m \
    --symbols AAPL MSFT GOOGL \
    --auto-targets \
    --top-n-targets 5 \
    --auto-features \
    --top-m-features 100
```

### Step 3: Model Training

The pipeline trains all selected model families on the selected targets with their selected features:

**Training Process:**
- Loads MTF data for all symbols
- Prepares cross-sectional datasets
- Trains each model family for each target
- Uses selected features per target (if provided)
- Saves models, metrics, and predictions

**Output:**
- `output_dir/training_results/` - Model artifacts, metrics, predictions
- Same structure as existing `train_with_strategies.py` output

**Example:**
```bash
# Full pipeline: ranking → selection → training
python TRAINING/train.py \
    --data-dir data/data_labeled/interval=5m \
    --symbols AAPL MSFT GOOGL \
    --auto-targets \
    --top-n-targets 5 \
    --auto-features \
    --top-m-features 100 \
    --families LightGBM XGBoost MLP
```

## Command-Line Arguments

### Required Arguments

- `--data-dir`: Data directory containing symbol data
- `--symbols`: List of symbols to train on

### Target Selection

- `--auto-targets`: Enable automatic target ranking (default: True)
- `--no-auto-targets`: Disable automatic target ranking
- `--top-n-targets`: Number of top targets to select (default: 5)
- `--max-targets-to-evaluate`: Limit number of targets to evaluate during ranking (default: evaluate all). Useful for faster E2E testing.
- `--targets`: Manual target list (overrides --auto-targets if provided)

### Feature Selection

- `--auto-features`: Enable automatic feature selection (default: True)
- `--no-auto-features`: Disable automatic feature selection
- `--top-m-features`: Number of top features per target (default: 100)
- `--features`: Manual feature list (overrides --auto-features if provided)

### Training

- `--families`: Model families to train (default: all enabled)
- `--strategy`: Training strategy - single_task, multi_task, cascade (default: single_task)
- `--output-dir`: Output directory (default: intelligent_output)

### Cache Control

- `--force-refresh`: Force refresh of cached rankings/selections
- `--no-refresh-cache`: Never refresh cache (use existing only)
- `--no-cache`: Disable caching entirely

### Data Limits (for testing)

- `--min-cs`: Minimum cross-sectional samples required (default: 10)
- `--max-rows-per-symbol`: Maximum rows to load per symbol
- `--max-rows-train`: Maximum training rows
- `--max-cs-samples`: Maximum cross-sectional samples per timestamp

### Config Files

- `--target-ranking-config`: Path to target ranking config YAML
- `--multi-model-config`: Path to feature selection config YAML

## Output Structure

**Note**: Output directories are automatically timestamped by default (format: `YYYYMMDD_HHMMSS`) to make runs distinguishable. For example, `intelligent_output` becomes `intelligent_output_20251208_143022`.

```
output_dir_YYYYMMDD_HHMMSS/
├── target_rankings/
│   ├── target_predictability_rankings.csv
│   └── feature_importances/
│       └── {target}/
│           └── {model}_importances.csv
├── feature_selections/
│   └── {target}/
│       ├── selected_features.txt
│       ├── feature_importance_multi_model.csv
│       ├── target_confidence.json
│       └── target_routing.json
├── target_confidence_summary.json  # Run-level summary (all targets)
├── target_confidence_summary.csv  # Human-readable summary table
├── training_results/
│   └── (model artifacts, metrics, predictions)
└── cache/
    ├── target_rankings.json
    └── feature_selections/
        └── {target}.json
```

To disable timestamping, use `add_timestamp=False` when initializing `IntelligentTrainer` programmatically.

## Caching Strategy

### Cache Benefits

- **Faster reruns**: Rankings/selections cached after first run
- **Incremental updates**: Re-rank targets periodically without full re-run
- **Cost savings**: Avoid expensive re-computation

### Cache Invalidation

- **Automatic**: Cache invalidated if symbols or configs change
- **Manual**: Use `--force-refresh` to force re-computation
- **Disable**: Use `--no-cache` to disable caching entirely

### Cache Keys

Cache keys are generated from:
- Symbol list
- Model families used
- Configuration hash

Same symbols + same configs = cache hit

## Examples

### Example 1: Quick Test Run

```bash
python TRAINING/train.py \
    --data-dir data/data_labeled/interval=5m \
    --symbols AAPL \
    --auto-targets \
    --top-n-targets 2 \
    --auto-features \
    --top-m-features 20 \
    --families LightGBM \
    --min-cs 3 \
    --max-rows-per-symbol 5000 \
    --max-rows-train 10000
```

**Use case**: Quick validation of pipeline functionality

### Example 2: Production Run

```bash
python TRAINING/train.py \
    --data-dir data/data_labeled/interval=5m \
    --symbols AAPL MSFT GOOGL AMZN TSLA \
    --auto-targets \
    --top-n-targets 10 \
    --auto-features \
    --top-m-features 100 \
    --families LightGBM XGBoost MLP Transformer \
    --strategy single_task \
    --output-dir production_training_results
```

**Use case**: Full production training run

### Example 3: Using Cached Results

```bash
# First run: Full computation
python TRAINING/train.py \
    --data-dir data/data_labeled/interval=5m \
    --symbols AAPL MSFT GOOGL \
    --auto-targets \
    --top-n-targets 5 \
    --auto-features \
    --top-m-features 100

# Second run: Uses cache (much faster)
python TRAINING/train.py \
    --data-dir data/data_labeled/interval=5m \
    --symbols AAPL MSFT GOOGL \
    --auto-targets \
    --top-n-targets 5 \
    --auto-features \
    --top-m-features 100 \
    --no-refresh-cache
```

**Use case**: Iterative development, testing different model families

## Integration with Existing Workflows

### Backward Compatibility

The intelligent training pipeline is **fully backward compatible** with existing workflows:

- **Manual targets/features**: Can still provide manual lists
- **Existing training pipeline**: Uses same `train_with_strategies.py` functions
- **Same output format**: Produces same model artifacts and metrics

### Migration Path

**Old workflow:**
```bash
# Step 1: Rank targets manually (deprecated)
python TRAINING/ranking/rank_target_predictability.py ...

# Step 2: Select features manually (deprecated)
python TRAINING/ranking/multi_model_feature_selection.py ...

# Step 3: Train with results
python TRAINING/train_with_strategies.py --targets ... --features ...
```

**New workflow:**
```bash
# All steps automated (recommended)
python TRAINING/train.py --auto-targets --auto-features
```

## Troubleshooting

### Issue: No targets selected

**Cause**: All targets filtered out (leakage, degenerate, etc.)

**Solution**: Check `output_dir/target_rankings/target_predictability_rankings.csv` for details

### Issue: Feature selection fails

**Cause**: Insufficient data or all features filtered

**Solution**: Check `output_dir/feature_selections/{target}/` for error logs

### Issue: Training fails

**Cause**: Data issues, memory limits, or model errors

**Solution**: Check training logs in `output_dir/training_results/`

### Issue: Cache not working

**Cause**: Cache key mismatch (symbols/configs changed)

**Solution**: Use `--force-refresh` to rebuild cache

## Best Practices

1. **Start small**: Test with 1-2 symbols and limited data first
2. **Use caching**: Enable caching for faster iterative development
3. **Monitor rankings**: Review target rankings to understand what's being selected
4. **Check features**: Verify selected features make sense for your targets
5. **Incremental updates**: Re-rank targets periodically (weekly/monthly)
6. **Production runs**: Use `--no-cache` for production to ensure fresh results

## Target Confidence and Routing

The pipeline automatically assesses target quality and routes targets into operational buckets:

**Confidence Metrics:**
- **Boruta coverage**: Number of confirmed/tentative/rejected features
- **Model coverage**: Ratio of successful models to available models
- **Score strength**: Mean/max scores, plus mean_strong_score (tree ensembles + CatBoost + NN)
- **Agreement ratio**: Fraction of top-K features appearing in ≥2 models
- **Score tier**: Orthogonal metric for signal strength (HIGH/MEDIUM/LOW)

**Confidence Buckets:**
- **HIGH**: Strong, robust signal with good agreement and Boruta support
- **MEDIUM**: Some signal present but not fully robust
- **LOW**: Weak signal or structural issues (with specific reason)

**Operational Routing:**
- **core**: Production-ready (HIGH confidence)
- **candidate**: Worth trying (MEDIUM confidence with decent scores)
- **experimental**: Fragile signal (LOW confidence, especially boruta_zero_confirmed)

**Configuration:**
All thresholds and routing rules are configurable in `CONFIG/feature_selection/multi_model.yaml` under the `confidence` section. See [Feature & Target Configs](../../02_reference/configuration/FEATURE_TARGET_CONFIGS.md#target-confidence--routing) for details.

**Output:**
- Per-target: `target_confidence.json`, `target_routing.json`
- Run-level: `target_confidence_summary.json`, `target_confidence_summary.csv` (human-readable table)

## Related Documentation

- [Ranking and Selection Consistency](RANKING_SELECTION_CONSISTENCY.md) - **NEW**: Unified pipeline behavior (interval handling, sklearn preprocessing, CatBoost configuration)
- [Model Training Guide](MODEL_TRAINING_GUIDE.md) - Manual training workflow
- [Feature Selection Tutorial](FEATURE_SELECTION_TUTORIAL.md) - Manual feature selection
- [Modular Config System](../../02_reference/configuration/MODULAR_CONFIG_SYSTEM.md) - Complete config system guide (includes `logging_config.yaml`)
- [Usage Examples](../../02_reference/configuration/USAGE_EXAMPLES.md) - Practical configuration examples
- [Intelligent Trainer API](../../02_reference/api/INTELLIGENT_TRAINER_API.md) - Complete API reference
- [CLI Reference](../../02_reference/api/CLI_REFERENCE.md) - Complete CLI documentation
- [Target Discovery](../../03_technical/research/TARGET_DISCOVERY.md) - Target research
- [Feature Importance Methodology](../../03_technical/research/FEATURE_IMPORTANCE_METHODOLOGY.md) - Feature importance research

