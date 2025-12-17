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
[3] Training Plan Generation → Routing decisions → Training jobs (NEW)
  ↓
[4] Model Training → Train models using plan (2-stage: CPU → GPU) (NEW)
  ↓
Trained Models
```

**Benefits:**
- **No manual steps**: Everything automated in one command
- **Intelligent selection**: Multi-model consensus for ranking/selection
- **GPU acceleration**: Target ranking and feature selection automatically use GPU (LightGBM, XGBoost, CatBoost) when available (NEW 2025-12-12)
- **Training routing**: Config-driven decisions about where to train (cross-sectional vs symbol-specific)
- **2-stage training**: Efficient CPU → GPU resource usage (all 20 models)
- **Plan-aware filtering**: Only approved targets and model families are trained
- **Cached results**: Rankings/selections cached for faster reruns
- **Leakage-free**: All existing safeguards preserved
- **Unified behavior**: Ranking and selection use consistent preprocessing and configuration
- **Reproducibility**: All parameters load from config (Single Source of Truth). Same config → same results across all pipeline stages (reproducibility ensured when using proper configs).

**Architecture Note**: The pipeline uses a modular structure internally (`TRAINING/training_strategies/`, `TRAINING/ranking/predictability/`, `TRAINING/models/specialized/`) but maintains 100% backward compatibility. All existing imports and workflows continue to work unchanged - this is purely an internal organization improvement for better maintainability.

**Pipeline Consistency:**
- **Interval handling**: Both ranking and selection respect `data.bar_interval` from experiment config (no spurious auto-detection warnings)
- **Sklearn preprocessing**: All sklearn-based models use shared `make_sklearn_dense_X()` helper for consistent NaN/dtype/inf handling
- **CatBoost configuration**: Auto-detects target type (classification vs regression) and sets appropriate loss function

## Quick Start

> **Important**: FoxML Core follows **Single Source of Truth (SST)** principles. Most settings come from config files, not CLI arguments. See [CLI vs Config Separation](../../03_technical/design/CLI_CONFIG_SEPARATION.md) for the complete policy.

### Config-Driven Approach (Recommended)

All settings are now in config files. CLI only provides:
- **Required inputs**: `--data-dir`, `--symbols`
- **Config selection**: `--experiment-config` (preferred)
- **Operational flags**: `--force-refresh`, `--no-cache`
- **Manual overrides**: `--targets`, `--features`, `--families` (explicit choices)

### Method 1: Using Default Config (Simplest)

Just provide data and symbols - all other settings come from `CONFIG/training_config/pipeline_config.yaml`:

```bash
python -m TRAINING.orchestration.intelligent_trainer \
    --data-dir "data/data_labeled/interval=5m" \
    --symbols AAPL MSFT GOOGL \
    --output-dir "my_training_run"
```

**Default settings** (from `pipeline_config.yaml`):
- `top_n_targets: 5`
- `top_m_features: 100`
- `min_cs: 10`
- `auto_targets: true`
- `auto_features: true`
- `strategy: single_task`

### Method 2: Test Config Auto-Detection (For Testing)

If your `--output-dir` contains "test" (case-insensitive), the system automatically uses test-friendly settings from `pipeline_config.yaml` → `test.intelligent_training`:

```bash
python -m TRAINING.orchestration.intelligent_trainer \
    --data-dir "data/data_labeled/interval=5m" \
    --symbols AAPL MSFT GOOGL TSLA NVDA \
    --output-dir "test_e2e_ranking_unified" \
    --families lightgbm xgboost random_forest catboost neural_network lasso mutual_information univariate_selection
```

**Test config settings** (automatically applied):
- `top_n_targets: 23`
- `max_targets_to_evaluate: 23`
- `top_m_features: 50`
- `min_cs: 3`
- `max_rows_per_symbol: 5000`
- `max_rows_train: 10000`

**Note**: The system detects "test" in the output directory name and automatically switches to test config. No CLI arguments needed for these settings!

### Method 3: Customize Config File

Edit `CONFIG/training_config/pipeline_config.yaml` to change defaults:

```yaml
intelligent_training:
  top_n_targets: 10  # Change default
  top_m_features: 50  # Change default
  min_cs: 5  # Change default
  # ... other settings
```

Then run with minimal CLI:
```bash
python -m TRAINING.orchestration.intelligent_trainer \
    --data-dir "data/data_labeled/interval=5m" \
    --symbols AAPL MSFT GOOGL
```

### Method 4: Using Experiment Configs (For Complex Experiments)

For experiments with many custom settings, create an experiment config:

**1. Create experiment config** (`CONFIG/experiments/my_experiment.yaml`):
```yaml
experiment:
  name: my_experiment
  description: "Custom experiment with specific settings"

data:
  data_dir: data/data_labeled/interval=5m
  symbols: [AAPL, MSFT, GOOGL]
  max_samples_per_symbol: 5000
  bar_interval: "5m"

targets:
  primary: fwd_ret_60m  # Required for experiment configs

feature_selection:
  top_n: 100
  model_families: [lightgbm, xgboost]

training:
  model_families: [lightgbm, xgboost]
```

**2. Run with experiment config:**
```bash
python -m TRAINING.orchestration.intelligent_trainer \
    --experiment-config my_experiment \
    --output-dir "my_experiment_results"
```

**Note**: Experiment configs require a `targets.primary` field. For auto-target selection, use Method 1, 2, or 3 instead.

### Manual Overrides (Allowed)

You can still override specific choices via CLI (these are explicit decisions, not config values):

```bash
# Override model families
python -m TRAINING.orchestration.intelligent_trainer \
    --data-dir "data/data_labeled/interval=5m" \
    --symbols AAPL MSFT GOOGL \
    --families lightgbm xgboost random_forest

# Override targets (manual selection)
python -m TRAINING.orchestration.intelligent_trainer \
    --data-dir "data/data_labeled/interval=5m" \
    --symbols AAPL MSFT \
    --targets fwd_ret_5m fwd_ret_15m

# Override features (manual selection)
python -m TRAINING.orchestration.intelligent_trainer \
    --data-dir "data/data_labeled/interval=5m" \
    --symbols AAPL MSFT \
    --features feature1 feature2 feature3
```

### Configuration File Locations

All settings are in config files:

- **Pipeline settings**: `CONFIG/training_config/pipeline_config.yaml`
  - `intelligent_training.*` - Intelligent trainer settings
  - `test.intelligent_training.*` - Test-friendly defaults (auto-detected)
  - `pipeline.data_limits.*` - Data sampling limits
- **Experiment configs**: `CONFIG/experiments/*.yaml` (for complex experiments)
- **Model configs**: `CONFIG/model_config/*.yaml` (model hyperparameters)
- **Feature selection**: `CONFIG/feature_selection/multi_model.yaml`

See [Config Basics](../configuration/CONFIG_BASICS.md) and [CLI vs Config Separation](../../03_technical/design/CLI_CONFIG_SEPARATION.md) for details.

### Using Cached Rankings (Faster)

```bash
python -m TRAINING.orchestration.intelligent_trainer \
    --data-dir "data/data_labeled/interval=5m" \
    --symbols AAPL MSFT GOOGL \
    --output-dir "my_training_run" \
    --no-refresh-cache
```

**Note**: All settings (top_n_targets, top_m_features, etc.) come from config. The `--no-refresh-cache` flag prevents re-running ranking/selection if cached results exist.

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

![Feature selection and stability analysis](../../images/feature_selection_stability.png)

*Feature selection showing multi-model consensus, importance analysis, and stability tracking*

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
- Same structure as existing training pipeline output
- Uses modular `TRAINING/training_strategies/` components internally

**Example:**
```bash
# Full pipeline: ranking → selection → training
# All settings from config (SST compliant)
python -m TRAINING.orchestration.intelligent_trainer \
    --data-dir "data/data_labeled/interval=5m" \
    --symbols AAPL MSFT GOOGL \
    --output-dir "full_pipeline_run" \
    --families lightgbm xgboost mlp
```

**Note**: `top_n_targets`, `top_m_features`, `min_cs`, etc. all come from config. Only `--families` is specified as a manual override (explicit choice).

## Command-Line Arguments

> **Note**: Most settings are now in config files. CLI only provides inputs, config selection, and operational flags. See [CLI vs Config Separation](../../03_technical/design/CLI_CONFIG_SEPARATION.md) for details.

### Required Arguments

- `--data-dir`: Data directory containing symbol data (required unless `--experiment-config` provided)
- `--symbols`: List of symbols to train on (required unless `--experiment-config` provided)

### Config File Selection (PREFERRED)

- `--experiment-config`: Experiment config name (without .yaml) from `CONFIG/experiments/` [PREFERRED - all settings from config]
- `--target-ranking-config`: Path to target ranking config YAML [LEGACY]
- `--multi-model-config`: Path to feature selection config YAML [LEGACY]

### Manual Overrides (Allowed - Explicit Choices)

- `--targets`: Manual target list (overrides config `auto_targets`)
- `--features`: Manual feature list (overrides config `auto_features`)
- `--families`: Model families to train (overrides config)

### Operational Flags

- `--output-dir`: Output directory (default: `intelligent_output`). Automatically timestamped by default.
- `--cache-dir`: Cache directory for rankings/selections (default: `output_dir/cache`)
- `--force-refresh`: Force refresh of cached rankings/selections
- `--no-refresh-cache`: Never refresh cache (use existing only)
- `--no-cache`: Disable caching entirely

### Testing Overrides (Not SST Compliant - Use Only for Testing)

- `--override-max-samples`: Override max samples per symbol (testing only, logs warning)
- `--override-max-rows`: Override max rows per symbol (testing only, logs warning)

### Configuration Settings (All in Config Files)

All of these settings are now in config files, not CLI:

- **Target Selection**: `intelligent_training.auto_targets`, `intelligent_training.top_n_targets`, `intelligent_training.max_targets_to_evaluate` (in `pipeline_config.yaml`)
- **Feature Selection**: `intelligent_training.auto_features`, `intelligent_training.top_m_features` (in `pipeline_config.yaml`)
- **Training Strategy**: `intelligent_training.strategy` (in `pipeline_config.yaml`)
- **Data Limits**: `pipeline.data_limits.*` or `intelligent_training.*` (in `pipeline_config.yaml`)
  - `max_rows_per_symbol`
  - `max_rows_train`
  - `min_cs`
  - `max_cs_samples`
- **Test Config**: `test.intelligent_training.*` (auto-detected when output-dir contains "test")

See [Config Basics](../configuration/CONFIG_BASICS.md) for how to modify these settings.

## Output Structure

**Note**: All runs are automatically organized in the `RESULTS/runs/` directory by comparison group metadata (dataset, task, routing, model family, feature set, split protocol). Output directories are automatically timestamped by default (format: `YYYYMMDD_HHMMSS`) to make runs distinguishable.

**Directory Organization**:
```
RESULTS/
└── runs/                                # All runs organized by comparison group
    └── cg-{hash}_n-{size}_fam-{family}/  # Comparison group directory
        └── {run_name}_YYYYMMDD_HHMMSS/
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
        ├── training_results/
        │   └── (model artifacts, metrics, predictions)
        ├── backups/                    # Config backups (organized by target)
        │   └── {target}/
        │       └── {timestamp}/
        │           ├── excluded_features.yaml
        │           ├── feature_registry.yaml
        │           └── manifest.json
        ├── REPRODUCIBILITY/            # Cohort-aware reproducibility tracking
        │   ├── TARGET_RANKING/
        │   │   └── {target}/
        │   │       └── cohort={cohort_id}/
        │   │           ├── metadata.json  # Includes symbols list, bin info
        │   │           ├── metrics.json
        │   │           └── drift.json
        │   ├── FEATURE_SELECTION/
        │   ├── TRAINING/
        │   ├── index.parquet
        │   └── stats.json
        └── cache/
            ├── target_rankings.json
            └── feature_selections/
                └── {target}.json
```

**Comparison Group Organization:**
- Runs are organized by **all outcome-influencing metadata** (dataset, task, routing, model family, feature set, split protocol)
- Directory format: `cg-{hash}_n-{sample_size}_fam-{model_family}`
  - `cg-{hash}`: 12-character hash of full comparison group key
  - `n-{sample_size}`: Human-readable sample size (e.g., `n-5000`, `n-25000`)
  - `fam-{model_family}`: Model family (e.g., `fam-lightgbm`, `fam-xgboost`)
- Example: `cg-abc123def456_n-5000_fam-lightgbm/`

**Benefits**:
- **Strict comparability**: Only runs with identical outcome-influencing metadata are grouped together
- **Audit-grade**: Comparison group key includes fingerprints for config, data, features, targets, and split protocol
- **Prevents fold drift**: Split protocol signature includes `split_seed` and `fold_assignment_hash`
- **Human navigable**: Directory names include sample size and model family for quick identification
- **Clean structure**: All runs under `RESULTS/runs/` keeps root directory clean
- **Early organization**: Comparison group computed at startup, runs created directly in final location

**Finding Runs**:
- **By comparison group**: `ls RESULTS/runs/cg-*/`
- **By sample size**: `ls RESULTS/runs/*_n-5000_*/`
- **By model family**: `ls RESULTS/runs/*_fam-lightgbm/`
- **By exact metadata**: Query `REPRODUCIBILITY/index.parquet`
- **By date**: Query `REPRODUCIBILITY/index.parquet` filtered by `created_at`
- **By cohort**: Query `REPRODUCIBILITY/index.parquet` filtered by `cohort_id`

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
- **Existing training pipeline**: Uses same `training_strategies/` module functions (backward compatible with `train_with_strategies.py` imports)
- **Same output format**: Produces same model artifacts and metrics

### Migration Path

**Old workflow:**
```bash
# Step 1: Rank targets manually (deprecated)
python -m TRAINING.ranking.predictability.main ...

# Step 2: Select features manually (deprecated)
python TRAINING/ranking/multi_model_feature_selection.py ...

# Step 3: Train with results
python -m TRAINING.training_strategies.main --targets ... --features ...
```

**Note**: The training system uses a modular structure internally (`TRAINING/training_strategies/`, `TRAINING/ranking/predictability/`, `TRAINING/models/specialized/`) but maintains 100% backward compatibility. The original files (`rank_target_predictability.py`, `train_with_strategies.py`, `specialized_models.py`) are now thin wrappers that re-export everything from the modular components. All existing imports continue to work unchanged - this is purely an internal organization improvement.

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
All thresholds and routing rules are configurable in `CONFIG/feature_selection/multi_model.yaml` under the `confidence` section. See [Feature & Target Configs](../../02_reference/configuration/FEATURE_TARGET_CONFIGS.md) for details.

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

