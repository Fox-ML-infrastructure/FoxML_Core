# CLI Reference

Command-line interface reference for FoxML Core.

> **Important**: FoxML Core follows **Single Source of Truth (SST)** principles. Most settings come from config files, not CLI arguments. CLI is primarily for:
> - Required inputs (data paths, symbols)
> - Config file selection (`--experiment-config`)
> - Operational flags (resume, dry-run, force-refresh)
>
> See [CLI vs Config Separation](../../03_technical/design/CLI_CONFIG_SEPARATION.md) for the complete policy.

## Intelligent Training Pipeline

### Main Training Script

The intelligent training pipeline automates target ranking, feature selection, and model training in a single command.

**Preferred Usage (Config-Driven):**

```bash
# Use experiment config (all settings from config)
python -m TRAINING.orchestration.intelligent_trainer \
    --experiment-config my_experiment

# Or provide minimal inputs (uses default configs)
python -m TRAINING.orchestration.intelligent_trainer \
    --data-dir data/data_labeled/interval=5m \
    --symbols AAPL MSFT GOOGL
```

**Required Options:**
- `--data-dir`: Data directory containing symbol data (required unless `--experiment-config` provided)
- `--symbols`: List of symbols to train on (required unless `--experiment-config` provided)

**Config File Selection (PREFERRED):**
- `--experiment-config`: Experiment config name (without .yaml) from `CONFIG/experiments/` [PREFERRED - all settings from config]
- `--target-ranking-config`: Path to target ranking config YAML [LEGACY]
- `--multi-model-config`: Path to feature selection config YAML [LEGACY]

**Manual Overrides (Use Sparingly):**
- `--targets`: Manual target list (overrides config `auto_targets`)
- `--features`: Manual feature list (overrides config `auto_features`)
- `--families`: Model families to train (overrides config)

**Operational Flags:**
- `--output-dir`: Output directory (default: `intelligent_output`). Automatically timestamped by default.
- `--cache-dir`: Cache directory for rankings/selections (default: `output_dir/cache`)
- `--force-refresh`: Force refresh of cached rankings/selections
- `--no-refresh-cache`: Never refresh cache (use existing only)
- `--no-cache`: Disable caching entirely

**Testing Overrides (Not SST Compliant - Use Only for Testing):**
- `--override-max-samples`: Override max samples per symbol (testing only, logs warning)
- `--override-max-rows`: Override max rows per symbol (testing only, logs warning)

**Configuration Settings (All in Config Files):**

All of these settings are now in config files, not CLI:

- **Target Selection**: `intelligent_training.auto_targets`, `intelligent_training.top_n_targets`, `intelligent_training.max_targets_to_evaluate` (in `pipeline_config.yaml` or experiment config)
- **Feature Selection**: `intelligent_training.auto_features`, `intelligent_training.top_m_features` (in `pipeline_config.yaml` or experiment config)
- **Training Strategy**: `intelligent_training.strategy` (in `pipeline_config.yaml` or experiment config)
- **Data Limits**: `pipeline.data_limits.*` (in `pipeline_config.yaml` or experiment config)
  - `max_samples_per_symbol`
  - `max_rows_per_symbol`
  - `max_rows_train`
  - `min_cs`
  - `max_cs_samples`
- **Model Families**: `model_families` in config files
- **Sequential Settings**: `pipeline.sequential.*` (backend, lookback, etc.)

See [CLI vs Config Separation](../../03_technical/design/CLI_CONFIG_SEPARATION.md) for complete details.

**Note**: The training system uses modular components internally (`TRAINING/training_strategies/`, `TRAINING/ranking/predictability/`) but all CLI commands and imports remain backward compatible. The original files are thin wrappers that re-export from the modular structure.

**Examples:**

```bash
# Preferred: Use experiment config (all settings from config)
python -m TRAINING.orchestration.intelligent_trainer \
    --experiment-config my_experiment

# Minimal inputs (uses default configs from pipeline_config.yaml)
python -m TRAINING.orchestration.intelligent_trainer \
    --data-dir data/data_labeled/interval=5m \
    --symbols AAPL MSFT GOOGL

# Manual target override (overrides config auto_targets)
python -m TRAINING.orchestration.intelligent_trainer \
    --data-dir data/data_labeled/interval=5m \
    --symbols AAPL MSFT \
    --targets fwd_ret_5m fwd_ret_15m

# Use cached results (faster)
python -m TRAINING.orchestration.intelligent_trainer \
    --experiment-config my_experiment \
    --no-refresh-cache

# Testing override (not SST compliant - logs warning)
python -m TRAINING.orchestration.intelligent_trainer \
    --experiment-config my_experiment \
    --override-max-samples 5000
    --top-n-targets 5 \
    --max-targets-to-evaluate 23
```

**See Also:**
- [Intelligent Training Tutorial](../../01_tutorials/training/INTELLIGENT_TRAINING_TUTORIAL.md) - Complete tutorial
- [Ranking and Selection Consistency](../../01_tutorials/training/RANKING_SELECTION_CONSISTENCY.md) - Unified pipeline behavior guide
- [Model Training Guide](../../01_tutorials/training/MODEL_TRAINING_GUIDE.md) - Manual training workflow
- [Modular Config System](../configuration/MODULAR_CONFIG_SYSTEM.md) - Config system guide (includes `logging_config.yaml`)
- [Usage Examples](../configuration/USAGE_EXAMPLES.md) - Practical examples

## Feature Selection

### Comprehensive Feature Ranking

```bash
python SCRIPTS/rank_features_comprehensive.py \
    --symbols AAPL,MSFT,GOOGL,TSLA,JPM \
    --target y_will_peak_60m_0.8 \
    --output-dir results/feature_ranking
```

**Options:**
- `--symbols`: Comma-separated list of symbols
- `--target`: Target column name (optional, for predictive ranking)
- `--output-dir`: Output directory for results

### Target Predictability Ranking

**Note**: These standalone scripts have been integrated into the intelligent training pipeline. Use the automated workflow instead:

```bash
# Automated target ranking (recommended)
python TRAINING/train.py \
    --data-dir data/data_labeled/interval=5m \
    --symbols AAPL MSFT GOOGL TSLA JPM \
    --auto-targets \
    --top-n-targets 10 \
    --output-dir results/target_rankings
```

**Legacy standalone script** (deprecated, use intelligent training pipeline):
```bash
# OLD WAY - Still works but not recommended
python -m TRAINING.ranking.predictability.main \
    --symbols AAPL,MSFT,GOOGL,TSLA,JPM \
    --output-dir results/target_rankings
```

### Multi-Model Feature Selection

**Note**: Feature selection is now automated in the intelligent training pipeline. Use the automated workflow instead:

```bash
# Automated feature selection (recommended)
python TRAINING/train.py \
    --data-dir data/data_labeled/interval=5m \
    --symbols AAPL MSFT \
    --targets y_will_peak_60m_0.8 \
    --auto-features \
    --top-m-features 60 \
    --output-dir results/feature_selection
```

**Legacy standalone script** (deprecated, use intelligent training pipeline):
```bash
# OLD WAY - Still works but not recommended
python TRAINING/ranking/multi_model_feature_selection.py \
    --target-column y_will_peak_60m_0.8 \
    --top-n 60 \
    --output-dir results/multi_model_selection
```

## Data Processing

### List Available Symbols

```bash
python SCRIPTS/list_available_symbols.py
```

### Remove Targets from Checkpoint

```bash
python SCRIPTS/remove_targets_from_checkpoint.py \
    --checkpoint models/checkpoint.pkl \
    --targets target1,target2
```

## Alpaca Trading

### Paper Trading Runner

```bash
python ALPACA_trading/SCRIPTS/paper_runner.py
```

### CLI Commands

```bash
# Check status
python ALPACA_trading/cli/paper.py status

# View positions
python ALPACA_trading/cli/paper.py positions

# View performance
python ALPACA_trading/cli/paper.py performance
```

## IBKR Trading

### Run Trading System

```bash
cd IBKR_trading
python run_trading_system.py
```

### Test Daily Models

```bash
python IBKR_trading/test_daily_models.py
```

### Comprehensive Testing

```bash
./IBKR_trading/test_all_models_comprehensive.sh
```

## See Also

- [Module Reference](MODULE_REFERENCE.md) - Python API (includes utility modules)
- [Intelligent Trainer API](INTELLIGENT_TRAINER_API.md) - Intelligent training pipeline API
- [Config Schema](CONFIG_SCHEMA.md) - Configuration reference
- [Config Loader API](../configuration/CONFIG_LOADER_API.md) - Configuration loading
- [Ranking and Selection Consistency](../../01_tutorials/training/RANKING_SELECTION_CONSISTENCY.md) - Unified pipeline behavior guide

