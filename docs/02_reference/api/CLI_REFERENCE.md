# CLI Reference

Command-line interface reference for FoxML Core.

## Intelligent Training Pipeline

### Main Training Script

The intelligent training pipeline automates target ranking, feature selection, and model training in a single command.

```bash
python TRAINING/train.py \
    --data-dir data/data_labeled/interval=5m \
    --symbols AAPL MSFT GOOGL \
    --auto-targets \
    --top-n-targets 5 \
    --auto-features \
    --top-m-features 100 \
    --families LightGBM XGBoost MLP
```

**Required Options:**
- `--data-dir`: Data directory containing symbol data
- `--symbols`: List of symbols to train on

**Target Selection:**
- `--auto-targets`: Enable automatic target ranking (default: True)
- `--no-auto-targets`: Disable automatic target ranking
- `--top-n-targets`: Number of top targets to select (default: 5)
- `--targets`: Manual target list (overrides --auto-targets)

**Feature Selection:**
- `--auto-features`: Enable automatic feature selection (default: True)
- `--no-auto-features`: Disable automatic feature selection
- `--top-m-features`: Number of top features per target (default: 100)
- `--features`: Manual feature list (overrides --auto-features)

**Training:**
- `--families`: Model families to train (default: all)
- `--strategy`: Training strategy - single_task, multi_task, cascade (default: single_task)
- `--output-dir`: Output directory (default: intelligent_output)

**Cache Control:**
- `--force-refresh`: Force refresh of cached rankings/selections
- `--no-refresh-cache`: Never refresh cache (use existing only)
- `--no-cache`: Disable caching entirely

**Data Limits (for testing):**
- `--min-cs`: Minimum cross-sectional samples (default: 10)
- `--max-rows-per-symbol`: Maximum rows to load per symbol
- `--max-rows-train`: Maximum training rows
- `--max-cs-samples`: Maximum cross-sectional samples per timestamp

**Config Files:**
- `--target-ranking-config`: Path to target ranking config YAML
- `--multi-model-config`: Path to feature selection config YAML

**Examples:**

```bash
# Fully automatic (defaults)
python TRAINING/train.py \
    --data-dir data/data_labeled/interval=5m \
    --symbols AAPL MSFT GOOGL

# Manual targets, auto features
python TRAINING/train.py \
    --data-dir data/data_labeled/interval=5m \
    --symbols AAPL MSFT \
    --targets fwd_ret_5m fwd_ret_15m \
    --auto-features \
    --top-m-features 50

# Use cached results (faster)
python TRAINING/train.py \
    --data-dir data/data_labeled/interval=5m \
    --symbols AAPL MSFT GOOGL \
    --auto-targets \
    --top-n-targets 5 \
    --no-refresh-cache
```

**See Also:**
- [Intelligent Training Tutorial](../../01_tutorials/training/INTELLIGENT_TRAINING_TUTORIAL.md) - Complete tutorial
- [Model Training Guide](../../01_tutorials/training/MODEL_TRAINING_GUIDE.md) - Manual training workflow

## Feature Selection

### Comprehensive Feature Ranking

```bash
python scripts/rank_features_comprehensive.py \
    --symbols AAPL,MSFT,GOOGL,TSLA,JPM \
    --target y_will_peak_60m_0.8 \
    --output-dir results/feature_ranking
```

**Options:**
- `--symbols`: Comma-separated list of symbols
- `--target`: Target column name (optional, for predictive ranking)
- `--output-dir`: Output directory for results

### Target Predictability Ranking

```bash
python scripts/rank_target_predictability.py \
    --symbols AAPL,MSFT,GOOGL,TSLA,JPM \
    --output-dir results/target_rankings
```

**Options:**
- `--symbols`: Comma-separated list of symbols
- `--model-families`: Model families to test (default: all)
- `--output-dir`: Output directory

### Multi-Model Feature Selection

```bash
python scripts/multi_model_feature_selection.py \
    --target-column y_will_peak_60m_0.8 \
    --top-n 60 \
    --output-dir results/multi_model_selection
```

## Data Processing

### List Available Symbols

```bash
python scripts/list_available_symbols.py
```

### Remove Targets from Checkpoint

```bash
python scripts/remove_targets_from_checkpoint.py \
    --checkpoint models/checkpoint.pkl \
    --targets target1,target2
```

## Alpaca Trading

### Paper Trading Runner

```bash
python ALPACA_trading/scripts/paper_runner.py
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

- [Module Reference](MODULE_REFERENCE.md) - Python API
- [Config Schema](CONFIG_SCHEMA.md) - Configuration reference

