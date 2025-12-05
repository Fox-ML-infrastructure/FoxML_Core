# CLI Reference

Command-line interface reference for Fox-v1-infra.

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

