# Simple Pipeline Usage Guide

The intelligent training pipeline can now be run with minimal command-line arguments by using configuration files.

**Pipeline Overview:** The system runs a 3-stage pipeline: (1) **Target Ranking** - ranks targets by predictability, (2) **Feature Selection** - selects optimal features per target, (3) **Training** - trains models with automatic routing decisions. Each stage evaluates targets in both **cross-sectional** (pooled across symbols) and **symbol-specific** (per-symbol) views for comprehensive analysis.

## Quick Start

### 1. Configure Once

Edit `CONFIG/pipeline/training/intelligent.yaml`:

```yaml
data:
  data_dir: "data/data_labeled/interval=5m"
  symbols:
    - AAPL
    - MSFT
    - GOOGL
    - TSLA
    - NVDA

targets:
  auto_targets: true
  top_n_targets: 10

features:
  auto_features: true
  top_m_features: 100

model_families:
  - lightgbm
  - xgboost
  - random_forest
  - catboost
  - neural_network
  - lasso
  - mutual_information
  - univariate_selection
```

### 2. Run Simple Command

```bash
python -m TRAINING.orchestration.intelligent_trainer \
    --output-dir "my_experiment"
```

That's it! All settings come from the config file.

## Command-Line Options

### Minimal (Recommended)

```bash
# Use all settings from config
python -m TRAINING.orchestration.intelligent_trainer \
    --output-dir "my_experiment"
```

### Quick Test Mode

```bash
# Fast test: 3 targets, 50 features
python -m TRAINING.orchestration.intelligent_trainer \
    --output-dir "test_run" \
    --quick
```

### Override Specific Settings

```bash
# Override data directory and symbols
python -m TRAINING.orchestration.intelligent_trainer \
    --output-dir "my_experiment" \
    --data-dir "data/data_labeled/interval=1m" \
    --symbols AAPL MSFT
```

### Manual Target/Feature Selection

```bash
# Use specific targets instead of auto-selection
python -m TRAINING.orchestration.intelligent_trainer \
    --output-dir "my_experiment" \
    --targets fwd_ret_5m fwd_ret_15m y_will_peak_60m_0.8
```

### Override Model Families

```bash
# Use only specific model families
python -m TRAINING.orchestration.intelligent_trainer \
    --output-dir "my_experiment" \
    --families lightgbm xgboost random_forest
```

## Configuration File Structure

The config file (`CONFIG/pipeline/training/intelligent.yaml`) supports:

### Data Configuration
- `data.data_dir`: Default data directory
- `data.symbols`: Default symbol list
- `data.max_rows_per_symbol`: Limit rows per symbol (null = no limit)
- `data.max_cs_samples`: Max cross-sectional samples per timestamp
- `data.min_cs`: Minimum cross-sectional samples per timestamp

### Target Selection
- `targets.auto_targets`: Enable automatic target ranking/selection
- `targets.top_n_targets`: Number of top targets to select
- `targets.max_targets_to_evaluate`: Limit evaluation (for faster testing)
- `targets.manual_targets`: Manual target list (overrides auto_targets)

### Feature Selection
- `features.auto_features`: Enable automatic feature selection
- `features.top_m_features`: Number of top features per target
- `features.manual_features`: Manual feature list (overrides auto_features)

### Model Families
- `model_families`: List of model families to train
  - Options: `lightgbm`, `xgboost`, `random_forest`, `catboost`, `neural_network`, `lasso`, `mutual_information`, `univariate_selection`

### Training Strategy
- `strategy`: Training strategy (`single_task`, `multi_task`, `ensemble`)

### Output & Cache
- `output.output_dir`: Default output directory
- `output.cache_dir`: Cache directory (null = auto-generate)
- `cache.use_cache`: Enable caching
- `cache.force_refresh`: Force refresh of cached results

### Advanced Options
- `advanced.run_leakage_diagnostics`: Run leakage diagnostics
- `advanced.enable_dual_view_ranking`: Enable dual-view target ranking
- `advanced.enable_loso`: Enable LOSO evaluation

### Test Mode
- `test.*`: Override settings when 'test' is in output_dir name

## Examples

### Full Production Run

```bash
# Edit config file first, then:
python -m TRAINING.orchestration.intelligent_trainer \
    --output-dir "production_run_2025_01_15"
```

### Quick Test

```bash
python -m TRAINING.orchestration.intelligent_trainer \
    --output-dir "test_quick" \
    --quick
```

### Custom Experiment

```bash
python -m TRAINING.orchestration.intelligent_trainer \
    --output-dir "custom_experiment" \
    --data-dir "data/data_labeled/interval=15m" \
    --symbols AAPL MSFT GOOGL \
    --targets fwd_ret_5m fwd_ret_15m \
    --families lightgbm xgboost
```

## Priority Order

Settings are applied in this order (later overrides earlier):

1. **Config file defaults** (`intelligent_training_config.yaml`)
2. **Test mode overrides** (if 'test' in output_dir)
3. **CLI arguments** (highest priority)

## Migration from Old Commands

### Old Way (Still Works)

```bash
python -m TRAINING.orchestration.intelligent_trainer \
    --data-dir "data/data_labeled/interval=5m" \
    --symbols AAPL MSFT GOOGL TSLA NVDA \
    --output-dir "test_e2e_ranking_unified" \
    --families lightgbm xgboost random_forest catboost neural_network lasso mutual_information univariate_selection
```

### New Way (Recommended)

1. Edit `CONFIG/pipeline/training/intelligent.yaml`
2. Run: `python -m TRAINING.orchestration.intelligent_trainer --output-dir "test_e2e_ranking_unified"`

## Tips

- **Use config files for production**: Keeps settings version-controlled and reproducible
- **Use CLI overrides for testing**: Quick experiments without editing config
- **Use `--quick` for fast iteration**: Limits evaluation for faster feedback
- **Check logs**: The pipeline logs which config values are being used
