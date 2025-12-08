# Intelligent Trainer API Reference

Complete API reference for the `IntelligentTrainer` class and intelligent training pipeline.

## Overview

The `IntelligentTrainer` class orchestrates target ranking, feature selection, and model training in a unified pipeline.

**Location**: `TRAINING/orchestration/intelligent_trainer.py`

## IntelligentTrainer Class

### Initialization

```python
from TRAINING.orchestration.intelligent_trainer import IntelligentTrainer
from pathlib import Path

trainer = IntelligentTrainer(
    data_dir=Path("data/data_labeled/interval=5m"),
    symbols=["AAPL", "MSFT", "GOOGL"],
    output_dir=Path("output"),
    cache_dir=None,  # Optional, defaults to output_dir/cache
    add_timestamp=True  # Optional, append timestamp to output_dir (default: True)
)
```

**Parameters:**
- `data_dir` (Path): Directory containing symbol data
- `symbols` (List[str]): List of symbols to train on
- `output_dir` (Path): Output directory for training results
- `cache_dir` (Optional[Path]): Cache directory for rankings/selections (default: output_dir/cache)
- `add_timestamp` (bool): Append timestamp to output_dir (format: `YYYYMMDD_HHMMSS`) to make runs distinguishable (default: True)

### Methods

#### `rank_targets_auto()`

Automatically rank targets and return top N.

```python
targets = trainer.rank_targets_auto(
    top_n=5,
    model_families=None,  # Optional list of model families
    multi_model_config=None,  # Optional multi-model config
    force_refresh=False,  # Ignore cache and re-rank
    use_cache=True  # Enable/disable caching
)
```

**Returns:** List[str] - List of top N target names

**Example:**
```python
# Rank and select top 5 targets
top_targets = trainer.rank_targets_auto(top_n=5)
# Returns: ['fwd_ret_5m', 'fwd_ret_15m', 'y_will_peak_60m_0.8', ...]
```

#### `select_features_auto()`

Automatically select top M features for a target.

```python
features = trainer.select_features_auto(
    target="fwd_ret_5m",
    top_m=100,
    model_families_config=None,  # Optional model families config
    multi_model_config=None,  # Optional multi-model config
    force_refresh=False,  # Ignore cache and re-select
    use_cache=True  # Enable/disable caching
)
```

**Returns:** List[str] - List of top M feature names

**Example:**
```python
# Select top 100 features for a target
features = trainer.select_features_auto(
    target="fwd_ret_5m",
    top_m=100
)
# Returns: ['rsi_10', 'sma_200', 'vwap_dev_high', ...]
```

#### `train_with_intelligence()`

Train models with intelligent target/feature selection.

```python
results = trainer.train_with_intelligence(
    auto_targets=True,  # Enable automatic target ranking
    top_n_targets=5,  # Number of top targets to select
    auto_features=True,  # Enable automatic feature selection
    top_m_features=100,  # Number of top features per target
    targets=None,  # Manual target list (overrides auto_targets)
    features=None,  # Manual feature list (overrides auto_features)
    families=None,  # Model families to train (default: all)
    strategy='single_task',  # Training strategy
    use_cache=True,  # Enable/disable caching
    **train_kwargs  # Additional training arguments
)
```

**Returns:** Dict[str, Any] - Training results dictionary

**Result Structure:**
```python
{
    'targets': ['fwd_ret_5m', 'fwd_ret_15m', ...],  # Selected targets
    'target_features': {  # Selected features per target
        'fwd_ret_5m': ['feat1', 'feat2', ...],
        'fwd_ret_15m': ['feat3', 'feat4', ...]
    },
    'strategy': 'single_task',
    'training_results': {...},  # Full training results
    'total_models': 15,  # Number of models trained
    'status': 'completed'
}
```

**Example:**
```python
# Full intelligent training pipeline
results = trainer.train_with_intelligence(
    auto_targets=True,
    top_n_targets=5,
    auto_features=True,
    top_m_features=100,
    families=['LightGBM', 'XGBoost', 'MLP'],
    strategy='single_task',
    min_cs=10,
    max_rows_train=50000
)

print(f"Trained {results['total_models']} models")
print(f"Targets: {results['targets']}")
```

## Module Functions

### `rank_targets()`

Rank targets using multiple model families.

```python
from TRAINING.ranking import rank_targets

rankings = rank_targets(
    targets_dict={...},  # Target configurations
    symbols=["AAPL", "MSFT"],
    data_dir=Path("data/data_labeled/interval=5m"),
    model_families=["lightgbm", "xgboost", "random_forest"],
    multi_model_config={...},  # Optional config
    output_dir=Path("output/target_rankings"),
    top_n=None  # Return all if None
)
```

**Returns:** List[TargetPredictabilityScore] - Ranked target scores

### `select_features_for_target()`

Select features for a specific target using multi-model consensus.

```python
from TRAINING.ranking import select_features_for_target

features, importance_results = select_features_for_target(
    target_column="fwd_ret_5m",
    symbols=["AAPL", "MSFT"],
    data_dir=Path("data/data_labeled/interval=5m"),
    model_families_config={...},  # Optional config
    multi_model_config={...},  # Optional config
    top_n=100,  # Number of top features
    output_dir=Path("output/feature_selections")
)
```

**Returns:** Tuple[List[str], FeatureImportanceResult] - Selected features and importance results

### `discover_targets()`

Discover all available targets from data.

```python
from TRAINING.ranking import discover_targets

targets_dict = discover_targets(
    symbol="AAPL",
    data_dir=Path("data/data_labeled/interval=5m")
)
```

**Returns:** Dict[str, TargetConfig] - Dictionary of target configurations

## Configuration

### Target Ranking Config

Default location: `CONFIG/training_config/target_ranking_config.yaml`

```yaml
target_ranking:
  enabled: true
  top_n_targets: 5
  min_predictability_score: 0.1
  model_families:
    - lightgbm
    - xgboost
    - random_forest
  max_samples_per_symbol: 10000
  cv_folds: 3
```

### Feature Selection Config

Default location: `CONFIG/multi_model_feature_selection.yaml`

```yaml
model_families:
  lightgbm:
    enabled: true
  random_forest:
    enabled: true
  neural_network:
    enabled: true
aggregation:
  require_min_models: 2
```

## Caching

### Cache Structure

```
output_dir/cache/
├── target_rankings.json  # Cached target rankings
└── feature_selections/
    ├── {target1}.json
    └── {target2}.json
```

### Cache Keys

Cache keys are generated from:
- Symbol list (sorted)
- Model families used
- Configuration hash (MD5)

Same symbols + same configs = cache hit

### Cache Control

```python
# Force refresh (ignore cache)
trainer.rank_targets_auto(top_n=5, force_refresh=True)

# Disable caching entirely
trainer.rank_targets_auto(top_n=5, use_cache=False)

# Use cache only (never refresh)
# Set force_refresh=False and use_cache=True
```

## Error Handling

### No Targets Selected

```python
try:
    targets = trainer.rank_targets_auto(top_n=5)
except ValueError as e:
    # All targets filtered out (leakage, degenerate, etc.)
    print(f"No targets selected: {e}")
```

### Feature Selection Failure

```python
try:
    features = trainer.select_features_auto(target="fwd_ret_5m", top_m=100)
except Exception as e:
    # Insufficient data or all features filtered
    print(f"Feature selection failed: {e}")
```

### Training Failure

```python
try:
    results = trainer.train_with_intelligence(...)
except Exception as e:
    # Data issues, memory limits, or model errors
    print(f"Training failed: {e}")
```

## Integration Examples

### Programmatic Usage

```python
from TRAINING.orchestration.intelligent_trainer import IntelligentTrainer
from pathlib import Path

# Initialize trainer
trainer = IntelligentTrainer(
    data_dir=Path("data/data_labeled/interval=5m"),
    symbols=["AAPL", "MSFT", "GOOGL"],
    output_dir=Path("output")
)

# Step 1: Rank targets
targets = trainer.rank_targets_auto(top_n=5)
print(f"Selected targets: {targets}")

# Step 2: Select features per target
target_features = {}
for target in targets:
    features = trainer.select_features_auto(target=target, top_m=100)
    target_features[target] = features
    print(f"{target}: {len(features)} features")

# Step 3: Train models
results = trainer.train_with_intelligence(
    targets=targets,
    target_features=target_features,
    families=['LightGBM', 'XGBoost'],
    strategy='single_task'
)

print(f"Trained {results['total_models']} models")
```

### Custom Config Loading

```python
from TRAINING.ranking import load_target_configs, load_multi_model_config
from pathlib import Path

# Load custom configs
target_config = load_target_configs(Path("custom_target_config.yaml"))
multi_model_config = load_multi_model_config(Path("custom_multi_model_config.yaml"))

# Use in trainer
trainer = IntelligentTrainer(...)
targets = trainer.rank_targets_auto(
    top_n=5,
    multi_model_config=multi_model_config
)
```

## Related Documentation

- [Intelligent Training Tutorial](../../01_tutorials/training/INTELLIGENT_TRAINING_TUTORIAL.md) - Complete tutorial
- [CLI Reference](CLI_REFERENCE.md) - Command-line interface
- [Module Reference](MODULE_REFERENCE.md) - Python API overview

