# Modular Configuration System

Complete guide to the new modular configuration system that prevents config "crossing" between pipeline components.

## Overview

The modular config system organizes configurations into **experiment-level**, **module-level**, and **system-level** files, with typed config classes (dataclasses) and a config builder for merging and validation.

### Key Benefits

- **No config "crossing"**: Each module has its own config directory
- **Type safety**: Typed config classes catch errors early
- **Validation**: Required fields and value ranges checked automatically
- **Backward compatible**: Legacy configs still work with deprecation warnings
- **Experiment configs**: Group related settings in one file

## Directory Structure

```
CONFIG/
├── experiments/              # Experiment-level configs (what are we running?)
│   └── *.yaml               # Example: fwd_ret_60m_test.yaml
│
├── feature_selection/        # Feature selection module configs
│   └── multi_model.yaml      # Model families, aggregation, sampling
│
├── target_ranking/           # Target ranking module configs
│   └── multi_model.yaml      # Model families, ranking settings
│
├── training/                 # Training module configs
│   └── models.yaml           # Model families for training
│
├── training_config/          # Legacy training configs (still used)
│   ├── pipeline_config.yaml
│   ├── gpu_config.yaml
│   └── ...
│
└── [legacy root configs]     # Still supported with deprecation warnings
    ├── multi_model_feature_selection.yaml
    └── ...
```

## Config Types

### 1. Experiment Configs (`CONFIG/experiments/`)

**Purpose:** Define what experiment you're running (data, targets, overrides).

**Structure:**
```yaml
experiment:
  name: my_experiment
  description: "Test run for fwd_ret_60m"

data:
  data_dir: data/data_labeled/interval=5m
  symbols: [AAPL, MSFT]
  interval: 5m
  max_samples_per_symbol: 3000

targets:
  primary: fwd_ret_60m

feature_selection:
  top_n: 30
  model_families: [lightgbm, xgboost]

target_ranking:
  min_samples: 100

training:
  model_families: [lightgbm, xgboost]
  cv_folds: 5
```

**Usage:**
```bash
python TRAINING/train.py --experiment-config my_experiment
```

### 2. Feature Selection Config (`CONFIG/feature_selection/multi_model.yaml`)

**Purpose:** Configure feature selection module (model families, aggregation, sampling).

**Key Settings:**
- `model_families`: Which models to use for feature importance
- `aggregation`: How to combine importance across models
- `sampling`: Cross-sectional sampling limits
- `top_n`: Default number of features to select

**Location:** `CONFIG/feature_selection/multi_model.yaml`

**Legacy:** `CONFIG/multi_model_feature_selection.yaml` (deprecated, still works)

### 3. Target Ranking Config (`CONFIG/target_ranking/multi_model.yaml`)

**Purpose:** Configure target ranking module (model families, ranking criteria).

**Key Settings:**
- `model_families`: Which models to use for predictability scoring
- `ranking`: Ranking algorithm settings
- `sampling`: Cross-sectional sampling limits
- `min_samples`: Minimum samples required per target

**Location:** `CONFIG/target_ranking/multi_model.yaml`

**Legacy:** Uses feature selection config (deprecated, still works)

### 4. Training Config (`CONFIG/training/models.yaml`)

**Purpose:** Configure training module (model families, CV settings).

**Key Settings:**
- `model_families`: Which models to train (references feature_selection config)
- `cv_folds`: Cross-validation folds
- `training`: Training-specific overrides

**Location:** `CONFIG/training/models.yaml`

## Typed Config Classes

All configs are loaded into typed dataclasses for validation:

### `ExperimentConfig`
```python
@dataclass
class ExperimentConfig:
    name: str
    data_dir: Path
    symbols: List[str]
    target: str
    interval: str = "5m"
    max_samples_per_symbol: int = 5000
    feature_selection_overrides: Dict[str, Any] = field(default_factory=dict)
    target_ranking_overrides: Dict[str, Any] = field(default_factory=dict)
    training_overrides: Dict[str, Any] = field(default_factory=dict)
```

### `FeatureSelectionConfig`
```python
@dataclass
class FeatureSelectionConfig:
    top_n: int
    model_families: Dict[str, Dict[str, Any]]
    aggregation: Dict[str, Any]
    # ... plus target/data info from experiment config
```

### `TargetRankingConfig`
```python
@dataclass
class TargetRankingConfig:
    model_families: Dict[str, Dict[str, Any]]
    ranking: Dict[str, Any]
    min_samples: int = 100
    # ... plus data info from experiment config
```

### `TrainingConfig`
```python
@dataclass
class TrainingConfig:
    model_families: Dict[str, Dict[str, Any]]
    cv_folds: int = 5
    pipeline: Dict[str, Any]
    # ... plus target/data info from experiment config
```

## Config Builder API

### Loading Experiment Config

```python
from CONFIG.config_builder import load_experiment_config

# Load from file
exp_cfg = load_experiment_config("fwd_ret_60m_test")
```

### Building Module Configs

```python
from CONFIG.config_builder import (
    build_feature_selection_config,
    build_target_ranking_config,
    build_training_config
)

# Build from experiment config
feature_cfg = build_feature_selection_config(exp_cfg)
ranking_cfg = build_target_ranking_config(exp_cfg)
training_cfg = build_training_config(exp_cfg)
```

### Validation

All configs are validated on load:
- Required fields must be present
- Value ranges checked (e.g., `cv_folds >= 2`)
- Type checking (paths converted to `Path` objects)

**Example:**
```python
# This will raise ValueError
bad_cfg = ExperimentConfig(
    name="",  # Empty name
    data_dir=Path("test"),
    symbols=[],  # Empty symbols
    target="test",
    max_samples_per_symbol=0  # Invalid value
)
```

## Migration Guide

### From Legacy to Modular Configs

**Old way:**
```bash
python TRAINING/train.py \
    --data-dir data/data_labeled/interval=5m \
    --symbols AAPL MSFT \
    --targets fwd_ret_60m \
    --multi-model-config CONFIG/multi_model_feature_selection.yaml
```

**New way:**
```bash
# 1. Create experiment config
cp CONFIG/experiments/fwd_ret_60m_test.yaml CONFIG/experiments/my_experiment.yaml
# Edit as needed

# 2. Use experiment config
python TRAINING/train.py --experiment-config my_experiment
```

### Config File Locations

| Old Location | New Location | Status |
|-------------|--------------|--------|
| `CONFIG/multi_model_feature_selection.yaml` | `CONFIG/feature_selection/multi_model.yaml` | ⚠️ Deprecated (still works) |
| (shared with feature selection) | `CONFIG/target_ranking/multi_model.yaml` | ✅ New |
| (various files) | `CONFIG/training/models.yaml` | ✅ New |

**Deprecation Warnings:**
When using legacy locations, you'll see:
```
⚠️  DEPRECATED: Using legacy config location: CONFIG/multi_model_feature_selection.yaml
   Please migrate to: CONFIG/feature_selection/multi_model.yaml
```

## Programmatic Usage

### In Code

```python
from CONFIG.config_builder import (
    load_experiment_config,
    build_feature_selection_config,
    build_target_ranking_config
)

# Load experiment config
exp_cfg = load_experiment_config("my_experiment")

# Build module configs
feature_cfg = build_feature_selection_config(exp_cfg)
ranking_cfg = build_target_ranking_config(exp_cfg)

# Use in pipeline
from TRAINING.ranking.feature_selector import select_features_for_target
from TRAINING.ranking.target_ranker import rank_targets

# Feature selection with typed config
features, importance_df = select_features_for_target(
    target_column=exp_cfg.target,
    symbols=exp_cfg.symbols,
    data_dir=exp_cfg.data_dir,
    feature_selection_config=feature_cfg
)

# Target ranking with typed config
rankings = rank_targets(
    targets=targets_dict,
    symbols=exp_cfg.symbols,
    data_dir=exp_cfg.data_dir,
    target_ranking_config=ranking_cfg
)
```

### In Intelligent Trainer

```python
from TRAINING.orchestration.intelligent_trainer import IntelligentTrainer
from CONFIG.config_builder import load_experiment_config

# Load experiment config
exp_cfg = load_experiment_config("my_experiment")

# Create trainer with experiment config
trainer = IntelligentTrainer(
    data_dir=exp_cfg.data_dir,
    symbols=exp_cfg.symbols,
    output_dir=Path("output"),
    experiment_config=exp_cfg  # Pass typed config
)

# Train (configs built automatically from experiment config)
results = trainer.train_with_intelligence(
    auto_targets=True,
    top_n_targets=5,
    max_targets_to_evaluate=23  # Limit for faster testing
)
```

## Best Practices

1. **Use experiment configs** for new projects
2. **Keep module configs separate** - don't mix feature selection and target ranking settings
3. **Override at experiment level** - use `feature_selection_overrides`, `target_ranking_overrides`, `training_overrides`
4. **Validate early** - configs are validated on load, catch errors before training
5. **Migrate gradually** - legacy configs still work, migrate when convenient

## Troubleshooting

### Config Not Found

```
FileNotFoundError: Experiment config not found: CONFIG/experiments/my_experiment.yaml
Available experiments: ['fwd_ret_60m_test']
```

**Solution:** Check available experiments or create a new one.

### Validation Errors

```
ValueError: ExperimentConfig.name cannot be empty
```

**Solution:** Check required fields in your experiment config YAML.

### Deprecation Warnings

```
⚠️  DEPRECATED: Using legacy config location: CONFIG/multi_model_feature_selection.yaml
```

**Solution:** Migrate to new location (`CONFIG/feature_selection/multi_model.yaml`).

## See Also

- [Configuration Overview](README.md) - Complete config system overview
- [Usage Examples](USAGE_EXAMPLES.md) - Practical examples
- [CLI Reference](../../api/CLI_REFERENCE.md) - Command-line options
- [Intelligent Training Tutorial](../../01_tutorials/training/INTELLIGENT_TRAINING_TUTORIAL.md) - Complete tutorial

