# Configuration Directory

All configuration files for the trader project.

## Files

### Model Configurations
- `model_config/*.yaml` - Per-model hyperparameters for TRAINING module
 - 17 model types (LightGBM, XGBoost, Ensemble, MultiTask, MLP, etc.)
 - Variants: conservative, balanced, aggressive

### Feature Selection
- `feature_selection_config.yaml` - LightGBM params, defaults, data filtering for feature selection
- `target_configs.yaml` - Target definitions for multi-target feature selection
- `feature_groups.yaml` - Feature family definitions for concept-based aggregation

### Legacy
- `base.yaml` - Base engine configuration (Aurora ruleset)

## Usage

### Load Model Config
```python
from CONFIG.config_loader import load_model_config

# Load with variant
config = load_model_config("lightgbm", variant="conservative")

# Load default
config = load_model_config("mlp")
```

### Load Feature Selection Config
```python
import yaml

with open("CONFIG/feature_selection_config.yaml") as f:
    config = yaml.safe_load(f)

lgbm_params = config['lightgbm']
defaults = config['defaults']
```

### Load Target Configs
```python
import yaml

with open("CONFIG/target_configs.yaml") as f:
    config = yaml.safe_load(f)

enabled_targets = {
    name: cfg for name, cfg in config['targets'].items()
    if cfg.get('enabled', True)
}
```

## Best Practices

1. **Never hardcode values** - Always use config files
2. **Version control configs** - Track changes in git
3. **Document changes** - Add comments explaining why values changed
4. **Test changes** - Run on small dataset before full production
5. **Create variants** - Don't modify defaults, create new variant configs

## Creating Custom Configs

```bash
# Copy existing config
cp CONFIG/feature_selection_config.yaml CONFIG/my_fast_selection.yaml

# Edit values
vim CONFIG/my_fast_selection.yaml

# Use it
python scripts/select_features.py --config CONFIG/my_fast_selection.yaml
```

For detailed documentation, see `INFORMATION/` directory.
