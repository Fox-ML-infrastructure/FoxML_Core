# Migration Notes

Migration status and notes for centralized configuration system.

## Migration Status

**Date:** November 13, 2025  
**Status:** COMPLETE - All production trainers migrated  
**Progress:** 17/17 production models (100%)

## Migrated Trainers

### Core Models
- LightGBMTrainer → `lightgbm.yaml`
- XGBoostTrainer → `xgboost.yaml`
- EnsembleTrainer → `ensemble.yaml`
- MultiTaskTrainer → `multi_task.yaml`

### Deep Learning
- MLPTrainer → `mlp.yaml`
- TransformerTrainer → `transformer.yaml`
- LSTMTrainer → `lstm.yaml`
- CNN1DTrainer → `cnn1d.yaml`

### Feature Engineering
- VAETrainer → `vae.yaml`
- GANTrainer → `gan.yaml`
- GMMRegimeTrainer → `gmm_regime.yaml`

### Probabilistic
- NGBoostTrainer → `ngboost.yaml`
- QuantileLightGBMTrainer → `quantile_lightgbm.yaml`

### Advanced
- ChangePointTrainer → `change_point.yaml`
- FTRLProximalTrainer → `ftrl_proximal.yaml`
- RewardBasedTrainer → `reward_based.yaml`
- MetaLearningTrainer → `meta_learning.yaml`

## Implementation Pattern

All migrated trainers follow:

```python
from CONFIG.config_loader import load_model_config

config = load_model_config("lightgbm", variant="conservative")
trainer = LightGBMTrainer(config)
```

## Benefits

1. **Centralized**: All configs in one location
2. **Versioned**: Configs tracked in git
3. **Variants**: Conservative, balanced, aggressive options
4. **Overridable**: Runtime parameter overrides

## See Also

- [Migration Notes](../../../INFORMATION/03_MIGRATION_NOTES.md) - Complete details
- [Config Basics](../../01_tutorials/configuration/CONFIG_BASICS.md) - Configuration tutorial

