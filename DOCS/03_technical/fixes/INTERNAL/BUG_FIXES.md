# Bug Fixes

History of significant bug fixes and improvements.

## Leakage Prevention

### Target Leakage

**Issue**: Target columns included in feature sets, causing data leakage.

**Fix**: Implemented `strip_targets()` function to remove target columns before training.

**Date**: 2025

### Forward Return Leakage

**Issue**: Forward returns calculated with look-ahead bias.

**Fix**: Corrected forward return calculation to use only past data.

**Date**: 2025

### Temporal Overlap

**Issue**: Walk-forward validation had temporal overlap between train and test sets.

**Fix**: Implemented strict temporal separation in fold builder.

**Date**: 2025

## Model Training Fixes

### Overfitting in Tree Models

**Issue**: LightGBM/XGBoost overfitting with default parameters.

**Fix**: Added conservative variants with higher regularization.

**Date**: 2025

### Early Stopping

**Issue**: Early stopping not implemented in all trainers.

**Fix**: Added automatic early stopping to all tree-based models.

**Date**: 2025

### Dropout Activation

**Issue**: Dropout inactive during training in MultiTaskStrategy.

**Fix**: Ensured `model.train()` mode during training, `model.eval()` during inference.

**Date**: 2025

## Data Processing Fixes

### Safe Target Pattern

**Issue**: Duplicate columns and target leakage in training data.

**Fix**: Implemented safe target pattern with `strip_targets()` and duplicate column handling.

**Date**: 2025

### Empty Data Handling

**Issue**: Indexing errors when data is empty.

**Fix**: Added guards for empty data, return HOLD with reason.

**Date**: 2025

## Configuration Fixes

### Centralized Configuration

**Issue**: Hardcoded parameters scattered across codebase.

**Fix**: Migrated all 17 production trainers to centralized YAML configs.

**Date**: November 2025

## See Also

- [Known Issues](KNOWN_ISSUES.md) - Current issues
- [Migration Notes](MIGRATION_NOTES.md) - Migration details

