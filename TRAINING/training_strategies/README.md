# Training Strategies

Training strategy implementations split from the original 2,523-line file for better maintainability.

**For detailed documentation, see:** [DOCS/03_technical/refactoring/TRAINING_STRATEGIES.md](../../../DOCS/03_technical/refactoring/TRAINING_STRATEGIES.md)

## Quick Reference

- **setup.py**: Bootstrap and setup
- **family_runners.py**: Family execution (427 lines)
- **utils.py**: Utility functions (442 lines)
- **data_preparation.py**: Data preparation (593 lines)
- **training.py**: Core training functions (989 lines)
- **strategies.py**: Strategy implementations
- **main.py**: Entry point

## Usage

```python
# Direct import (recommended)
from TRAINING.training_strategies.training import train_models_for_interval_comprehensive

# Backward compatible (still works)
from TRAINING.train_with_strategies import train_models_for_interval_comprehensive
```
