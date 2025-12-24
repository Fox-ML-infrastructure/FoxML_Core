# Specialized Models

Specialized model implementations split from the original 4,518-line file for better maintainability.

**For detailed documentation, see:** [DOCS/03_technical/refactoring/SPECIALIZED_MODELS.md](../../../DOCS/03_technical/refactoring/SPECIALIZED_MODELS.md)

## Quick Reference

- **wrappers.py**: Model wrapper classes (TFSeriesRegressor, GMMRegimeRegressor, OnlineChangeHeuristic)
- **predictors.py**: Predictor classes (GANPredictor, ChangePointPredictor)
- **trainers.py**: Core training functions (817 lines)
- **trainers_extended.py**: Extended training functions (1,204 lines)
- **metrics.py**: Metrics functions
- **data_utils.py**: Data loading/preparation (989 lines)
- **core.py**: Main orchestration (1,391 lines)
- **constants.py**: Shared constants

## Usage

```python
# Direct import (recommended)
from TRAINING.models.specialized.core import train_model

# Backward compatible (still works)
from TRAINING.models.specialized_models import train_model
```
