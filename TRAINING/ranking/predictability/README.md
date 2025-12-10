# Target Predictability Ranking

Target predictability ranking system split from the original 3,454-line file for better maintainability.

**For detailed documentation, see:** [DOCS/03_technical/refactoring/TARGET_PREDICTABILITY_RANKING.md](../../../../DOCS/03_technical/refactoring/TARGET_PREDICTABILITY_RANKING.md)

## Quick Reference

- **scoring.py**: TargetPredictabilityScore class
- **composite_score.py**: Composite score calculation
- **data_loading.py**: Config and data loading (360 lines)
- **leakage_detection.py**: Leakage detection (1,964 lines)
- **model_evaluation.py**: Model training & evaluation (2,542 lines)
- **reporting.py**: Report generation
- **main.py**: Entry point

## Usage

```python
# Direct import (recommended)
from TRAINING.ranking.predictability.model_evaluation import evaluate_target_predictability

# Backward compatible (still works)
from TRAINING.ranking.rank_target_predictability import evaluate_target_predictability
```
