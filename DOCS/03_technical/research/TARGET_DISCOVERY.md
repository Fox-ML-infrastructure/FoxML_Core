# Target Discovery

Methods for discovering and ranking predictive targets.

## Overview

Target discovery identifies which prediction targets are most predictable and suitable for model training.

## Target Ranking

### Predictability Ranking

Rank targets by their predictability across multiple models:

```bash
# Using the modular main module (recommended)
python -m TRAINING.ranking.predictability.main \
    --symbols AAPL,MSFT,GOOGL,TSLA,JPM \
    --output-dir results/target_rankings

# Or using the backward-compatible wrapper (still works)
python TRAINING/ranking/rank_target_predictability.py \
    --symbols AAPL,MSFT,GOOGL,TSLA,JPM \
    --output-dir results/target_rankings
```

**Note**: The ranking system uses a modular structure (`TRAINING/ranking/predictability/`) but maintains 100% backward compatibility. The original `rank_target_predictability.py` is now a thin wrapper that imports from the modular components.

**Metrics:**
- Information Coefficient (IC)
- R² score
- Sharpe ratio (for returns)
- Model agreement

### Multi-Model Evaluation

Test targets across multiple model families:

```python
model_families = ['lightgbm', 'xgboost', 'random_forest', 'neural_network']
for family in model_families:
    score = evaluate_target(y, model_family=family)
```

## Target Types

### Barrier Targets

**Format**: `y_will_peak_60m_0.8`  
**Meaning**: Will price peak 0.8% above current within 60 minutes  
**Use Case**: Entry/exit signals

### Forward Returns

**Format**: `target_fwd_ret_5m`  
**Meaning**: Forward return over 5-minute horizon  
**Use Case**: Return prediction

### Excess Returns

**Format**: `target_excess_ret_15m`  
**Meaning**: Return above benchmark  
**Use Case**: Relative performance

## Selection Criteria

### High Predictability

- IC > 0.1
- R² > 0.15
- Consistent across models

### Low Cost

- Low trading costs
- High liquidity
- Minimal market impact

### Practical Utility

- Actionable signals
- Reasonable frequency
- Aligned with strategy

## Best Practices

1. **Rank First**: Always rank targets before feature selection
2. **Multi-Model**: Test across multiple model families
3. **Validate**: Use walk-forward validation
4. **Monitor**: Track target predictability over time

## See Also

- [Intelligent Training Tutorial](../../01_tutorials/training/INTELLIGENT_TRAINING_TUTORIAL.md) - Automated target ranking workflow (recommended)
- [Target Ranking Module](../../../TRAINING/ranking/predictability/) - Ranking implementation (modular structure, see [detailed docs](../refactoring/TARGET_PREDICTABILITY_RANKING.md))
- [Feature Selection Guide](../implementation/FEATURE_SELECTION_GUIDE.md) - Complete feature selection methodology

