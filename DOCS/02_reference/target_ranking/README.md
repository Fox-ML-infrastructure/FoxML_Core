# Target Ranking Reference

## Overview

The target ranking system evaluates which targets (labels) are most predictable and worth pursuing for model training. It now supports **dual-view evaluation** to ensure consistency with downstream feature selection and training.

**NEW (2025-12-12)**: GPU acceleration is automatically enabled for target ranking. LightGBM, XGBoost, and CatBoost use GPU when available, providing 10-50x speedup on large datasets. See [GPU Setup Guide](../../01_tutorials/setup/GPU_SETUP.md) for configuration.

## Dual-View Evaluation

Each target is evaluated in two views:

1. **CROSS_SECTIONAL**: Pooled cross-sectional samples across all symbols
2. **SYMBOL_SPECIFIC**: Separate evaluation on each symbol's own time series
3. **LOSO** (optional): Leave-One-Symbol-Out evaluation for generalization testing

### Why Dual-View?

- **View consistency**: Target ranking → feature ranking → training must use the same view
- **Better routing**: Some targets work cross-sectionally, others work symbol-specifically
- **Generalization testing**: LOSO view tests if cross-sectional learning generalizes to unseen symbols

## Routing Decisions

After dual-view evaluation, each target gets a routing decision:

- **CROSS_SECTIONAL**: Strong cross-sectional performance with good symbol coverage
- **SYMBOL_SPECIFIC**: Weak cross-sectional but some symbols work well
- **BOTH**: Strong cross-sectional but performance is concentrated in specific symbols
- **BLOCKED**: Suspiciously high scores (likely leakage) - requires manual review

## Usage

### Basic Usage

```python
from TRAINING.ranking import rank_targets

results = rank_targets(
    targets=targets_dict,
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    data_dir=Path("data"),
    model_families=['lightgbm', 'random_forest'],
    output_dir=Path("results")
)
```

### Accessing Routing Decisions

```python
from TRAINING.ranking.target_routing import load_routing_decisions

routing = load_routing_decisions(
    Path("results/REPRODUCIBILITY/TARGET_RANKING/routing_decisions.json")
)

for target_name, route_info in routing.items():
    print(f"{target_name}: {route_info['route']} - {route_info['reason']}")
```

## Configuration

See `CONFIG/target_ranking_config.yaml` for:
- `enable_symbol_specific`: Enable/disable symbol-specific evaluation
- `enable_loso`: Enable/disable LOSO evaluation
- `routing.*`: Routing decision thresholds

## Reproducibility

Results are stored in:
```
REPRODUCIBILITY/TARGET_RANKING/{target}/
  CROSS_SECTIONAL/cohort={cohort_id}/metadata.json
  SYMBOL_SPECIFIC/symbol={symbol}/cohort={cohort_id}/metadata.json
  routing_decisions.json
```

## Integration with Feature Selection

Feature selection automatically respects the view from target ranking:

```python
# For cross-sectional targets
features = select_features_for_target(
    target_column="y_will_peak_60m_0.8",
    symbols=all_symbols,
    view="CROSS_SECTIONAL"
)

# For symbol-specific targets
features = select_features_for_target(
    target_column="y_will_peak_60m_0.8",
    symbols=[symbol],
    view="SYMBOL_SPECIFIC",
    symbol=symbol
)
```

## See Also

- [Dual-View Target Ranking Implementation](../../03_technical/implementation/DUAL_VIEW_TARGET_RANKING.md)
- [Simple Pipeline Usage](../../01_tutorials/SIMPLE_PIPELINE_USAGE.md)
