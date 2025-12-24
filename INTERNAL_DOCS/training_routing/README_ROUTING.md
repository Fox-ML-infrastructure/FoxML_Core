# Training Routing System

**The "quant infra brain" that makes reproducible, config-driven decisions about where to train models.**

## Overview

The training routing system determines, for each `(target, symbol)` pair, whether to:
- Train **cross-sectional** models (pooled across symbols)
- Train **symbol-specific** models
- Train **both** (ensemble approach)
- Train **experimental only** (unstable but promising signals)
- **Block** training (leakage, insufficient data, etc.)

This decision is based on metrics from:
- Feature selection (scores, model family failures)
- Stability analysis (feature importance consistency)
- Leakage detection (data quality flags)

## Quick Start

### 1. Generate Routing Plan

After feature selection completes, the routing plan is automatically generated. You can also run it manually:

```bash
python -m TRAINING.orchestration.generate_routing_plan \
    --output-dir results/feature_selections \
    --targets y_will_swing_low_10m_0.20 y_will_peak_60m_0.8 \
    --symbols AAPL MSFT GOOGL TSLA
```

### 2. Check Routing Plan & Training Plan

**Routing Plan** (`METRICS/routing_plan/`):
- `routing_plan.json` - Machine-readable plan
- `routing_plan.yaml` - YAML format
- `routing_plan.md` - Human-readable report

**Training Plan** (`METRICS/training_plan/`) ⭐ NEW:
- `training_plan.json` - Actionable training jobs
- `training_plan.yaml` - YAML format
- `training_plan.md` - Human-readable report with job details

### 3. Use Routing Plan in Training

```python
from TRAINING.orchestration.routing_integration import (
    load_routing_plan,
    should_train_cross_sectional,
    should_train_symbol_specific
)

plan = load_routing_plan(Path("METRICS/routing_plan"))

# Check if CS training should be enabled
if should_train_cross_sectional(plan, "y_will_swing_low_10m_0.20"):
    # Train CS model
    ...

# Check if symbol-specific training should be enabled
if should_train_symbol_specific(plan, "y_will_swing_low_10m_0.20", "AAPL"):
    # Train symbol-specific model
    ...
```

## Configuration

Edit `CONFIG/training_config/routing_config.yaml` to adjust:

- **Score thresholds**: `min_score`, `strong_score` for CS and symbol-specific
- **Stability requirements**: Which stability categories are allowed
- **Sample size minimums**: Minimum rows required for CS vs symbol training
- **Experimental lane**: Enable/disable and thresholds
- **Both-strong behavior**: What to do when both CS and symbol are strong

## Routing Decision Logic

The router uses priority-ordered rules:

1. **Hard blocks**: Leakage detected, insufficient data → `ROUTE_BLOCKED`
2. **CS strong, local weak**: → `ROUTE_CROSS_SECTIONAL`
3. **Local strong, CS weak**: → `ROUTE_SYMBOL_SPECIFIC`
4. **Both strong**: → `ROUTE_BOTH` (or prefer one based on config)
5. **Experimental lane**: Unstable but promising → `ROUTE_EXPERIMENTAL_ONLY`
6. **Fallback**: → `ROUTE_BLOCKED`

## Metrics Aggregation

The system automatically collects metrics from:

- `feature_selections/{target}/model_metadata.json` - Per-symbol scores
- `feature_selections/{target}/target_confidence.json` - CS confidence
- Stability snapshots from `TRAINING/stability/feature_importance/`
- Leakage detection outputs (if available)

## Integration

The routing system integrates automatically with `IntelligentTrainer`:

1. Feature selection runs per target
2. After all feature selections complete, routing plan is generated
3. Routing plan saved to `METRICS/routing_plan/`
4. **Training plan automatically generated** → `METRICS/training_plan/`
5. Training can consume either plan to decide which models to train

**Note:** The training plan contains actionable job specifications with priorities, model families, and execution details.

## Example Output

```
[ROUTER] target=y_will_swing_low_10m_0.20: CS=STRONG (AUC=0.62, STABLE)
[ROUTER]   AAPL → ROUTE_BOTH (local strong)
[ROUTER]   MSFT → ROUTE_CROSS_SECTIONAL (local sample too small)
[ROUTER]   TSLA → ROUTE_SYMBOL_SPECIFIC (CS weak, local strong)
[ROUTER]   NFLX → ROUTE_BLOCKED (leakage detected)
```

## Architecture

```
TRAINING/orchestration/
├── training_router.py          # Core routing decision engine
├── metrics_aggregator.py      # Collects metrics from pipeline outputs
├── routing_integration.py      # Integration hooks for pipeline
├── training_plan_generator.py  # Converts routing → training jobs ⭐ NEW
└── generate_routing_plan.py   # CLI entry point

CONFIG/training_config/
└── routing_config.yaml        # Routing policy configuration
```

## Training Plan Artifact

The system now generates a **training plan** that converts routing decisions into actionable training jobs:

```python
from TRAINING.orchestration.training_plan_generator import (
    generate_training_plan_from_routing
)

plan = generate_training_plan_from_routing(
    routing_plan_path=Path("METRICS/routing_plan/routing_plan.json"),
    output_dir=Path("METRICS/training_plan"),
    model_families=["lightgbm", "xgboost"]
)
```

Each job in the plan includes:
- `job_id`: Unique identifier
- `target`: Target name
- `symbol`: Symbol name (None for CS)
- `route`: Routing decision
- `training_type`: "cross_sectional" or "symbol_specific"
- `model_families`: List of families to train
- `priority`: Job priority (higher = more important)
- `reason`: Why this job was created
- `metadata`: Additional context

The training plan is automatically generated after routing plan creation.

## Advanced Usage

### Custom Stability Classification

Override stability classification by providing custom thresholds in `routing_config.yaml`:

```yaml
routing:
  stability_classification:
    stable_overlap_min: 0.75
    stable_tau_min: 0.65
    drifting_overlap_min: 0.55
    max_std_overlap: 0.15
```

### Feature-Level Leakage Filtering

If `require_safe_features_only: true`, the router will only use features marked as `SAFE` in leakage detection outputs.

### Experimental Lane Capping

The experimental lane respects `max_fraction_symbols_per_target` to prevent experiment explosion. Only the top-N experimental candidates per target are allowed.

## Troubleshooting

**No routing candidates found:**
- Ensure feature selection has completed
- Check that `feature_selections/{target}/` directories exist
- Verify `model_metadata.json` files are present

**All routes are BLOCKED:**
- Check sample sizes meet minimums
- Verify leakage status is not `BLOCKED`
- Review stability metrics (may all be `DIVERGED`)

**CS metrics missing:**
- CS metrics are aggregated from per-symbol outputs
- Ensure multiple symbols were processed
- Check `target_confidence.json` exists

## See Also

- `CONFIG/training_config/routing_config.yaml` - Full configuration schema
- `TRAINING/orchestration/training_router.py` - Implementation details
- `TRAINING/stability/feature_importance/` - Stability tracking system
