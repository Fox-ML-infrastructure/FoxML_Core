# Training Routing System - Implementation Summary

## What Was Built

A comprehensive, config-driven training routing system that makes reproducible decisions about where to train models for each `(target, symbol)` pair.

### Core Components

1. **Routing Configuration** (`CONFIG/training_config/routing_config.yaml`)
   - Score thresholds (CS vs symbol-specific)
   - Stability requirements
   - Sample size minimums
   - Experimental lane settings
   - Feature safety requirements

2. **Training Router** (`TRAINING/orchestration/training_router.py`)
   - Priority-ordered decision rules
   - Stability classification from metrics
   - CS and symbol eligibility evaluation
   - Route combination logic
   - Plan generation with JSON/YAML/Markdown outputs

3. **Metrics Aggregator** (`TRAINING/orchestration/metrics_aggregator.py`)
   - Collects metrics from feature selection outputs
   - Loads stability metrics from snapshots
   - Aggregates into routing candidates DataFrame
   - Saves to parquet + JSON

4. **Integration Hooks** (`TRAINING/orchestration/routing_integration.py`)
   - `generate_routing_plan_after_feature_selection()` - Main integration point
   - `load_routing_plan()` - Load plan from disk
   - `should_train_cross_sectional()` - Check CS eligibility
   - `should_train_symbol_specific()` - Check symbol eligibility

5. **CLI Entry Point** (`TRAINING/orchestration/generate_routing_plan.py`)
   - Standalone script to generate routing plans
   - Command-line interface with all options

6. **Auto-Integration** (updated `intelligent_trainer.py`)
   - Automatically generates routing plan after feature selection
   - Non-blocking (fails gracefully if metrics unavailable)

## Routing States

- `ROUTE_CROSS_SECTIONAL` - Train CS models only
- `ROUTE_SYMBOL_SPECIFIC` - Train symbol-specific models only
- `ROUTE_BOTH` - Train both (ensemble approach)
- `ROUTE_EXPERIMENTAL_ONLY` - Experimental lane (unstable but promising)
- `ROUTE_BLOCKED` - No training (leakage, insufficient data, etc.)

## Decision Logic Flow

```
For each (target, symbol):
  1. Evaluate CS eligibility (score, stability, sample size, leakage)
  2. Evaluate symbol eligibility (same criteria)
  3. Apply priority-ordered rules:
     - Hard blocks → BLOCKED
     - CS strong, local weak → CROSS_SECTIONAL
     - Local strong, CS weak → SYMBOL_SPECIFIC
     - Both strong → BOTH (or prefer one)
     - Experimental lane → EXPERIMENTAL_ONLY
     - Fallback → BLOCKED
```

## Artifacts Generated

1. **Routing Candidates** (`METRICS/routing_candidates.parquet` + `.json`)
   - One row per target (CS) or (target, symbol) pair
   - Contains: scores, stability, sample sizes, leakage status, etc.

2. **Routing Plan** (`METRICS/routing_plan/`)
   - `routing_plan.json` - Machine-readable
   - `routing_plan.yaml` - YAML format
   - `routing_plan.md` - Human-readable report with tables

## Usage Examples

### Automatic (Integrated)

The routing plan is automatically generated after feature selection in `IntelligentTrainer`:

```python
trainer = IntelligentTrainer(...)
results = trainer.train_with_intelligence(
    auto_targets=True,
    auto_features=True,
    ...
)
# Routing plan automatically generated in METRICS/routing_plan/
```

### Manual CLI

```bash
python -m TRAINING.orchestration.generate_routing_plan \
    --output-dir results/feature_selections \
    --targets y_will_swing_low_10m_0.20 y_will_peak_60m_0.8 \
    --symbols AAPL MSFT GOOGL TSLA
```

### Programmatic

```python
from TRAINING.orchestration.routing_integration import (
    generate_routing_plan_after_feature_selection,
    load_routing_plan,
    should_train_cross_sectional,
    should_train_symbol_specific
)

# Generate plan
plan = generate_routing_plan_after_feature_selection(
    output_dir=Path("results/feature_selections"),
    targets=["y_will_swing_low_10m_0.20"],
    symbols=["AAPL", "MSFT"]
)

# Use plan
if should_train_cross_sectional(plan, "y_will_swing_low_10m_0.20"):
    # Train CS model
    ...

if should_train_symbol_specific(plan, "y_will_swing_low_10m_0.20", "AAPL"):
    # Train symbol-specific model
    ...
```

## Configuration

Edit `CONFIG/training_config/routing_config.yaml` to adjust:

- **Thresholds**: `min_score`, `strong_score` for CS and symbol
- **Stability**: Which categories are allowed (`STABLE`, `DRIFTING`, etc.)
- **Sample sizes**: Minimum rows required
- **Experimental**: Enable/disable, thresholds, max fraction
- **Both-strong behavior**: `ROUTE_BOTH`, `PREFER_CS`, or `PREFER_SYMBOL`

## Key Features

✅ **Reproducible**: All decisions are deterministic based on metrics + config  
✅ **Config-driven**: No hardcoded thresholds  
✅ **Comprehensive**: Uses scores, stability, leakage, sample sizes  
✅ **Extensible**: Easy to add new rules or metrics  
✅ **Non-invasive**: Fails gracefully if metrics unavailable  
✅ **Observable**: Detailed logging and human-readable reports  

## Integration Points

1. **After Feature Selection**: Automatically generates plan
2. **Before Training**: Can check plan to decide which models to train
3. **Standalone**: Can run independently to analyze existing metrics

## Automatic Integration with Training Phase ✅

**The training plan is now automatically consumed by the training phase!**

When `IntelligentTrainer.train_with_intelligence()` runs:

1. Feature selection completes
2. Routing plan is generated → `METRICS/routing_plan/`
3. Training plan is automatically generated → `METRICS/training_plan/`
4. **Training phase automatically filters targets/symbols based on training plan**
5. Only approved jobs are executed

**Implementation:**
- `TRAINING/orchestration/training_plan_consumer.py` - Filters targets/symbols
- Integrated into `intelligent_trainer.py` after routing plan generation
- Automatically filters before calling `train_models_for_interval_comprehensive()`

**Result:** Only targets/symbols with approved training jobs in the plan are trained.

## Next Steps (Future Enhancements)

- Feature-level leakage filtering (currently placeholder)
- Confidence intervals from CV (currently None)
- Feature set hashing for tracking
- Model family failure tracking (needs total families attempted)
- Time-based stability analysis (trend detection)
- Ensemble weight recommendations based on route
- Symbol-specific training execution (currently only CS filtering implemented)

## Files Created/Modified

**New Files:**
- `CONFIG/training_config/routing_config.yaml`
- `TRAINING/orchestration/training_router.py`
- `TRAINING/orchestration/metrics_aggregator.py`
- `TRAINING/orchestration/routing_integration.py`
- `TRAINING/orchestration/generate_routing_plan.py`
- `TRAINING/orchestration/README_ROUTING.md`
- `TRAINING/orchestration/ROUTING_SYSTEM_SUMMARY.md`

**Modified Files:**
- `TRAINING/orchestration/intelligent_trainer.py` - Added routing plan generation hook

## Testing

The system is designed to work with existing feature selection outputs. To test:

1. Run feature selection for some targets/symbols
2. Generate routing plan: `python -m TRAINING.orchestration.generate_routing_plan ...`
3. Check `METRICS/routing_plan/routing_plan.md` for human-readable report
4. Use plan in training logic to filter which models to train

## Documentation

- `README_ROUTING.md` - User guide
- `ROUTING_SYSTEM_SUMMARY.md` - This file (implementation summary)
- `routing_config.yaml` - Inline comments explain all settings
