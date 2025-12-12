# Training Routing & Plan System

**The "quant infra brain" that makes reproducible, config-driven decisions about where to train models.**

## Overview

The training routing system determines, for each `(target, symbol)` pair, whether to:
- Train **cross-sectional** models (pooled across symbols) ✅ **Implemented**
- Train **symbol-specific** models ⚠️ **Plan generated, execution pending**
- Train **both** (ensemble approach) ⚠️ **Plan generated, execution pending**
- Train **experimental only** (unstable but promising signals) ⚠️ **Plan generated, execution pending**
- **Block** training (leakage, insufficient data, etc.) ✅ **Implemented**

This decision is based on metrics from:
- Feature selection (scores, model family failures)
- Stability analysis (feature importance consistency)
- Leakage detection (data quality flags)

**Current Status:**
- ✅ **Cross-sectional training**: Fully implemented with automatic filtering based on routing plan
- ⚠️ **Symbol-specific training**: Routing decisions and training plan generated, but per-symbol model execution requires additional pipeline integration (future enhancement)

## Quick Start

### One-Command Pipeline

```bash
python -m TRAINING.orchestration.intelligent_trainer \
    --data-dir data \
    --symbols AAPL MSFT GOOGL \
    --auto-targets \
    --auto-features
```

**What happens:**
1. Ranks targets → Selects top N
2. Selects features per target
3. Generates routing plan → training plan
4. Trains models using plan (2-stage: CPU → GPU)

See `END_TO_END_FLOW.md` for complete flow details.

### Training with Plan

```bash
# Train all models with auto-detected plan
python -m TRAINING.training_strategies.main \
    --data-dir data \
    --symbols AAPL MSFT GOOGL \
    --model-types sequential
```

See `QUICK_START.md` and `ONE_COMMAND_TRAINING.md` for more examples.

## Artifacts

### Routing Plan (`METRICS/routing_plan/`)
- `routing_plan.json` - Machine-readable plan
- `routing_plan.yaml` - YAML format
- `routing_plan.md` - Human-readable report

### Training Plan (`METRICS/training_plan/`)
- `master_training_plan.json` - **Single source of truth**
- `training_plan.json` - Convenience mirror
- `training_plan.yaml` - YAML format
- `training_plan.md` - Human-readable report
- `by_target/<target>.json` - Derived views
- `by_symbol/<symbol>.json` - Derived views

## Configuration

Edit `CONFIG/training_config/routing_config.yaml` to adjust:
- **Score thresholds**: `min_score`, `strong_score` for CS and symbol-specific
- **Stability requirements**: Which stability categories are allowed
- **Sample size minimums**: Minimum rows required for CS vs symbol training
- **Experimental lane**: Enable/disable and thresholds
- **Both-strong behavior**: What to do when both CS and symbol are strong

## Documentation

### User Guides (Operational)
- `QUICK_START.md` - Quick start guide
- `END_TO_END_FLOW.md` - Complete end-to-end flow
- `ONE_COMMAND_TRAINING.md` - One-command examples
- `TWO_STAGE_TRAINING.md` - 2-stage training guide
- `SUMMARY.md` - Quick reference
- `INDEX.md` - Navigation guide

### Internal Documentation (Planning & Architecture)
**For architecture and implementation details**, see the internal documentation.

## Key Concepts

**Routing Plan** - Decisions about where to train (CS, symbol-specific, both, experimental, blocked)
- Location: `METRICS/routing_plan/`

**Training Plan** - Actionable job specifications derived from routing decisions
- Location: `METRICS/training_plan/`
- Master file: `master_training_plan.json` (single source of truth)

**Metrics Aggregation** - Collecting metrics from feature selection, stability, leakage detection
- Location: `METRICS/routing_candidates.parquet` (or `.csv`)

**Training Plan Consumption** - Filtering training based on the plan
- Implementation: `TRAINING/orchestration/training_plan_consumer.py`

## Implementation Files

**Core Implementation:**
- `TRAINING/orchestration/metrics_aggregator.py` - Metrics collection
- `TRAINING/orchestration/training_router.py` - Routing decisions
- `TRAINING/orchestration/training_plan_generator.py` - Job spec generation
- `TRAINING/orchestration/training_plan_consumer.py` - Plan consumption
- `TRAINING/orchestration/routing_integration.py` - Integration hooks
- `TRAINING/orchestration/intelligent_trainer.py` - Training orchestrator

**Configuration:**
- `CONFIG/training_config/routing_config.yaml` - Routing policy
