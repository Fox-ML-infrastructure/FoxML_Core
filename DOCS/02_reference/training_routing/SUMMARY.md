# Training Routing & Plan System - Summary

**One-command pipeline: Target Ranking → Feature Selection → Training Plan → Training**

## Overview

The system now supports a **complete end-to-end pipeline** in a single command. When you run `intelligent_trainer`, it:

1. **Ranks targets** by predictability
2. **Selects features** per target
3. **Generates training plan** (routing decisions → job specs)
4. **Trains models** using the plan (2-stage: CPU → GPU)

The training plan is **automatically passed** to the training phase - no manual steps required.

## One-Command Usage

```bash
python -m TRAINING.orchestration.intelligent_trainer \
    --data-dir data \
    --symbols AAPL MSFT GOOGL \
    --auto-targets \
    --auto-features
```

**What happens:**
- ✅ Ranks all targets, selects top N
- ✅ Selects features for each target
- ✅ Generates routing plan (`METRICS/routing_plan/`)
- ✅ Generates training plan (`globals/training_plan/master_training_plan.json` - primary, `METRICS/training_plan/` - legacy fallback)
- ✅ Trains models using the plan (2-stage: CPU → GPU)
- ✅ All 20 models trained (sequential + cross-sectional)

## Sequential Mode = Pipeline Mode

`--model-types sequential` is now a **pipeline mode** (not "LSTM-only"):

- Trains **all 20 models** (both sequential AND cross-sectional)
- Uses **2-stage approach**: CPU models first, then GPU models
- **Stage 1 (CPU)**: 10 CPU-only models (LightGBM, XGBoost, etc.)
- **Stage 2 (GPU)**: 10 GPU models (4 TF + 6 Torch)

> **Note**: In this configuration, XGBoost runs on CPU. If GPU XGBoost is enabled in the future, it should be moved to Stage 2.

## Plan-Aware Training

Both stages respect the **training plan**:

- **Targets are filtered** using the master training plan before training starts
- **Model families** can be filtered per job based on the plan (future extension)
- The plan is auto-detected from common locations, or can be explicitly specified

## Artifacts

### Training Plan (Single Source of Truth)

- `globals/training_plan/master_training_plan.json` - Canonical plan (primary location)
- `globals/training_plan/training_plan.json` - Convenience mirror
- `globals/training_plan/by_target/<target>.json` - Derived views
- `globals/training_plan/by_symbol/<symbol>.json`
- `globals/training_plan/by_type/<type>.json`
- Note: `METRICS/training_plan/` is supported as legacy fallback for backward compatibility

### Routing Plan

- `METRICS/routing_plan/routing_plan.json` - Routing decisions
- `METRICS/routing_candidates.parquet` - Aggregated metrics

## Benefits

✅ **One Command** - Complete pipeline from ranking to training
✅ **Plan-Driven** - Training respects intelligent routing decisions
✅ **2-Stage** - Efficient CPU → GPU resource usage
✅ **Auto-Detection** - Training phase finds plan automatically
✅ **No Thread Pollution** - CPU and GPU models run separately
✅ **Cached** - Ranking and feature selection results are cached

## Quick Reference

### Full Pipeline (One Command)

```bash
python -m TRAINING.orchestration.intelligent_trainer \
    --data-dir data \
    --symbols AAPL MSFT GOOGL \
    --auto-targets \
    --auto-features
```

### Train Only (With Auto-Detected Plan)

```bash
python -m TRAINING.training_strategies.main \
    --data-dir data \
    --symbols AAPL MSFT GOOGL \
    --model-types sequential
```

### Train Only (Explicit Plan)

```bash
python -m TRAINING.training_strategies.main \
    --data-dir data \
    --symbols AAPL MSFT GOOGL \
    --model-types sequential \
    --training-plan-dir results/globals/training_plan
```

## Documentation

- `END_TO_END_FLOW.md` - Complete flow documentation
- `TWO_STAGE_TRAINING.md` - 2-stage approach details
**For architecture details**, see the internal documentation.
- `QUICK_START.md` - Quick start guide
- `ONE_COMMAND_TRAINING.md` - One-command examples
