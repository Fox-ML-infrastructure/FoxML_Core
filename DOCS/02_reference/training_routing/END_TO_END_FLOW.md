# End-to-End Training Flow

**One-command pipeline: Target Ranking → Feature Selection → Training Plan → Training**

## Overview

The system now supports a **complete end-to-end pipeline** in a single command:

1. **Target Ranking** - Ranks all available targets by predictability
2. **Feature Selection** - Selects best features per target
3. **Training Plan Generation** - Creates routing plan and training plan
4. **Training Execution** - Trains models using the plan (2-stage: CPU → GPU)

## One-Command Flow

### Using IntelligentTrainer (Full Pipeline)

```bash
python -m TRAINING.orchestration.intelligent_trainer \
    --data-dir data \
    --symbols AAPL MSFT GOOGL \
    --auto-targets \
    --auto-features
```

**What happens:**

1. **Target Ranking Phase**
   - Discovers all targets from data
   - Ranks them by predictability metrics
   - Selects top N targets (configurable)

2. **Feature Selection Phase**
   - For each selected target:
     - Runs feature selection algorithms
     - Detects leakage
     - Selects best features

3. **Routing Plan Generation**
   - Aggregates metrics from ranking + feature selection
   - Generates routing decisions (CS vs symbol-specific)
   - Creates `METRICS/routing_plan/`

4. **Training Plan Generation**
   - Converts routing plan into training jobs
   - Creates `globals/training_plan/master_training_plan.json` (primary location)
   - Generates derived views (by_target, by_symbol, etc.)

5. **Training Execution**
   - Loads training plan
   - Filters targets based on plan
   - Trains models in 2-stage order (CPU → GPU)
   - Respects model families from plan

**Output:**
```
🎯 Ranking targets (top 10)...
✅ Target ranking complete: 10 targets selected

📊 Feature selection for 10 targets...
✅ Feature selection complete

[ROUTER] ✅ Training routing plan generated
[ROUTER] ✅ Training plan generated: globals/training_plan (primary location)
📋 Training plan filter applied: 10 → 7 targets

📊 Stage 1 (CPU): 10 models
📊 Stage 2 (GPU): 10 models
```

## Two-Step Flow (Alternative)

If you want to separate planning from execution:

### Step 1: Generate Training Plan

```bash
python -m TRAINING.orchestration.intelligent_trainer \
    --data-dir data \
    --symbols AAPL MSFT GOOGL \
    --auto-targets \
    --auto-features \
    --skip-training  # Only generate plan, don't train
```

This creates:
- `METRICS/routing_plan/`
- `globals/training_plan/master_training_plan.json` (primary location)

### Step 2: Train Using Plan

```bash
python -m TRAINING.training_strategies.main \
    --data-dir data \
    --symbols AAPL MSFT GOOGL \
    --model-types sequential \
    --training-plan-dir results/globals/training_plan
```

Or with auto-detection:

```bash
python -m TRAINING.training_strategies.main \
    --data-dir data \
    --symbols AAPL MSFT GOOGL \
    --model-types sequential
# Auto-detects plan from results/globals/training_plan (primary) or results/METRICS/training_plan (legacy fallback)
```

## Artifacts Created

### During Planning Phase

```
results/
  METRICS/
    routing_candidates.parquet  # Aggregated metrics
    routing_plan/
      routing_plan.json
      routing_plan.yaml
      routing_plan.md
    training_plan/
      master_training_plan.json  # Single source of truth
      training_plan.json          # Convenience mirror
      training_plan.yaml
      training_plan.md
      by_target/
        <target>.json
      by_symbol/
        <symbol>.json
      by_type/
        cross_sectional.json
        symbol_specific.json
```

### During Training Phase

```
output/
  <timestamp>/
    models/          # Trained models
    metrics/         # Training metrics
    logs/            # Training logs
```

## Configuration

### IntelligentTrainer Options

```bash
python -m TRAINING.orchestration.intelligent_trainer \
    --data-dir data \
    --symbols AAPL MSFT GOOGL \
    --auto-targets \              # Auto-rank and select targets
    --auto-features \              # Auto-select features per target
    --top-n-targets 10 \           # Number of top targets to use
    --skip-training \              # Only generate plan, don't train
    --force-refresh                # Ignore cache, re-run everything
```

### Training Options

```bash
python -m TRAINING.training_strategies.main \
    --data-dir data \
    --symbols AAPL MSFT GOOGL \
    --model-types sequential \    # 2-stage pipeline (CPU → GPU)
    --training-plan-dir <path> \   # Explicit plan location
    --no-training-plan             # Skip plan, train all targets
```

## Plan-Aware Training

The training phase automatically:

1. **Loads the training plan** (auto-detected or explicit)
2. **Filters targets** - Only trains targets with jobs in plan
3. **Uses model families** - Respects families specified in plan (per-target if available)
4. **Applies 2-stage ordering** - CPU models first, then GPU models

### Example: Plan Filters Targets

```json
{
  "jobs": [
    {
      "job_id": "cs_y_will_swing_low_10m_0.20",
      "target": "y_will_swing_low_10m_0.20",
      "training_type": "cross_sectional",
      "model_families": ["LightGBM", "XGBoost", "LSTM"]
    }
  ]
}
```

**Result:**
- Only `y_will_swing_low_10m_0.20` is trained
- Only `LightGBM`, `XGBoost`, `LSTM` are trained for this target
- Other targets/families are skipped

## Benefits

✅ **One Command** - Complete pipeline from ranking to training
✅ **Plan-Driven** - Training respects intelligent routing decisions
✅ **Cached** - Ranking and feature selection results are cached
✅ **Flexible** - Can run planning and training separately
✅ **Auto-Detection** - Training phase finds plan automatically
✅ **2-Stage** - Efficient CPU → GPU resource usage

## Workflow Examples

### Example 1: Full Pipeline (One Command)

```bash
# Everything in one go
python -m TRAINING.orchestration.intelligent_trainer \
    --data-dir data \
    --symbols AAPL MSFT GOOGL \
    --auto-targets \
    --auto-features
```

### Example 2: Plan First, Train Later

```bash
# Step 1: Generate plan
python -m TRAINING.orchestration.intelligent_trainer \
    --data-dir data \
    --symbols AAPL MSFT GOOGL \
    --auto-targets \
    --auto-features \
    --skip-training

# Step 2: Train (auto-detects plan)
python -m TRAINING.training_strategies.main \
    --data-dir data \
    --symbols AAPL MSFT GOOGL \
    --model-types sequential
```

### Example 3: Train Without Plan

```bash
# Skip planning, train all targets
python -m TRAINING.training_strategies.main \
    --data-dir data \
    --symbols AAPL MSFT GOOGL \
    --model-types sequential \
    --no-training-plan
```

## Summary

**Yes, it's one command!** The `intelligent_trainer` orchestrates:

1. Target ranking
2. Feature selection  
3. Training plan generation
4. Training execution

All in a single run, with the training plan automatically passed to the training phase.
