# Phase 3 Training Plan Integration

**How Phase 3 (Sequential Model Training) uses the training plan system.**

## Current Status

**Phase 3 is integrated** - Sequential models (LSTM, Transformer, CNN1D) are trained through:
1. `IntelligentTrainer` - ✅ **Fully integrated** with training plan
2. `main.py` (training_strategies/main.py) - ✅ **Now supports** training plan via `--training-plan-dir`

Sequential models can be trained by:
- Using `--model-types sequential` in main.py
- Passing sequential families (LSTM, Transformer, CNN1D) in `--families`
- Using `IntelligentTrainer` with sequential families

## Integration Options

### Option 1: Use IntelligentTrainer (Recommended)

**Best approach:** Use `IntelligentTrainer` which already has full training plan integration.

```python
from TRAINING.orchestration.intelligent_trainer import IntelligentTrainer
from pathlib import Path

trainer = IntelligentTrainer(
    data_dir=Path("data"),
    symbols=["AAPL", "MSFT", "GOOGL"],
    output_dir=Path("results/phase3")
)

# IntelligentTrainer automatically:
# 1. Generates routing plan after feature selection
# 2. Generates training plan
# 3. Filters targets/symbols based on plan
# 4. Uses model families from plan
results = trainer.train_with_intelligence(
    auto_targets=True,
    auto_features=True,
    families=["LSTM", "Transformer", "CNN1D"],  # Sequential models
    strategy="single_task"
)
```

**Benefits:**
- ✅ Full training plan integration (automatic)
- ✅ Target/symbol filtering
- ✅ Model family filtering
- ✅ Consistent with intelligent pipeline

### ✅ Option 2: Use main.py with Training Plan (Now Supported + Auto-Detection)

**main.py now supports training plan integration with auto-detection!**

#### Simplest Command (Auto-Detects Plan)

```bash
# Train all sequential models with auto-detected training plan
python -m TRAINING.training_strategies.main \
    --data-dir data \
    --symbols AAPL MSFT GOOGL \
    --model-types sequential \
    --output-dir output/sequential_models
```

**What it does:**
- ✅ Auto-detects training plan from common locations
- ✅ Trains all 6 sequential models (CNN1D, LSTM, Transformer, TabCNN, TabLSTM, TabTransformer)
- ✅ Filters targets based on plan (if found)
- ✅ Uses model families from plan (if found)

#### With Explicit Plan Location

```bash
python -m TRAINING.training_strategies.main \
    --data-dir data \
    --symbols AAPL MSFT GOOGL \
    --model-types sequential \
    --training-plan-dir results/METRICS/training_plan \
    --output-dir output/sequential_models
```

**What it does:**
- ✅ Loads training plan from `--training-plan-dir`
- ✅ Filters targets based on plan
- ✅ Uses model families from plan (per-target if available)
- ✅ Works with sequential models (all 6 by default)

### Option 3: Manual Training Plan Integration (For Custom Scripts)

If you need a custom Phase 3 script (for EXPERIMENTS workflow), integrate training plan manually:

```python
from pathlib import Path
from TRAINING.orchestration.training_plan_consumer import (
    load_training_plan,
    filter_targets_by_training_plan,
    get_model_families_for_job
)
from TRAINING.training_strategies.training import train_models_for_interval_comprehensive

# Load training plan (if available)
training_plan_dir = Path("results/METRICS/training_plan")
training_plan = load_training_plan(training_plan_dir)

# Filter targets based on plan
targets = ["target1", "target2", "target3"]  # Your targets
if training_plan:
    filtered_targets = filter_targets_by_training_plan(
        targets=targets,
        training_plan=training_plan,
        training_type="cross_sectional"  # or "symbol_specific" if applicable
    )
    logger.info(f"Filtered {len(targets)} → {len(filtered_targets)} targets")
else:
    filtered_targets = targets
    logger.info("No training plan found, using all targets")

# Get model families from plan (if available)
families = ["LSTM", "Transformer", "CNN1D"]
target_families_map = {}
if training_plan:
    for target in filtered_targets:
        plan_families = get_model_families_for_job(
            training_plan,
            target=target,
            symbol=None,
            training_type="cross_sectional"
        )
        if plan_families:
            # Filter to only sequential models
            seq_families = [f for f in plan_families if f in families]
            if seq_families:
                target_families_map[target] = seq_families

# Train with filtered targets and families
results = train_models_for_interval_comprehensive(
    interval="cross_sectional",
    targets=filtered_targets,
    mtf_data=mtf_data,
    families=families,
    strategy="single_task",
    output_dir="output/sequential_models",
    target_features=target_features,  # From Phase 1
    target_families=target_families_map if target_families_map else None
)
```

## Recommended Implementation

### For EXPERIMENTS Workflow

If Phase 3 needs to fit into the EXPERIMENTS workflow structure:

```python
# phase3_sequential_models/run_phase3.py

import argparse
from pathlib import Path
from TRAINING.orchestration.training_plan_consumer import (
    load_training_plan,
    filter_targets_by_training_plan,
    get_model_families_for_job
)
from TRAINING.training_strategies.training import train_models_for_interval_comprehensive

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--metadata-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--training-plan-dir", help="Path to training plan (optional)")
    args = parser.parse_args()
    
    # Load Phase 1 artifacts
    metadata_dir = Path(args.metadata_dir)
    with open(metadata_dir / "top_50_features.json") as f:
        selected_features = json.load(f)
    
    # Load data
    mtf_data = load_mtf_data(args.data_dir, symbols=["AAPL", "MSFT", ...])
    
    # Load training plan if available
    training_plan = None
    if args.training_plan_dir:
        training_plan_dir = Path(args.training_plan_dir)
        training_plan = load_training_plan(training_plan_dir)
    
    # Determine targets (from config or plan)
    targets = load_targets_from_config(args.config)
    
    # Filter targets based on training plan
    if training_plan:
        filtered_targets = filter_targets_by_training_plan(
            targets=targets,
            training_plan=training_plan,
            training_type="cross_sectional"
        )
    else:
        filtered_targets = targets
    
    # Get model families from plan
    families = ["LSTM", "Transformer", "CNN1D"]
    target_families_map = {}
    if training_plan:
        for target in filtered_targets:
            plan_families = get_model_families_for_job(
                training_plan,
                target=target,
                symbol=None,
                training_type="cross_sectional"
            )
            if plan_families:
                seq_families = [f for f in plan_families if f in families]
                if seq_families:
                    target_families_map[target] = seq_families
    
    # Train sequential models
    results = train_models_for_interval_comprehensive(
        interval="cross_sectional",
        targets=filtered_targets,
        mtf_data=mtf_data,
        families=families,
        strategy="single_task",
        output_dir=args.output_dir,
        target_features={t: selected_features for t in filtered_targets},
        target_families=target_families_map if target_families_map else None
    )
    
    return results

if __name__ == "__main__":
    main()
```

## Integration Points

### 1. Training Plan Location

Phase 3 should look for training plan in:
- `METRICS/training_plan/` (relative to output_dir)
- Or accept `--training-plan-dir` argument

### 2. Target Filtering

- Filter targets based on `cross_sectional` jobs in plan
- Log filtering results: `"Filtered X → Y targets"`

### 3. Model Family Filtering

- Extract families from plan per target
- Filter to only sequential models (LSTM, Transformer, CNN1D)
- Use per-target families if available

### 4. Symbol Filtering (Future)

- When symbol-specific training is implemented, filter symbols per target
- Use `filter_symbols_by_training_plan()` for each target

## Backward Compatibility

**Important:** Phase 3 should work **without** training plan (backward compatible):

```python
training_plan = load_training_plan(training_plan_dir)
if training_plan:
    # Use plan for filtering
    filtered_targets = filter_targets_by_training_plan(...)
else:
    # Fall back to all targets
    filtered_targets = targets
    logger.info("No training plan found, using all targets")
```

## Testing

To test Phase 3 with training plan:

1. **Generate training plan first:**
   ```bash
   # Run intelligent trainer to generate plan
   python -m TRAINING.orchestration.intelligent_trainer ...
   ```

2. **Run Phase 3 with plan:**
   ```bash
   python phase3_sequential_models/run_phase3.py \
       --data-dir data \
       --metadata-dir metadata \
       --training-plan-dir results/METRICS/training_plan \
       --output-dir output/sequential_models
   ```

3. **Verify filtering:**
   - Check logs for "Filtered X → Y targets"
   - Verify only approved targets are trained
   - Verify model families match plan

## Summary

**Current Status:** ✅ Phase 3 is integrated and supports training plan!

**How to Use:**

1. **Via IntelligentTrainer** (Recommended):
   ```python
   trainer = IntelligentTrainer(...)
   trainer.train_with_intelligence(families=["LSTM", "Transformer", "CNN1D"])
   ```
   - ✅ Full training plan integration (automatic)
   - ✅ Target/symbol filtering
   - ✅ Model family filtering

2. **Via main.py** (Now Supported):
   ```bash
   python -m TRAINING.training_strategies.main \
       --model-types sequential \
       --training-plan-dir results/METRICS/training_plan \
       ...
   ```
   - ✅ Training plan integration via `--training-plan-dir`
   - ✅ Target filtering
   - ✅ Model family filtering

3. **Custom Scripts** (If needed):
   - Use training plan consumer functions manually
   - See "Option 3" above for code examples

**Integration Points:**
- ✅ `load_training_plan()` - Load plan from disk
- ✅ `filter_targets_by_training_plan()` - Filter targets
- ✅ `get_model_families_for_job()` - Get families per target
- ✅ `train_models_for_interval_comprehensive()` - Accepts `target_families` parameter
- ✅ `main.py` - Now supports `--training-plan-dir` argument

**Phase 3 is ready to use with training plan!** 🎉
