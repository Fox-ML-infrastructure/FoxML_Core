# One-Command Training Guide

**Simplified commands for training with automatic training plan integration.**

## Sequential Models (All 20 Models in 2-Stage Approach)

### Simplest Command

```bash
python -m TRAINING.training_strategies.main \
    --data-dir data \
    --symbols AAPL MSFT GOOGL \
    --model-types sequential
```

**What it does:**
- ✅ Trains **all 20 models** (both sequential AND cross-sectional)
- ✅ Uses **2-stage approach**: CPU models first, then GPU models
  - **Stage 1 (CPU)**: LightGBM, XGBoost, QuantileLightGBM, RewardBased, NGBoost, GMMRegime, ChangePoint, FTRLProximal, Ensemble, MetaLearning
  - **Stage 2 (GPU)**: MLP, VAE, GAN, MultiTask (TensorFlow) + CNN1D, LSTM, Transformer, TabCNN, TabLSTM, TabTransformer (PyTorch)
- ✅ Auto-detects training plan from common locations
- ✅ Filters targets based on plan (if found)
- ✅ Uses model families from plan (if found)
- ✅ Creates timestamped output directory

### Even Simpler (Convenience Module)

```bash
python -m TRAINING.training_strategies.train_sequential data AAPL MSFT GOOGL
```

**Or with defaults:**
```bash
python -m TRAINING.training_strategies.train_sequential
# Uses: data_dir=data, symbols=AAPL MSFT GOOGL TSLA
```

## How Auto-Detection Works

The system automatically looks for training plans in these locations (in order):

1. `output_dir/../METRICS/training_plan/` (same level as output)
2. `output_dir/METRICS/training_plan/` (inside output_dir)
3. `results/METRICS/training_plan/` (common results location)
4. `./results/METRICS/training_plan/` (current directory)

**If found:**
- Logs: `"📋 Auto-detected training plan: ..."`
- Filters targets based on plan
- Uses model families from plan

**If not found:**
- Trains all targets (backward compatible)
- Uses all model families

## Examples

### Example 1: Basic Sequential Training

```bash
python -m TRAINING.training_strategies.main \
    --data-dir data \
    --symbols AAPL MSFT GOOGL \
    --model-types sequential \
    --output-dir output/sequential
```

**Output:**
```
📋 Auto-detected training plan: results/METRICS/training_plan
📋 Training plan filter applied: 10 → 7 targets
🎯 Training all sequential models: ['CNN1D', 'LSTM', 'Transformer', 'TabCNN', 'TabLSTM', 'TabTransformer']
```

### Example 2: With Explicit Plan

```bash
python -m TRAINING.training_strategies.main \
    --data-dir data \
    --symbols AAPL MSFT GOOGL \
    --model-types sequential \
    --training-plan-dir results/METRICS/training_plan \
    --output-dir output/sequential
```

### Example 3: Without Plan (Train All)

```bash
python -m TRAINING.training_strategies.main \
    --data-dir data \
    --symbols AAPL MSFT GOOGL \
    --model-types sequential \
    --no-training-plan \
    --output-dir output/sequential
```

## All Models (Full Zoo)

### Train Everything (20 Models)

```bash
python -m TRAINING.training_strategies.main \
    --data-dir data \
    --symbols AAPL MSFT GOOGL \
    --model-types both
```

**Trains:**
- 14 cross-sectional models
- 6 sequential models
- Total: 20 models

### Cross-Sectional Only (14 Models)

```bash
python -m TRAINING.training_strategies.main \
    --data-dir data \
    --symbols AAPL MSFT GOOGL \
    --model-types cross-sectional
```

## Model Families Reference

### Sequential Models (6)
- CNN1D
- LSTM
- Transformer
- TabCNN
- TabLSTM
- TabTransformer

### Cross-Sectional Models (14)
- LightGBM
- XGBoost
- MLP
- Ensemble
- RewardBased
- QuantileLightGBM
- NGBoost
- GMMRegime
- ChangePoint
- FTRLProximal
- VAE
- GAN
- MetaLearning
- MultiTask

## Workflow

### Recommended: Two-Step Process

**Step 1: Generate Training Plan**
```bash
# Run intelligent trainer (generates routing + training plan)
python -m TRAINING.orchestration.intelligent_trainer \
    --data-dir data \
    --symbols AAPL MSFT GOOGL \
    --auto-targets --auto-features
```

**Step 2: Train Sequential Models**
```bash
# Sequential models auto-detect the plan
python -m TRAINING.training_strategies.main \
    --data-dir data \
    --symbols AAPL MSFT GOOGL \
    --model-types sequential
```

### Or: One-Step (No Plan)

```bash
# Train everything without plan (trains all targets)
python -m TRAINING.training_strategies.main \
    --data-dir data \
    --symbols AAPL MSFT GOOGL \
    --model-types sequential \
    --no-training-plan
```

## Tips

1. **Use `--model-types sequential`** - Automatically trains all 6 sequential models
2. **Auto-detection is smart** - Finds plan in common locations
3. **Plan is optional** - System works without it (trains all targets)
4. **Output is timestamped** - Each run gets unique directory
5. **Check logs** - Look for "📋 Auto-detected training plan" message

## Troubleshooting

**Plan not detected?**
- Check if `METRICS/training_plan/master_training_plan.json` exists
- Or specify `--training-plan-dir` explicitly
- Or use `--no-training-plan` to disable

**Want to train specific models?**
```bash
# Train only LSTM and Transformer
python -m TRAINING.training_strategies.main \
    --data-dir data \
    --symbols AAPL MSFT \
    --families LSTM Transformer \
    --model-types sequential
```

**Want custom output location?**
```bash
python -m TRAINING.training_strategies.main \
    --data-dir data \
    --symbols AAPL MSFT GOOGL \
    --model-types sequential \
    --output-dir my_custom_output
```
