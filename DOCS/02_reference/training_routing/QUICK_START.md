# Training Plan Quick Start

**One-command training with automatic training plan integration.**

## Sequential Mode (2-Stage Pipeline) - All 20 Models

### Simplest Command

```bash
# Train all models (sequential + cross-sectional) with 2-stage approach
python -m TRAINING.training_strategies.main \
    --data-dir data \
    --symbols AAPL MSFT GOOGL \
    --model-types sequential \
    --output-dir output/sequential_models
```

**What it does:**
- ✅ Trains **all 20 models** (both sequential AND cross-sectional)
- ✅ **2-stage approach**: CPU models first, then GPU models
  - **Stage 1 (CPU)**: 10 CPU-only models (LightGBM, XGBoost, etc.)
  - **Stage 2 (GPU)**: 10 GPU models (MLP, VAE, LSTM, Transformer, etc.)
- ✅ Auto-detects training plan from common locations
- ✅ Filters targets based on plan (if found)
- ✅ Uses model families from plan (if found)

### With Convenience Module

```bash
# Even simpler - uses defaults
python -m TRAINING.training_strategies.train_sequential data AAPL MSFT GOOGL
```

### With Explicit Training Plan

```bash
# Specify training plan location
python -m TRAINING.training_strategies.main \
    --data-dir data \
    --symbols AAPL MSFT GOOGL \
    --model-types sequential \
    --training-plan-dir results/globals/training_plan \
    --output-dir output/sequential_models
```

### Disable Training Plan

```bash
# Skip training plan (train all targets)
python -m TRAINING.training_strategies.main \
    --data-dir data \
    --symbols AAPL MSFT GOOGL \
    --model-types sequential \
    --no-training-plan \
    --output-dir output/sequential_models
```

## Auto-Detection

The system automatically looks for training plans in:
1. `output_dir/globals/training_plan/` (primary - new location)
2. `output_dir/../globals/training_plan/` (same level as output)
3. `output_dir/METRICS/training_plan/` (legacy fallback - inside output_dir)
4. `output_dir/../METRICS/training_plan/` (legacy fallback - same level as output)
5. `results/METRICS/training_plan/` (legacy fallback - common results location)
6. `./results/METRICS/training_plan/` (legacy fallback - current directory)

If found, it automatically:
- Filters targets based on plan
- Uses model families from plan
- Logs: `"📋 Auto-detected training plan: ..."`

## All Models (Full Zoo)

### Train Everything

```bash
# Train all 20 models (cross-sectional + sequential)
python -m TRAINING.training_strategies.main \
    --data-dir data \
    --symbols AAPL MSFT GOOGL \
    --model-types both \
    --output-dir output/all_models
```

### Train Cross-Sectional Only

```bash
# Train all 14 cross-sectional models
python -m TRAINING.training_strategies.main \
    --data-dir data \
    --symbols AAPL MSFT GOOGL \
    --model-types cross-sectional \
    --output-dir output/cross_sectional
```

## Model Families

**Sequential Mode trains all 20 models in 2-stage approach:**

**Stage 1 (CPU - 10 models):**
- LightGBM, QuantileLightGBM, XGBoost
- RewardBased, NGBoost, GMMRegime, ChangePoint
- FTRLProximal, Ensemble, MetaLearning

**Stage 2 (GPU - 10 models):**
- TensorFlow (4): MLP, VAE, GAN, MultiTask
- PyTorch (6): CNN1D, LSTM, Transformer, TabCNN, TabLSTM, TabTransformer

> **Note**: `--model-types sequential` is a **pipeline mode** (2-stage CPU→GPU schedule), not "LSTM-only models."
- TabCNN
- TabLSTM
- TabTransformer

**Cross-Sectional Models (14):**
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

**Total: 20 models**

## Training Plan Integration

### Automatic (Recommended)

Just run the command - training plan is auto-detected if available:

```bash
python -m TRAINING.training_strategies.main \
    --data-dir data \
    --symbols AAPL MSFT GOOGL \
    --model-types sequential
```

### Manual

Specify training plan location:

```bash
python -m TRAINING.training_strategies.main \
    --data-dir data \
    --symbols AAPL MSFT GOOGL \
    --model-types sequential \
    --training-plan-dir results/globals/training_plan
```

### What Gets Filtered

- **Targets**: Only targets with jobs in training plan
- **Model Families**: Only families specified in plan (per-target if available)
- **Symbols**: (Future) Symbols filtered per target

## Examples

### Example 1: Sequential Models with Auto-Detected Plan

```bash
python -m TRAINING.training_strategies.main \
    --data-dir data \
    --symbols AAPL MSFT GOOGL TSLA \
    --model-types sequential \
    --output-dir output/sequential
```

**Output:**
```
📋 Auto-detected training plan: results/globals/training_plan (or results/METRICS/training_plan as fallback)
📋 Training plan filter applied: 10 → 7 targets
🎯 Training only sequential models: 6 models
```

### Example 2: All Models with Training Plan

```bash
python -m TRAINING.training_strategies.main \
    --data-dir data \
    --symbols AAPL MSFT GOOGL \
    --model-types both \
    --training-plan-dir results/globals/training_plan \
    --output-dir output/all_models
```

### Example 3: Custom Families with Plan

```bash
python -m TRAINING.training_strategies.main \
    --data-dir data \
    --symbols AAPL MSFT \
    --families LSTM Transformer CNN1D \
    --model-types sequential \
    --training-plan-dir results/globals/training_plan
```

## Tips

1. **Run IntelligentTrainer first** to generate training plan, then sequential models will auto-detect it
2. **Use `--model-types sequential`** to train all 20 models in 2-stage approach (CPU → GPU)
3. **Training plan is optional** - system works without it (trains all targets)
4. **Auto-detection is smart** - checks common locations automatically

## Troubleshooting

**No training plan detected:**
- Check if `globals/training_plan/master_training_plan.json` exists (or `METRICS/training_plan/master_training_plan.json` as fallback)
- Or specify `--training-plan-dir` explicitly
- Or use `--no-training-plan` to disable

**All targets filtered out:**
- Check training plan has jobs for your targets
- Review `globals/training_plan/training_plan.md` for details (or `METRICS/training_plan/training_plan.md` as fallback)
- Use `--no-training-plan` to train all targets

**Wrong model families:**
- Training plan specifies families per target
- Check plan: `globals/training_plan/master_training_plan.json` (or `METRICS/training_plan/master_training_plan.json` as fallback)
- Or use `--no-training-plan` to use all families
