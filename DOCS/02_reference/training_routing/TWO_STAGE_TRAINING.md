# 2-Stage Training Approach

**CPU models first, then GPU models - prevents thread pollution and optimizes resource usage.**

## Overview

`--model-types sequential` is now a **pipeline mode** that trains **all 20 models** (both sequential and cross-sectional) using a **2-stage approach**:

1. **Stage 1 (CPU)**: CPU-only / CPU-preferred models run first
2. **Stage 2 (GPU)**: GPU-capable models run second

This prevents thread pollution between CPU and GPU libraries and ensures efficient resource usage.

> **Note**: In this context, `sequential` refers to the **pipeline mode** (2-stage CPU→GPU schedule), not "LSTM-only models." The mode trains both sequential models (LSTM, Transformer, etc.) and cross-sectional models (LightGBM, XGBoost, etc.) in a specific order.

## Stage 1: CPU Models (10 models)

These models run first, using CPU resources:

- **LightGBM** - Gradient boosting (CPU)
- **QuantileLightGBM** - Quantile regression (CPU)
- **XGBoost** - Gradient boosting (CPU)
  > **Note**: In this configuration, XGBoost runs on CPU (no GPU histogram build). If GPU XGBoost is enabled in the future, it should be moved to Stage 2.
- **RewardBased** - Reward-based learning (CPU)
- **NGBoost** - Natural gradient boosting (CPU)
- **GMMRegime** - Gaussian mixture models (CPU)
- **ChangePoint** - Change point detection (CPU)
- **FTRLProximal** - FTRL proximal optimizer (CPU)
- **Ensemble** - Ensemble methods (CPU)
- **MetaLearning** - Meta-learning (CPU, sklearn-based)

## Stage 2: GPU Models (10 models)

These models run second, using GPU when available:

### TensorFlow Models (4 models)
- **MLP** - Multi-layer perceptron (GPU)
- **VAE** - Variational autoencoder (GPU)
- **GAN** - Generative adversarial network (GPU)
- **MultiTask** - Multi-task learning (GPU)

### PyTorch Models (6 models)
- **CNN1D** - 1D convolutional neural network (GPU)
- **LSTM** - Long short-term memory (GPU)
- **Transformer** - Transformer architecture (GPU)
- **TabCNN** - Tabular CNN (GPU)
- **TabLSTM** - Tabular LSTM (GPU)
- **TabTransformer** - Tabular Transformer (GPU)

## Why 2-Stage?

### Thread Pollution Prevention

CPU models (especially tree-based like LightGBM, XGBoost) use OpenMP threading. GPU models use CUDA/TensorFlow/PyTorch threading. Running them together can cause:
- Thread conflicts
- Memory fragmentation
- Slower training times
- Resource contention

### Resource Optimization

1. **CPU Stage**: Uses all CPU cores efficiently
2. **GPU Stage**: Uses GPU memory and compute efficiently
3. **Clean Separation**: No interference between stages

## Usage

### Basic Command

```bash
python -m TRAINING.training_strategies.main \
    --data-dir data \
    --symbols AAPL MSFT GOOGL \
    --model-types sequential
```

**Output:**
```
🎯 Sequential mode: training all models (sequential + cross-sectional): 20 models
📊 Stage 1 (CPU): 10 models - ['LightGBM', 'QuantileLightGBM', 'XGBoost', 'RewardBased', 'NGBoost']...
📊 Stage 2 (GPU): 10 models - 4 TF, 6 Torch, 0 others
```

### Training Order

The system automatically orders models as:
1. **CPU families** (Stage 1)
2. **TF families** (Stage 2, TensorFlow)
3. **Torch families** (Stage 2, PyTorch)
4. **Other families** (Stage 2, if any)

## Plan-Aware Training

Both stages respect the **training plan**:

- **Targets are filtered** using the master training plan before training starts
- **Model families** can be filtered per job based on the plan (future extension)
- The plan is auto-detected from common locations, or can be explicitly specified

## Benefits

✅ **No Thread Conflicts**: CPU and GPU models run separately
✅ **Efficient Resource Usage**: CPU cores used fully in Stage 1, GPU in Stage 2
✅ **Predictable Performance**: Consistent training times
✅ **Memory Management**: Clean memory state between stages
✅ **Error Isolation**: Problems in one stage don't affect the other
✅ **Plan-Driven**: Training respects intelligent routing decisions

## Comparison

### Old Behavior (Sequential Only)
- Only trained 6 sequential models
- No cross-sectional models
- No 2-stage ordering

### New Behavior (Sequential Mode)
- Trains all 20 models (sequential + cross-sectional)
- 2-stage ordering (CPU → GPU)
- Better resource utilization
- Prevents thread pollution

## Technical Details

### Family Classifications

```python
# CPU families (Stage 1)
CPU_FAMS = {
    "LightGBM", "QuantileLightGBM", "XGBoost", "RewardBased",
    "NGBoost", "GMMRegime", "ChangePoint", "FTRLProximal",
    "Ensemble", "MetaLearning"
}

# TensorFlow families (Stage 2)
TF_FAMS = {"MLP", "VAE", "GAN", "MultiTask"}

# PyTorch families (Stage 2)
TORCH_FAMS = {
    "CNN1D", "LSTM", "Transformer",
    "TabCNN", "TabLSTM", "TabTransformer"
}
```

### Ordering Logic

1. Filter families by type (if `--model-types sequential`, include both)
2. Separate into CPU and GPU groups
3. Within GPU: separate TF and Torch
4. Final order: CPU → TF → Torch → Others

## Examples

### Example 1: Full Training (20 Models)

```bash
python -m TRAINING.training_strategies.main \
    --data-dir data \
    --symbols AAPL MSFT GOOGL \
    --model-types sequential
```

**Trains:**
- Stage 1: 10 CPU models
- Stage 2: 10 GPU models (4 TF + 6 Torch)

### Example 2: With Training Plan

```bash
python -m TRAINING.training_strategies.main \
    --data-dir data \
    --symbols AAPL MSFT GOOGL \
    --model-types sequential \
    --training-plan-dir results/METRICS/training_plan
```

**Filters targets and families based on plan, then applies 2-stage ordering.**

### Example 3: Custom Families

```bash
python -m TRAINING.training_strategies.main \
    --data-dir data \
    --symbols AAPL MSFT \
    --families LightGBM XGBoost LSTM Transformer \
    --model-types sequential
```

**Trains:**
- Stage 1: LightGBM, XGBoost (CPU)
- Stage 2: LSTM, Transformer (GPU/Torch)

## Notes

- The 2-stage approach is **automatic** when using `--model-types sequential`
- You can still use `--model-types cross-sectional` for only cross-sectional models
- The ordering is applied **after** filtering by training plan (if provided)
- CPU models use OpenMP threading, GPU models use CUDA/TF/Torch threading
