# Training Plan System - Ready to Use

**The system is fully implemented, hardened, and ready for one-command usage.**

## ✅ What's Ready

### 1. Master Training Plan Structure
- ✅ `master_training_plan.json` - Single source of truth
- ✅ Derived views (by_target, by_symbol, by_type, by_route)
- ✅ Full metadata (run_id, git_commit, config_hash, etc.)

### 2. Training Plan Integration
- ✅ Automatic generation after routing
- ✅ Automatic consumption in training phase
- ✅ Auto-detection from common locations
- ✅ Backward compatible (works without plan)

### 3. Sequential Mode (2-Stage Pipeline)
- ✅ Fully integrated with training plan
- ✅ Auto-detects plan automatically
- ✅ Trains all 20 models (6 sequential + 14 cross-sectional) in 2-stage approach
- ✅ One-command usage

### 4. Error Handling
- ✅ Comprehensive input validation
- ✅ Graceful error handling
- ✅ Safe fallbacks
- ✅ Clear error messages

## 🚀 Quick Start

### Sequential Models (Simplest)

```bash
# Train all 6 sequential models with auto-detected plan
python -m TRAINING.training_strategies.main \
    --data-dir data \
    --symbols AAPL MSFT GOOGL \
    --model-types sequential
```

**That's it!** The system will:
1. Auto-detect training plan (if available)
2. Train all 20 models in 2-stage approach (CPU → GPU)
3. Filter targets based on plan
4. Use model families from plan

### Or Use Convenience Module

```bash
python -m TRAINING.training_strategies.train_sequential \
    data AAPL MSFT GOOGL
```

## 📋 What Gets Trained

### Sequential Mode (2-Stage Pipeline)
When you use `--model-types sequential`, it trains **all 20 models** in a 2-stage approach:

**Stage 1 (CPU - 10 models):**
- LightGBM, QuantileLightGBM, XGBoost
- RewardBased, NGBoost, GMMRegime, ChangePoint
- FTRLProximal, Ensemble, MetaLearning

**Stage 2 (GPU - 10 models):**
- TensorFlow (4): MLP, VAE, GAN, MultiTask
- PyTorch (6): CNN1D, LSTM, Transformer, TabCNN, TabLSTM, TabTransformer

> **Note**: `--model-types sequential` is a **pipeline mode** (2-stage CPU→GPU schedule), not "LSTM-only models." It trains both sequential and cross-sectional models in an optimized order.

## 🔍 Auto-Detection

Training plan is automatically detected from:
1. `output_dir/globals/training_plan/` (primary - new location)
2. `output_dir/../globals/training_plan/` (same level as output)
3. `output_dir/METRICS/training_plan/` (legacy fallback - inside output_dir)
4. `output_dir/../METRICS/training_plan/` (legacy fallback - same level as output)
5. `results/METRICS/training_plan/` (legacy fallback - common results location)
6. `./results/METRICS/training_plan/` (legacy fallback - current directory)

**No need to specify `--training-plan-dir`** unless you want a custom location!

## 📚 Documentation

- `QUICK_START.md` - Quick reference guide
- `ONE_COMMAND_TRAINING.md` - Detailed one-command examples
**For architecture and implementation details**, see the internal documentation.

## ✨ Features

- ✅ **One-command usage** - Just specify `--model-types sequential`
- ✅ **Auto-detection** - Finds training plan automatically
- ✅ **All models** - Trains all 20 models (sequential + cross-sectional) in 2-stage approach
- ✅ **Plan integration** - Filters targets and families automatically
- ✅ **Error handling** - Comprehensive validation and fallbacks
- ✅ **Backward compatible** - Works without training plan
- ✅ **2-stage optimization** - CPU models first, then GPU models (prevents thread pollution)

## 🎯 Example Workflow

```bash
# Step 1: Generate training plan (optional)
python -m TRAINING.orchestration.intelligent_trainer \
    --data-dir data \
    --symbols AAPL MSFT GOOGL \
    --auto-targets --auto-features

# Step 2: Train all models with 2-stage approach (auto-detects plan)
python -m TRAINING.training_strategies.main \
    --data-dir data \
    --symbols AAPL MSFT GOOGL \
    --model-types sequential
```

**Or skip step 1 and train without plan:**
```bash
python -m TRAINING.training_strategies.main \
    --data-dir data \
    --symbols AAPL MSFT GOOGL \
    --model-types sequential \
    --no-training-plan
```

## 🛡️ Safety Features

- ✅ Input validation on all entry points
- ✅ Type checking before operations
- ✅ Safe defaults on errors
- ✅ Graceful degradation
- ✅ Clear error messages
- ✅ Comprehensive logging

## 📊 Status

**System Status:** ✅ **Production Ready**

- All core features implemented
- Error handling comprehensive
- Documentation complete
- One-command usage available
- Auto-detection working
- Backward compatible

**Ready to use in production!** 🎉
