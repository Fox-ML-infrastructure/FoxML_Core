# GPU Setup Guide

Configure GPU acceleration for target ranking, feature selection, and model training.

## Overview

GPU acceleration provides 10-50x speedup for target ranking, feature selection, and training, especially with large datasets (>100k samples).

**NEW (2025-12-12)**: GPU acceleration is now enabled for target ranking and feature selection in addition to model training. LightGBM, XGBoost, and CatBoost automatically use GPU when available.

## Prerequisites

- NVIDIA GPU with CUDA support (current)
- CUDA toolkit installed
- OpenCL drivers (for LightGBM GPU)
- 7GB+ VRAM recommended

**Note**: ROCm (AMD GPU) support is planned for the future once the major architecture is solidified. The current implementation focuses on NVIDIA CUDA.

## Quick Start

### 1. Check GPU Setup

```bash
python SCRIPTS/check_gpu_setup.py
```

This verifies:
- LightGBM GPU support
- OpenCL installation
- GPU memory availability

### 2. Enable GPU Mode

Edit `feature_selection_config.yaml` in the `CONFIG/` directory (see [Configuration Reference](../../02_reference/configuration/README.md)):

```yaml
lightgbm:
  device: "gpu"  # Change from "cpu" to "gpu"
  gpu_platform_id: 0
  gpu_device_id: 0
  max_bin: 63  # Reduces VRAM usage (63 or 127 for 7GB VRAM)
```

### 3. Run Feature Selection

```bash
python SCRIPTS/select_features.py
```

The script automatically:
- Detects GPU mode
- Tests GPU availability
- Falls back to CPU if GPU fails
- Processes symbols sequentially on GPU

## Memory Optimization

### For 7GB VRAM

```yaml
lightgbm:
  device: "gpu"
  max_bin: 63
  num_leaves: 31
  max_depth: 5
```

### For 11GB+ VRAM

```yaml
lightgbm:
  device: "gpu"
  max_bin: 255
  num_leaves: 127
  max_depth: 7
```

## Performance Expectations

### Target Ranking
- **CPU**: ~5-15 minutes per target (with 11 model families)
- **GPU**: ~1-3 minutes per target (10-50x speedup on large datasets)

### Feature Selection
- **CPU**: ~2-5 minutes per symbol (421 features)
- **GPU (7GB)**: ~10-30 seconds per symbol
- **GPU (11GB+)**: ~5-15 seconds per symbol

### Model Training
- **CPU**: Varies by model family
- **GPU**: 10-50x speedup for supported families (LightGBM, XGBoost, CatBoost, neural networks)

## Troubleshooting

### GPU Not Being Used

**Check Logs:**
- Look for `✅ Using GPU (CUDA) for [Model]` - GPU is active
- Look for `⚠️ [Model] GPU test failed` - GPU not available, using CPU

**Common Issues:**
1. **XGBoost not compiled with GPU support**
   - Install XGBoost with GPU: `pip install xgboost --upgrade`
   - Or build from source with CUDA support

2. **CatBoost GPU not available**
   - Ensure CatBoost was installed with GPU support
   - Check CUDA drivers: `nvidia-smi`

3. **LightGBM GPU not detected**
   - Verify CUDA installation: `nvidia-smi`
   - Check OpenCL: `clinfo` (for OpenCL fallback)
   - Reinstall LightGBM with GPU: `pip install lightgbm --install-option=--gpu`

### GPU Test Disabled

To skip GPU test for faster startup (not recommended):
```yaml
gpu:
  lightgbm:
    test_enabled: false  # Skip GPU test
  xgboost:
    test_enabled: false
  catboost:
    test_enabled: false
```

### Out of Memory

For model training, reduce VRAM usage in model configs:
- LightGBM: Reduce `max_bin` to 63 or 31
- XGBoost: Reduce `max_depth` or `tree_method` complexity
- CatBoost: Reduce `iterations` or `depth`

### Fallback to CPU

If GPU fails, the system automatically falls back to CPU. Check logs for specific error messages explaining why GPU failed.

## Future: ROCm Support

ROCm (AMD GPU) support is planned for future development once the major architecture is solidified. This will enable:

- TensorFlow GPU acceleration on AMD hardware
- XGBoost GPU support via ROCm
- LightGBM GPU support on AMD GPUs
- Same abstraction layer as CUDA for seamless switching

The implementation will follow the same patterns as CUDA support but use the ROCm backend. This expansion will increase accessibility and deployment options for users with AMD hardware.

## Configuration Reference

All GPU settings are in `CONFIG/training_config/gpu_config.yaml`. See:
- [Training Pipeline Configs](../../02_reference/configuration/TRAINING_PIPELINE_CONFIGS.md) - GPU configuration details
- [Configuration System Overview](../../02_reference/configuration/README.md) - Complete config system guide

## See Also

- [Auto Target Ranking](../../training/AUTO_TARGET_RANKING.md) - How to use GPU-accelerated target ranking
- [Feature Selection Tutorial](../../training/FEATURE_SELECTION_TUTORIAL.md) - GPU-accelerated feature selection
- [Roadmap](../../../ROADMAP.md) - See Phase 4 for multi-GPU support timeline
- [Known Issues](../../02_reference/KNOWN_ISSUES.md) - GPU troubleshooting and limitations

