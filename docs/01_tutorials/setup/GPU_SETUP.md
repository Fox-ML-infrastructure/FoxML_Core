# GPU Setup Guide

Configure GPU acceleration for feature selection and model training.

## Overview

GPU acceleration provides 10-50x speedup for feature selection and training, especially with large datasets.

## Prerequisites

- NVIDIA GPU with CUDA support
- CUDA toolkit installed
- OpenCL drivers (for LightGBM GPU)
- 7GB+ VRAM recommended

## Quick Start

### 1. Check GPU Setup

```bash
python scripts/check_gpu_setup.py
```

This verifies:
- LightGBM GPU support
- OpenCL installation
- GPU memory availability

### 2. Enable GPU Mode

Edit `CONFIG/feature_selection_config.yaml`:

```yaml
lightgbm:
  device: "gpu"  # Change from "cpu" to "gpu"
  gpu_platform_id: 0
  gpu_device_id: 0
  max_bin: 63  # Reduces VRAM usage (63 or 127 for 7GB VRAM)
```

### 3. Run Feature Selection

```bash
python scripts/select_features.py
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

- **CPU**: ~2-5 minutes per symbol (421 features)
- **GPU (7GB)**: ~10-30 seconds per symbol
- **GPU (11GB+)**: ~5-15 seconds per symbol

## Troubleshooting

### GPU Not Detected

1. Verify CUDA installation: `nvidia-smi`
2. Check OpenCL: `clinfo`
3. Reinstall LightGBM with GPU: `pip install lightgbm --install-option=--gpu`

### Out of Memory

Reduce `max_bin` to 63 or 31, or reduce `num_leaves`.

### Fallback to CPU

If GPU fails, the system automatically falls back to CPU. Check logs for error messages.

## See Also

- [GPU Feature Selection Guide](../../../dep/GPU_FEATURE_SELECTION_GUIDE.md) - Detailed guide
- [Quick Start GPU](../../../dep/QUICK_START_GPU.md) - Quick reference

