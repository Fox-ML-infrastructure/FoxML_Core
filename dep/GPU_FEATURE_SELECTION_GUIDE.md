# GPU-Accelerated Feature Selection Guide

## Overview

The feature selection pipeline supports GPU acceleration via LightGBM, providing 10-50x speedup compared to CPU processing, especially for large datasets with many features.

## Why GPU for Feature Selection?

- Parallel Processing: GPUs have thousands of cores that can calculate tree splits simultaneously
- High Memory Bandwidth: Much faster data transfer than system RAM
- Efficiency: Even with 7GB VRAM, GPU is typically faster than 100GB+ system RAM

## Quick Start

### 1. Check GPU Setup

Run the verification script:

```bash
python SCRIPTS/check_gpu_setup.py
```

This will check:
- LightGBM GPU support
- OpenCL installation
- GPU memory availability

### 2. Enable GPU Mode

Edit `CONFIG/feature_selection_config.yaml`:

```yaml
lightgbm:
  device: "gpu" # Change from "cpu" to "gpu"
  gpu_platform_id: 0
  gpu_device_id: 0
  max_bin: 63 # Reduces VRAM usage (63 or 127 recommended for 7GB VRAM)
```

### 3. Run Feature Selection

```bash
python SCRIPTS/select_features.py
```

The script will:
- Automatically detect GPU mode
- Test GPU availability
- Fall back to CPU if GPU fails
- Process symbols sequentially on GPU (faster than parallel CPU)
- Auto-convert data to float32 (saves 50% VRAM)

## Memory Optimization for 7GB VRAM

The configuration includes several optimizations for limited VRAM:

### 1. Reduced Binning (`max_bin: 63`)
- Default LightGBM uses 255 bins
- Reducing to 63 saves significant VRAM
- Minimal impact on model quality for feature importance

### 2. Single Precision (`gpu_use_dp: false`)
- Uses float32 instead of float64
- Cuts memory usage in half
- Sufficient precision for feature selection

### 3. Sequential Processing
- Processes one symbol at a time on GPU
- Prevents VRAM overflow from parallel processing
- Each symbol trains much faster than CPU

### 4. Auto Data Conversion
- Script automatically converts DataFrames to float32 when GPU is enabled
- No manual intervention needed

## Installation Guide

### For NVIDIA GPUs

1. Install OpenCL and CUDA:

```bash
sudo apt update
sudo apt install nvidia-opencl-dev ocl-icd-opencl-dev
# Also ensure CUDA toolkit is installed
```

2. Reinstall LightGBM with GPU support:

```bash
pip uninstall lightgbm
pip install lightgbm --install-option=--gpu
```

Or build from source:

```bash
git clone --recursive https://github.com/microsoft/LightGBM
cd LightGBM && mkdir build && cd build
cmake -DUSE_GPU=1 ..
make -j4
cd ../python-package && python setup.py install
```

### For AMD GPUs

1. Install ROCm (follow AMD's official guide)

2. Build LightGBM with OpenCL:

```bash
git clone --recursive https://github.com/microsoft/LightGBM
cd LightGBM && mkdir build && cd build
cmake -DUSE_GPU=1 -DOpenCL_LIBRARY=/opt/rocm/opencl/lib/libOpenCL.so ..
make -j4
cd ../python-package && python setup.py install
```

### Verify Installation

```bash
# Check OpenCL devices
clinfo

# Test LightGBM GPU
python SCRIPTS/check_gpu_setup.py
```

## Configuration Reference

### GPU Parameters

```yaml
lightgbm:
  # Device selection
  device: "gpu" # "gpu" or "cpu"
  gpu_platform_id: 0 # OpenCL platform (usually 0)
  gpu_device_id: 0 # GPU device ID (0 for first GPU)

  # Memory optimization (for 7GB VRAM)
  max_bin: 63 # Histogram bins (63, 127, or 255)
  gpu_use_dp: false # Single precision (float32)
  max_cat_to_onehot: 4 # Limit categorical encoding

  # Other settings
  force_col_wise: false # Must be false for GPU
  n_jobs: 1 # GPU handles parallelism internally
```

### Performance Tuning

For larger VRAM (16GB+):

```yaml
max_bin: 127 # or 255 for maximum quality
```

For smaller VRAM (<7GB):

```yaml
max_bin: 31
num_leaves: 15 # Reduce tree complexity
```

For extremely large datasets:
- Consider sampling: Process a subset of symbols first
- Reduce `n_estimators` from 500 to 100-200
- Use `subsample: 0.5` for row sampling

## Troubleshooting

### Error: "GPU initialization failed"

Cause: OpenCL not installed or GPU not detected

Solution:
1. Run `clinfo` to check OpenCL
2. Install OpenCL drivers (see installation guide above)
3. Verify GPU is not in use by other processes

### Error: "Out of memory"

Cause: Dataset too large for 7GB VRAM

Solutions:
1. Reduce `max_bin` to 31
2. Reduce `num_leaves` to 15
3. Process fewer symbols at once: `--symbols AAPL,MSFT`
4. Use sampling in your data pipeline

### Error: "LightGBM not compiled with GPU support"

Cause: Standard pip install doesn't include GPU

Solution: Reinstall with `--install-option=--gpu` or build from source

### Warning: "GPU slower than expected"

Check:
1. GPU isn't throttling (temperature/power)
2. No other GPU processes running (`nvidia-smi` or `rocm-smi`)
3. Dataset is large enough to benefit (>10k rows, >50 features)

## Performance Expectations

### Typical Speedups (vs 12-core CPU)

| Dataset Size | Features | Speedup |
|-------------|----------|---------|
| Small (10k rows, 50 features) | 2-5x | Better to use CPU parallelism |
| Medium (100k rows, 100 features) | 10-20x | GPU shines here |
| Large (1M+ rows, 200+ features) | 20-50x | Maximum GPU advantage |

### Example: Processing 20 Symbols

CPU Mode (12 workers):
- Time per symbol: ~30-60 seconds
- Total time: ~10-20 minutes
- Memory: 5-10GB RAM per worker

GPU Mode (sequential):
- Time per symbol: ~3-5 seconds
- Total time: ~1-2 minutes
- Memory: 3-5GB VRAM total

## Switching Between CPU and GPU

You can easily switch modes in the config file:

```yaml
# Use GPU
lightgbm:
  device: "gpu"

# Use CPU
lightgbm:
  device: "cpu"
```

Or override via environment variable (future feature):

```bash
LIGHTGBM_DEVICE=cpu python SCRIPTS/select_features.py
```

## FAQ

Q: Can I run multiple feature selection jobs on one GPU?
A: Not recommended. GPU memory is shared, and LightGBM works best with exclusive access.

Q: Will GPU mode give different feature rankings?
A: Results should be nearly identical. Tiny differences may occur due to floating-point precision, but rankings are consistent.

Q: My GPU has 4GB VRAM. Will it work?
A: Possibly. Try `max_bin: 31` and `num_leaves: 15`. Process fewer symbols at once.

Q: Should I use GPU for small datasets?
A: Not necessarily. For <50k rows, CPU parallelism may be faster due to GPU initialization overhead.

Q: Can I use multiple GPUs?
A: LightGBM supports one GPU at a time. You could manually distribute symbols across GPUs using `gpu_device_id`.

## Next Steps

1. Run `python SCRIPTS/check_gpu_setup.py`
2. Enable GPU in config: `device: "gpu"`
3. Test with one symbol: `python SCRIPTS/select_features.py --symbols AAPL`
4. Run full feature selection: `python SCRIPTS/select_features.py`
5. Monitor GPU usage: `watch -n 1 nvidia-smi` (or `rocm-smi`)

## Resources

- [LightGBM GPU Tutorial](https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html)
- [OpenCL Installation Guide](https://github.com/microsoft/LightGBM/tree/master/docs)
- [LightGBM Parameters Reference](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
