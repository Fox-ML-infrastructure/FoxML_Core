# XGBoost GPU Build Guide

This guide explains how to build XGBoost from source with CUDA support, as conda and pip packages don't expose GPU functionality in their Python bindings.

## Why Build from Source?

Even though XGBoost conda packages show `USE_CUDA=True` in build info, the Python bindings don't expose the `gpu_hist` tree method. This is a known limitation of pre-built packages. To use XGBoost with GPU acceleration, you must build from source.

## Prerequisites

1. **CUDA Toolkit** installed in your conda environment:
   ```bash
   conda install -c conda-forge cuda-toolkit cuda-nvcc -y
   ```

2. **CUDA Headers** must be available. Verify with:
   ```bash
   find $CONDA_PREFIX -name "cuda_runtime.h" | grep -v tensorflow
   ```
   Should find: `$CONDA_PREFIX/targets/x86_64-linux/include/cuda_runtime.h`

3. **CMake** and build tools:
   ```bash
   conda install cmake make gcc_linux-64 gxx_linux-64 -y
   ```

4. **Git** (for cloning XGBoost)

## Quick Build

Use the provided build script:

```bash
bash scripts/build_xgboost_cuda.sh
```

This script:
- Clones XGBoost from source
- Builds with CUDA support
- Installs the Python package
- Tests GPU functionality

**Build time:** 10-20 minutes

## Manual Build

If the script doesn't work for your system, follow these steps:

### 1. Clone XGBoost

```bash
cd /tmp
git clone --recursive https://github.com/dmlc/xgboost.git
cd xgboost
```

### 2. Configure CUDA Paths

**For Conda Environments:**

Find your CUDA paths:
```bash
CUDA_INCLUDE="$CONDA_PREFIX/targets/x86_64-linux/include"
CUDA_LIB="$CONDA_PREFIX/targets/x86_64-linux/lib"
CUDA_COMPILER="$CONDA_PREFIX/bin/nvcc"
```

**For System CUDA:**

```bash
CUDA_INCLUDE="/usr/local/cuda/include"
CUDA_LIB="/usr/local/cuda/lib64"
CUDA_COMPILER="/usr/local/cuda/bin/nvcc"
```

### 3. Determine GPU Compute Capability

Find your GPU's compute capability:
- RTX 3080: `86`
- RTX 3090: `86`
- RTX 4090: `89`
- A100: `80`
- V100: `70`

Check with:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

### 4. Build XGBoost

```bash
mkdir build && cd build

cmake .. \
    -DUSE_CUDA=ON \
    -DCUDA_ARCHITECTURES="86" \  # Replace with your GPU's compute capability
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
    -DCUDA_TOOLKIT_ROOT_DIR="$CONDA_PREFIX" \
    -DCUDA_INCLUDE_DIRS="$CUDA_INCLUDE" \
    -DCUDA_CUDART_LIBRARY="$CUDA_LIB/libcudart.so" \
    -DCMAKE_CUDA_COMPILER="$CUDA_COMPILER" \
    -DCMAKE_CUDA_HOST_COMPILER="$(which g++)"

make -j$(nproc)
```

### 5. Install Python Package

```bash
cd ../python-package
pip install -e . --no-deps
```

### 6. Verify Installation

```python
import xgboost as xgb
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

test_data = xgb.DMatrix([[1, 2, 3]], label=[1])
# XGBoost 3.x API: use device='cuda' with tree_method='hist' (not gpu_hist)
xgb.train({"device": "cuda", "tree_method": "hist", "max_depth": 1}, test_data, num_boost_round=1)
print("✅ XGBoost GPU support is working!")
```

## Common Issues

### Issue: "cuda_runtime.h: No such file or directory"

**Solution:**
```bash
conda install -c conda-forge cuda-toolkit -y
```

Verify headers exist:
```bash
find $CONDA_PREFIX -name "cuda_runtime.h" | grep -v tensorflow
```

### Issue: CMake can't find CUDA

**Solution:** Set environment variables explicitly:
```bash
export CUDA_TOOLKIT_ROOT_DIR="$CONDA_PREFIX"
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"
```

### Issue: Wrong CUDA architecture

**Solution:** Check your GPU's compute capability and update `-DCUDA_ARCHITECTURES`:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

### Issue: Build fails with compiler errors

**Solution:** Ensure compatible compiler versions:
```bash
conda install gcc_linux-64 gxx_linux-64 -y
```

### Issue: "gpu_hist" still not available after build

**Solution:** XGBoost 3.x uses a different API! Use `device='cuda'` with `tree_method='hist'` instead of `tree_method='gpu_hist'`:

```python
# Old API (XGBoost 2.x):
xgb.train({"tree_method": "gpu_hist", ...}, ...)

# New API (XGBoost 3.x):
xgb.train({"device": "cuda", "tree_method": "hist", ...}, ...)
```

If you still have issues:
1. Verify the build completed successfully
2. Uninstall old XGBoost: `pip uninstall xgboost -y`
3. Reinstall from the built package
4. Restart Python/conda environment

## System-Specific Configuration

### Conda Environment (Recommended)

The build script is configured for conda environments. It automatically:
- Finds CUDA in `$CONDA_PREFIX/targets/x86_64-linux/`
- Sets all necessary environment variables
- Uses conda's CUDA compiler

### System-Wide CUDA Installation

If using system CUDA instead of conda:

1. Update the build script to point to system CUDA:
   ```bash
   CUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda"
   CUDA_INCLUDE_DIRS="/usr/local/cuda/include"
   CUDA_LIB="/usr/local/cuda/lib64"
   ```

2. Ensure CUDA version matches your driver:
   ```bash
   nvcc --version
   nvidia-smi
   ```

### Multiple CUDA Versions

If you have multiple CUDA installations:

1. Set `CUDA_TOOLKIT_ROOT_DIR` to the version you want
2. Ensure `nvcc` in PATH matches that version
3. Verify library compatibility

## Troubleshooting

### Check CUDA Installation

```bash
# Verify nvcc
nvcc --version

# Verify GPU
nvidia-smi

# Verify headers
find $CONDA_PREFIX -name "cuda_runtime.h" | head -1

# Verify libraries
ls $CONDA_PREFIX/targets/x86_64-linux/lib/libcudart.so*
```

### Check XGBoost Build

```python
import xgboost as xgb
build_info = xgb.build_info()
print(f"USE_CUDA: {build_info.get('USE_CUDA')}")
print(f"CUDA_VERSION: {build_info.get('CUDA_VERSION')}")
```

### Test GPU Functionality

```python
import xgboost as xgb
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# This should work if GPU support is properly built
# XGBoost 3.x API: use device='cuda' with tree_method='hist'
test_data = xgb.DMatrix([[1, 2, 3]], label=[1])
try:
    xgb.train({"device": "cuda", "tree_method": "hist", "max_depth": 1}, test_data, num_boost_round=1)
    print("✅ GPU support working!")
except (ValueError, RuntimeError) as e:
    print(f"❌ GPU support not available: {e}")
```

## Performance Notes

- GPU acceleration is most beneficial for large datasets (>100K samples)
- For smaller datasets, CPU may be faster due to GPU overhead
- Memory usage on GPU is higher - monitor VRAM usage
- XGBoost will automatically fall back to CPU if GPU isn't available

## References

- [XGBoost Build Documentation](https://xgboost.readthedocs.io/en/stable/build.html)
- [CUDA Compute Capabilities](https://developer.nvidia.com/cuda-gpus)
- [Conda CUDA Packages](https://anaconda.org/conda-forge/cuda-toolkit)

