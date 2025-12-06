# TensorFlow Executable Stack Error Fix

## Error

```
ImportError: libtensorflow_cc.so.2: cannot enable executable stack as shared object requires: Invalid argument
```

## Cause

This is a system-level security issue. The TensorFlow library (`libtensorflow_cc.so.2`) requires an executable stack, but your system's security policy (likely SELinux or PaX) is preventing it.

## Solution

### Option 1: Fix TensorFlow Libraries (Recommended)

Use `execstack` to mark the TensorFlow libraries as requiring an executable stack:

```bash
# Install execstack if not available
sudo pacman -S execstack  # Arch Linux
# or
sudo apt-get install execstack  # Debian/Ubuntu

# Find all TensorFlow libraries
find ~/miniconda3/envs/trader_env/lib/python*/site-packages/tensorflow -name "*.so*" -type f

# Fix all TensorFlow shared libraries
find ~/miniconda3/envs/trader_env/lib/python*/site-packages/tensorflow -name "*.so*" -type f -exec execstack -c {} \;

# Verify the fix
find ~/miniconda3/envs/trader_env/lib/python*/site-packages/tensorflow -name "*.so*" -type f -exec scanelf -qe {} \; | grep -i "rwx"
```

### Option 2: Use CPU-Only TensorFlow (Workaround)

If you can't fix the executable stack issue, install CPU-only TensorFlow:

```bash
conda activate trader_env
pip uninstall tensorflow tensorflow-gpu
pip install tensorflow-cpu
```

**Note**: This disables GPU acceleration but allows TensorFlow to work.

### Option 3: Disable Security Policy (Not Recommended)

If you have root access and understand the security implications:

```bash
# Disable PaX/ASLR for TensorFlow (NOT RECOMMENDED for production)
# This is a system-level change and may have security implications
```

## Verification

After applying the fix, test TensorFlow import:

```bash
conda activate trader_env
python3 -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU available:', len(tf.config.list_physical_devices('GPU')) > 0)"
```

## Child Process Note

If the error appears in child processes (isolation runner), ensure the fix is applied to **all** TensorFlow libraries, not just the main one. Child processes load TensorFlow from the same conda environment, so the fix should apply to both parent and child.

## See Also

- [GPU Setup Guide](../../01_tutorials/setup/GPU_SETUP.md) - GPU configuration
- [XGBoost GPU Build](../setup/XGBOOST_GPU_BUILD.md) - Related GPU setup

