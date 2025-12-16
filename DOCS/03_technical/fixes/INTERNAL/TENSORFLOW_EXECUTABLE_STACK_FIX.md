# TensorFlow Executable Stack Error Fix

## Error

```
ImportError: libtensorflow_cc.so.2: cannot enable executable stack as shared object requires: Invalid argument
```

## Cause

This is a system-level security issue. The TensorFlow library (`libtensorflow_cc.so.2`) requires an executable stack, but your system's security policy (likely SELinux or PaX) is preventing it.

**Note**: Unlike XGBoost, TensorFlow does **NOT** need to be built from source. Pre-built TensorFlow packages from conda/pip already include GPU support. This is a system security policy issue, not a compilation issue. Building TensorFlow from source is possible but extremely complex (requires Bazel, takes hours) and may not solve the executable stack issue if it's a kernel-level security policy (PaX/grsecurity).

## Solution

### Option 1: Fix TensorFlow Libraries (Recommended)

Use `execstack -c` to **CLEAR** the executable stack flag (the library is marked as requiring it, but your system blocks it):

```bash
# Install execstack if not available
sudo pacman -S execstack  # Arch Linux
# or
sudo apt-get install execstack  # Debian/Ubuntu

# Step 1: Find the main TensorFlow library
python3 << 'EOF'
import os, tensorflow as tf
tf_dir = os.path.dirname(tf.__file__)
lib_path = os.path.join(tf_dir, "libtensorflow_cc.so.2")
print(lib_path)
EOF

# Step 2: Check current flag (should show 'X' = requires executable stack)
execstack -q /path/to/libtensorflow_cc.so.2

# Step 3: CLEAR the flag (use -c to clear, not set)
execstack -c /path/to/libtensorflow_cc.so.2

# Step 4: Verify (should show '-' = does NOT require executable stack)
execstack -q /path/to/libtensorflow_cc.so.2

# Step 5: Fix ALL TensorFlow libraries
find ~/miniconda3/envs/trader_env/lib/python*/site-packages/tensorflow -name "*.so*" -type f -exec execstack -c {} \;

# Step 6: Test TensorFlow import
python3 -c "import tensorflow as tf; print('✅ TensorFlow works!'); print('GPUs:', len(tf.config.list_physical_devices('GPU')))"
```

**Important**: Use `execstack -c` to **CLEAR** the flag (make it `-`), not set it. The library comes with `X` (requires executable stack) which your system blocks, so we clear it to `-` (does not require).

### Option 2: Use CPU-Only TensorFlow (Workaround)

If you can't fix the executable stack issue, install CPU-only TensorFlow:

```bash
conda activate trader_env
pip uninstall tensorflow tensorflow-gpu
pip install tensorflow-cpu
```

**Note**: This disables GPU acceleration but allows TensorFlow to work.

### Option 3: Build TensorFlow from Source (Not Recommended)

Building TensorFlow from source is possible but **extremely complex** and may not solve the executable stack issue:

- Requires Bazel build system
- Takes 2-4+ hours to compile
- Very complex configuration
- May not fix the issue if it's a kernel-level security policy (PaX/grsecurity)
- Pre-built binaries already have GPU support

**Recommendation**: Use Option 1 (execstack fix) or Option 2 (CPU-only) instead.

### Option 4: Disable Security Policy (Not Recommended)

If you have root access and understand the security implications:

```bash
# Disable PaX/ASLR for TensorFlow (NOT RECOMMENDED for production)
# This is a system-level change and may have security implications
# Only if execstack doesn't work and you understand the risks
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

