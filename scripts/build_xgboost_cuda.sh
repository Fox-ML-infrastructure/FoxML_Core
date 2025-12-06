#!/bin/bash
# Build XGBoost with CUDA support from source
# Auto-detects system configuration (conda vs system CUDA, GPU compute capability)

set -e

echo "🔧 Building XGBoost with CUDA support..."

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo "❌ nvcc not found. CUDA toolkit may not be installed."
    echo "   Install with: conda install -c conda-forge cuda-toolkit -y"
    exit 1
fi

# Auto-detect GPU compute capability
echo "🔍 Detecting GPU compute capability..."
if command -v nvidia-smi &> /dev/null; then
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -1 | tr -d '.')
    if [ -n "$COMPUTE_CAP" ]; then
        CUDA_ARCH="$COMPUTE_CAP"
        echo "   ✅ Detected compute capability: $CUDA_ARCH"
    else
        echo "   ⚠️  Could not detect compute capability, defaulting to 86 (RTX 3080/3090)"
        CUDA_ARCH="86"
    fi
else
    echo "   ⚠️  nvidia-smi not available, defaulting to 86 (RTX 3080/3090)"
    CUDA_ARCH="86"
fi

# Create temporary build directory
BUILD_DIR="/tmp/xgboost_build"
if [ -d "$BUILD_DIR" ]; then
    echo "📁 Removing existing build directory..."
    rm -rf "$BUILD_DIR"
fi

echo "📥 Cloning XGBoost repository..."
git clone --recursive --depth 1 https://github.com/dmlc/xgboost.git "$BUILD_DIR"
cd "$BUILD_DIR"

echo "🔨 Building XGBoost with CUDA support..."
mkdir -p build
cd build

# Auto-detect CUDA installation (conda vs system)
echo "🔍 Detecting CUDA installation..."

# Try conda environment first
if [ -n "$CONDA_PREFIX" ] && [ -f "$CONDA_PREFIX/targets/x86_64-linux/include/cuda_runtime.h" ]; then
    echo "   ✅ Found conda CUDA installation"
    CUDA_TOOLKIT_ROOT="$CONDA_PREFIX"
    CUDA_INCLUDE="$CONDA_PREFIX/targets/x86_64-linux/include"
    CUDA_LIB="$CONDA_PREFIX/targets/x86_64-linux/lib"
    CUDA_COMPILER="$CONDA_PREFIX/bin/nvcc"
elif [ -n "$CONDA_PREFIX" ] && [ -f "$CONDA_PREFIX/include/cuda_runtime.h" ]; then
    echo "   ✅ Found conda CUDA installation (alternative location)"
    CUDA_TOOLKIT_ROOT="$CONDA_PREFIX"
    CUDA_INCLUDE="$CONDA_PREFIX/include"
    CUDA_LIB="$CONDA_PREFIX/lib"
    CUDA_COMPILER="$CONDA_PREFIX/bin/nvcc"
# Try system CUDA
elif [ -f "/usr/local/cuda/include/cuda_runtime.h" ]; then
    echo "   ✅ Found system CUDA installation"
    CUDA_TOOLKIT_ROOT="/usr/local/cuda"
    CUDA_INCLUDE="/usr/local/cuda/include"
    CUDA_LIB="/usr/local/cuda/lib64"
    CUDA_COMPILER="/usr/local/cuda/bin/nvcc"
# Try common system locations
elif [ -f "/opt/cuda/include/cuda_runtime.h" ]; then
    echo "   ✅ Found system CUDA installation (alternative location)"
    CUDA_TOOLKIT_ROOT="/opt/cuda"
    CUDA_INCLUDE="/opt/cuda/include"
    CUDA_LIB="/opt/cuda/lib64"
    CUDA_COMPILER="/opt/cuda/bin/nvcc"
else
    echo "❌ ERROR: Could not find CUDA headers (cuda_runtime.h)"
    echo ""
    echo "   Searched locations:"
    echo "     - $CONDA_PREFIX/targets/x86_64-linux/include/"
    echo "     - $CONDA_PREFIX/include/"
    echo "     - /usr/local/cuda/include/"
    echo "     - /opt/cuda/include/"
    echo ""
    echo "   Solutions:"
    echo "     1. For conda: conda install -c conda-forge cuda-toolkit -y"
    echo "     2. For system: Install CUDA toolkit from NVIDIA"
    echo "     3. Set CUDA_TOOLKIT_ROOT_DIR manually if CUDA is in a custom location"
    exit 1
fi

# Verify CUDA headers exist
if [ ! -f "$CUDA_INCLUDE/cuda_runtime.h" ]; then
    echo "❌ ERROR: cuda_runtime.h not found at $CUDA_INCLUDE"
    exit 1
fi

echo "📁 CUDA toolkit root: $CUDA_TOOLKIT_ROOT"
echo "📁 CUDA include: $CUDA_INCLUDE"
echo "📁 CUDA lib: $CUDA_LIB"
echo "📁 CUDA compiler: $CUDA_COMPILER"

# Set environment variables for CMake
export CUDA_TOOLKIT_ROOT_DIR="$CUDA_TOOLKIT_ROOT"
export CUDA_PATH="$CUDA_TOOLKIT_ROOT"
export CUDA_HOME="$CUDA_TOOLKIT_ROOT"
export PATH="$(dirname $CUDA_COMPILER):$PATH"
export LD_LIBRARY_PATH="$CUDA_LIB:$LD_LIBRARY_PATH"

# Determine install prefix (use conda if available, otherwise system default)
if [ -n "$CONDA_PREFIX" ]; then
    INSTALL_PREFIX="$CONDA_PREFIX"
else
    INSTALL_PREFIX="/usr/local"
fi

echo "📦 Install prefix: $INSTALL_PREFIX"

cmake .. \
    -DUSE_CUDA=ON \
    -DCUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_TOOLKIT_ROOT" \
    -DCUDA_INCLUDE_DIRS="$CUDA_INCLUDE" \
    -DCUDA_CUDART_LIBRARY="$CUDA_LIB/libcudart.so" \
    -DCMAKE_CUDA_COMPILER="$CUDA_COMPILER" \
    -DCMAKE_CUDA_HOST_COMPILER="$(which g++)"

make -j$(nproc)

echo "📦 Installing XGBoost Python package..."
cd ../python-package
pip install -e . --no-deps

echo "✅ XGBoost with CUDA support installed!"
echo ""
echo "Testing installation..."
python3 << 'PYEOF'
import xgboost as xgb
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

try:
    test_data = xgb.DMatrix([[1, 2, 3]], label=[1])
    xgb.train({"tree_method": "gpu_hist", "max_depth": 1}, test_data, num_boost_round=1)
    print("✅ SUCCESS! XGBoost GPU support is working!")
except Exception as e:
    print(f"❌ GPU support not working: {e}")
PYEOF

echo ""
echo "🎉 Done! XGBoost GPU support should now be available."

