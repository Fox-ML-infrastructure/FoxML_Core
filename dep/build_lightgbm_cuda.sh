#!/bin/bash
# Build LightGBM with CUDA support on Arch Linux

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Building LightGBM with CUDA Support for Arch Linux       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check prerequisites
echo "🔍 Checking prerequisites..."
echo ""

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo "❌ CUDA not found!"
    echo ""
    echo "Install CUDA with:"
    echo "   sudo pacman -S cuda cudnn"
    echo ""
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9.]*\).*/\1/p')
echo "✅ CUDA found: version $CUDA_VERSION"

# Check for nvidia-smi
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found! NVIDIA driver may not be installed."
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
echo "✅ GPU detected: $GPU_NAME (${GPU_MEM}MB VRAM)"

# Check for required build tools
echo ""
echo "🔍 Checking build tools..."

for tool in cmake gcc make git; do
    if ! command -v $tool &> /dev/null; then
        echo "❌ $tool not found!"
        echo "   Install with: sudo pacman -S $tool"
        exit 1
    fi
done
echo "✅ All build tools found"

# Confirm before proceeding
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Ready to build LightGBM with CUDA support"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "This will:"
echo "  1. Clone LightGBM repository to /tmp"
echo "  2. Build with CUDA support (may take 5-10 minutes)"
echo "  3. Uninstall old LightGBM from pip"
echo "  4. Install new CUDA-enabled version"
echo ""
read -p "Continue? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Set up build directory
BUILD_DIR="/tmp/LightGBM_build_$(date +%s)"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo ""
echo "📦 Cloning LightGBM repository..."
git clone --recursive --depth 1 https://github.com/microsoft/LightGBM.git
cd LightGBM

echo ""
echo "🔧 Configuring CMake with CUDA support..."
mkdir build && cd build

# Detect GPU compute capability
if command -v nvidia-smi &> /dev/null; then
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.' | tr -d ' ')
    echo "   Detected compute capability: $COMPUTE_CAP"
    
    # Use CUDA_ARCHITECTURES for newer CMake
    CMAKE_ARGS="-DUSE_CUDA=1 -DCUDA_ARCHITECTURES=$COMPUTE_CAP"
else
    CMAKE_ARGS="-DUSE_CUDA=1"
fi

/usr/bin/cmake $CMAKE_ARGS ..

echo ""
echo "🔨 Building LightGBM (this may take 5-10 minutes)..."
echo "   Using $(nproc) CPU cores..."
make -j$(nproc)

echo ""
echo "📦 Installing Python package..."
cd ../python-package

# Uninstall old version
echo "   Removing old LightGBM installation..."
pip uninstall -y lightgbm 2>/dev/null || true

# Install new version
echo "   Installing CUDA-enabled LightGBM..."
pip install -e . --no-build-isolation

echo ""
echo "✅ Build complete!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Testing installation..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Test installation
cd /home/Jennifer/trader
python SCRIPTS/check_gpu_setup.py

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 LightGBM with CUDA support is ready!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Next steps:"
echo "  1. Test with one symbol:"
echo "     python SCRIPTS/select_features.py --symbols AAPL"
echo ""
echo "  2. Run full feature selection:"
echo "     python SCRIPTS/select_features.py"
echo ""
echo "  3. Monitor GPU usage while running:"
echo "     watch -n 1 nvidia-smi"
echo ""
echo "Build directory: $BUILD_DIR"
echo "You can safely delete it after verifying everything works."
echo ""

