#!/bin/bash
# Build script for IBKR Trading Engine C++ components

set -e

echo "🚀 Building IBKR Trading Engine C++ Components"
echo "=============================================="

# Configuration
BUILD_DIR="build"
INSTALL_DIR="install"
CMAKE_BUILD_TYPE="Release"
PARALLEL_JOBS=$(nproc)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check dependencies
check_dependencies() {
    print_status "Checking dependencies..."
    
    # Check for required tools
    local missing_deps=()
    
    if ! command -v cmake &> /dev/null; then
        missing_deps+=("cmake")
    fi
    
    if ! command -v make &> /dev/null; then
        missing_deps+=("make")
    fi
    
    if ! command -v g++ &> /dev/null; then
        missing_deps+=("g++")
    fi
    
    if ! command -v python3 &> /dev/null; then
        missing_deps+=("python3")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_error "Missing dependencies: ${missing_deps[*]}"
        print_status "Please install missing dependencies and try again"
        exit 1
    fi
    
    print_success "All dependencies found"
}

# Check Python packages
check_python_packages() {
    print_status "Checking Python packages..."
    
    local missing_packages=()
    
    if ! python3 -c "import pybind11" &> /dev/null; then
        missing_packages+=("pybind11")
    fi
    
    if ! python3 -c "import numpy" &> /dev/null; then
        missing_packages+=("numpy")
    fi
    
    if [ ${#missing_packages[@]} -ne 0 ]; then
        print_warning "Missing Python packages: ${missing_packages[*]}"
        print_status "Installing missing packages..."
        pip3 install ${missing_packages[*]}
    fi
    
    print_success "Python packages ready"
}

# Check system libraries
check_system_libraries() {
    print_status "Checking system libraries..."
    
    # Check for OpenMP
    if ! pkg-config --exists omp; then
        print_warning "OpenMP not found, trying alternative..."
        if ! pkg-config --exists openmp; then
            print_error "OpenMP not found. Please install libomp-dev or libgomp-dev"
            exit 1
        fi
    fi
    
    # Check for BLAS
    if ! pkg-config --exists blas; then
        print_warning "BLAS not found, trying alternative..."
        if ! pkg-config --exists openblas; then
            print_error "BLAS not found. Please install libblas-dev or libopenblas-dev"
            exit 1
        fi
    fi
    
    # Check for LAPACK
    if ! pkg-config --exists lapack; then
        print_warning "LAPACK not found, trying alternative..."
        if ! pkg-config --exists openblas; then
            print_error "LAPACK not found. Please install liblapack-dev or libopenblas-dev"
            exit 1
        fi
    fi
    
    print_success "System libraries ready"
}

# Create build directory
create_build_directory() {
    print_status "Creating build directory..."
    
    if [ -d "$BUILD_DIR" ]; then
        print_warning "Build directory exists, cleaning..."
        rm -rf "$BUILD_DIR"
    fi
    
    mkdir -p "$BUILD_DIR"
    mkdir -p "$INSTALL_DIR"
    
    print_success "Build directory created"
}

# Configure with CMake
configure_cmake() {
    print_status "Configuring with CMake..."
    
    cd "$BUILD_DIR"
    
    cmake .. \
        -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
        -DCMAKE_INSTALL_PREFIX="../$INSTALL_DIR" \
        -DCMAKE_CXX_COMPILER=g++ \
        -DCMAKE_CXX_STANDARD=20 \
        -DCMAKE_CXX_STANDARD_REQUIRED=ON \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DCMAKE_VERBOSE_MAKEFILE=ON
    
    if [ $? -ne 0 ]; then
        print_error "CMake configuration failed"
        exit 1
    fi
    
    cd ..
    print_success "CMake configuration complete"
}

# Build the project
build_project() {
    print_status "Building project..."
    
    cd "$BUILD_DIR"
    
    make -j"$PARALLEL_JOBS"
    
    if [ $? -ne 0 ]; then
        print_error "Build failed"
        exit 1
    fi
    
    cd ..
    print_success "Build complete"
}

# Run tests
run_tests() {
    print_status "Running tests..."
    
    cd "$BUILD_DIR"
    
    # Run unit tests
    if [ -f "test_inference" ]; then
        print_status "Running inference tests..."
        ./test_inference
    fi
    
    if [ -f "test_features" ]; then
        print_status "Running feature tests..."
        ./test_features
    fi
    
    if [ -f "test_market_data" ]; then
        print_status "Running market data tests..."
        ./test_market_data
    fi
    
    cd ..
    print_success "Tests completed"
}

# Run benchmarks
run_benchmarks() {
    print_status "Running benchmarks..."
    
    cd "$BUILD_DIR"
    
    if [ -f "benchmark_inference" ]; then
        print_status "Running inference benchmark..."
        ./benchmark_inference
    fi
    
    if [ -f "benchmark_features" ]; then
        print_status "Running feature benchmark..."
        ./benchmark_features
    fi
    
    if [ -f "benchmark_market_data" ]; then
        print_status "Running market data benchmark..."
        ./benchmark_market_data
    fi
    
    cd ..
    print_success "Benchmarks completed"
}

# Install the project
install_project() {
    print_status "Installing project..."
    
    cd "$BUILD_DIR"
    
    make install
    
    if [ $? -ne 0 ]; then
        print_error "Installation failed"
        exit 1
    fi
    
    cd ..
    print_success "Installation complete"
}

# Create Python package
create_python_package() {
    print_status "Creating Python package..."
    
    # Create setup.py for the Python module
    cat > setup.py << 'EOF'
from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
import os

# Get the build directory
build_dir = "build"

# Define the extension
ext_modules = [
    Pybind11Extension(
        "ibkr_trading_engine_py",
        [
            "python_bindings/inference_engine_bindings.cpp",
            "python_bindings/feature_pipeline_bindings.cpp",
            "python_bindings/market_data_bindings.cpp",
            "python_bindings/linear_algebra_bindings.cpp",
        ],
        include_dirs=[
            "include",
            pybind11.get_include(),
        ],
        libraries=["ibkr_trading_engine"],
        library_dirs=[os.path.join(build_dir, "src")],
        language='c++',
        cxx_std=20,
    ),
]

setup(
    name="ibkr_trading_engine",
    version="1.0.0",
    author="IBKR Trading Team",
    description="High-performance C++ trading engine with Python bindings",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
)
EOF
    
    # Build Python package
    python3 setup.py build_ext --inplace
    
    if [ $? -ne 0 ]; then
        print_error "Python package creation failed"
        exit 1
    fi
    
    print_success "Python package created"
}

# Main build process
main() {
    echo "Starting build process..."
    echo "Build type: $CMAKE_BUILD_TYPE"
    echo "Parallel jobs: $PARALLEL_JOBS"
    echo ""
    
    check_dependencies
    check_python_packages
    check_system_libraries
    create_build_directory
    configure_cmake
    build_project
    run_tests
    run_benchmarks
    install_project
    create_python_package
    
    echo ""
    print_success "🎉 Build process completed successfully!"
    echo ""
    echo "📁 Build artifacts:"
    echo "  - C++ library: $INSTALL_DIR/lib/libibkr_trading_engine.so"
    echo "  - Python module: ibkr_trading_engine_py.so"
    echo "  - Headers: $INSTALL_DIR/include/"
    echo ""
    echo "🚀 Next steps:"
    echo "  1. Test the Python module: python3 -c 'import ibkr_trading_engine_py'"
    echo "  2. Run benchmarks: ./build/benchmark_inference"
    echo "  3. Integrate with Python trading system"
}

# Run main function
main "$@"
