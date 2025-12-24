#!/usr/bin/env python3
"""
Quick diagnostic script to verify LightGBM GPU support and usage.

This will:
1. Check if LightGBM has GPU support compiled in
2. Test CUDA and OpenCL devices
3. Show actual GPU usage during training
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add project root to path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

print("=" * 70)
print("LightGBM GPU Diagnostic")
print("=" * 70)
print()

# Check environment
print("Environment:")
print(f"  CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES', 'not set')}")
print()

# Check LightGBM installation
try:
    import lightgbm as lgb
    print(f"✅ LightGBM version: {lgb.__version__}")
    
    # Check build info
    try:
        build_info = lgb.basic.Booster().get_params()
        print(f"  Build info available: {bool(build_info)}")
    except:
        pass
    
    # Check if GPU is available
    print()
    print("Testing GPU devices:")
    print("-" * 70)
    
    # Generate test data
    X = np.random.rand(10000, 100).astype(np.float32)
    y = np.random.rand(10000).astype(np.float32)
    
    # Test CUDA
    print("\n1. Testing CUDA device...")
    try:
        model_cuda = lgb.LGBMRegressor(device='cuda', n_estimators=50, verbose=1, n_jobs=1)
        print("   Training with CUDA...")
        model_cuda.fit(X, y)
        print("   ✅ CUDA works!")
        
        # Check if model actually used GPU
        if hasattr(model_cuda, 'device'):
            print(f"   Device attribute: {model_cuda.device}")
    except Exception as e:
        print(f"   ❌ CUDA failed: {e}")
        print("   This usually means:")
        print("     - LightGBM not built with CUDA support")
        print("     - CUDA libraries not in LD_LIBRARY_PATH")
        print("     - GPU not accessible")
    
    # Test OpenCL
    print("\n2. Testing OpenCL device...")
    try:
        model_opencl = lgb.LGBMRegressor(device='gpu', n_estimators=50, verbose=1, n_jobs=1)
        print("   Training with OpenCL...")
        model_opencl.fit(X, y)
        print("   ✅ OpenCL works!")
        
        if hasattr(model_opencl, 'device'):
            print(f"   Device attribute: {model_opencl.device}")
    except Exception as e:
        print(f"   ❌ OpenCL failed: {e}")
    
    # Test CPU (baseline)
    print("\n3. Testing CPU device (baseline)...")
    try:
        model_cpu = lgb.LGBMRegressor(device='cpu', n_estimators=50, verbose=-1, n_jobs=4)
        print("   Training with CPU...")
        model_cpu.fit(X, y)
        print("   ✅ CPU works!")
    except Exception as e:
        print(f"   ❌ CPU failed: {e}")
    
    print()
    print("=" * 70)
    print("Recommendations:")
    print("=" * 70)
    print("If GPU tests failed:")
    print("  1. Check if LightGBM was built with GPU support:")
    print("     - pip install lightgbm --install-option=--gpu")
    print("     - Or use conda: conda install -c conda-forge lightgbm-gpu")
    print("  2. Verify CUDA is accessible:")
    print("     - nvidia-smi should show GPU")
    print("     - Check LD_LIBRARY_PATH includes CUDA libs")
    print("  3. For small datasets, CPU may be faster (GPU overhead)")
    print()
    print("If GPU works but seems slow:")
    print("  - GPU is most efficient for large datasets (>100k samples)")
    print("  - Small datasets may be faster on CPU due to GPU overhead")
    print("  - Check GPU utilization with: nvidia-smi -l 1")
    print()
    
except ImportError:
    print("❌ LightGBM not installed")
    print("   Install with: pip install lightgbm")
    sys.exit(1)

